import os
import gc
import torch
import monai
import torch.nn as nn
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from registration_svf.losses.regularisation.jacobian import Jacobianloss, compute_jacobian_determinant_3d, compute_jacobian_determinant, jacobian_determinant_3d
from registration_svf.registration import RegistrationModule
from registration_svf.utils.grid_utils import warp, compose, displacement2grid
from registration_svf.losses.regularisation.magnitude import  MagnitudeLoss
from registration_svf.losses.regularisation.gradient import  Grad3d


class RegistrationTrainingModule(pl.LightningModule):
    """
    Registration training module for 3D image registration
    """
    def __init__(self, model : RegistrationModule, learning_rate: float= 0.001, save_path: str = "./",
                 lambda_sim=1.0, lambda_seg=0.0, lambda_reg=0.0):
        """
        Registration training module for 3D image registration
        :param model: RegistrationModule
        :param learning_rate: Learning rate for the optimizer
        :param save_path: Path to save the model
        :param lambda_sim: Loss factor - intensity image
        :param lambda_seg: Loss factor - segmentation map
        :param lambda_reg: Loss factor - segmentation map
        """

        super().__init__()
        self.reg_model = model
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.dice_metric = DiceMetric(include_background=True, reduction="none", ignore_empty=False)
        self.dice_max = 0
        self.sim_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.seg_loss = nn.MSELoss()
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.lambda_reg = lambda_reg

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the registration module
        :param source: Source image
        :param target: Target image
        :return: Flow field if RegistrationModule else
        """
        return self.reg_model(torch.cat([source, target], dim=1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.dice_max = 0
        self.reg_model.train()

    def training_step(self, batch) -> torch.Tensor:
        source_img = batch['source_image'][tio.DATA].float().to(self.device)
        target_img = batch['target_image'][tio.DATA].float().to(self.device)
        source_label = batch['source_label'][tio.DATA].float().to(self.device)
        target_label = batch['target_label'][tio.DATA].float().to(self.device)

        loss_errors = torch.zeros([3], device=self.device)
        velocity = self.forward(source_img, target_img)
        t = torch.rand(1, device=self.device)
        disp_j = self.reg_model.velocity2displacement(velocity * t)
        disp_i = self.reg_model.velocity2displacement(velocity * (t - 1.))
        if self.lambda_sim > 0:
            jw = warp(source_img, disp_j)
            iw = warp(target_img, disp_i)
            loss_errors[0] = self.lambda_sim * self.sim_loss(jw, iw)
        if self.lambda_seg > 0:
            jw = warp(source_label, disp_j)
            iw = warp(target_label, disp_i)
            loss_errors[1] = self.lambda_seg * (self.seg_loss(jw, iw))
        # Gradient loss
        if self.lambda_grad > 0:
            t = torch.rand(1, device=self.device)
            v = t * velocity
            flow_j = self.reg_model.velocity2displacement(v)
            v = (t - 1.) * velocity
            flow_i = self.reg_model.velocity2displacement(v)
            flow_v = self.reg_model.velocity2displacement(velocity)
            flow__v = self.reg_model.velocity2displacement(-velocity)
            flow_vi = compose(flow_v, flow_i)
            flow__vj = compose(flow__v, flow_j)
            phi_vi = displacement2grid(flow_vi)
            phi__vj = displacement2grid(flow__vj)
            phi_j = displacement2grid(flow_j)
            phi_i = displacement2grid(flow_i)
            loss_errors[2] += torch.nn.MSELoss()(phi_vi, phi_j) + torch.nn.MSELoss()(phi__vj, phi_i)
            loss_errors[2] = self.lambda_grad * loss_errors[2]
        loss = loss_errors.sum()
        self.log_dict({
            "Global loss": loss,
            "Intensity" : loss_errors[0],
            "Segmentation": loss_errors[1],
            "Regulation": loss_errors[2]
        }, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss



    def on_train_epoch_end(self):
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_model.pth")
        gc.collect()

    def validation_step(self, batch):
        source_img = batch['source_image'][tio.DATA].float().to(self.device)
        target_img = batch['target_image'][tio.DATA].float().to(self.device)
        source_label = batch['source_label'][tio.DATA].float().to(self.device)
        target_label = batch['target_label'][tio.DATA].float().to(self.device)
        with torch.no_grad():
            flow = self.forward(source_img, target_img)
            disp = self.reg_model.velocity2displacement(flow)
            neg_det_j = torch.nn.functional.relu(-compute_jacobian_determinant_3d(disp))
            neg_det_j = (neg_det_j > 0).sum().item()
            if self.jacobian_nb_value < neg_det_j:
                self.jacobian_nb_value = neg_det_j
        warped_source_label = torch.argmax(warp(source_label, disp), dim=1)
        warped_source_label = tio.LabelMap(tensor=warped_source_label.cpu().int(), affine=batch['source_label']['affine'].squeeze())
        one_hot = tio.OneHot()
        warped_source_label = one_hot(warped_source_label)[tio.DATA].to(self.device).unsqueeze(0)

        target_one_hot = tio.LabelMap(tensor=torch.argmax(target_label, dim=1), affine=batch['target_label']['affine'].squeeze())
        target_one_hot = one_hot(target_one_hot)[tio.DATA].to(self.device).unsqueeze(0)
        self.dice_metric(warped_source_label, target_one_hot)


    def on_validation_epoch_start(self) -> None:
        self.jacobian_nb_value = 0


    def on_validation_end(self) -> None:
        dice_scores = self.dice_metric.get_buffer()
        self.dice_metric.reset()
        mean_dices =  dice_scores.mean().item()
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.reg_model.state_dict(), self.save_path + "/best_model.pth")
        self.logger.experiment.add_scalar("Mean dice", mean_dices, self.current_epoch)
        self.logger.experiment.add_scalar("Mean cortex", dice_scores[:, 2].mean().item(), self.current_epoch)
        self.logger.experiment.add_scalar("Worst jacobian", self.jacobian_nb_value, self.current_epoch)
        print(f"Validation mean dice: {mean_dices}, cortex dice: {dice_scores[:, 2].mean().item()}, worst jacobian: {self.jacobian_nb_value}")
        gc.collect()

    def save(self, path: str):
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.reg_model.state_dict(), os.path.join(path, "model.pth"))
