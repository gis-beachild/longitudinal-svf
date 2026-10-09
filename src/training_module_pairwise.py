"""PyTorch Lightning training module for pairwise SVF registration (intensity, segmentation, and regularization losses).

Author: Fl0rian
"""
import os
import gc
import torch
import monai
import torch.nn as nn
import torchio as tio
import pytorch_lightning as pl
from monai.metrics import DiceMetric # type: ignore
from losses.jacobian import compute_jacobian_determinant_3d
from modules.svf_registration import SVFRegistrationModule
from utils.grid_utils import warp, compose, displacement2grid


class RegistrationTrainingModule(pl.LightningModule):
    """
    Registration training module for 3D image registration
    """
    def __init__(self, model : SVFRegistrationModule, learning_rate: float= 0.001, save_path: str = "./",
                 lambda_sim: float = 1.0, lambda_seg: float = 0.0, lambda_reg: float = 0.0) -> None:
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
        self.sim_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21) # type: ignore
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

    def training_step(self, batch: dict) -> torch.Tensor:
        """Compute the combined intensity/segmentation/regularization loss for one source-target pair."""
        source_img = batch['source_image'][tio.DATA].float()
        target_img = batch['target_image'][tio.DATA].float()
        source_label = batch['source_label'][tio.DATA].float()
        target_label = batch['target_label'][tio.DATA].float()

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
        if self.lambda_reg > 0:
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
            loss_errors[2] = self.lambda_reg * loss_errors[2]
        loss = loss_errors.sum()
        self.log_dict({
            "Global loss": loss,
            "Intensity" : loss_errors[0],
            "Segmentation": loss_errors[1],
            "Regulation": loss_errors[2]
        }, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def on_train_epoch_end(self) -> None:
        """Checkpoint the registration model at the end of every training epoch."""
        torch.save(self.reg_model.state_dict(), self.save_path + "/last_model.pth")
        gc.collect()

    def validation_step(self, batch: dict) -> None:
        """Warp the source label with the predicted flow and accumulate its Dice against the target label."""
        source_img = batch['source_image'][tio.DATA].float()
        target_img = batch['target_image'][tio.DATA].float()
        source_label = batch['source_label'][tio.DATA].float()
        target_label = batch['target_label'][tio.DATA].float()
        with torch.no_grad():
            flow = self.forward(source_img, target_img)
            disp = self.reg_model.velocity2displacement(flow)
            neg_det_j = torch.nn.functional.relu(-compute_jacobian_determinant_3d(disp))
            neg_det_j = (neg_det_j > 0).sum().item()
            if self.jacobian_nb_value < neg_det_j:
                self.jacobian_nb_value = neg_det_j
        warped_labels = torch.argmax(warp(source_label, disp), dim=1)
        warped_one_hot = torch.nn.functional.one_hot(
            warped_labels, num_classes=source_label.shape[1]
        ).movedim(-1, 1).float()
        self.dice_metric(warped_one_hot, target_label)


    def on_validation_epoch_start(self) -> None:
        self.jacobian_nb_value = 0


    def on_validation_end(self) -> None:
        dice_scores = self.dice_metric.get_buffer()
        self.dice_metric.reset()

        mean_dices =  dice_scores.mean().item() # type: ignore
        if self.dice_max < mean_dices:
            self.dice_max = mean_dices
            torch.save(self.reg_model.state_dict(), self.save_path + "/best_model.pth")
        self.logger.experiment.add_scalar("Mean dice", mean_dices, self.current_epoch) # type: ignore
        self.logger.experiment.add_scalar("Worst jacobian", self.jacobian_nb_value, self.current_epoch) # type: ignore
        print(f"Validation mean dice: {mean_dices}, worst jacobian: {self.jacobian_nb_value}")
        gc.collect()

    def save(self, path: str) -> None:
        """
        Save the model
        :param path: Path to save the model
        """
        torch.save(self.reg_model.state_dict(), os.path.join(path, "model.pth"))
