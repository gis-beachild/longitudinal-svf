"""PyTorch Lightning training module for longitudinal SVF registration (alternating SVF/temporal-MLP optimization).

Author: Fl0rian
"""
import os
from typing import Tuple, Union

import monai.losses
import torch
import pytorch_lightning as pl
import torchio as tio
import matplotlib.pyplot as plt
from torch import Tensor
from monai.metrics import DiceMetric # type: ignore
from torchvision.transforms.functional import rotate
from losses.jacobian import compute_jacobian_determinant_3d
from utils.grid_utils import compose, warp, displacement2grid, plt_grid
from utils.utils import normalize_to_0_1
from src.modules.longitudinal_model import LongitudinalDeformation
import random

class LongitudinalTrainingModule(pl.LightningModule):
    """LightningModule for 4D Longitudinal Deformation Estimation"""

    def __init__(
            self,
            model: LongitudinalDeformation,
            learning_rate_svf : float = 1e-3,
            learning_rate_mlp: float = 1e-3,
            save_path: str = "./",
            num_inter_by_epoch: int = 1,
            lambda_reg: float = 0.05,
            lambda_seg: float = 0.05,
            lambda_sim: float = 0.05
    ) -> None:
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.learning_rate_svf = learning_rate_svf
        self.learning_rate_mlp = learning_rate_mlp
        self.save_path = save_path
        self.num_inter_by_epoch = num_inter_by_epoch
        self.dice_metric = DiceMetric(include_background=True, reduction="sum", ignore_empty=False)
        self.dice_max = 0.0
        self.automatic_optimization = False
        self.loss_seg = torch.nn.MSELoss(reduction='mean')
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21) # type: ignore

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """Predict the stationary velocity field between ``source`` and ``target``."""
        return self.model((source, target))

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
        """Build the SVF optimizer, plus a second optimizer for the temporal MLP when ``time_mode == 'mlp'``."""
        opt_svf = torch.optim.Adam(self.model.svf_model.parameters(), lr=self.learning_rate_svf)
        if self.model.time_mode == 'mlp':
            opt_mlp = torch.optim.Adam(self.model.temp_model.parameters(), lr=self.learning_rate_mlp) # type: ignore
            return (opt_svf, opt_mlp)
        else:
            return opt_svf

    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def on_train_start(self) -> None:
        self.subjt_t0 = self.trainer.train_dataloader.dataset[0] # type: ignore
        self.subjt_t1 = self.trainer.train_dataloader.dataset[-1] # type: ignore
        for i in range(self.trainer.train_dataloader.dataset.num_subjects): # type: ignore
            subject = self.trainer.train_dataloader.dataset[i] # type: ignore
            if subject['age'] == 0:
                self.subjt_t0 = subject
            if subject['age'] == 1:
                self.subjt_t1 = subject
        self.subjt_t0['image'].data = self.subjt_t0['image'].data.float().unsqueeze(dim=0).to(self.device)
        self.subjt_t0['label'].data = self.subjt_t0['label'].data.float().unsqueeze(dim=0).to(self.device)
        self.subjt_t1['image'].data = self.subjt_t1['image'].data.float().unsqueeze(dim=0).to(self.device)
        self.subjt_t1['label'].data = self.subjt_t1['label'].data.float().unsqueeze(dim=0).to(self.device)
        self.inp = torch.cat([self.subjt_t1['image'].data.float(), self.subjt_t1['image'].data.float()], dim=1)


    def train_mlp(self, mlp_opt) -> None:
        """Take one manual-optimization pass over the temporal MLP, using the current (detached) SVF velocity."""
        one = torch.tensor(1.0, device=self.device)

        with torch.no_grad():
            velocity = self.model.forward(self.inp).detach()
        for i in random.sample(range(1, self.trainer.train_dataloader.dataset.num_subjects), self.trainer.train_dataloader.dataset.num_subjects - 1): # type: ignore
            loss = torch.zeros((), device=self.device)
            k = self.trainer.train_dataloader.dataset[i] # type: ignore
            t = torch.tensor([k['age']], device=self.device)
            time = self.model.encode_time(t)
            disp_j = self.model.svf_model.velocity2displacement(velocity * time)  # φ_time
            disp_i = self.model.svf_model.velocity2displacement(velocity * (time - one))  # φ_{time-1}
            if self.lambda_sim > 0:
                jw = warp(self.subjt_t0['image'].data, disp_j)  # t0 → time
                iw = warp(self.subjt_t1['image'].data, disp_i)  # t1 → time
                tgt_img = k["image"].data.to(self.device).unsqueeze(dim=0)
                loss += self.lambda_sim * (self.loss_sim(jw, tgt_img) + self.loss_sim(iw, tgt_img))
            if self.lambda_seg > 0:
                jw_lab = warp(self.subjt_t0['label'].data, disp_j)
                iw_lab = warp(self.subjt_t1['label'].data, disp_i)
                tgt_lab = k["label"].data.to(self.device).unsqueeze(dim=0)
                loss += self.lambda_seg * (self.loss_seg(jw_lab, tgt_lab) + self.loss_seg(iw_lab, tgt_lab))
            mlp_opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            mlp_opt.step()
        self.log_dict(
            {
                "Loss MLP": loss,
            },
            prog_bar=True,
        )

    def train_svf(self, svf_opt) -> None:
        """Take one manual-optimization pass over the SVF backbone (intensity, segmentation, and inverse-consistency losses)."""
        one = torch.tensor(1.0, device=self.device)

        for i in random.sample(range(1, self.trainer.train_dataloader.dataset.num_subjects), self.trainer.train_dataloader.dataset.num_subjects - 1): # type: ignore
            int_loss = torch.zeros((), device=self.device)
            seg_loss = torch.zeros((), device=self.device)
            reg_loss = torch.zeros((), device=self.device)
            velocity = self.model.forward(self.inp)  # shape: (B,C,*,*,*)
            flow_v = self.model.svf_model.velocity2displacement(velocity)  # φ_{+1}
            flow__v = self.model.svf_model.velocity2displacement(-velocity)  # φ_{-1}
            k = self.trainer.train_dataloader.dataset[i] # type: ignore
            t = torch.tensor([k['age']], device=self.device)
            with torch.no_grad():
                time = self.model.encode_time(t).detach()
            disp_j = self.model.svf_model.velocity2displacement(velocity * time)
            disp_i = self.model.svf_model.velocity2displacement(velocity * (time - one))
            if self.lambda_sim > 0:
                jw = warp(self.subjt_t0['image'].data, disp_j)
                iw = warp(self.subjt_t1['image'].data, disp_i)
                tgt_img = k["image"].data.to(self.device).unsqueeze(dim=0)
                int_loss += self.loss_sim(jw, tgt_img) + self.loss_sim(iw, tgt_img)
            if self.lambda_seg > 0:
                jw_lab = warp(self.subjt_t0['label'].data.float(), disp_j)
                tgt_lab = k["label"].data.to(self.device).unsqueeze(dim=0).float()
                seg_loss += self.loss_seg(jw_lab, tgt_lab)
            flow_vi = compose(flow_v, disp_i)
            flow_vj = compose(flow__v, disp_j)
            grid_vi = displacement2grid(flow_vi)
            grid_vj = displacement2grid(flow_vj)
            reg_loss += torch.mean((grid_vi - grid_vj) ** 2)
            svf_opt.zero_grad(set_to_none=True)
            loss = self.lambda_sim * int_loss + self.lambda_seg * seg_loss + 100 * reg_loss
            self.manual_backward(loss)
            svf_opt.step()
        self.log_dict(
            {
                "Loss Global": loss,
                "Loss SVF-Int": int_loss,
                "Loss SVF-Seg": seg_loss,
                "Loss Reg": reg_loss,
            },
            prog_bar=True,
        )

    def training_step(self, _) -> None:
        """Run one manual-optimization step over the SVF backbone, then (in mlp mode) the temporal MLP."""
        mlp_opt = None
        if self.model.time_mode == 'mlp':
            svf_opt, mlp_opt = self.optimizers() # type: ignore
        else:
            svf_opt = self.optimizers()
        self.train_svf(svf_opt)
        # Start the MLP training after few epoch to stabilize the training
        if self.model.time_mode == 'mlp':
            self.train_mlp(mlp_opt)


    def on_train_epoch_end(self) -> None:
        """Checkpoint the full longitudinal model at the end of every training epoch."""
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "last_model.pth"))

    def validation_step(self, _) -> None:
        """Warp the ``t0`` subject to every subject's age, log Dice/negative-Jacobian, and save debug volumes."""
        if self.trainer.val_dataloaders.dataset.num_subjects < 2: # type: ignore
            raise ValueError("Validation dataset must contain at least two subjects with age 0 and 1.")
        subjt_t0 = self.trainer.val_dataloaders.dataset[0] # type: ignore
        subjt_t1 = self.trainer.val_dataloaders.dataset[1] # type: ignore

        for i in range(self.trainer.val_dataloaders.dataset.num_subjects): # type: ignore
            subject = self.trainer.val_dataloaders.dataset[i] # type: ignore
            if subject['age'] == 0:
                subjt_t0 = subject
            if subject['age'] == 1:
                subjt_t1 = subject
        max_jac_neg = 0.0
        input = torch.cat([subjt_t0['image'].data.float(), subjt_t1['image'].data.float()],
                          dim=0).unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            velocity = self.model.forward(input)
        for subject in self.trainer.val_dataloaders.dataset: # type: ignore
            time = self.model.encode_time(torch.tensor([subject['age']], device=self.device))
            disp = self.model.svf_model.velocity2displacement(time * velocity)
            warped_label = torch.argmax(warp(subjt_t0["label"].data.unsqueeze(0).to(self.device).float(), disp),
                                        dim=1)
            warped_label = tio.LabelMap(tensor=warped_label, affine=subjt_t0['label'].affine)
            warped_label = tio.transforms.OneHot()(warped_label).data.unsqueeze(0)
            target = tio.LabelMap(tensor=torch.argmax(subject["label"].data.unsqueeze(0).to(self.device), dim=1),
                                  affine=subjt_t0['label'].affine)
            target_one_hot = tio.transforms.OneHot()(target).data.unsqueeze(0)
            print(time)
            self.dice_metric(
                warped_label,
                target_one_hot
            )

        neg_det_j = torch.nn.functional.relu(-compute_jacobian_determinant_3d(disp))
        neg_det_j = (neg_det_j > 0).sum().item()
        if max_jac_neg < neg_det_j:
            max_jac_neg = neg_det_j
        warped_image = warp(subjt_t0["image"].data.unsqueeze(0).float().to(self.device), disp)
        tio.LabelMap(tensor=torch.argmax(warped_label, dim=1).cpu().numpy(),
                     affine=subjt_t0['label'].affine).save(
            os.path.join(self.save_path, f"label_warped_source{i}.nii.gz"))
        tio.ScalarImage(tensor=warped_image.squeeze(0).cpu().numpy(), affine=subjt_t0['image'].affine).save(
            os.path.join(self.save_path, f"image_warped_source{i}.nii.gz"))
        disp_voxel = self.model.svf_model.velocity2displacement(velocity)
        disp_phys = disp_voxel * torch.tensor(subjt_t0.spacing,device=self.device).view(1,3,1,1,1)
        tio.ScalarImage(tensor=disp_phys.squeeze(0).cpu().numpy(), affine=subjt_t0['image'].affine).save(
            os.path.join(self.save_path, f"forward_dvf{i}.nii.gz"))
        # Metric calculation
        dice_scores = self.dice_metric.aggregate(reduction="none")
        self.dice_metric.reset()
        mean_dice = dice_scores.mean().item() # type: ignore
        plt.figure(figsize=(10, 6))  # Set width and height in inches
        plt.plot(dice_scores.mean(dim=1).cpu().numpy()) # type: ignore
        plt.ylim(0, 1)
        plt.title("mDice score")
        plt.xlabel('')  # Removes X-axis label
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.save_path + "/mDice.png")
        plt.close()
        self.dice_metric.reset()
        # Save best model
        if self.dice_max < mean_dice:
            self.dice_max = mean_dice
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_best.pth"))
            self.log("Dice max", self.dice_max, prog_bar=True, on_epoch=True, sync_dist=True)

        self.log_dict({
            "Mean dice": mean_dice,
            "Negative Jacobian": max_jac_neg
        }, prog_bar=True, on_epoch=True, sync_dist=True)
        xyz = displacement2grid(disp)
        _, D, H, W, _ = xyz.shape
        mid_slice = normalize_to_0_1(warped_image.squeeze())[..., W // 2:W // 2 + 1].permute(2, 0, 1).cpu()
        self.logger.experiment.add_image("Registered image", rotate(mid_slice, 90, expand=True).numpy(), self.current_epoch) # type: ignore

    def save(self, path: str) -> None:
        """Saves the model state dicts to disk."""
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))

