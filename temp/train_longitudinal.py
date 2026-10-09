import os
import random
import torch
import pytorch_lightning as pl
import torchio as tio
import matplotlib.pyplot as plt
from torch import Tensor
from monai.metrics import DiceMetric
from torchvision.transforms.functional import rotate
from losses.regularisation.jacobian import Jacobianloss, compute_jacobian_determinant_3d
from losses.regularisation.regularization import MagnitudeLoss, Grad3d
from utils.grid_utils import compose, warp, displacement2grid
from utils.visualize import plt_grid
from src.utils.utils import normalize_to_0_1
from modules.longitudinal_deformation import OurLongitudinalDeformation
from src.utils.utils import compute_sdf_3d
class LongitudinalTrainingModule(pl.LightningModule):
    """LightningModule for 4D Longitudinal Deformation Estimation"""

    def __init__(
        self,
        model: OurLongitudinalDeformation,
        learning_rate: float = 1e-3,
        save_path: str = "./",
        num_inter_by_epoch: int = 1,
        penalize: str = 'v',
        lambda_reg: float = 0.05
    ):
        super().__init__()
        if penalize not in ['v', 'd']:
            raise ValueError("penalize must be 'v' or 'd'")

        self.model = model
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.penalize = penalize
        self.num_inter_by_epoch = num_inter_by_epoch
        self.dice_metric = DiceMetric(include_background=True, reduction="sum")
        self.dice_max = 0.0

    def forward(self, source: Tensor, target: Tensor):
        return self.model((source, target))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def on_train_epoch_start(self):
        self.model.train()

    def on_validation_epoch_start(self):
        self.model.eval()

    def training_step(self, _):
        subject_t0 = self.trainer.train_dataloader.dataset[self.trainer.train_dataloader.dataset.index_t0_subject]
        subject_t1 = self.trainer.train_dataloader.dataset.get_subject_based_on_subject_transform(self.trainer.train_dataloader.dataset.index_t1_subject, subject_t0)
        seg_loss =  torch.tensor(0.0, device=self.device)
        reg_loss = torch.tensor(0.0, device=self.device)
        velocity = self.model(torch.cat([subject_t0['image'][tio.DATA], subject_t1['image'][tio.DATA]], dim=0).unsqueeze(0).to(self.device))
        index = random.sample(range(1, self.trainer.train_dataloader.dataset.num_subjects), 4)

        one_velocity = self.model.encode_time(torch.Tensor([1.]).to(self.device)) * velocity
        minus_one_velocity =  self.model.encode_time(torch.Tensor([-1.]).to(self.device)) * velocity


        for i in index:
            subject_k = self.trainer.train_dataloader.dataset.get_subject_based_on_subject_transform(i, subject_t0)
            timed_velocity = self.model.encode_time(torch.Tensor([subject_k['age']]).to(self.device)) * velocity
            flow_J = self.model.reg_model.velocity2displacement(timed_velocity)
            timed_velocity = self.model.encode_time(torch.Tensor([subject_k['age'] - 1]).to(self.device)) * velocity
            flow_I = self.model.reg_model.velocity2displacement(timed_velocity)

            Iw = warp(subject_t1['label'][tio.DATA].unsqueeze(0).float().to(self.device), flow_I)
            Jw = warp(subject_t0['label'][tio.DATA].unsqueeze(0).float().to(self.device), flow_J)

            seg_loss += torch.nn.MSELoss()(subject_k['label'][tio.DATA].unsqueeze(0).float().to(self.device), Jw) + \
                        torch.nn.MSELoss()(subject_k['label'][tio.DATA].unsqueeze(0).float().to(self.device), Iw)

            flow_VI = warp(one_velocity, flow_I) + flow_I
            flow__VJ = warp(minus_one_velocity, flow_J) + flow_J

            reg_loss += 0.5 * (torch.nn.MSELoss()(flow__VJ, flow_I) +
                               torch.nn.MSELoss()(flow_VI, flow_J))
        reg_loss *= self.lambda_reg
        loss = seg_loss + reg_loss

        self.log_dict({
            "Global loss": loss,
            "Segmentation": seg_loss,
            "Regu": reg_loss
        }, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "last_model_reg.pth"))

    def validation_step(self, _):
        subject_t0 = self.trainer.val_dataloaders.dataset[self.trainer.train_dataloader.dataset.index_t0_subject]
        subject_t1 = self.trainer.val_dataloaders.dataset.get_subject_based_on_subject_transform(self.trainer.train_dataloader.dataset.index_t1_subject, subject_t0)
        with torch.no_grad():
            velocity = self.model(
                torch.cat([subject_t0['image'][tio.DATA], subject_t1['image'][tio.DATA]], dim=0).unsqueeze(0).to(self.device))
            for i in range(self.trainer.train_dataloader.dataset.num_subjects):
                subject_k = self.trainer.val_dataloaders.dataset.get_subject_based_on_subject_transform(i, subject_t0)
                label_k = subject_k['label'][tio.DATA].to(self.device)
                age_t = torch.tensor([subject_k['age']], device=self.device)
                disp = self.model.reg_model.velocity2displacement(self.model.encode_time(age_t) * velocity)
                warped = warp(subject_t0["label"][tio.DATA].unsqueeze(0).to(self.device).float(), disp, mode='nearest')
                self.dice_metric(
                    torch.nn.functional.one_hot(warped.squeeze().long(), 20).permute(3, 0, 1, 2).unsqueeze(0),
                    torch.nn.functional.one_hot(label_k.squeeze().long(), 20).permute(3, 0, 1, 2).to(self.device).unsqueeze(0)
                )

            final_disp = self.model.reg_model.velocity2displacement(self.model.encode_time(torch.tensor([1.0], device=self.device)) * velocity)
            warped_label = warp(subject_t0["label"][tio.DATA].unsqueeze(0).float().to(self.device), final_disp, mode='nearest')
            warped_image = warp(subject_t0["image"][tio.DATA].unsqueeze(0).float().to(self.device), final_disp)

            # Metric calculation
            dice_scores = self.dice_metric.aggregate(reduction="none")
            self.dice_metric.reset()
            mean_dice = dice_scores.mean().item()
            cortex_mean = dice_scores[:, 3:5].mean().item()
            jac_neg = (compute_jacobian_determinant_3d(final_disp.squeeze(0)) < 0).sum().cpu().item()

            plt.figure(figsize=(10, 6))  # Set width and height in inches
            plt.plot(dice_scores.mean(dim=1).cpu().numpy())
            plt.ylim(0, 1)
            plt.title("mDice score")
            plt.xlabel('')  # Removes X-axis label
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(self.save_path + "/mDice.png")
            self.dice_metric.reset()

            # Save best model
            if self.dice_max < mean_dice:
                self.dice_max = mean_dice
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "model_reg_best.pth"))
                self.log("Dice max", self.dice_max, prog_bar=True, on_epoch=True, sync_dist=True)

            self.log_dict({
                "Mean dice": mean_dice,
                "Cortex mean": cortex_mean,
                "Negative Jacobian": jac_neg
            }, prog_bar=True, on_epoch=True, sync_dist=True)

            # Save visuals and nifti outputs

            xyz = displacement2grid(final_disp, grid_normalize=True)
            xyz = xyz.permute(0, 2, 3, 4, 1)

            _, D, H, W, _ = xyz.shape
            xy = torch.cat(
                [xyz[0, :, :, D // 2, 0].unsqueeze(-1), xyz[0, :, :, D // 2, 1].unsqueeze(-1)],
                dim=-1)
            grid_plot = plt_grid(xy=xy.cpu().numpy(), factor=1)
            grid_plot.savefig(os.path.join(self.save_path, "grid.png"))


            mid_slice = normalize_to_0_1(warped_image.squeeze())[..., W // 2:W // 2+1].permute(2, 0, 1).cpu()
            self.logger.experiment.add_image("Registered image", rotate(mid_slice,90, expand=True).numpy(), self.current_epoch)
            self.logger.experiment.add_figure("Forward grid", grid_plot, self.current_epoch)

            # Save warps
            tio.LabelMap(tensor=warped_label.squeeze(0).cpu().numpy(), affine=subject_t0['label'].affine).save(
                os.path.join(self.save_path, "label_warped_source.nii.gz"))
            tio.ScalarImage(tensor=warped_image.squeeze(0).cpu().numpy(), affine=subject_t0['image'].affine).save(
                os.path.join(self.save_path, "image_warped_source.nii.gz"))
            disp_phys = final_disp * torch.tensor(subject_t0.spacing, device=self.device).view(1, 3, 1, 1, 1)
            tio.ScalarImage(tensor=disp_phys.squeeze(0).cpu().numpy(), affine=subject_t0['image'].affine).save(
                os.path.join(self.save_path, "forward_dvf.nii.gz"))

    def save(self, path: str):
        """Saves the model state dicts to disk."""
        torch.save(self.model.reg_model.state_dict(), os.path.join(path, "reg_model.pth"))
        if self.model.mode == 'mlp':
            torch.save(self.mlp_model.state_dict(), os.path.join(path, "temporal_model.pth"))