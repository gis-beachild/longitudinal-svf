"""Lightning training for longitudinal SVF registration on sequence batches."""
import os
import random

import matplotlib.pyplot as plt
import monai.losses
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchio as tio
from monai.metrics import DiceMetric
from torch import Tensor

from losses.jacobian import compute_jacobian_determinant_3d
from utils.grid_utils import compose, displacement2grid, warp
from src.modules.longitudinal_model import LongitudinalDeformation


class LongitudinalTrainingModule(pl.LightningModule):
    """Alternate SVF and temporal-MLP updates for each loaded subject sequence."""

    def __init__(self, model: LongitudinalDeformation,
                 learning_rate_svf: float = 1e-3, learning_rate_mlp: float = 1e-3,
                 save_path: str = './', num_inter_by_epoch: int = 1,
                 lambda_reg: float = 0.05, lambda_seg: float = 0.05,
                 lambda_sim: float = 0.05) -> None:
        super().__init__()
        self.model = model
        self.learning_rate_svf = learning_rate_svf
        self.learning_rate_mlp = learning_rate_mlp
        self.save_path = save_path
        self.num_inter_by_epoch = num_inter_by_epoch
        self.lambda_reg = lambda_reg
        self.lambda_seg = lambda_seg
        self.lambda_sim = lambda_sim
        self.automatic_optimization = False
        self.loss_seg = torch.nn.MSELoss()
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.dice_metric = DiceMetric(include_background=True, reduction='mean', ignore_empty=False)
        self.dice_max = 0.0
        os.makedirs(save_path, exist_ok=True)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return self.model(torch.cat((source, target), dim=1))

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(self.model.svf_model.parameters(), lr=self.learning_rate_svf)]
        if self.model.time_mode == 'mlp':
            optimizers.append(torch.optim.Adam(self.model.mlp_model.parameters(), lr=self.learning_rate_mlp))
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999) for opt in optimizers]
        return optimizers, schedulers

    def _prepare_batch(self, batch):
        """Convert (1,T,1,D,H,W) batches to sequences and encode labels consistently."""
        images, segs, ages = batch
        if images.ndim != 6 or images.shape[0] != 1 or images.shape[2] != 1:
            raise ValueError('Expected images with shape (1, T, 1, D, H, W)')
        images = images[0].to(self.device).float()
        ages = ages[0].to(self.device).float()
        if ages.ndim != 1 or ages.numel() != images.shape[0] or ages.numel() < 2:
            raise ValueError('Each sequence must contain at least two images and matching ages')
        if not torch.isfinite(ages).all() or torch.any(ages[1:] < ages[:-1]) or ages[-1] <= ages[0]:
            raise ValueError('Sequence ages must be finite, sorted, and span a nonzero interval')
        # The predicted velocity connects this sequence's endpoints, not global dataset ages.
        times = (ages - ages[0]) / (ages[-1] - ages[0])
        labels = None
        if segs.numel():
            if segs.shape != batch[0].shape:
                raise ValueError('Segmentations must have the same shape as images')
            segs = segs[0, :, 0].to(self.device).long()
            labels = F.one_hot(segs, num_classes=int(segs.max().item()) + 1).movedim(-1, 1).float()
        return images, labels, times

    def _data_losses(self, images, labels, index, disp_start, disp_end):
        intensity = images.new_zeros(())
        segmentation = images.new_zeros(())
        if self.lambda_sim > 0:
            target = images[index:index + 1]
            intensity = (self.loss_sim(warp(images[:1], disp_start), target)
                         + self.loss_sim(warp(images[-1:], disp_end), target))
        if self.lambda_seg > 0 and labels is not None:
            target = labels[index:index + 1]
            segmentation = (self.loss_seg(warp(labels[:1], disp_start), target)
                            + self.loss_seg(warp(labels[-1:], disp_end), target))
        return intensity, segmentation

    def train_svf(self, optimizer, images, labels, times) -> None:
        schedulers = self.lr_schedulers()
        scheduler = schedulers[0] if isinstance(schedulers, (list, tuple)) else schedulers
        for index in random.sample(range(1, len(times)), len(times) - 1):
            velocity = self(images[:1], images[-1:])
            with torch.no_grad():
                time = self.model.encode_time(times[index:index + 1])
            disp_start = self.model.svf_model.velocity2displacement(velocity * time)
            disp_end = self.model.svf_model.velocity2displacement(velocity * (time - 1))
            intensity, segmentation = self._data_losses(images, labels, index, disp_start, disp_end)
            regularization = velocity.new_zeros(())
            if self.lambda_reg > 0:
                forward = self.model.svf_model.velocity2displacement(velocity)
                backward = self.model.svf_model.velocity2displacement(-velocity)
                grid_start = displacement2grid(compose(forward, disp_end))
                grid_end = displacement2grid(compose(backward, disp_start))
                regularization = (grid_start - grid_end).square().mean()
            loss = self.lambda_sim * intensity + self.lambda_seg * segmentation + self.lambda_reg * regularization
            if loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                optimizer.step()
                scheduler.step()
            self.log_dict({'Loss Global': loss.detach(), 'Loss SVF-Int': intensity.detach(),
                           'Loss SVF-Seg': segmentation.detach(), 'Loss Reg': regularization.detach()},
                          prog_bar=True, batch_size=1)

    def train_mlp(self, optimizer, images, labels, times) -> None:
        # Endpoint times are fixed by the monotonic MLP and provide no temporal supervision.
        indices = [i for i in range(1, len(times) - 1) if 0 < times[i] < 1]
        if not indices or (self.lambda_sim <= 0 and (self.lambda_seg <= 0 or labels is None)):
            return
        with torch.no_grad():
            velocity = self(images[:1], images[-1:]).detach()
        for index in random.sample(indices, len(indices)):
            time = self.model.encode_time(times[index:index + 1])
            disp_start = self.model.svf_model.velocity2displacement(velocity * time)
            disp_end = self.model.svf_model.velocity2displacement(velocity * (time - 1))
            intensity, segmentation = self._data_losses(images, labels, index, disp_start, disp_end)
            loss = self.lambda_sim * intensity + self.lambda_seg * segmentation
            optimizer.zero_grad(set_to_none=True)
            self.manual_backward(loss)
            optimizer.step()
            self.lr_schedulers()[1].step()
            self.log('Loss MLP', loss.detach(), prog_bar=True, batch_size=1)

    def training_step(self, batch, batch_idx) -> None:
        images, labels, times = self._prepare_batch(batch)
        if self.lambda_sim <= 0 and (self.lambda_seg <= 0 or labels is None):
            raise ValueError('Training requires an active image loss or available segmentations with lambda_seg > 0')
        optimizers = self.optimizers()
        svf_optimizer = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        self.train_svf(svf_optimizer, images, labels, times)
        if self.model.time_mode == 'mlp':
            self.train_mlp(optimizers[1], images, labels, times)

    def on_train_epoch_end(self) -> None:
        if self.trainer.is_global_zero:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'last_model.pth'))

    def on_validation_epoch_start(self) -> None:
        self.validation_dice = []

    def validation_step(self, batch, batch_idx) -> None:
        images, labels, times = self._prepare_batch(batch)
        velocity = self(images[:1], images[-1:])
        max_negative = 0
        scores = []
        # Retrieve the transformed affine, since output tensors live on the resized grid.
        subject = None
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            dataset = self.trainer.val_dataloaders.dataset
            subject_index = list(self.trainer.val_dataloaders.sampler)[batch_idx]
            subject = dataset.get_subject(subject_index, 0)
            if dataset.transform is not None:
                subject = dataset.transform(subject)
        for index in range(1, len(times)):
            time = self.model.encode_time(times[index:index + 1])
            displacement = self.model.svf_model.velocity2displacement(velocity * time)
            negative = (compute_jacobian_determinant_3d(displacement[0]) < 0).sum().item()
            max_negative = max(max_negative, negative)
            prediction = None
            if labels is not None:
                prediction = warp(labels[:1], displacement).argmax(dim=1)
                prediction_one_hot = F.one_hot(prediction, num_classes=labels.shape[1]).movedim(-1, 1)
                score = self.dice_metric(prediction_one_hot, labels[index:index + 1]).mean()
                self.dice_metric.reset()
                scores.append(score)
            if subject is not None:
                suffix = f'subject_{batch_idx}_time_{index}'
                warped_image = warp(images[:1], displacement)
                tio.ScalarImage(tensor=warped_image[0].cpu(), affine=subject.image.affine).save(
                    os.path.join(self.save_path, f'image_warped_{suffix}.nii.gz'))
                if prediction is not None:
                    tio.LabelMap(tensor=prediction.cpu(), affine=subject.image.affine).save(
                        os.path.join(self.save_path, f'label_warped_{suffix}.nii.gz'))
                spacing = displacement.new_tensor(subject.image.spacing).view(3, 1, 1, 1)
                tio.ScalarImage(tensor=(displacement[0] * spacing).cpu(), affine=subject.image.affine).save(
                    os.path.join(self.save_path, f'forward_dvf_{suffix}.nii.gz'))
        if scores:
            mean_dice = torch.stack(scores).mean()
            self.validation_dice.append(mean_dice.detach())
            self.log('Mean dice', mean_dice, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log('Negative Jacobian', float(max_negative), prog_bar=True,
                 on_epoch=True, sync_dist=True, batch_size=1)

    def on_validation_epoch_end(self) -> None:
        if not self.validation_dice:
            return
        scores = torch.stack(self.validation_dice)
        mean_dice = self.all_gather(scores.mean()).mean().item()
        if self.trainer.sanity_checking:
            return
        if mean_dice > self.dice_max:
            self.dice_max = mean_dice
            if self.trainer.is_global_zero:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_best.pth'))
        self.log('Dice max', self.dice_max, prog_bar=True, sync_dist=True)
        if self.trainer.is_global_zero:
            plt.figure(figsize=(10, 6))
            plt.plot(scores.cpu().numpy())
            plt.ylim(0, 1)
            plt.title('mDice per sequence')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, 'mDice.png'))
            plt.close()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
