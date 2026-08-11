"""Unified Hydra training entry point for pairwise and longitudinal (linear/mlp) SVF registration.

Selects the datamodule, model, and training module to use based on ``cfg.mode``
(``pairwise`` / ``longitudinal_linear`` / ``longitudinal_mlp``).

Author: Fl0rian
"""
import os
import gc
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.modules.svf_registration import SVFRegistrationModule
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from data.pairwise_datamodule import PairwiseRegistrationDataModule
from data.longitudinal_datamodule import LongitudinalDataModule
from training_module_pairwise import RegistrationTrainingModule
from training_module_longitudinal import LongitudinalTrainingModule
from src.modules.longitudinal_model import LongitudinalDeformation

gc.collect()
torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Build the model/datamodule/training-module for ``cfg.mode`` and run PyTorch Lightning training."""
    torch.set_float32_matmul_precision('high')
    print(OmegaConf.to_yaml(cfg))

    save_dir = f'./results/{cfg.data.name}/{cfg.mode}/'
    os.makedirs(save_dir, exist_ok=True)
    sub_save_dir = os.path.join(save_dir, os.path.basename(HydraConfig.get().runtime.output_dir))
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(save_dir, 'log'), name=None, version='') # type: ignore
    svf_model: SVFRegistrationModule = hydra.utils.instantiate(cfg.svf_model)

    if cfg.mode == "pairwise":
        datamodule: pl.LightningDataModule = PairwiseRegistrationDataModule(
            data_dir=cfg.data.csv_path,
            batch_size=cfg.data.batch_size,
            rsize=cfg.data.rsize,
            csize=cfg.data.csize,
            num_workers=cfg.data.num_workers,
            num_classes=cfg.data.num_classes)
        if cfg.train_pair.load != "":
            svf_model.load_state_dict(torch.load(cfg.train_pair.load))
        training_module = RegistrationTrainingModule(
            model=svf_model,
            save_path=sub_save_dir,
            learning_rate=cfg.train_pair.learning_rate,
            lambda_sim=cfg.train_pair.lambda_sim,
            lambda_reg=cfg.train_pair.lambda_reg,
            lambda_seg=cfg.train_pair.lambda_seg)
        max_steps = cfg.train_pair.max_steps
        checkpoint = cfg.train_pair.checkpoint
    else:  # "longitudinal_linear" or "longitudinal_mlp"
        datamodule: pl.LightningDataModule = LongitudinalDataModule(
            data_dir=cfg.data.csv_path,
            batch_size=cfg.data.batch_size,
            rsize=cfg.data.rsize,
            csize=cfg.data.csize,
            t0=cfg.data.t0,
            t1=cfg.data.t1,
            num_workers=cfg.data.num_workers,
            date_format=cfg.data.date_format,
            num_classes=cfg.data.num_classes)
        if cfg.train_long.load_svf != "":
            svf_model.load_state_dict(torch.load(cfg.train_long.load_svf))
        model: LongitudinalDeformation = LongitudinalDeformation(
            svf_model=svf_model, time_mode=cfg.train_long.time_mode, t0=cfg.data.t0, t1=cfg.data.t1)
        if cfg.train_long.load_model != "":
            model.load_state_dict(torch.load(cfg.train_long.load_model))
        training_module = LongitudinalTrainingModule(
            model=model,
            save_path=sub_save_dir,
            learning_rate_svf=cfg.train_long.learning_rate_svf,
            learning_rate_mlp=cfg.train_long.learning_rate_mlp,
            lambda_reg=cfg.train_long.lambda_reg,
            lambda_sim=cfg.train_long.lambda_sim,
            lambda_seg=cfg.train_long.lambda_seg,
            num_inter_by_epoch=cfg.train_long.num_inter_by_epoch)
        max_steps = cfg.train_long.max_steps
        checkpoint = cfg.train_long.checkpoint

    if os.environ.get("SVF_SMOKE_TEST"):
        print("SVF_SMOKE_TEST set: model/datamodule/training_module built successfully, exiting before trainer.fit")
        return

    trainer = pl.Trainer(max_steps=max_steps, precision=32, num_sanity_val_steps=10, logger=tensorboard_logger,
                         callbacks=[ModelCheckpoint(every_n_train_steps=100, dirpath=sub_save_dir, save_last=True)],
                         check_val_every_n_epoch=5, gradient_clip_algorithm='norm',
                         enable_progress_bar=True)

    trainer.fit(model=training_module,
                datamodule=datamodule,
                ckpt_path=checkpoint if checkpoint != "" else None)
    print("Training finished")
    print("Saving model")
    training_module.save(save_dir)


if __name__ == '__main__':
    main()
