import os
import gc
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from registration_svf.registration import RegistrationModule
from omegaconf import DictConfig, OmegaConf
from datamodule import LongitudinalDataModule
from longitudinal_model import LongitudinalDeformation
from training_module import LongitudinalTrainingModule
from hydra.core.hydra_config import HydraConfig

gc.collect()
torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="../../../configs/", config_name="config_long")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')
    print(OmegaConf.to_yaml(cfg))

    save_dir = f'./results/{cfg.data.name}/svf_{cfg.train_long.time_mode}/'
    if not  os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_save_dir = os.path.join(save_dir, HydraConfig.get().runtime.output_dir)


    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(save_dir, 'log'), name=None, version='')
    model: RegistrationModule = hydra.utils.instantiate(cfg.svf_model)
    if cfg.svf_model.load != "":
        model.load_state_dict(torch.load(cfg.svf_model.load))
    datamodule: pl.LightningDataModule = LongitudinalDataModule(
        data_dir=cfg.data.csv_path,
        batch_size=cfg.data.batch_size,
        rsize=cfg.data.rsize,
        csize=cfg.data.csize,
        t0=cfg.data.t0,
        t1=cfg.data.t1,
        num_workers=cfg.data.num_workers,
        date_format=cfg.data.date_format,
        num_classes=cfg.data.num_classes
    )
    model : LongitudinalDeformation = LongitudinalDeformation(svf_model=model, time_mode=cfg.train_long.time_mode, t0=cfg.data.t0, t1=cfg.data.t1)
    if cfg.train_long.load != "":
        model.load_state_dict(torch.load(cfg.train_long.load))
    training_module = LongitudinalTrainingModule(model=model,save_path=sub_save_dir,
                                                 learning_rate_svf=cfg.train_long.learning_rate,
                                                 learning_rate_mlp=cfg.train_long.learning_rate,
                                                 lambda_reg=cfg.train_long.lambda_reg,
                                                 lambda_sim=cfg.train_long.lambda_sim,
                                                 lambda_seg=cfg.train_long.lambda_seg,
                                                 num_inter_by_epoch=cfg.train_long.module.num_inter_by_epoch)
    trainer = pl.Trainer(max_steps=cfg.train_long.max_steps, precision=32, num_sanity_val_steps=10, logger=tensorboard_logger,
                         callbacks= [ModelCheckpoint(every_n_train_steps=100, dirpath=sub_save_dir, save_last=True)],
                         check_val_every_n_epoch=5, gradient_clip_algorithm='norm',
                         enable_progress_bar=True)
    checkpoint = None
    if cfg.train_long.checkpoint != "":
        checkpoint = cfg.train_long.checkpoint
    trainer.fit(model=training_module,
                datamodule=datamodule,
                ckpt_path=checkpoint)
    print("Training finished")
    print("Saving model")
    training_module.save(save_dir)

if __name__ == '__main__':
    main()