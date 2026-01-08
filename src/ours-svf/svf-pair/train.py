import os
import gc
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import registration_svf
from registration_svf.registration import RegistrationModule
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from datamodule import PairwiseRegistrationDataModule
from training_module import RegistrationTrainingModule

gc.collect()
torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="../../../configs/", config_name="config_pair")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')
    hydra_cfg = HydraConfig.get()
    save_dir = f'./results/{cfg.data.name}/"svf_pair"/'
    if not  os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_save_dir = os.path.join(save_dir, HydraConfig.get().runtime.output_dir)
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(save_dir, 'log'), name=None, version='')
    model: RegistrationModule = hydra.utils.instantiate(cfg.svf_model)

    datamodule: pl.LightningDataModule = PairwiseRegistrationDataModule(
        data_dir=cfg.data.csv_path,
        batch_size=cfg.data.batch_size,
        rsize=cfg.data.rsize,
        csize=cfg.data.csize,
        num_classes=cfg.data.num_classes)
    if cfg.train_pair.load != "":
        model.load_state_dict(torch.load(cfg.train_pair.load))
    training_module = RegistrationTrainingModule(model=model,
                                                 save_path=sub_save_dir,
                                                 learning_rate=cfg.train_pair.learning_rate,
                                                 lambda_sim=cfg.train_pair.lambda_sim,
                                                 lambda_reg=cfg.train_pair.lambda_reg,
                                                 lambda_seg=cfg.train_pair.lambda_seg)

    trainer = pl.Trainer(max_steps=cfg.train_pair.max_steps, precision=32, num_sanity_val_steps=10, logger=tensorboard_logger,
                         callbacks= [ModelCheckpoint(every_n_train_steps=100, dirpath=sub_save_dir, save_last=True)],
                         check_val_every_n_epoch=5, gradient_clip_algorithm='norm',
                         enable_progress_bar=True)
    
    checkpoint = None
    if cfg.train.checkpoint != "":
        checkpoint = cfg.train_pair.checkpoint
    trainer.fit(model=training_module,
                datamodule=datamodule,
                ckpt_path=checkpoint)
    print("Training finished")
    print("Saving model")
    training_module.save(save_dir)

if __name__ == '__main__':
    main()