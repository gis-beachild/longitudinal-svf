"""PyTorch Lightning DataModule for pairwise registration, driving the merged train.py/predict.py pipeline.

Author: Fl0rian
"""
from typing import Optional

import torchio as tio
import pytorch_lightning as pl
import pandas as pd
import random
from .pairwise_dataset import PairwiseSubjectsDataset, PairwiseSubjectsDatasetValidation
from data.json_manifest import load_sessions


class PairwiseRegistrationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str,
                 rsize: int | tuple[int, int, int],
                 csize: int | tuple[int, int, int],
                 batch_size: int = 1,
                 num_workers: int = 15,
                 seed: int = 42,
                 num_classes: int = 20,
                 root_dir: str | None = None,
                 train_json: str | None = None,
                 val_json: str | None = None,
                 merge_labels_0_1: bool = False) -> None:
        """
        Data module for registration task.
        :param data_dir: Path to the CSV file containing image and label paths.
        :param rsize: Resize the image to this size.
        :param csize: Crop or pad the image to this size before resizing.
        :param batch_size: Batch size for training.
        :param num_workers: Number of workers for data loading.
        :param seed: Random seed for shuffling data.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rsize = rsize
        self.csize = csize
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.seed = seed
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.train_json = train_json
        self.val_json = val_json
        self.merge_labels_0_1 = merge_labels_0_1

    def prepare_data(self) -> None:
        """Download or prepare data (if needed)."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load subjects from the CSV into ``self.subjects``."""
        if self.train_json is not None:
            if self.root_dir is None:
                raise ValueError("root_dir is required with train_json")
            train_rows = [session for subject in load_sessions(self.root_dir, self.train_json) for session in subject]
            val_rows = [session for subject in load_sessions(self.root_dir, self.val_json or self.train_json) for session in subject]
        else:
            train_rows = pd.read_csv(self.data_dir).to_dict("records")
            val_rows = train_rows

        def make_subjects(rows):
            subjects = []
            for row in rows:
                label = tio.ScalarImage(row['label'])
                if label.data.shape[0] == 1:
                    label = tio.LabelMap(row['label'])
                subjects.append(tio.Subject(image=tio.ScalarImage(row['image']), label=label))
            return subjects

        self.subjects = make_subjects(train_rows)
        self.val_subjects = make_subjects(val_rows)
        if self.seed is not None:
            random.seed(self.seed)
        #random.shuffle(subjects)

    def _label_transform(self):
        if self.merge_labels_0_1:
            return tio.Lambda(lambda data: (data - 1).clamp_min(0), include=['label'])
        return None


    def train_dataloader(self) -> tio.SubjectsLoader:
        transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(self.csize),
            tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), include=['image']),
            *([self._label_transform()] if self.merge_labels_0_1 else []),
            tio.transforms.OneHot(self.num_classes),
        ])
        train_dataset = PairwiseSubjectsDataset(self.subjects, transform=transform)
        return tio.SubjectsLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  persistent_workers=False,
                                  pin_memory=False,
                                  prefetch_factor=2,
                                  shuffle=True)

    def val_dataloader(self) -> tio.SubjectsLoader:
        transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(self.csize),
            tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), include=['image']),
            *([self._label_transform()] if self.merge_labels_0_1 else []),
            tio.transforms.OneHot(self.num_classes),
        ])
        val_dataset = PairwiseSubjectsDatasetValidation(self.val_subjects, transform=transform)
        return tio.SubjectsLoader(val_dataset,
                                  batch_size=1,
                                  num_workers=self.num_workers,
                                  persistent_workers=False,
                                  pin_memory=False,
                                  shuffle=False,
                                  prefetch_factor=2)

    def test_dataloader(self) -> tio.SubjectsLoader:
        transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(self.csize),
            tio.transforms.Resize(self.rsize),
            tio.transforms.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), masking_method='label'),
            *([self._label_transform()] if self.merge_labels_0_1 else []),
            tio.transforms.OneHot(self.num_classes),
        ])
        test_dataset = PairwiseSubjectsDatasetValidation(self.val_subjects, transform=transform)
        return tio.SubjectsLoader(test_dataset, batch_size=1,
                                  num_workers=self.num_workers,
                                  persistent_workers=False)
