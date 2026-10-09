"""PyTorch Lightning DataModule for longitudinal registration, driving the merged train.py/predict.py pipeline.

Author: Fl0rian
"""
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
from datetime import datetime
from .longitudinal_dataset import LongitudinalDatasetValidation
from .json_manifest import load_sessions


class LongitudinalDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 t0: int,
                 t1: int,
                 rsize: int | tuple[int, int, int],
                 csize: int | tuple[int, int, int],
                 batch_size: int = 1,
                 num_workers: int = 8,
                 num_classes: int = 20,
                 date_format: str | None = None,
                 seed: int = 42,
                 root_dir: str | None = None,
                 train_json: str | None = None,
                 val_json: str | None = None,
                 merge_labels_0_1: bool = False) -> None:
        """
        Data module for longitudinal registration task.
        :param data_dir: Path to the CSV file containing image and label paths.
        :param t0: Time point 0.
        :param t1: Time point 1.
        :param rsize: Resize the image to this size.
        :param csize: Crop or pad the image to this size before resizing.
        :param batch_size: Batch size for training.
        :param num_workers: Number of workers for data loading.
        :param num_classes: Number of classes for segmentation.
        """
        super().__init__()
        self.data_dir = data_dir
        self.t0 = t0
        self.t1 = t1
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rsize = rsize
        self.csize = csize
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.seed = seed
        self.num_classes = num_classes
        self.date_format = date_format
        self.root_dir = root_dir
        self.train_json = train_json
        self.val_json = val_json
        self.merge_labels_0_1 = merge_labels_0_1


    def prepare_data(self) -> None:
        """Download or prepare data (if needed)."""
        pass

    def _get_subjects(self, manifest: str | None = None) -> list:
        """Normalize JSON ages against t0/t1, or CSV ages against endpoint row indices."""
        if manifest is not None:
            if self.root_dir is None:
                raise ValueError("root_dir is required with a JSON manifest")
            sequences = load_sessions(self.root_dir, manifest)
            if len(sequences) != 1:
                raise ValueError("longitudinal training requires one subject per manifest")
            if self.t1 <= self.t0:
                raise ValueError("tn must be greater than t0")
            return [tio.Subject(
                image=tio.ScalarImage(session['image']),
                label=tio.LabelMap(session['label']),
                age=(session['age'] - self.t0) / (self.t1 - self.t0),
            ) for session in sequences[0]]
        subjects = []
        reference_date = None
        df = pd.read_csv(self.data_dir)
        for index, row in df.iterrows():
            if tio.ScalarImage(row['label']).data.shape[0] > 1:
                subject = tio.Subject(
                    image=tio.ScalarImage(row['image']),
                    label=tio.ScalarImage(row['label']),
                    string_age=str(row['age'])
                )
            else:
                subject = tio.Subject(
                    image=tio.ScalarImage(row['image']),
                    label=tio.LabelMap(row['label']),
                    string_age=str(row['age'])
                )
            subjects.append(subject)

        reference_date_t0 = datetime.strptime(subjects[self.t0]['string_age'], self.date_format).timestamp() if self.date_format else float(
            subjects[self.t0]['string_age'])
        reference_date_t1 = datetime.strptime(subjects[self.t1]['string_age'], self.date_format).timestamp() if self.date_format else float(
            subjects[self.t1]['string_age'])
        for i in range(len(subjects)):
            age = datetime.strptime(subjects[i]['string_age'], self.date_format).timestamp() if self.date_format else float(
                subjects[i]['string_age'])
            subjects[i]['age'] = (float(age) - float(reference_date_t0)) / float(reference_date_t1 - reference_date_t0)

        return subjects

    def setup(self, stage=None) -> None:
        self.train_subjects = self._get_subjects(self.train_json)
        for age, name in ((0, "t0"), (1, "tn")):
            if not any(subject['age'] == age for subject in self.train_subjects):
                raise ValueError(f"Training data must contain a session at {name}")
        self.val_subjects = self._get_subjects(self.val_json or self.train_json)

    def _label_transform(self):
        if self.merge_labels_0_1:
            return tio.Lambda(lambda data: (data - 1).clamp_min(0), include=['label'])
        return None


    def train_dataloader(self) -> tio.SubjectsLoader:
        transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(self.csize),
            tio.transforms.Resize(self.rsize),
            tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9), include=['image']),
            tio.transforms.Clamp(out_min=0, out_max=1, include=['image']),
            *([self._label_transform()] if self.merge_labels_0_1 else []),
            tio.transforms.OneHot(self.num_classes)
        ])
        if self.train_subjects is None:
            self.setup("fit")
        train_dataset = LongitudinalDatasetValidation(self.train_subjects, transform=transform)
        return tio.SubjectsLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> tio.SubjectsLoader:
        transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(self.csize),
            tio.transforms.Resize(self.rsize),
            tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9), include=['image']),
            tio.transforms.Clamp(out_min=0, out_max=1, include=['image']),
            *([self._label_transform()] if self.merge_labels_0_1 else []),
            tio.transforms.OneHot(self.num_classes)
        ])
        if self.val_subjects is None:
            self.setup("validate")
        val_dataset = LongitudinalDatasetValidation(self.val_subjects, transform=transform)
        return tio.SubjectsLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> tio.SubjectsLoader:
        transform = tio.transforms.Compose([
            tio.transforms.CropOrPad(self.csize),
            tio.transforms.Resize(self.rsize),
            tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9), include=['image']),
            tio.transforms.Clamp(out_min=0, out_max=1, include=['image']),
            *([self._label_transform()] if self.merge_labels_0_1 else []),
            tio.transforms.OneHot(self.num_classes)
        ])
        if self.val_subjects is None:
            self.setup("test")
        test_dataset = LongitudinalDatasetValidation(self.val_subjects, transform=transform)
        return tio.SubjectsLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
