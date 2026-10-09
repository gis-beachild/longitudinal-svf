"""Regression checks for JSON longitudinal datasets and held-out validation."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip('torchio')
pytest.importorskip('pytorch_lightning')
import torch
import torchio as tio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data.longitudinal_datamodule import LongitudinalDataModule


def make_manifest(root, name, ages):
    sessions = []
    for age in ages:
        image = f'image_{age}.nii.gz'
        label = f'label_{age}.nii.gz'
        tio.ScalarImage(tensor=torch.rand(1, 8, 8, 8)).save(root / image)
        labels = torch.arange(512).reshape(1, 8, 8, 8) % 5
        tio.LabelMap(tensor=labels).save(root / label)
        sessions.append(dict(image='/' + image, segmentation='/' + label, age=age))
    (root / name).write_text(json.dumps({'subjects': [{'sessions': sessions}]}))


def module(root, **kwargs):
    return LongitudinalDataModule(
        data_dir='', root_dir=str(root), train_json='train.json',
        t0=92, t1=161, csize=(8, 8, 8), rsize=(8, 8, 8),
        num_classes=4, merge_labels_0_1=True, num_workers=0, **kwargs)


def test_json_ages_labels_and_held_out_validation(tmp_path):
    make_manifest(tmp_path, 'train.json', [161, 92])
    make_manifest(tmp_path, 'val.json', [120])
    dm = module(tmp_path, val_json='val.json')
    dm.setup('fit')
    assert [s['age'] for s in dm.train_subjects] == [0, 1]
    assert dm.val_subjects[0]['age'] == pytest.approx(28 / 69)
    subject = dm.train_dataloader().dataset[0]
    assert subject['label'].data.shape == (4, 8, 8, 8)
    assert torch.all(subject['label'].data.sum(0) == 1)
    assert next(iter(dm.val_dataloader()))['age'].item() == pytest.approx(28 / 69)


def test_missing_endpoint_is_rejected(tmp_path):
    make_manifest(tmp_path, 'train.json', [92, 120])
    with pytest.raises(ValueError, match='session at tn'):
        module(tmp_path).setup('fit')


def test_validation_defaults_to_training_manifest(tmp_path):
    make_manifest(tmp_path, 'train.json', [161, 92])
    dm = module(tmp_path)
    dm.setup('fit')
    assert [s['age'] for s in dm.val_subjects] == [0, 1]
