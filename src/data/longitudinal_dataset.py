"""Datasets exposing a longitudinal subject series for training and single-item validation.

Author: Fl0rian
"""
import numpy
import torchio as tio
import torch
from typing import Sequence


class LongitudinalDataset(tio.SubjectsDataset):
    '''
        LongitudinalSubjectDataset is a subclass of torchio.SubjectsDataset with a fixed length of 1.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.transforms.Compose) -> None:
        """
        :param subjects:
        :param random_flip: If True, randomly flip is applied to the subjects.
        :param transform:
        """
        super().__init__(subjects, transform=None)
        self.ages = torch.tensor([subject['age'] for subject in subjects], dtype=torch.float)
        self.transform = transform
        self.num_subjects = len(self._subjects)
        self.index_t0_subject = next((i for i, subject in enumerate(self._subjects) if subject['age'] == 0), None)
        self.index_t1_subject = next((i for i, subject in enumerate(self._subjects) if subject['age'] == 1), None)

    def __len__(self) -> int:
        return len(self._subjects)

    def get_subject_based_on_subject_transform(self, index: int, subject: tio.Subject) -> tio.Subject:
        """
        Get the transformed subjects based on a given subject.
        :param subject: A torchio.Subject object.
        :return: A torchio.SubjectsDataset object with transformed subjects.
        """

        transformation_parameters = subject.get_composed_history()
        subject_transformed = transformation_parameters(self._subjects[index])
        return subject_transformed


    def __getitem__(self, idx: int) -> tio.Subject:
        return self.transform(tio.SubjectsDataset.__getitem__(self, idx))



class LongitudinalDatasetValidation(tio.SubjectsDataset):
    '''
        LongitudinalSubjectDataset is a subclass of torchio.SubjectsDataset with a fixed length of 1.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.transforms.Compose) -> None:
        """
        :param subjects:
        :param random_flip: If True, randomly flip is applied to the subjects.
        :param transform:
        """
        super().__init__(subjects, transform=None)
        self.ages = torch.tensor([subject['age'] for subject in subjects], dtype=torch.float)
        self.transform = transform
        self.num_subjects = len(self._subjects)
        self.index_t0_subject = next((i for i, subject in enumerate(self._subjects) if subject['age'] == 0), None)
        self.index_t1_subject = next((i for i, subject in enumerate(self._subjects) if subject['age'] == 1), None)

    def __len__(self) -> int:
        return 1


    def __getitem__(self, idx: int) -> tio.Subject:
        subject = tio.SubjectsDataset.__getitem__(self, idx)
        transformed_subject = self.transform(subject)
        transformed_subject['age']  = subject['age']
        return transformed_subject
