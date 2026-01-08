import numpy
import torchio as tio
import torch
from typing import Sequence


class LongitudinalDataset(tio.SubjectsDataset):
    '''
        LongitudinalSubjectDataset is a subclass of torchio.SubjectsDataset with a fixed length of 1.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose = None):
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

    def get_subject_based_on_subject_transform(self, index, subject: tio.Subject) -> tio.Subject:
        """
        Get the transformed subjects based on a given subject.
        :param subject: A torchio.Subject object.
        :return: A torchio.SubjectsDataset object with transformed subjects.
        """

        transformation_parameters = subject.get_composed_history()
        subject_transformed = transformation_parameters(self._subjects[index])
        return subject_transformed


    def series_transform_based_on_subject(self, subject: tio.Subject) -> (torch.Tensor, torch.Tensor):
        """
        Get the data and label tensors from a subject.
        :param subject: A torchio.Subject object.
        :return: A tuple of (data_tensor, label_tensor).
        """
        transformation_parameters = subject.get_composed_history()
        shape_label = subject['label'][tio.DATA].shape
        labels = torch.zeros((self.num_subjects, shape_label[0], shape_label[1], shape_label[2], shape_label[3]),
                             dtype=torch.float)
        images = torch.zeros((self.num_subjects, shape_label[1], shape_label[2], shape_label[3]), dtype=torch.float)
        for i in range(len(self._subjects)):
            subject_transformed = transformation_parameters(self._subjects[i])
            labels[i] = subject_transformed['label'][tio.DATA].float()
            images[i] = subject_transformed['image'][tio.DATA].float().squeeze(dim=0)

        return {"images": images,
                "labels": labels,
                "ages": self.ages,
                }


    def __getitem__(self, idx) -> tio.Subject:
        return self.transform(tio.SubjectsDataset.__getitem__(self, idx))



class LongitudinalDatasetValidation(tio.SubjectsDataset):
    '''
        LongitudinalSubjectDataset is a subclass of torchio.SubjectsDataset with a fixed length of 1.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose = None):
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

    def get_subject_based_on_subject_transform(self, index, subject: tio.Subject) -> tio.Subject:
        """
        Get the transformed subjects based on a given subject.
        :param subject: A torchio.Subject object.
        :return: A torchio.SubjectsDataset object with transformed subjects.
        """

        transformation_parameters = subject.get_composed_history()
        subject_transformed = transformation_parameters(self._subjects[index])
        return subject_transformed


    def series_transform_based_on_subject(self, subject: tio.Subject) -> (torch.Tensor, torch.Tensor):
        """
        Get the data and label tensors from a subject.
        :param subject: A torchio.Subject object.
        :return: A tuple of (data_tensor, label_tensor).
        """
        transformation_parameters = subject.get_composed_history()
        shape_label = subject['label'][tio.DATA].shape
        labels = torch.zeros((self.num_subjects, shape_label[0], shape_label[1], shape_label[2], shape_label[3]),
                             dtype=torch.float)
        images = torch.zeros((self.num_subjects, shape_label[1], shape_label[2], shape_label[3]), dtype=torch.float)
        for i in range(len(self._subjects)):
            subject_transformed = transformation_parameters(self._subjects[i])
            labels[i] = subject_transformed['label'][tio.DATA].float()
            images[i] = subject_transformed['image'][tio.DATA].float().squeeze(dim=0)

        return {"images": images,
                "labels": labels,
                "ages": self.ages,
                }


    def __getitem__(self, idx) -> tio.Subject:
        return self.transform(tio.SubjectsDataset.__getitem__(self, idx))
