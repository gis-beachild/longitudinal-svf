import torch
import torchio as tio
from typing import Sequence
import random
import itertools


class PairwiseSubjectsDataset(torch.utils.data.Dataset):
    '''
        TorchIO SubjectsDataset that return a pair of subjects within all possible combinaisons.
        The dataset length is the number of possible combinaisons.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''

        combinations = list(itertools.combinations(range(len(subjects)), 2))
        self.all_pair = combinations
        self.data_subjects = []
        self.transform = transform
        for i in range(len(subjects)):
            self.data_subjects.append(transform(subjects[i]))

    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.all_pair)

    def __getitem__(self, idx: int) -> dict[str, tio.Subject]:
        '''
            Get the pair of subjects at index idx
            :param idx: index of the pair of subjects
        '''
        subject_registration_pair = tio.Subject(
            {
                'source_image': self.data_subjects[self.all_pair[idx][0]]['image'],
                'target_image': self.data_subjects[self.all_pair[idx][1]]['image'],
                'source_label': self.data_subjects[self.all_pair[idx][0]]['label'],
                'target_label': self.data_subjects[self.all_pair[idx][1]]['label'],
            }
        )
        return subject_registration_pair


class PairwiseSubjectsDatasetValidation(torch.utils.data.Dataset):
    '''
        TorchIO SubjectsDataset that return a pair of subjects within all possible combinaisons.
        The dataset length is the number of possible combinaisons.
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose):
        '''
        PairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        combinations = list(itertools.combinations(range(len(subjects)), 2))
        self.all_pair = combinations
        self.data_subjects = []
        self.transform = transform
        for i in range(len(subjects)):
            self.data_subjects.append(transform(subjects[i]))

    def __len__(self):
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.data_subjects) - 1

    def __getitem__(self, idx: int) -> tio.Subject:
        '''
            Get the pair of subjects at index idx
            :param idx: index of the pair of subjects
        '''
        subject_registration_pair = tio.Subject(
            {
                'source_image': self.data_subjects[self.all_pair[idx][0]]['image'],
                'target_image': self.data_subjects[self.all_pair[idx][1]]['image'],
                'source_label': self.data_subjects[self.all_pair[idx][0]]['label'],
                'target_label': self.data_subjects[self.all_pair[idx][1]]['label'],
            }
        )
        return subject_registration_pair


class RandomPairwiseSubjectsDataset(tio.SubjectsDataset):
    '''
        Torch subjects_dataset that return a random pair of subjects within all possible combinaisons
    '''
    def __init__(self, subjects: Sequence[tio.Subject], transform: tio.Compose):
        '''
        RandomPairwiseSubjectsDataset
        :param subjects: Sequence of subjects
        :param transform: Transform composition applying to the subjects
        '''
        super().__init__(subjects, transform=transform)
        combinations = list(itertools.combinations(range(len(subjects)), 2))
        self.all_pair = combinations + [(j, i) for i, j in combinations]


    def __len__(self) -> int:
        '''
            Return the number of subjects in the dataset
        '''
        return len(self.all_pair)

    def __getitem__(self, idx: int) -> dict[str, tio.Subject]:
        '''
            Get a random pair of subjects for the subject at index idx.
            :param idx: index of the pair of subjects
        '''
        rand = random.randint(0, len(self.all_pair) - 1)
        subject_pair = {
            str(0): tio.SubjectsDataset.__getitem__(self, self.all_pair[rand][0]),
            str(1): tio.SubjectsDataset.__getitem__(self, self.all_pair[rand][1])
        }
        return subject_pair