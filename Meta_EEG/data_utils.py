import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import zipfile as zp
import os
from copy import deepcopy
import pathlib


def experiment_storage(experiment_name: str):
    p = 1
    path = pathlib.Path(pathlib.Path.cwd())
    if os.path.exists((pathlib.Path(path, str(experiment_name)))):
        while os.path.exists(pathlib.Path(path, str(experiment_name) + str(p))):
            p += 1
        os.mkdir(pathlib.Path(path, str(experiment_name) + str(p)))
        experiment_name = str(experiment_name) + str(p)
    else:
        os.mkdir(pathlib.Path(path, str(experiment_name)))
    return experiment_name


class SubjDataset(Dataset):   # create train dataset for one of subjects
    def __init__(self, data):
        x = []
        for a in data['X']:
            x.append(a.transpose(0, 1)[np.newaxis, :, :])
        self.inputs = x
        self.targets = data['Y'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MetaDataset:
    def __init__(self, dataset_name: str, subjects: list = None, description: str = None):
        self.dataset_name = dataset_name
        if subjects is None:
            subjects = []
        else:
            self.subjects = subjects
        if description is None:
            self.description = 'Description not provided'
        else:
            self.description = description
        self.data = {}

    def load_data(self, filename: str | None = None, logging=False):
        if filename is None:
            raise ValueError('Dataset filename not provided')
        if not os.path.isfile(filename + '.zip'):
            raise ValueError('Specified filename does not exist')
        if logging: print('Dataset found')
        with zp.ZipFile(filename + '.zip') as zf:
            if logging: print('dataset file contains: ' + str(zf.namelist()))
            with zf.open("subjects.pkl", "r") as fp:
                self.subjects = pickle.load(fp)
            if logging: print('subjects loaded: ' + str(self.subjects))
            if 'description.txt' in zf.namelist():
                with zf.open("description.txt", "r") as fp:
                    self.description = fp.read()
                if logging: print('description loaded: ' + str(self.description))
            else:
                self.description = 'Description not provided'
            for subject in self.subjects:
                if 'subj_' + str(subject) + '/test_data.pkl' in zf.namelist():
                    self.data[subject] = {}
                    with zf.open('subj_' + str(subject) + '/test_data.pkl') as fp:
                        self.data[subject]['test'] = pd.read_pickle(fp)
                    with zf.open('subj_' + str(subject) + '/train_data.pkl') as fp:
                        self.data[subject]['train'] = pd.read_pickle(fp)
                else:
                    raise ValueError('Specified subject does not exist or dataset is broken')

    def save_data(self, filename: str):
        if os.path.isfile(filename + '.zip'):
            raise ValueError('Specified dataset already exists')
        else:
            with zp.ZipFile(filename + '.zip', 'w') as zf:
                zf.write('subjects.pkl')
                with zf.open("subjects.pkl", "w") as fp:
                    pickle.dump(self.subjects,fp)
                zf.write('description.txt')
                with zf.open("description.txt", "w") as fp:
                    fp.write(self.description.encode('utf-8'))
                for subject in self.subjects:
                    zf.write('subj_' + str(subject) + '/test_data.pkl')
                    with zf.open('subj_' + str(subject) + '/test_data.pkl', 'w') as fp:
                        self.data[subject]['test'].to_pickle(fp)
                    zf.write('subj_' + str(subject) + '/train_data.pkl')
                    with zf.open('subj_' + str(subject) + '/train_data.pkl') as fp:
                        self.data[subject]['train'].to_pickle(fp)

    def add_subject_from_xy(self, subject_id: int, x: list, y: list, test_size: float = 0.2, rewrite: bool = False):
        if subject_id in self.subjects:
            if rewrite:
                self.subjects.remove(subject_id)
            else:
                raise ValueError('Specified subject already exists')
        self.subjects.append(subject_id)
        self.data[subject_id] = {}
        data_dict = {'X': x, 'Y': y}
        n = len(list(set(y)))
        df = pd.DataFrame.from_dict(data_dict)
        t_n = int(len(df.index) * test_size / n)
        test_data = []
        for i in range(n):
            test_data.append(df.loc[df['Y'] == i][-t_n:])
        self.data[subject_id]['test'] = pd.concat(test_data)
        self.data[subject_id]['train'] = df.drop(self.data[subject_id]['test'].index)

    def part_data_subj(self, subj: int, rs: int, n: int = 1):
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        data = self.data[subj]['train']
        train_data = data.sample(n=n, random_state=rs)
        val_data = data.drop(train_data.index)
        test_data = self.data[subj]['test']
        return SubjDataset(train_data), SubjDataset(val_data), SubjDataset(test_data)

    def all_data_subj(self, subj: int):
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        data = pd.concat([self.data[subj]['train'], self.data[subj]['test']], ignore_index=True)
        return SubjDataset(data)

    def test_data_subj(self, subj: int):
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        return SubjDataset(self.data[subj]['test'])

    def other_data_subj(self, subj, subjects):
        data = pd.DataFrame()
        using_subjects = deepcopy(subjects)
        using_subjects.remove(subj)
        for sub in using_subjects:
            data = pd.concat([data, self.data[sub]['train']], ignore_index=True)
        return SubjDataset(data)

    def last_n_data_subj(self, subj, train: int, rs=42, return_dataset=True):
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        data = self.data[subj]['train']
        n = data['Y'].nunique(dropna=True)
        if train < n:
            n = train
        train_data = pd.DataFrame()
        for i in range(n):
            train_data = pd.concat([train_data,
                                    data.loc[data['Y'] == i].sample(n=int(train/n), random_state=rs)])
        test_data = self.data[subj]['test']
        if return_dataset:
            return SubjDataset(train_data), SubjDataset(test_data)
        else:
            return train_data.to_numpy(copy=True), test_data.to_numpy(copy=True)







