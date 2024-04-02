import braindecode
import os
import numpy as np
import pandas as pd
import pathlib
import braindecode.datasets
import pickle
import optuna
from .EEGnet_model import *
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
import joblib


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


def part_data_subj(subj, j: int, dataset_name: str, n=1):
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name, 'subj_'+str(subj))
    data = pd.read_pickle(pathlib.Path(path, 'train_data.pkl'))
    train_data = data.sample(n=n, random_state=j)
    val_data = data.drop(train_data.index)
    test_data = pd.read_pickle(pathlib.Path(path, 'test_data.pkl'))
    return SubjDataset(train_data), SubjDataset(val_data), SubjDataset(test_data)


def all_data_subj(subj, dataset_name: str):
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name, 'subj_'+str(subj))
    data = pd.read_pickle(pathlib.Path(path, 'train_data.pkl'))
    return SubjDataset(data)


def test_data_subj(subj, dataset_name: str):
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name, 'subj_'+str(subj))
    data = pd.read_pickle(pathlib.Path(path, 'test_data.pkl'))
    data = data.sample(frac=0.2)
    return SubjDataset(data)


def other_data_subj(subj, subjects, dataset_name: str):
    data = pd.DataFrame()
    using_subjects = deepcopy(subjects)
    using_subjects.remove(subj)
    for sub in using_subjects:
        path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name, 'subj_'+str(sub))
        data = pd.concat([data, pd.read_pickle(pathlib.Path(path, 'train_data.pkl'))], ignore_index=True)
    return SubjDataset(data)


def last_n_data_subj(subj, dataset_name: str, r_s=42, train=2, test_frac=0.2, return_dataset=True):
    # r_s is random state for train data sampling
    # todo resolve different number of classes
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name, 'subj_'+str(subj))
    data = pd.read_pickle(pathlib.Path(path, 'train_data.pkl'))
    train_data = pd.concat([data.loc[data['Y'] == 1].sample(n=int(train/4), random_state=r_s),
                            data.loc[data['Y'] == 0].sample(n=int(train/4), random_state=r_s)])
    train_data = pd.concat([train_data,
                            data.loc[data['Y'] == 2].sample(n=int(train/4), random_state=r_s)])
    train_data = pd.concat([train_data,
                            data.loc[data['Y'] == 3].sample(n=int(train/4), random_state=r_s)])
    test_data = pd.read_pickle(pathlib.Path(path, 'test_data.pkl'))
    if return_dataset:
        return SubjDataset(train_data), SubjDataset(test_data)
    else:
        return train_data.to_numpy(copy=True), test_data.to_numpy(copy=True)


def data_from_windows_dataset(dataset, subjects: list,
                              dataset_name: str = None, description: str = None, test_size=0.2):
    """
    :param - dataset: is windows dataset from braindecode
    :param - subjects: list of subjects in dataset
    :param - dataset_name: name of dataset if it is not specified function will return the base name
    :param - description: some description of dataset (notes about preprocessing)
    :param - test_size: size of test data which vil contain equal datapoints of each class and will be taken from the end
     of dataset
    :return: - name of dataset (str)
    """
    X = []
    Y = []
    Window = []
    p = 1
    path = pathlib.Path(pathlib.Path.cwd())
    if dataset_name is None:
        while os.path.exists(pathlib.Path(path, 'data_' + str(p))):
            p += 1
        os.mkdir(pathlib.Path(path, 'data_' + str(p)))
        dataset_name = str(p)
    elif os.path.exists((pathlib.Path(path, 'data_' + str(dataset_name)))):
        while os.path.exists(pathlib.Path(path, 'data_' + str(dataset_name) + str(p))):
            p += 1
        os.mkdir(pathlib.Path(path, 'data_' + str(dataset_name) + str(p)))
        dataset_name = str(dataset_name) + str(p)
    else:
        os.mkdir(pathlib.Path(path, 'data_' + str(dataset_name)))
    path = (pathlib.Path(path, 'data_' + dataset_name))
    dsub = dataset.split('subject')  # split for subjects
    for sub in subjects:
        sub_path = pathlib.Path(path, "subj_" + str(sub))
        os.mkdir(sub_path)
        for x, y, window in dsub[str(sub)]:
            X.append(np.array(x))
            Y.append(y)
            Window.append(window)
        data_dict = {'X': X, 'Y': Y, 'window_data': Window}
        n = len(list(set(Y)))
        df = pd.DataFrame.from_dict(data_dict)
        t_n = int(len(df.index) * test_size / n)
        test_data = []
        for i in range(n):
            test_data.append(df.loc[df['Y'] == i][-t_n:])
        test_data = pd.concat(test_data)
        train_data = df.drop(test_data.index)
        pt = str(pathlib.Path(sub_path, 'train_data.pkl').resolve())
        train_data.to_pickle(pt)
        pt = str(pathlib.Path(sub_path, 'test_data.pkl').resolve())
        test_data.to_pickle(pt)
    with open('./data_' + str(dataset_name) + "/subjects.pkl", "wb") as fp:
        pickle.dump(subjects, fp)
    if description is not None:
        with open('./data_' + str(dataset_name) + "/subjects.txt", "wb") as fp:
            pickle.dump(str(description), fp)
    return dataset_name


def meta_step(weights_before: OrderedDict, weights_after: OrderedDict,  model, outestepsize=0.8,
              outestepsize1: float = None, iteration=None, epochs=None, meta_optimizer=None):
    if iteration is None or epochs is None or epochs == 0:
        raise ValueError('iterative error in meta step definition')
    if meta_optimizer is None:
        if outestepsize1 is None:
            outerstepsize0 = outestepsize * (1 - iteration / epochs)
            state_dict = {name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize0
                          for name in weights_before}
            model.load_state_dict(state_dict)
        else:
            outerstepsize0 = outestepsize * (1 - iteration / epochs)
            outerstep1 = outestepsize1 * (1 - iteration / epochs)
            state_dict = {name: weights_before[name] + (
                    weights_after[name] - weights_before[name]) * outerstepsize0 if name.startswith(
                'conv_features') else weights_before[name] + (weights_after[name] - weights_before[name]) * outerstep1 for
                          name in weights_before}
            model.load_state_dict(state_dict)
    else:
        rd = []
        meta_optimizer.zero_grad()
        model.load_state_dict(weights_before)
        model.train()
        for name in weights_before:
            if name.endswith('weight') or name.endswith('bias'):
                rd.append(weights_after[name] - weights_before[name])
        for p,d in zip(model.parameters(), rd):
            p.grad = d
        meta_optimizer.step()
        model.eval()
    return model


def params_pretrain(trial, model_params, dataset_name, tr_sub, tst_sub, double_meta_step=False, meta_optimizer=False):
    if not meta_optimizer:
        if double_meta_step:
            params = {
                'innerepochs': trial.suggest_int('innerepochs', 1, 20),
                'oterepochs': trial.suggest_int('oterepochs', 2, 70),
                'outerstepsize0': trial.suggest_float('outerstepsize0', 0.01, 0.9),
                'outerstepsize1': trial.suggest_float('outerstepsize1', 0.01, 0.9),
                'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
                'in_datasamples': trial.suggest_int('in_datasamples', 1, 50)
            }
        else:
            params = {
                'innerepochs': trial.suggest_int('innerepochs', 1, 20),
                'oterepochs': trial.suggest_int('oterepochs', 2, 70),
                'outerstepsize0': trial.suggest_float('outerstepsize0', 0.01, 0.9),
                'outerstepsize1': None,
                'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
                'in_datasamples': trial.suggest_int('in_datasamples', 1, 50)
            }
    else:
        params = {
            'innerepochs': trial.suggest_int('innerepochs', 1, 20),
            'oterepochs': trial.suggest_int('oterepochs', 2, 70),
            'outerstepsize0': trial.suggest_float('outerstepsize0', 0.000006, 0.1),
            'outerstepsize1': None,
            'in_lr': trial.suggest_float('in_lr', 0.000006, 0.05),
            'in_datasamples': trial.suggest_int('in_datasamples', 1, 50)
        }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_from_params(model_params)   # todo create model from param realization DONE
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
    if meta_optimizer is not None:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['outerstepsize0'])
    n = params['in_datasamples']
    for iteration in range(params['oterepochs']):
        task_t = []
        task_v = []
        weights_before = deepcopy(model.state_dict())
        # Generate task
        for i in range(len(tr_sub)):
            train_data, val_data, test_data = part_data_subj(subj=tr_sub[i], dataset_name=dataset_name,
                                                             n=n, j=42 + iteration)
            task_t.append(DataLoader(train_data, batch_size=n, drop_last=True, shuffle=True))
            task_v.append(DataLoader(val_data, batch_size=n, drop_last=True, shuffle=True))
        # Do SGD on this task
        for i in range(len(task_t)):
            best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), task_t[i],
                                         epochs=params['innerepochs'], device=device, logging=False)
            model = meta_step(weights_before=weights_before, weights_after=model.state_dict(),
                              outestepsize=params['outerstepsize0'], outestepsize1=params['outerstepsize1'],
                              epochs=params['oterepochs'], iteration=iteration, meta_optimizer=meta_optimizer,
                              model=model)

    stat = 0
    for test_sub in tst_sub:
        train_data, test_data, _ = part_data_subj(subj=test_sub, n=n, j=42, dataset_name=dataset_name)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            stat += torch.sum(correct).item() / len(targets)
    return stat / len(tst_sub)


def meta_params(dataset_name: str, tr_sub: list, tst_sub: list, model_params, trials=50, jobs=1, double_meta_step=False, meta_optimizer=False):   # todo : automatic description in data folder
    p = 1
    lib_name = 'params_for_' + model_params['model_type'] + '_model_'
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name)
    while os.path.exists(pathlib.Path(path, lib_name + str(p))):
        p += 1
    os.mkdir(pathlib.Path(path, lib_name + str(p)))
    path = pathlib.Path(path, lib_name + str(p))
    file_path = str(path.resolve()) + "/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    func = lambda trial: params_pretrain(trial, model_params=model_params, dataset_name=dataset_name, tr_sub=tr_sub,
                                         tst_sub=tst_sub, double_meta_step=double_meta_step,
                                         meta_optimizer=meta_optimizer)
    study = optuna.create_study(storage=storage, study_name='optuna_meta_params', direction='maximize',
                                load_if_exists=True)
    study.optimize(func, n_trials=trials, n_jobs=jobs)
    params = study.best_params
    return params


def pretrain(params: dict, model_params: dict, all_subjects: list, target_sub: list, dataset_name: str,
             meta_optimizer=False):
    stat = []
    stat1 = []
    if 'outerstepsize1' not in params.keys():
        params.update(outerstepsize1=None)
    p = 1
    path = './data_' + dataset_name + '/'
    os.mkdir(path + 'models/')
    path = path + 'models/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_from_params(model_params)
    model.to(device)
    start_weights = deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()   # todo resolve criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
    optim = torch.optim.Adam(model.parameters(), lr=params['in_lr']*params['outerstepsize0']/params['oterepochs'])
    if meta_optimizer:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['outerstepsize0'])
    else:
        meta_optimizer = None
    for sub in target_sub:
        minimal_len = 10000
        model.load_state_dict(start_weights)
        subjects = deepcopy(all_subjects)
        subjects.remove(sub)
        for iteration in range(params['oterepochs']):   # iterations for outer epochs
            task_t = []
            iter_t = []
            #task_v = []
            weights_before = deepcopy(model.state_dict())
            # Generate task
            for i in range(len(subjects)):   # creation of dataset for all-1 sub
                train_data = all_data_subj(subj=subjects[i], dataset_name=dataset_name)
                task_t.append(DataLoader(train_data, batch_size=params['in_datasamples'], drop_last=True, shuffle=True))
                if minimal_len > len(task_t[i]):
                    minimal_len = len(task_t[i])
                iter_t.append(iter(task_t[i]))
                #task_v.append(DataLoader(val_data, batch_size=n, drop_last=True, shuffle=True))
            # Do SGD on this task
            for j in range(minimal_len):   # iterations for number of samples for each sub
                for i in range(len(task_t)):   # iterations for subjects
                    #best_model, best_acc = nnm.train(model, optimizer, torch.nn.CrossEntropyLoss(), task_t[i][j],
                    #                             epochs=innerepochs, device=device, logging=False)
                    batch = next(iter_t[i])
                    start_weights = deepcopy(model.state_dict())
                    best_model, best_acc = single_batch_train(model, optimizer, torch.nn.CrossEntropyLoss(), batch,
                                                              epochs=params['innerepochs'], device=device, logging=False)
                    # Interpolate between current weights and trained weights from this task
                    # I.e. (weights_before - weights_after) is the meta-gradient
                    weights_after = deepcopy(model.state_dict())
                    model = meta_step(weights_before=start_weights, weights_after=weights_after,
                                      outestepsize=params['outerstepsize0'], outestepsize1=params['outerstepsize1'],
                                      epochs=params['oterepochs'], iteration=iteration, meta_optimizer=meta_optimizer,
                                      model=model)

        torch.save(model.state_dict(), (path + str(sub) + "-reptile.pkl"))
        test_data = test_data_subj(subj=sub, dataset_name=dataset_name)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            stat.append(torch.sum(correct).item()/len(targets))
        data = other_data_subj(sub, all_subjects, dataset_name=dataset_name)
        data = DataLoader(data, batch_size=64, drop_last=True, shuffle=True)
        model.load_state_dict(start_weights)
        best_model, best_acc = train(model, optim, torch.nn.CrossEntropyLoss(), data,
                                     epochs=params['innerepochs'] * params['oterepochs'], device=device, logging=False)
        torch.save(model.state_dict(), (path + str(sub) + "-baseline.pkl"))
        test_data = test_data_subj(subj=sub,  dataset_name=dataset_name)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            stat1.append(torch.sum(correct).item()/len(targets))
    print("Pretraining competed with mean cold AUC ROC = " + str(sum(stat)/len(stat)))
    with open(path+'reptile_cold_stats.txt', 'w') as fp:
        for i in range(len(target_sub)):
            fp.write("reptile on subject " + str(target_sub[i]) + " ACC is " + str(stat[i]) + '\n\n')
        fp.write("Mean ACC = " + str(sum(stat)/len(stat)) + '\n\n')
        fp.write("Params: " + '\n\n')
        for k, v in params.items():
            fp.write(str(k) + ' == ' + str(v) + '\n\n')
        for i in range(len(target_sub)):
            fp.write("baseline on subject " + str(target_sub[i]) + " ACC is " + str(stat1[i]) + '\n\n')
    return sum(stat)/len(stat)


def p_aftrain(trial, model_params, dataset_name, tst_sub):
    a = trial.suggest_int('a_ep', 2, 20)
    params = {
              'lr': trial.suggest_float('lr', 0.00003, 0.05),
              'a': a,
              'b': trial.suggest_int('b_ep', 0, 4*a-1)
              }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_from_params(model_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    model.to(device)
    stat = 0
    for sub in tst_sub:
        for j in range(0, 51, 4):
            model.load_state_dict(torch.load("data_" + dataset_name + "/models/" + str(sub) + "-reptile.pkl"))
            if not j == 0:
                train_data, test_data = last_n_data_subj(dataset_name=dataset_name, subj=sub, train=j, r_s=42,
                                                         test_frac=0.2)
                data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                             epochs=int(j*params['a']-params['b']), device=device, logging=False)

            else:
                train_data, test_data = last_n_data_subj(dataset_name=dataset_name, subj=sub, train=4,
                                                         r_s=42, test_frac=0.2)

            #train_data, test_data = ut.n_data_subj(subj=using_subject, train=params['n'], r_s=42, test_frac=0.2)
            test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
            with torch.no_grad():
                for batch in test_data_loader:
                    inputs, targets = batch
                inputs = inputs.to(device=device, dtype=torch.float)
                output = model(inputs)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device=device, dtype=torch.float)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                stat += torch.sum(correct).item()/len(targets)
    return stat/(12 * len(tst_sub))


def aftrain_params(dataset_name: str, model_params: dict, tst_subj: list, trials: int, jobs: int):
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name)
    os.mkdir(pathlib.Path(path, 'af_params'))
    path = pathlib.Path(path, 'af_params')
    file_path = "./data_" + dataset_name + "/af_params/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    func = lambda trial: p_aftrain(trial, model_params=model_params, dataset_name=dataset_name, tst_sub=tst_subj)
    study = optuna.create_study(storage=storage, study_name='optuna_af_params', direction='maximize')
    study.optimize(func, n_trials=trials, n_jobs=jobs)
    af_params = study.best_params
    with open("./data_" + dataset_name + "/af_params/aftrain_params.txt", 'w') as fp:
        for k, v in af_params.items():
            fp.write(str(k) + ' == ' + str(v) + '\n\n')
    return af_params


def aftrain(target_sub, model_params, af_params, dataset_name, iterations=1, logging=False):
    dt = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_from_params(model_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=af_params['lr'])
    model.to(device)
    start_weights = deepcopy(model.state_dict())
    for sub in target_sub:
        if logging:
            print('afftrain for ' + str(sub) + ' subject of ' + str(len(target_sub)))
        data_points = []
        stat = []
        rstat = []
        for k in range(iterations):
            if logging:
                print('afftrain iteration ' + str(k+1) + ' of ' + str(iterations) + ' started')
            in_data_points = []
            in_stat = []
            r_in_stat = []
            for j in range(0, 51, 4):   # todo resolve 4 to number of dataclasses
                if logging:
                    print('performed ' + str(j) + ' steps')
                in_data_points.append(j)
                model.load_state_dict(torch.load("data_" + dataset_name + "/models/" + str(sub) + "-reptile.pkl"))
                if not j == 0:
                    train_data, test_data = last_n_data_subj(dataset_name=dataset_name, subj=sub, train=j,
                                                             r_s=42+k, test_frac=0.2)
                    data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                    best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                                 epochs=int(j*af_params['a_ep']-af_params['b_ep']),
                                                 device=device, logging=False)

                else:
                    train_data, test_data = last_n_data_subj(dataset_name=dataset_name, subj=sub,
                                                             train=4, r_s=42+k, test_frac=0.2)
                test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
                with torch.no_grad():
                    for batch in test_data_loader:
                        inputs, targets = batch
                    inputs = inputs.to(device=device, dtype=torch.float)
                    output = model(inputs)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to(device=device, dtype=torch.float)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    r_in_stat.append(torch.sum(correct).item()/len(targets))
                model.load_state_dict(torch.load("data_" + dataset_name + "/models/" + str(sub) + "-baseline.pkl"))
                if not j == 0:
                    train_data, test_data = last_n_data_subj(dataset_name=dataset_name, subj=sub,
                                                             train=j, r_s=42+k, test_frac=0.2)
                    data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                    best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                                 epochs=int(j*af_params['a_ep']-af_params['b_ep']),
                                                 device=device, logging=False)

                else:
                    train_data, test_data = last_n_data_subj(dataset_name=dataset_name, subj=sub,
                                                             train=4, r_s=42+k, test_frac=0.2)
                test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
                with torch.no_grad():
                    for batch in test_data_loader:
                        inputs, targets = batch
                    inputs = inputs.to(device=device, dtype=torch.float)
                    output = model(inputs)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to(device=device, dtype=torch.float)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    in_stat.append(torch.sum(correct).item()/len(targets))
            rstat.append(r_in_stat)
            stat.append(in_stat)
            data_points.append(in_data_points)
        dt[str(sub) + '_baseline_data'] = deepcopy(stat)
        dt[str(sub) + '_reptile_data'] = deepcopy(rstat)
        dt[str(sub) + '_datapoints'] = deepcopy(data_points)
    if logging:
        print('aftraining complete')
    path = pathlib.Path(pathlib.Path.cwd(), 'data_' + dataset_name)
    os.mkdir(pathlib.Path(path, 'af_run'))
    path = pathlib.Path(path, 'af_run')
    fig, ax = plt.subplots(len(target_sub)+1, sharex=False, sharey=True, figsize=(12, 9*(len(target_sub)+1)))
    i = 0
    A_acc = []
    A_accr = []
    for sub in target_sub:
        acc = np.mean(dt[str(sub) + '_baseline_data'], axis=0)
        A_acc.append(acc)
        std = np.std(dt[str(sub) + '_baseline_data'], axis=0)
        accr = np.mean(dt[str(sub) + '_reptile_data'], axis=0)
        A_accr.append(accr)
        stdr = np.std(dt[str(sub) + '_reptile_data'], axis=0)
        num = dt[str(sub) + '_datapoints'][0]
        ax[i].plot(num, acc, label='EEGNet')
        ax[i].fill_between(num, acc - std, acc + std, alpha=0.2)
        ax[i].plot(num, accr, label='EEGNet with Reptile')
        ax[i].fill_between(num, accr - stdr, accr + stdr, alpha=0.2)
        ax[i].set_title('Learning curve for subject' + str(sub))
        ax[i].set_xlabel('Train data size')
        ax[i].set_ylabel('ACC')
        ax[i].legend(loc='best')
        i += 1
    acc = np.mean(A_acc, axis=0)
    std = np.std(A_acc, axis=0)
    accr = np.mean(A_accr, axis=0)
    stdr = np.std(A_accr, axis=0)
    num = dt[str(sub) + '_datapoints'][0]
    ax[i].plot(num, acc, label='EEGNet')
    ax[i].fill_between(num, acc - std, acc + std, alpha = 0.2)
    ax[i].plot(num, accr, label='EEGNet with Reptile')
    ax[i].fill_between(num, accr - stdr, accr + stdr, alpha = 0.2)
    ax[i].set_title('Mean learning curve')
    ax[i].set_xlabel('Train data size')
    ax[i].set_ylabel('ACC')
    ax[i].legend(loc='best')
    joblib.dump(dt, pathlib.Path(path, 'all_data_af_test.sav'))
    plt.savefig(pathlib.Path(path, "af_Learn_ACC_ALL" + ".pdf"), format="pdf", bbox_inches="tight")
