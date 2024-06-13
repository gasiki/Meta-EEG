import optuna
from .EEGnet_model import *
from .data_utils import *
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
import joblib
import multiprocessing
from functools import partial


def meta_step(weights_before: OrderedDict, weights_after: OrderedDict, meta_weights: OrderedDict,  model,
              outestepsize=0.8, outestepsize1: float = None, iteration=None, epochs=None, meta_optimizer=None,
              task_len: int = 1):
    if iteration is None or epochs is None or epochs == 0:
        raise ValueError('iterative error in meta step definition')
    if not meta_optimizer:
        if outestepsize1 is None:
            outerstepsize0 = outestepsize * (1 - iteration / epochs) / task_len
            state_dict = {name: meta_weights[name] + (weights_after[name] - weights_before[name]) * outerstepsize0
                          for name in weights_before}
            #model.load_state_dict(state_dict)
        else:
            outerstepsize0 = outestepsize * (1 - iteration / epochs) / task_len
            outerstep1 = outestepsize1 * (1 - iteration / epochs) / task_len
            state_dict = {name: meta_weights[name] + (
                    weights_after[name] - weights_before[name]) * outerstepsize0 if name.startswith(
                'conv_features') else meta_weights[name] + (weights_after[name] - weights_before[name]) * outerstep1 for
                          name in weights_before}
            #model.load_state_dict(state_dict)
    else:
        rd = []
        meta_optimizer.zero_grad()
        model.load_state_dict(meta_weights)
        model.train()
        for name in weights_before:
            if name.endswith('weight') or name.endswith('bias'):
                rd.append((weights_after[name] - weights_before[name])/task_len)
        for p, d in zip(model.parameters(), rd):
            p.grad = d
        meta_optimizer.step()
        model.eval()
        state_dict = deepcopy(model.state_dict())
    return state_dict


def meta_learner(model, task, epochs: int, batch_size: int, in_epochs: int, optimizer,
                 meta_optimizer, lr1, lr2, device, mode='epoch', early_stopping=0, val=None):
    if mode == 'epoch':
        n = 128   # todo resolve batch size
    elif mode == 'batch':
        n = batch_size
    elif mode == 'single_batch':
        n = batch_size
    else:
        raise ValueError('incorrect meta-training mode')
    flag = 0
    best_stat = 0
    best_stat_epoch = 0
    early_model = copy.deepcopy(model.state_dict())
    for iteration in range(epochs):
        meta_weights = deepcopy(model.state_dict())
        # Do SGD on this task
        task_len = len(task[0])
        for j in range(task_len):
            weights_before = deepcopy(meta_weights)
            for i in range(len(task)):
                dat = DataLoader(task[i][j], batch_size=n, drop_last=False, shuffle=False)
                val_loader = None
                if val is not None:
                    val_loader = DataLoader(val[i], batch_size=len(val[i]), drop_last=False, shuffle=False)
                model.load_state_dict(weights_before)
                best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), dat, val_loader,
                                             epochs=in_epochs, device=device, logging=False)
                if val is not None:
                    weights_after = deepcopy(best_model.state_dict())
                else:
                    weights_after = deepcopy(model.state_dict())
                meta_weights = meta_step(weights_before=weights_before, weights_after=weights_after,
                                         meta_weights=meta_weights, outestepsize=lr1,
                                         outestepsize1=lr2, epochs=epochs,
                                         iteration=iteration, meta_optimizer=meta_optimizer, model=model,
                                         task_len=len(task[i][j]) * len(task[i]))   # fixme size of task?
            if early_stopping != 0:
                stat = 0.0
                model.load_state_dict(meta_weights)
                for val_d in val:
                    test_data_loader = DataLoader(val_d, batch_size=500, drop_last=False, shuffle=False)
                    with torch.no_grad():
                        for batch in test_data_loader:
                            inputs, targets = batch
                        inputs = inputs.to(device=device, dtype=torch.float)
                        output = model(inputs)
                        targets = targets.type(torch.LongTensor)
                        targets = targets.to(device=device, dtype=torch.float)
                        correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                        stat += torch.sum(correct).item() / len(targets)
                stat = stat / len(val)
                if stat > best_stat:
                    print('in process ' + str(multiprocessing.current_process().name) +
                          '\nnew best stat: {}, on epoch: {}/{}, batch: {}/{}'.format(stat, iteration, epochs,
                                                                                      j, task_len))
                    best_stat = stat
                    flag = 0
                elif flag == 0:
                    early_model = copy.deepcopy(model.state_dict())
                    best_stat_epoch = iteration
                    flag = 1
                else:
                    flag += 1
                if flag >= early_stopping:
                    print('Early stopping! With best stat: {}, on epoch: {}'.format(best_stat, best_stat_epoch))
                    model.load_state_dict(early_model)
                    return model
        model.load_state_dict(meta_weights)

    return model


def params_pretrain(trial, model, metadataset: MetaDataset, tr_sub, tst_sub, double_meta_step=False,
                    mode='epoch', meta_optimizer=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    if not meta_optimizer:
        if double_meta_step:
            params = {
                'innerepochs': trial.suggest_int('innerepochs', 1, 50),
                'oterepochs': trial.suggest_int('oterepochs', 2, 100),
                'outerstepsize0': trial.suggest_float('outerstepsize0', 0.01, 0.9),
                'outerstepsize1': trial.suggest_float('outerstepsize1', 0.01, 0.9),
                'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
                'in_datasamples': trial.suggest_int('in_datasamples', 1, 64)
            }
        else:
            params = {
                'innerepochs': trial.suggest_int('innerepochs', 1, 10),
                'oterepochs': trial.suggest_int('oterepochs', 2, 20),
                'outerstepsize0': trial.suggest_float('outerstepsize0', 0.0006, 0.5),
                'outerstepsize1': None,
                'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
                'in_datasamples': trial.suggest_int('in_datasamples', 1, 64)
            }
    else:
        params = {
            'innerepochs': trial.suggest_int('innerepochs', 1, 20),
            'oterepochs': trial.suggest_int('oterepochs', 2, 20),
            'outerstepsize0': trial.suggest_float('outerstepsize0', 0.0006, 0.3),
            'outerstepsize1': None,
            'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
            'in_datasamples': trial.suggest_int('in_datasamples', 1, 50)
        }
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['outerstepsize0'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
    n = params['in_datasamples']
    # Generate task
    task_t = []
    for i in range(len(tr_sub)):
        train_tasks, _ = metadataset.all_data_subj(subj=tr_sub[i], n=n, mode=mode)
        task_t.append(train_tasks)  # todo size of meta batches
    model = meta_learner(model, task_t, params['oterepochs'], n, params['innerepochs'], optimizer, meta_optimizer,
                         params['outerstepsize0'], params['outerstepsize1'], device, mode)
    stat = 0
    for test_sub in tst_sub:
        train_data, test_data, _ = metadataset.part_data_subj(subj=test_sub, n=n, rs=42)
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


def meta_params(metadataset: MetaDataset, tr_sub: list, tst_sub: list, model, trials=50, jobs=1, mode='single_batch',
                double_meta_step=False, meta_optimizer=False, experiment_name='experiment'):
    p = 1
    lib_name = 'params_for_meta_training'
    path = pathlib.Path(pathlib.Path.cwd(), experiment_name)
    while os.path.exists(pathlib.Path(path, lib_name + str(p))):
        p += 1
    os.mkdir(pathlib.Path(path, lib_name + str(p)))
    path = pathlib.Path(path, lib_name + str(p))
    file_path = str(path.resolve()) + "/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    func = lambda trial: params_pretrain(trial, model=model, metadataset=metadataset, tr_sub=tr_sub,
                                         tst_sub=tst_sub, double_meta_step=double_meta_step, mode=mode,
                                         meta_optimizer=meta_optimizer)
    study = optuna.create_study(storage=storage, study_name='optuna_meta_params', direction='maximize',
                                load_if_exists=True)
    study.optimize(func, n_trials=trials, n_jobs=jobs)
    params = study.best_params
    return params


def meta_train(params: dict, model, metadataset: MetaDataset, wal_sub, path, name: str = None,
               mode='single_batch', meta_optimizer=False, subjects: list = None, loging=True, baseline=True,
               early_stopping=0):
    if name is None:
        name = str(wal_sub)
    if subjects is None:
        subjects = metadataset.subjects
    if 'outerstepsize1' not in params.keys():
        params.update(outerstepsize1=None)
    if loging:
        print("mata train for sub: " + str(subjects) + " started  in process"
              + str(multiprocessing.current_process().name))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    bs_model = deepcopy(model)
    bs_weights = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
    if meta_optimizer:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['outerstepsize0'])
    n = params['in_datasamples']
    # Generate task
    task_t = []
    val_task = []
    for i in range(len(subjects)):
        train_tasks, val = metadataset.all_data_subj(subj=subjects[i], n=n, mode=mode, early_stopping=early_stopping)
        task_t.append(train_tasks)  # todo move dataloading to meta-learner
        val_task.append(val)
    model = meta_learner(model, task_t, params['oterepochs'], n, params['innerepochs'], optimizer, meta_optimizer,
                         params['outerstepsize0'], params['outerstepsize1'], device, mode, early_stopping, val=val_task)
    torch.save(model.state_dict(), (path + str(name) + "-reptile.pkl"))
    if loging:
        print("meta train for sub: " + str(subjects) + "completed")
    test_data = metadataset.test_data_subj(subj=wal_sub)
    test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, targets = batch
        inputs = inputs.to(device=device, dtype=torch.float)
        output = model(inputs)
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device=device, dtype=torch.float)
        correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
        stat = torch.sum(correct).item()/len(targets)
    if baseline:
        model = bs_model
        model.load_state_dict(bs_weights)
        optim = torch.optim.Adam(model.parameters(), lr=params['in_lr']*params['outerstepsize0']/params['oterepochs'])
        data = metadataset.multiple_data(subjects)
        data = DataLoader(data, batch_size=64, drop_last=True, shuffle=True)
        best_model, best_acc = train(model, optim, torch.nn.CrossEntropyLoss(), data,
                                     epochs=params['innerepochs'] * params['oterepochs'], device=device, logging=loging)
        torch.save(model.state_dict(), (path + name + "-baseline.pkl"))
        test_data = metadataset.test_data_subj(subj=wal_sub)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            stat1 = torch.sum(correct).item()/len(targets)
        return [stat, stat1]
    else:
        return stat


def meta_exp(params: dict, model, target_sub: list, metadataset: MetaDataset, mode='single_batch',
             meta_optimizer=False, num_workers=1, experiment_name='experiment', all_subjects: list = None,
             early_stopping=0):
    if all_subjects is None:
        all_subjects = metadataset.subjects
    path = './' + experiment_name + '/'
    os.mkdir(path + 'models/')
    path = path + 'models/'
    stat = []
    stat1 = []
    task = []
    #step = partial(meta_train, params=params, model=model, metadataset=metadataset, path=path,
    #               meta_optimizer=meta_optimizer)
    for sub in target_sub:
        target_subjects = deepcopy(all_subjects)
        target_subjects.remove(sub)
        task.append(tuple((params, model, metadataset, sub, path, None, mode, meta_optimizer, target_subjects, True,
                           True, early_stopping)))
    with multiprocessing.Pool(num_workers) as p:
        res = p.starmap(meta_train, task)
        p.close()
        p.join()
    for st in res:
        stat.append(st[0])
        stat1.append(st[1])
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


def old_meta_train(params: dict, model, all_subjects: list, target_sub: list, metadataset: MetaDataset,   # TODO remove
               meta_optimizer=False, num_workers=0, experiment_name='experiment'):
    stat = []
    stat1 = []
    if 'outerstepsize1' not in params.keys():
        params.update(outerstepsize1=None)
    p = 1
    path = './' + experiment_name + '/'
    os.mkdir(path + 'models/')
    path = path + 'models/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    base_weights = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
    optim = torch.optim.Adam(model.parameters(), lr=params['in_lr']*params['outerstepsize0']/params['oterepochs'])
    if meta_optimizer:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['outerstepsize0'])
    else:
        meta_optimizer = None
    for sub in target_sub:
        model.load_state_dict(base_weights)
        subjects = deepcopy(all_subjects)
        subjects.remove(sub)
        for iteration in range(params['oterepochs']):   # iterations for outer epochs
            task_t = []
            # Generate task
            for i in range(len(subjects)):
                train_data, val_data, test_data = metadataset.part_data_subj(subj=subjects[i],
                                                                             n=params['in_datasamples'],
                                                                             rs=42 + iteration)
                task_t.append(DataLoader(train_data, batch_size=params['in_datasamples'],
                                         drop_last=True, shuffle=True))
            # Do SGD on this task
            for i in range(len(task_t)):
                start_weights = deepcopy(model.state_dict())
                best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), task_t[i],
                                             epochs=params['innerepochs'], device=device, logging=False)
                weights_after = deepcopy(model.state_dict())
                model = meta_step(weights_before=start_weights, weights_after=weights_after,
                                  outestepsize=params['outerstepsize0'], outestepsize1=params['outerstepsize1'],
                                  epochs=params['oterepochs'], iteration=iteration, meta_optimizer=meta_optimizer,
                                  model=model)
        torch.save(model.state_dict(), (path + str(sub) + "-reptile.pkl"))
        test_data = metadataset.test_data_subj(subj=sub)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False, num_workers=num_workers)
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            stat.append(torch.sum(correct).item()/len(targets))
        data = metadataset.other_data_subj(sub, all_subjects)
        data = DataLoader(data, batch_size=64, drop_last=True, shuffle=True, num_workers=num_workers)
        model.load_state_dict(base_weights)
        best_model, best_acc = train(model, optim, torch.nn.CrossEntropyLoss(), data,
                                     epochs=params['innerepochs'] * params['oterepochs'], device=device, logging=False)
        torch.save(model.state_dict(), (path + str(sub) + "-baseline.pkl"))
        test_data = metadataset.test_data_subj(subj=sub)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False,
                                      num_workers=num_workers)
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


def p_aftrain(trial, model, metadataset: MetaDataset, tst_sub, experiment_name, last_layer=False):
    a = trial.suggest_int('a_ep', 2, 20)
    params = {
              'lr': trial.suggest_float('lr', 0.00003, 0.05),
              'a': a,
              'b': trial.suggest_int('b_ep', 0, 4*a-1)
              }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    model.to(device)
    if last_layer:
        for n, p in model.named_parameters():
            p.requires_grad = False
            if n.startswith('out_features'):
                p.requires_grad = True
    stat = 0
    for sub in tst_sub:
        for j in range(0, 51, 4):
            model.load_state_dict(torch.load(experiment_name + "/models/" + str(sub) + "-reptile.pkl"))
            if not j == 0:
                train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=j, rs=42)
                data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                             epochs=int(j*params['a']-params['b']), device=device, logging=False)

            else:
                train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=1, rs=42)
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


def aftrain_params(metadataset: MetaDataset, model, tst_subj: list, trials: int, jobs: int, experiment_name='experiment',
                   last_layer=False):
    path = pathlib.Path(pathlib.Path.cwd(), experiment_name)
    os.mkdir(pathlib.Path(path, 'af_params'))
    path = pathlib.Path(path, 'af_params')
    file_path = "./" + experiment_name + "/af_params/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    func = lambda trial: p_aftrain(trial, model=model, metadataset=metadataset, tst_sub=tst_subj,
                                   experiment_name=experiment_name, last_layer=last_layer)
    study = optuna.create_study(storage=storage, study_name='optuna_af_params', direction='maximize')
    study.optimize(func, n_trials=trials, n_jobs=jobs)
    af_params = study.best_params
    with open("./" + experiment_name + "/af_params/aftrain_params.txt", 'w') as fp:
        for k, v in af_params.items():
            fp.write(str(k) + ' == ' + str(v) + '\n\n')
    return af_params


def aftrain(target_sub, model, af_params, metadataset: MetaDataset, iterations=1, logging=False,
            experiment_name='experiment', last_layer=False):
    dt = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=af_params['lr'])
    model.to(device)
    if last_layer:
        for n, p in model.named_parameters():
            p.requires_grad = False
            if n.startswith('out_features'):
                p.requires_grad = True
    cn = metadataset.data[target_sub[0]]['train']['Y'].nunique(dropna=True)
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
            for j in range(0, 51, cn):   # todo resolve 4 to number of dataclasses DONE
                if logging:
                    print('performed ' + str(j) + ' steps')
                in_data_points.append(j)
                model.load_state_dict(torch.load(experiment_name + "/models/" + str(sub) + "-reptile.pkl"))
                if not j == 0:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=j, rs=42+k)
                    data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                    best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                                 epochs=int(j*af_params['a_ep']-af_params['b_ep']),
                                                 device=device, logging=False)

                else:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=1, rs=42+k)
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
                model.load_state_dict(torch.load(experiment_name + "/models/" + str(sub) + "-baseline.pkl"))
                if not j == 0:
                    train_data, test_data = metadataset.last_n_data_subj( subj=sub, train=j, rs=42+k,)
                    data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                    best_model, best_acc = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                                 epochs=int(j*af_params['a_ep']-af_params['b_ep']),
                                                 device=device, logging=False)

                else:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=1, rs=42+k)
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
    path = pathlib.Path(pathlib.Path.cwd(), experiment_name)
    os.mkdir(pathlib.Path(path, 'af_run'))
    path = pathlib.Path(path, 'af_run')
    fig, ax = plt.subplots(len(target_sub)+1, sharex=False, sharey=True, figsize=(12, 9*(len(target_sub)+1)))
    i = 0
    A_acc = []
    A_accr = []
    for sub in target_sub:
        acc = np.mean(np.array(dt[str(sub) + '_baseline_data']), axis=0)
        A_acc.append(acc)
        std = np.std(np.array(dt[str(sub) + '_baseline_data']), axis=0)
        accr = np.mean(np.array(dt[str(sub) + '_reptile_data']), axis=0)
        A_accr.append(accr)
        stdr = np.std(np.array(dt[str(sub) + '_reptile_data']), axis=0)
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
    sub = target_sub[0]
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
