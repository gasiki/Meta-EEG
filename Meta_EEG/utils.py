import braindecode
import os
import numpy as np
import pandas as pd
import pathlib
import braindecode.datasets
import pickle


def data_from_windows_dataset(dataset: braindecode.datasets.base.BaseConcatDataset, subjects: list,
                              dataset_name:str=None, description:str =None, test_size=0.2):

    """
    :param dataset: is windows dataset from braindecode
    :param subjects: list of subjects in dataset
    :param dataset_name: name of dataset if it is not specified function will return the base name
    :param description: some description of dataset (notes about preprocessing)
    :param test_size: size of test data which vil contain equal datapoints of each class and will be taken from the end
     of dataset
    :return: name of dataset (str)
    :
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
        with open('./data_' + str(dataset_name) + "/subjects.txt", "w") as fp:
            pickle.dump(description, fp)
    return dataset_name


def BCI_IV_pretrain(trial):
    params = {
        'innerepochs': trial.suggest_int('inn_ep', 1, 20),
        'oterepochs': trial.suggest_int('out_ep', 2, 70),
        'outerstepsize0': trial.suggest_float('out_sts0', 0.01, 0.9),
        'outerstepsize1': trial.suggest_float('out_sts1', 0.01, 0.9),
        'in_lr': trial.suggest_float('lr', 0.000006, 0.005),
        'in_datasamples': trial.suggest_int('local_batch_size', 1, 50)
    }
    subjects = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    test_subjs = [3, 24, 25, 26]
    t_id = trial.number
    progress = Bar('Trial ' + str(t_id), max=params['oterepochs'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = nnm.inEEG_Net(num_classes=2, dropout1=0.52, dropout2=0.36, time_of_interest=2000, lowpass=50, point_reducer= 5)
    model.to(device)
    #criterion = nn.SoftMarginLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
    n = params['in_datasamples']
    for iteration in range(params['oterepochs']):
        task_t = []
        task_v = []
        weights_before = deepcopy(model.state_dict())
        progress.next()
        # Generate task
        for i in range(len(subjects)):
            train_data, val_data, test_data = bci4_part_data_subj(subj=subjects[i], n=n, j=42+iteration)
            task_t.append(DataLoader(train_data, batch_size=n, drop_last=True, shuffle=True))
            task_v.append(DataLoader(val_data, batch_size=n, drop_last=True, shuffle=True))
        # Do SGD on this task
        for i in range(len(task_t)):
            best_model, best_acc = nnm.train(model, optimizer, torch.nn.CrossEntropyLoss(), task_t[i], epochs=params['innerepochs'], device=device, logging=False)
            # Interpolate between current weights and trained weights from this task
            # I.e. (weights_before - weights_after) is the meta-gradient
            weights_after = model.state_dict()
            outerstepsize = params['outerstepsize0'] * (1 - iteration / params['oterepochs'])  # linear schedule
            outerstepsize1 = params['outerstepsize1'] * (1 - iteration / params['oterepochs'])  # linear schedule
            model.load_state_dict({name: weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize if name.startswith('conv_features') else weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize1 for name in weights_before})


    stat = 0
    for test_sub in test_subjs:
        train_data, test_data, _ = bci4_part_data_subj(subj=test_sub, n=n, j=42)
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
    progress.finish()
    return stat/len(test_subjs)

def meta_params():
    p = 1
    path = pathlib.Path(pathlib.Path.cwd(), 'Lee_data')
    while os.path.exists(pathlib.Path(path, 'params' + str(p))):
        p += 1
    os.mkdir(pathlib.Path(path, 'params' + str(p)))
    path = pathlib.Path(path, 'params' + str(p))
    file_path = "./Lee_data/params" + str(p) + "/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock("./Lee_data/params" + str(p) + "/journal.log")

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    #progress = []
    #manager = enlighten.get_manager()
    #for i in range(15):
    #    progress.append(manager.counter(total=350, desc='Trial -1', unit="Epochs", color="green"))
    #func = lambda trial: simple_pretrain(trial, progress=progress)
    study = optuna.create_study(storage=storage, study_name='optuna_meta_params', direction='maximize', load_if_exists=True)
    study.optimize(BCI_IV_pretrain, n_trials=50, n_jobs=4)
    joblib.dump(study, pathlib.Path(path, 'optuna_meta_params.pkl'))

    return params
