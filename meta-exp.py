import Meta_EEG as me
import json
import shutil
from sys import argv


def main(inputs_file):
    with open(str(inputs_file) + '.json', 'r') as f:
        config = json.load(f)
    experiment_name = me.experiment_storage(experiment_name=config['dataset']["experiment_name"])
    dataset = me.MetaDataset(dataset_name='bci_iv_2a')
    dataset.load_data('data_BCI_IV_2a')
    print('Dataset ' + dataset.dataset_name + ' loaded.')
    model = me.model_from_params(config['model'])
    params = me.meta_params(metadataset=dataset, tr_sub=config['dataset']['pretrain_sub'],
                            tst_sub=config['dataset']['val_sub'], model=model,
                            trials=config['pretrain_trials'], jobs=config['pretrain_jobs'],
                            double_meta_step=config['double_meta_step'],
                            meta_optimizer=config['meta_optimizer'], experiment_name=experiment_name)
    if inputs_file == 'test':
        print('Test params are used')
        params = {
            'innerepochs': 2,
            'oterepochs': 2,
            'outerstepsize0': 0.745,
            'in_lr': 0.0017,
            'in_datasamples': 2
        }
    pretraining_subjects = config['dataset']['target_sub'] + config['dataset']['val_sub']
    pretraining_auc = me.meta_train(params=params, model=model, all_subjects=config['dataset']['subject_ids'],
                                    target_sub=pretraining_subjects, metadataset=dataset,
                                    meta_optimizer=config['meta_optimizer'], experiment_name=experiment_name)
    shutil.make_archive('models_for_' + str(inputs_file), 'zip', str(experiment_name) + '/models')
    print('Pretraining completed. Mean auc for pretraining: ' + str(pretraining_auc))
    af_params = me.aftrain_params(metadataset=dataset, model=model,
                                  tst_subj=config['dataset']['val_sub'],
                                  trials=config['aftrain_trials'], jobs=config['aftrain_jobs'],
                                  experiment_name=experiment_name)
    if inputs_file == 'test':
        print('Test af_params are used')
        af_params = {
            'lr': 0.0005,
            'a_ep': 2,
            'b_ep': 0
        }
    me.aftrain(target_sub=config['dataset']['target_sub'], model=model,
               af_params=af_params, metadataset=dataset, iterations=config['aftrain_iterations'],
               experiment_name=experiment_name)
    shutil.make_archive('results_for_' + str(inputs_file), 'zip', str(experiment_name))


inputs_num = argv
if __name__ == '__main__':
    main(inputs_num[1])
    print('Done!')
