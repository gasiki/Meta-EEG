from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import (exponential_moving_standardize, preprocess, Preprocessor)
from braindecode.datautil.windowers import create_windows_from_events
import Meta_EEG as me
import json
import shutil
from sys import argv


def load_data(dataset=None, subject_ids=None, tst_s=0.2):
    dataset_name = None
    if dataset is None:
        ValueError('Dataset name is required')
    else:
        if dataset == 'BCI_IV_2a':
            ds = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_ids)
            low_cut = 4.
            high_cut = 38.
            factor_new = 1e-3
            init_block_size = 1000
            preprocessors = [
                Preprocessor('pick_types', eeg=True, meg=False, stim=False),
                Preprocessor(lambda x: x*1e6),
                Preprocessor('filter', l_freq=low_cut, h_freq=high_cut),
                Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size)
            ]
            preprocess(ds, preprocessors)
            sfreq = ds.datasets[0].raw.info['sfreq']
            assert all([d.raw.info['sfreq'] == sfreq for d in ds.datasets])
            trial_start_ofset_samples = 0
            w_ds = create_windows_from_events(
                ds,
                trial_start_offset_samples=trial_start_ofset_samples,
                trial_stop_offset_samples=0,
                preload=True
            )
            dataset_name = me.data_from_windows_dataset(dataset=w_ds, dataset_name=dataset, subjects=subject_ids,
                                                        description='filtration 4-38+exp_m_sd (only EEG channels)',
                                                        test_size=tst_s)
        else:
            ValueError('Unrecognized dataset name')
    return dataset_name


def main(inputs_file):
    with open(str(inputs_file) + '.json', 'r') as f:
        config = json.load(f)
    dataset_name = load_data(config['dataset']['dataset_name'], config['dataset']['subject_ids'],
                             config['dataset']['test_size'])
    print('Dataset ' + dataset_name + ' loaded.')
    model = me.model_from_params(config['model'])
    params = me.meta_params(dataset_name=dataset_name, tr_sub=config['dataset']['pretrain_sub'],
                            tst_sub=config['dataset']['val_sub'], model_params=config['model'],
                            trials=config['pretrain_trials'], jobs=config['pretrain_jobs'],
                            double_meta_step=config['double_meta_step'],
                            meta_optimizer=config['meta_optimizer'])
    pretraining_auc = me.pretrain(params=params, model_params=config['model'],
                                  all_subjects=config['dataset']['subject_ids'],
                                  target_sub=config['dataset']['target_sub'],
                                  dataset_name=dataset_name, meta_optimizer=config['meta_optimizer'])
    print('Pretraining completed. Mean auc for pretraining: ' + str(pretraining_auc))
    af_params = me.aftrain_params(dataset_name=dataset_name, model_params=config['model'],
                                  tst_subj=config['dataset']['val_sub'],
                                  trials=config['aftrain_trials'], jobs=config['aftrain_jobs'])
    me.aftrain(target_sub=config['dataset']['target_sub'], model_params=config['model'],
               af_params=af_params, dataset_name=dataset_name, iterations=config['aftrain_iterations'])
    shutil.make_archive('results_for_' + str(inputs_file), 'zip', 'data_' + str(dataset_name))


script, inputs_num = argv
if __name__ == '__main__':
    main(inputs_num)
    print('Done!')
