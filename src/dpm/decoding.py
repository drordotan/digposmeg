#
# Functions for decoding
#

from __future__ import division

import os
import multiprocessing
import pickle
import time
import tempfile
import copy

import pmne.split
import pmne

import dpm.files
from dpm import write_to_log

'''
#------------------------------------------------------------------
class StratifiedKFoldFiltered(StratifiedKFold):

    def __init__(self, y, y_include_in_test_set, n_folds=3, shuffle=False, random_state=None):
        super(StratifiedKFoldFiltered, self).__init__(n_folds, shuffle, random_state)

        if len(y_include_in_test_set) != len(y):
            raise ValueError("StratifiedKFoldFiltered err: len(y) != len(y_include_in_test_set)")

        self.y_include_in_test_set = np.array([bool(x) for x in y_include_in_test_set])


    def __iter__(self):
        ind = np.arange(self.n)
        for test_index in self._iter_test_masks():
            train_index = np.logical_not(test_index)
            train_index = ind[train_index]
            test_index = test_index & self.y_include_in_test_set
            test_index = ind[test_index]
            yield train_index, test_index
'''


#------------------------------------------------------------------
def create_evoked(epochs, cond_names):
    return [epochs[cond].average() for cond in cond_names]


#=============================================================================================================
#  Functions to fit classifiers
#=============================================================================================================

#--------------------------------------------------------------------------------
def run_fit_multi_subjects(classifier_specs, subj_ids, out_dir, data_filenames, filename_suffix,
                           load_error_trials=False, decim=1, meg_channels=True,
                           env_per_subj=False, decoding_method='standard',
                           sliding_window_size=100, sliding_window_step=20,
                           tmin=-0.1, tmax=1.0, baseline=(None, 0), reject=None, generalize_across_time=True, create_dir=False):

    if create_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(out_dir):
        raise Exception('Output directory does not exist: {:}'.format(out_dir))

    log_fn = '{:}/fit_{:}_{:.0f}_{:.0f}.log'.format(out_dir, decoding_method, tmin*1000, tmax*1000)
    write_to_log(log_fn, 'Started at {:}\n'.format(time.strftime('%D %H:%M:%S')))
    write_to_log(log_fn, 'Processing subjects {:}\n'.format(subj_ids))

    if env_per_subj:
        #-- Run each subject in a separate python environment (interpreter)
        script_path = _generate_script('fit_one_subject.py',
                                       {'DECIM': decim,
                                        'MEG_CHANNELS': meg_channels,
                                        'DECODING_METHOD': decoding_method,
                                        'SLIDING_WINDOW_SIZE': sliding_window_size,
                                        'SLIDING_WINDOW_STEP': sliding_window_step,
                                        'TMIN': tmin,
                                        'TMAX': tmax,
                                        'BASELINE': baseline,
                                        'GAT': generalize_across_time,
                                        'filename_suffix': filename_suffix,
                                        'load_error_trials': load_error_trials,
                                        },
                                       {'CLASSIFIER_SPECS': classifier_specs['code_writer']})
        for subj in subj_ids:
            _run_script(script_path, [subj, out_dir, data_filenames, log_fn])

    else:
        #-- Run everything in the current interpreter
        for subj in subj_ids:
            fit_one_subject(subj, data_filenames=data_filenames, classifier_specs=classifier_specs, out_dir=out_dir, log_fn=log_fn,
                            load_error_trials=load_error_trials, decim=decim, meg_channels=meg_channels,
                            decoding_method=decoding_method, sliding_window_size=sliding_window_size,
                            sliding_window_step=sliding_window_step, tmin=tmin, tmax=tmax, baseline=baseline, reject=reject,
                            generalize_across_time=generalize_across_time, filename_suffix=filename_suffix)

    with open(log_fn, 'a') as fp:
        fp.write('\nFinished all subjects.\n')


#--------------------------------------------------------------------------------------
def _generate_script(script_name, params, explicit_string_params=None):

    template_fn = os.path.dirname(__file__) + '/' + script_name + '.templ'
    with open(template_fn, 'r') as fp:
        script = fp.read()

    for key, value in params.items():
        keyword = '${:}$'.format(key)
        if isinstance(value, str):
            value = "'{:}'".format(value)
        else:
            value = str(value)

        script = script.replace(keyword, value)

    explicit_string_params = explicit_string_params or {}
    for key, value in explicit_string_params.items():
        keyword = '${:}$'.format(key)
        value = str(value)
        script = script.replace(keyword, value)

    script_path = tempfile.gettempdir() + '/' + script_name
    with open(script_path, 'w') as fp:
        fp.write(script)

    return script_path


#--------------------------------------------------------------------------------------
def _run_script(script_path, args):

    cmd = "python " + script_path + " " + " ".join(['"' + str(a) + '"' for a in args])
    print('>>> Running: {:}'.format(cmd))
    os.system(cmd)


#--------------------------------------------------------------------------------
# noinspection PyIncorrectDocstring
def fit_one_subject(subj_id, data_filenames, classifier_specs, out_dir, filename_suffix,
                    load_error_trials=False, log_fn=None, decim=1, meg_channels=True,
                    decoding_method='standard', sliding_window_size=None, sliding_window_step=None,
                    tmin=-0.1, tmax=1.0, baseline=(None, 0), lopass_filter=None, reject=None, generalize_across_time=True):
    """
    :param classifier_specs: list of dicts, each having 2 entries: y_label_func and epoch_filter
    """

    print('\n\n\n\n========================  Processing subject {:} =======================\n\n\n'.format(subj_id))

    write_to_log(log_fn, 'Started processing subject {:} at {:}\n'.format(subj_id, time.strftime('%D %H:%M:%S')))

    sdata = dpm.files.load_subj_data(subj_id, data_filenames, lopass_filter, load_error_trials=load_error_trials)

    epochs = _create_stim_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject)

    fit_results = fit_classifier(epochs, classifier_specs, decoding_method, sliding_window_size, sliding_window_step, only_fit=False,
                                 generalize_across_time=generalize_across_time)

    print('\n\n----- {:}: Saving results to {:}'.format(subj_id, out_dir))
    _save_extra_config_params(fit_results, dict(tmin=tmin, tmax=tmax, lopass_filter=lopass_filter, baseline=baseline,
                              decim=decim, meg_channels=meg_channels))
    filename_id = _filename(subj_id, decoding_method, tmin * 1000, tmax * 1000, filename_suffix)
    _save_fitting_results(fit_results, out_dir, filename_id)

    scoring_results = dict(scores=fit_results['scores'], times=fit_results['times'])
    _save_decoding_scores(scoring_results, out_dir, filename_id, classifier_specs)

    write_to_log(log_fn, 'Finished processing subject {:}\n'.format(subj_id))


#--------------------------------------------------------------------------------
def fit_epochs(epochs, classifier_specs, out_dir, subj_id, filename_suffix,
               decoding_method='standard', sliding_window_size=None, sliding_window_step=None):

    print('\n\n\n\n========================  Processing subject {:} =======================\n\n\n'.format(subj_id))

    decoding_result = fit_classifier(epochs, classifier_specs, decoding_method, sliding_window_size, sliding_window_step, only_fit=True,
                                     generalize_across_time=False)

    print('\n\n----- {:}: Saving results to {:}'.format(subj_id, out_dir))
    filename_id = _filename(subj_id, decoding_method, epochs.tmin * 1000, epochs.tmax * 1000, filename_suffix)
    _save_fitting_results(decoding_result, out_dir, filename_id)
    # There's no scoring here, so this is probably not needed: _save_decoding_scores(decoding_result, out_dir, filename_id, classifier_specs)


#-------------------------
def _create_stim_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject=None):
    # noinspection PyProtectedMember
    return umne._epochs.create_epochs_from_raw(raw=sdata.raw, events=sdata.stimulus_events, metadata=sdata.stimulus_metadata,
                                               meg_channels=meg_channels, tmin=tmin, tmax=tmax, decim=decim, baseline=baseline, reject=reject)


#--------------------------
def _create_response_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject=None):
    """
    The window tmin tmax should contain mainly negative times as we want the last event to be the button press.
    """
    # noinspection PyProtectedMember
    return umne._epochs.create_epochs_from_raw(raw=sdata.raw, events=sdata.response_events, metadata=sdata.response_metadata,
                                               meg_channels=meg_channels, tmin=tmin, tmax=tmax, decim=decim, baseline=baseline, reject=reject)


#-------------------------
def fit_classifier(epochs, classifier_specs, decoding_method, sliding_window_size, sliding_window_step, only_fit, generalize_across_time):

    decoding_result = pmne.decoding.decode(epochs=epochs,
                                           get_y_label_func=classifier_specs['y_label_func'],
                                           epoch_filter=classifier_specs['epoch_filter'],
                                           decoding_method=decoding_method,
                                           sliding_window_size=sliding_window_size,
                                           sliding_window_step=sliding_window_step,
                                           n_jobs=multiprocessing.cpu_count(),
                                           only_fit=only_fit,
                                           generalize_across_time=generalize_across_time)

    return decoding_result


#-------------------------
def _save_extra_config_params(fit_results, params):
    for param, value in params.items():
        fit_results['config'][param] = value


#-------------------------
def _save_fitting_results(fit_result, out_dir, filename, epochs=None):

    filename = "{:}/decoder_{:}.pkl".format(out_dir, filename)
    print('Saving classifier fits to ' + filename)

    n_folds = fit_result['cv']
    data_to_save = dict(n_folds=n_folds,
                        epochs=None if epochs is None else epochs.metadata)

    if n_folds == 1:
        data_to_save['estimator'] = fit_result['estimator']
    else:
        data_to_save['estimators'] = fit_result['estimators']

    if 'config' in fit_result:
        data_to_save['config'] = fit_result['config']

    with open(filename, 'wb') as fp:
        pickle.dump(data_to_save, fp)


#-------------------------
def _filename(*args):
    return "_".join([_to_str(v) for v in args if v is not None])


def _to_str(v):
    return '{:.0f}'.format(v) if isinstance(v, float) else str(v)


#-------------------------
def _save_decoding_scores(decoding_results, out_dir, filename, classifier_specs=None):

    assert 'scores' in decoding_results, 'Invalid decoding results, keys={}'.format(list(decoding_results.keys()))
    assert 'times' in decoding_results, 'Invalid decoding results, keys={}'.format(list(decoding_results.keys()))

    filename = "{:}/scores_{:}.pkl".format(out_dir, filename)

    print('Saving decoding scores to ' + filename)

    data = copy.copy(decoding_results)

    if classifier_specs is not None:
        data['classifier_type'] = classifier_specs['code_writer']

    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


#=============================================================================================================
#  Fit classifiers to some of the data, and score on the rest of the data
#=============================================================================================================

#--------------------------------------------------------------------------------
def run_fit_decoder(subj_ids, classifier_specs, data_filenames, out_dir,
                    filename_suffix, load_error_trials=False, decim=1, meg_channels=True,
                    decoding_method='standard', sliding_window_size=100, sliding_window_step=10,
                    tmin=-0.1, tmax=1.0, baseline=(None, 0), lopass_filter=None, reject=None, generalize_across_time=True,
                    env_per_subj=False):

    if not os.path.exists(out_dir):
        raise Exception('Output directory does not exist: {:}'.format(out_dir))

    log_fn = '{:}/fit_sep_{:}.log'.format(out_dir, _filename(decoding_method, tmin*1000, tmax*1000))
    write_to_log(log_fn, 'Started at {:}\n'.format(time.strftime('%D %H:%M:%S')))
    write_to_log(log_fn, 'Processing subjects {:}\n'.format(subj_ids))

    if env_per_subj:
        #-- Run each subject in a separate python environment (interpreter)
        assert False  # todo later
        # noinspection PyUnreachableCode
        script_path = _generate_script('fit_score_separate.py',
                                       {})
        for subj in subj_ids:
            _run_script(script_path, [subj, out_dir, data_filenames, log_fn])

    else:
        #-- Run everything in the current interpreter
        for subj_id in subj_ids:
            fit_decoder(subj_id, classifier_specs, data_filenames, out_dir, load_error_trials=load_error_trials,
                        filename_suffix=filename_suffix, log_fn=log_fn, decim=decim, meg_channels=meg_channels,
                        decoding_method=decoding_method, sliding_window_size=sliding_window_size, sliding_window_step=sliding_window_step,
                        tmin=tmin, tmax=tmax, baseline=baseline, lopass_filter=lopass_filter, reject=reject,
                        generalize_across_time=generalize_across_time)

    with open(log_fn, 'a') as fp:
        fp.write('\nFinished all subjects.\n')


#--------------------------------------------------------------------------------
def fit_decoder(subj_id, classifier_specs, data_filenames, out_dir, filename_suffix,
                load_error_trials=False, log_fn=None, decim=1, meg_channels=True,
                decoding_method='standard', sliding_window_size=None, sliding_window_step=None,
                tmin=-0.1, tmax=1.0, baseline=(None, 0), lopass_filter=None, reject=None, generalize_across_time=True):

    print('\n\n\n\n========================  Processing subject {:} =======================\n\n\n'.format(subj_id))

    write_to_log(log_fn, 'Started processing subject {:} at {:}\n'.format(subj_id, time.strftime('%D %H:%M:%S')))

    sdata = dpm.files.load_subj_data(subj_id, data_filenames, lopass_filter, load_error_trials=load_error_trials)

    epochs = _create_stim_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject)

    fit_results = fit_classifier(epochs, classifier_specs, decoding_method, sliding_window_size, sliding_window_step, True,
                                 generalize_across_time=generalize_across_time)

    _save_extra_config_params(fit_results, dict(tmin=tmin, tmax=tmax, lopass_filter=lopass_filter, baseline=baseline, reject=reject,
                                                decim=decim, meg_channels=meg_channels))

    filename_id = _filename(subj_id, decoding_method, tmin * 1000, tmax * 1000)
    _save_fitting_results(fit_results, out_dir, _filename(filename_id, filename_suffix))

    write_to_log(log_fn, 'Finished processing subject {:}\n'.format(subj_id))


#=============================================================================================================
#  Fit classifiers to some of the data, and score on the rest of the data
#=============================================================================================================

#--------------------------------------------------------------------------------
def run_fit_and_score_on_separate_trials(subj_ids, classifier_specs_train, data_filenames, out_dir, grouping_metadata_fields,
                                         filename_suffix_fit, filename_suffix_scores=None, epoch_filter=None,
                                         load_error_trials=False, training_group_size=0.5,
                                         classifier_specs_test=None,
                                         decim=1, meg_channels=True,
                                         decoding_method='standard', sliding_window_size=100, sliding_window_step=20,
                                         tmin=-0.1, tmax=1.0, baseline=(None, 0), lopass_filter=None, reject=None, generalize_across_time=True,
                                         env_per_subj=False, on_response=False, cv=None, train_epoch_filter=None, test_epoch_filter=None,
                                         create_dir=False):

    if create_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(out_dir):
        dpm.util.create_folder(out_dir)
        # raise Exception('Output directory does not exist: {:}'.format(out_dir))

    log_fn = '{:}/fit_sep_{:}.log'.format(out_dir, _filename(decoding_method, tmin*1000, tmax*1000))
    write_to_log(log_fn, 'Started at {:}\n'.format(time.strftime('%D %H:%M:%S')))
    write_to_log(log_fn, 'Processing subjects {:}\n'.format(subj_ids))

    if env_per_subj:
        #-- Run each subject in a separate python environment (interpreter)
        script_path = _generate_script('fit_score_separate.py',
                                       {'grouping_metadata_fields': grouping_metadata_fields,
                                        'training_group_size': training_group_size,
                                        'load_error_trials': load_error_trials,
                                        'decim': decim,
                                        'meg_channels': meg_channels,
                                        'decoding_method': decoding_method,
                                        'sliding_window_size': sliding_window_size,
                                        'sliding_window_step': sliding_window_step,
                                        'tmin': tmin,
                                        'tmax': tmax,
                                        'baseline': baseline,
                                        'lopass_filter': lopass_filter,
                                        'filename_suffix_fit': filename_suffix_fit,
                                        'filename_suffix_scores': filename_suffix_scores,
                                        'generalize_across_time': generalize_across_time,
                                        'reject': reject,
                                        'on_response': on_response,
                                        'cv': cv,
                                        'train_epoch_filter': train_epoch_filter,
                                        'test_epoch_filter': test_epoch_filter
                                        },
                                       {'classifier_specs_train': classifier_specs_train})
        for subj in subj_ids:
            _run_script(script_path, [subj, out_dir, data_filenames, log_fn])

    else:
        #-- Run everything in the current interpreter
        for subj_id in subj_ids:
            fit_and_score_on_separate_trials(subj_id, classifier_specs_train, data_filenames, out_dir,
                                             grouping_metadata_fields=grouping_metadata_fields,
                                             filename_suffix_fit=filename_suffix_fit,
                                             filename_suffix_scores=filename_suffix_scores,
                                             epoch_filter=epoch_filter,
                                             load_error_trials=load_error_trials,
                                             training_group_size=training_group_size,
                                             classifier_specs_test=classifier_specs_test,
                                             log_fn=log_fn, decim=decim, meg_channels=meg_channels,
                                             decoding_method=decoding_method,
                                             sliding_window_size=sliding_window_size,
                                             sliding_window_step=sliding_window_step,
                                             tmin=tmin, tmax=tmax, baseline=baseline, lopass_filter=lopass_filter, reject=reject,
                                             generalize_across_time=generalize_across_time, on_response=on_response,
                                             cv=cv, train_epoch_filter=train_epoch_filter, test_epoch_filter=test_epoch_filter)

    with open(log_fn, 'a') as fp:
        fp.write('\nFinished all subjects.\n')


#--------------------------------------------------------------------------------
def fit_and_score_on_separate_trials(subj_id, classifier_specs_train, data_filenames, out_dir, grouping_metadata_fields,
                                     filename_suffix_fit, filename_suffix_scores=None, epoch_filter=None,
                                     load_error_trials=False, training_group_size=0.5,
                                     classifier_specs_test=None,
                                     log_fn=None, decim=1, meg_channels=True,
                                     decoding_method='standard', sliding_window_size=100, sliding_window_step=20,
                                     tmin=-0.1, tmax=1.0, baseline=(None, 0), lopass_filter=None, reject=None,
                                     generalize_across_time=True, on_response=False, cv=None, train_epoch_filter=None, test_epoch_filter=None):

    print('\n\n\n\n========================  Processing subject {:} =======================\n\n\n'.format(subj_id))

    classifier_specs_test = classifier_specs_test or [classifier_specs_train]
    if filename_suffix_scores is None:
        filename_suffix_scores = [filename_suffix_fit]
    assert len(classifier_specs_test) == len(filename_suffix_scores)

    write_to_log(log_fn, 'Started processing subject {:} at {:}\n'.format(subj_id, time.strftime('%D %H:%M:%S')))

    sdata = dpm.files.load_subj_data(subj_id, data_filenames, lopass_filter, load_error_trials=load_error_trials)

    if on_response:
        all_epochs = _create_response_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject)
    else:
        all_epochs = _create_stim_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject)

    if epoch_filter is not None:
        all_epochs = all_epochs[epoch_filter]

    if cv is None:
        train_epochs, test_epochs = _get_train_test_epochs_for_one_fold(all_epochs, grouping_metadata_fields, test_epoch_filter,
                                                                        train_epoch_filter, training_group_size)
        n_folds = 1

    else:
        if train_epoch_filter is not None:
            Exception('There is a filter for training and testing but you want to do cross validation. This is not compatible.')
        train_epochs, test_epochs = pmne.split.split_train_test_cv(all_epochs, grouping_metadata_fields, cv=cv)
        n_folds = cv

    #-- Fit
    fit_results = _fit_for_multiple_folds(classifier_specs_train, decoding_method, generalize_across_time, n_folds, sliding_window_size,
                                          sliding_window_step, train_epochs)

    _save_extra_config_params(fit_results, dict(tmin=tmin, tmax=tmax, lopass_filter=lopass_filter, baseline=baseline,
                                                decim=decim, meg_channels=meg_channels))

    filename_id = _filename(subj_id, decoding_method, tmin*1000, tmax*1000)
    _save_fitting_results(fit_results, out_dir, _filename(filename_id, filename_suffix_fit, '' if n_folds == 1 else '_folds'))

    #-- Score according to each classifier_specs_test
    for cspec, suffix_save in zip(classifier_specs_test, filename_suffix_scores):
        scoring_results = _score_multiple_folds(fit_results, train_epochs, test_epochs, cspec, n_folds)
        _save_decoding_scores(scoring_results, out_dir, _filename(filename_id, suffix_save), cspec)

    write_to_log(log_fn, 'Finished processing subject {:}\n'.format(subj_id))


#------------------------------------------
def _get_train_test_epochs_for_one_fold(all_epochs, grouping_metadata_fields, test_epoch_filter, train_epoch_filter, training_group_size):

    train_epochs, test_epochs = umne.split.split_by_event_type(all_epochs, grouping_metadata_fields, training_group_size)

    if train_epoch_filter is not None:
        train_epochs = train_epochs[train_epoch_filter]
        test_epochs = test_epochs[test_epoch_filter]

    print('Fitting on {:} epochs, scoring on {:} epochs\n'.format(len(train_epochs), len(test_epochs)))
    train_epochs = [train_epochs]
    test_epochs = [test_epochs]

    return train_epochs, test_epochs


#------------------------------------------
def _fit_for_multiple_folds(classifier_specs_train, decoding_method, generalize_across_time, n_folds, sliding_window_size, sliding_window_step,
                            train_epochs):
    estimators = []
    fit_results = None

    for i_fold in range(n_folds):
        fit_results = fit_classifier(train_epochs[i_fold], classifier_specs_train, decoding_method, sliding_window_size, sliding_window_step, True,
                                     generalize_across_time=generalize_across_time)
        estimators.append(fit_results['estimator'])
        del fit_results['estimator']

    fit_results['estimators'] = estimators

    return fit_results


#------------------------------------------
def _score_multiple_folds(fit_results, train_epochs, test_epochs, cspec, n_folds):

    estimators = fit_results['estimators']

    all_folds_scores = []
    for i_fold in range(n_folds):
        curr_fold_scores = _score_estimator(test_epochs[i_fold], fit_results['preprocess'], estimators[i_fold], cspec['y_label_func'],
                                            cspec['epoch_filter'])
        all_folds_scores.append(curr_fold_scores)

    scoring_results = dict(procedure=fit_results['procedure'],
                           estimators=estimators,
                           scores=all_folds_scores,
                           pipeline=fit_results['pipeline'],
                           preprocess=fit_results['preprocess'],
                           cv=fit_results['cv'],
                           times=fit_results['times'],
                           config=fit_results['config'],
                           train_epochs=[e.metadata for e in train_epochs],
                           test_epochs=[e.metadata for e in test_epochs])

    return scoring_results


#--------------------------------------------------------------------------------
def _score_estimator(epochs, preprocess_pipeline, estimator, y_label_func, epochs_filter):
    """
    Compute score of an already-existing classifier for a new set of epochs

    The scores are updated on fit_results
    """
    if epochs_filter is not None:
        epochs = epochs[epochs_filter]

    X = preprocess_pipeline.fit_transform(epochs.get_data())
    y_true = y_label_func(epochs)

    return estimator.score(X, y_true)


#=============================================================================================================
#  Functions to apply an existing classifier to new data
#=============================================================================================================

def run_score_existing_decoder(subj_ids, fit_results_filename, data_filenames, out_dir, get_y_label_func,
                               filename_suffix, tmin=None, tmax=None, baseline=None):

    # get from the decoder: decim, meg_channels, decoding method, sliding window size/step,
    # tmin/tmax (we can override it to narrow down or to test time generalization), lopass_filter
    # baseline

    if not os.path.exists(out_dir):
        raise Exception('Output directory does not exist: {:}'.format(out_dir))

    log_fn = '{:}/apply_{:}.log'.format(out_dir, os.path.basename(fit_results_filename))
    write_to_log(log_fn, 'Started at {:}\n'.format(time.strftime('%D %H:%M:%S')), False)
    write_to_log(log_fn, 'Processing subjects {:}\n'.format(subj_ids))

    for subj in subj_ids:
        score_existing_decoder(subj, fit_results_filename=fit_results_filename, data_filenames=data_filenames,
                               out_dir=out_dir, get_y_label_func=get_y_label_func, filename_suffix=filename_suffix,
                               log_fn=log_fn, tmin=tmin, tmax=tmax, baseline=baseline)

    write_to_log(log_fn, '\nFinished scoring all subjects\n')


#--------------------------------------------------------------------------------
def score_existing_decoder(subj_id, fit_results_filename, data_filenames, out_dir, get_y_label_func, filename_suffix,
                           load_error_trials=False, log_fn=None, tmin=None, tmax=None, baseline=None, reject=None):

    print('\n\n\n\n========================  Processing subject {:} =======================\n\n\n'.format(subj_id))

    write_to_log(log_fn, 'Started processing subject {:} at {:}\n'.format(subj_id, time.strftime('%D %H:%M:%S')))

    with open(out_dir+'/'+fit_results_filename.format(subj_id), 'rb') as fp:
        fit_results = pickle.load(fp)

    estimator = _get_estimator(fit_results)
    lopass_filter, decim, meg_channels, decoding_method, tmax, tmin, baseline = _get_fit_params(fit_results, tmax, tmin, baseline)

    sdata = dpm.files.load_subj_data(subj_id, data_filenames, lopass_filter, load_error_trials=load_error_trials)

    epochs = _create_stim_epochs(sdata, baseline, decim, meg_channels, tmax, tmin, reject)

    scores = _score_estimator(epochs, fit_results['preprocess'], estimator, get_y_label_func, None)  # todo epochs_filter

    scoring_result = dict(scores=scores, times=fit_results['times'])
    _save_decoding_scores(scoring_result, out_dir, _filename(subj_id, decoding_method, tmin * 1000, tmax * 1000, filename_suffix))

    write_to_log(log_fn, 'Finished processing subject {:}\n'.format(subj_id))


#-----------------------------------------------------------
def _get_estimator(fit_results):

    #-- support previous file format
    if 'estimators' in fit_results:
        assert len(fit_results['estimators']) == 1
        return fit_results['estimators'][0]

    assert 'estimator' in fit_results, "Invalid format for fit-results file {:}".format(filename)
    return fit_results['estimator']


#-----------------------------------------------------------
def _get_fit_params(fit_results, tmax, tmin, baseline):

    cfg = fit_results['config']
    lopass_filter = cfg['lopass_filter']
    decim = cfg['decim']
    meg_channels = cfg['meg_channels']
    decoding_method = cfg['decoding_method']
    tmax = cfg['tmax'] if tmax is None else tmax
    tmin = cfg['tmin'] if tmin is None else tmin
    baseline = cfg['baseline'] if baseline is None else baseline

    return lopass_filter, decim, meg_channels, decoding_method, tmax, tmin, baseline
