import numpy as np
import re

import dpm

from dpm.rsa_old import dissimilarity as dis
import matplotlib.pyplot as plt


subjs = ['ag', 'am', 'at', 'bl', 'bo', 'cc', 'eb', 'ga', 'jm0', 'lb', 'lj', 'mn', 'mp']


dpm.rsa_old.gen_dissimilarity_multi_subj(subjs, dpm.consts.comparison_raw_files,
                                         tmin=0, tmax=0.7, out_dir='/temp/dissim', out_filename_previx='dissimilarity',
                                         decim=2, meg_channels=True,
                                         event_ids=dpm.stimuli.all_event_ids(ndigits=2), n_pca=30,
                                         riemann=False, corr_metric='spearmanr', riemann_metric='riemann',
                                         sliding_window_size=50, sliding_window_step=25, sliding_window_min_size=None,
                                         zscore=False, include_diagonal=True, averaging_method='linear')

#---------------------------------------------------------------
def run_one_subject(subj_id):

    print('\n\n\n\n=============================================')
    print('====  Working on subject {:}\n\n'.format(subj_id))

    subj_dir = dpm.consts.subj_path[subj_id]

    subj_data = dpm.files.load_raw(subj_dir, dpm.comparison_raw_files)

    dissimilarity_predictor_funcs = dis.decade_id, dis.unit_id, dis.location, dis.retinotopic_id
    dissimilarity_predictor_descs = 'Different decades', 'Different units', 'Location', 'Different digits in same location'

    dpm.rsa_old.regress_one_subject(subj_id, subj_data, dissimilarity_predictor_funcs, dissimilarity_predictor_descs,
                                    tmin=0, tmax=0.7, decim=1, meg_channels=True,
                                    target_numbers=dpm.stimuli.all_targets(ndigits=2), target_positions=range(5),
                                    riemann=True, window_size=0.1,
                                    metric=['riemann', 'kullback', 'logeuclid', 'euclid', 'wasserstein'],
                                    n_pca=30, results_file_name='rsa_results', delete_prev_results=True)

    dpm.rsa_old.regress_one_subject(subj_id, subj_data, dissimilarity_predictor_funcs, dissimilarity_predictor_descs,
                                    tmin=0, tmax=0.7, decim=4, meg_channels=True,
                                    target_numbers=dpm.stimuli.all_targets(ndigits=2), target_positions=range(5),
                                    riemann=False, window_size=None,
                                    metric=['spearman', 'euclidean'],
                                    n_pca=30, results_file_name='rsa_results', delete_prev_results=False)



#---------------------------------------------------------------

def plot_results(subj_ids, key_patterns, filename='rsa_results'):

    if isinstance(key_patterns, str):
        key_patterns = key_patterns,

    #-- A list of matching entries from all subjects. Each entry is
    entries = [e for sid in subj_ids for e in _load_subject(sid, key_patterns, filename)]
    if len(entries) == 0:
        print('Nothing to plot')
        return

    times = [e['tmin'] for e in entries[0]]
    pred_names = list(entries[0][0]['predictors'])

    beta = _get_data_as_matrix(entries)
    avg_beta = np.mean(beta, axis=0)

    if len(times) == 1 and avg_beta.shape[0] > 1:
        times = range(avg_beta.shape[0])

    plt.plot(times, avg_beta)
    plt.legend(pred_names)



#------------------------
def _get_data_as_matrix(entries):
    """
    Return a subjects x timepoints x predictors matrix
    """

    beta = []  # np.zeros(n_entries, n_times, n_predictors)
    for entry in entries:
        if len(entry) == 1 and len(entry[0]['result'].shape) == 2:
            beta.append(entry[0]['result'])
        else:
            entry_results = []
            for rr in entry:
                entry_results.append(rr['result'])
            beta.append(entry_results)

    #-- Omit the last predictor, which is the const
    return np.array(beta)[:,:,:-1]


def _load_subject(subj_id, key_patterns, filename):
    fn = dpm.consts.results_dir + 'rsa/' + subj_id + "_" + filename + ".npy"
    data = dict(np.load(fn).item())
    matching_entries = [data[k] for k in data.keys() if _key_matches(key_patterns, k)]
    return matching_entries


def _key_matches(key_patterns, key):
    for pat in key_patterns:
        if not re.match('.*' + pat + '.*', key):
            return False
    return True

#---------------------------------------------------------------

# run_one_subject('ga')

plot_results(['am', 'at', 'bo', 'ga', 'mp'], ['metric=rieman', 'riemann=True'])
