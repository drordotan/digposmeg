import os
from math import floor

import numpy as np

import dpm
import umne
import umne.util



#=============================================================================
class remapmatrix:
    """
    Remap dissimilarity matrices into new matrices
    """

    #-----------------------------------------------------------------------------
    @staticmethod
    def to_retinotopic(matrix, stimuli):
        """
        Map a dissimilarity matrix into an equivalent-but-larger matrix which should make it easier to visualize
        retinotopic effect. In the new matrix, two stimuli that share a digit in the same location will be adjacent
        """
        new_groups = [dict(digit=d, location=loc) for loc in range(6) for d in dpm.stimuli.decade_digits]

        def stimulus_matches_group_digit(stimulus, group):
            target = stimulus['target']
            stim_loc = stimulus['location']
            unit = target % 10
            if target < 10:
                return stim_loc == group['location'] and unit == group['digit']
            else:
                decade = int(np.floor(target / 10))
                return (stim_loc == group['location'] and decade == group['digit']) or \
                        (stim_loc == group['location'] - 1 and unit == group['digit'])

        return umne.rsa_old.remap_stimuli(matrix, stimuli, new_groups,
                                          filter_old_by_new_func=stimulus_matches_group_digit,
                                          sort_old_func=lambda a, b: a['target'] - b['target'],
                                          merge_stim_func=lambda s, g, i: merge_dicts([s, g, dict(ind_in_group=i)]))


#------------------------------------------------------------------------------------
def merge_dicts(list_of_dicts):
    """
    Merge several dict's into one.

    :param list_of_dicts: a list of dict objects. Later entries in the list will override preceding ones.
    """
    result = dict()
    for d in list_of_dicts:
        result.update(d)
    return result


#=============================================================================
class sortmatrix:
    """
    Sort dissimilarity matrices
    """

    @staticmethod
    def bytarget(rev=False):
        """
        This function returns a comparator function that sorts stimuli by the target number

        :param rev: if True, targets will be sorted in reverse order
        """
        if rev:
            def sortfunc(a, b):
                return - ((a['target']-b['target'])*10+(a['location']-b['location']))
        else:
            def sortfunc(a, b):
                return (a['target'] - b['target']) * 10 + (a['location'] - b['location'])
        return sortfunc

    @staticmethod
    def bylocation(rev=False):
        """
        This function returns a comparator function that sorts stimuli by their location

        :param rev: if True, locations will be sorted in reverse order
        """
        if rev:
            def sortfunc(a, b):
                return - ((a['location'] - b['location']) * 100 + (a['target'] - b['target']))
        else:
            def sortfunc(a, b):
                return (a['location'] - b['location']) * 100 + (a['target'] - b['target'])
        return sortfunc


#============================================================================================
#           Compute observed dissimilarity
#============================================================================================

#-------------------------------------------------------------------------------------------
def gen_dissimilarity_multi_subj(subj_ids, data_files, tmin, tmax, out_dir, out_filename_previx='dissimilarity',
                                 decim=1, meg_channels=True,
                                 event_ids=dpm.stimuli.all_event_ids(ndigits=2), n_pca=30,
                                     riemann=True, corr_metric='spearmanr', riemann_metric='riemann',
                                 sliding_window_size=None, sliding_window_step=None, sliding_window_min_size=None,
                                 zscore=False, include_diagonal=True, averaging_method='square'):
    """

    :param subj_ids:
    :param data_files:
    :param tmin:
    :param tmax:
    :param out_dir:
    :param out_filename_previx:
    :param decim:
    :param meg_channels:
    :param event_ids:
    :param n_pca:
    :param riemann:
    :param corr_metric:
    :param riemann_metric:
    :param sliding_window_size:
    :param sliding_window_step:
    :param sliding_window_min_size:
    :param zscore:
    :param include_diagonal:
    :param averaging_method: how to average the result matrices across subject. Use 'squared' for correlation metrics,
              'linear' for distance metrics
    """

    #todo: very important - save the stimuli in the file

    if sliding_window_size is None:
        time_points = range(int(tmin*1000), int(tmax*1000)+1, decim)
    else:
        #-- compute the center of each time window
        sliding_window_duration = sliding_window_size * decim
        sliding_window_step_duration = sliding_window_step * decim
        sliding_window_min_duration = (sliding_window_min_size or sliding_window_size) * decim
        time_points = np.array(range(int(tmin*1000),
                                     int(tmax*1000) - sliding_window_min_duration + 1,
                                     sliding_window_step_duration))
        last_window_time_duration = int(tmax*1000) - time_points[-1]
        time_points[:-1] += sliding_window_duration / 2
        time_points[-1] += last_window_time_duration / 2

    run_params = dict(meg_channels=meg_channels,
                      tmin=tmin,
                      tmax=tmax,
                      decim=decim,
                      event_ids=event_ids,
                      n_pca=n_pca,
                      riemann=riemann,
                      corr_metric=corr_metric,
                      riemann_metric=riemann_metric,
                      sliding_window_size=sliding_window_size,
                      sliding_window_step=sliding_window_step,
                      sliding_window_min_size=sliding_window_min_size,
                      zscore=zscore,
                      include_diagonal=include_diagonal,
                      time_points_ms=time_points)

    def _save_dissimilarity(filename, sid, dissimilarity_matrices):
        np.save(filename,
                dict(subj_ids=sid, dissim_matrices=dissimilarity_matrices, params=run_params))

    per_subj_dir = out_dir + os.sep + 'persubj'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(per_subj_dir):
        os.mkdir(per_subj_dir)

    dm_all_subjects = []

    for subj_id in subj_ids:
        sdata = dpm.files.load_raw(dpm.subj_path[subj_id], data_files)

        epochs = \
            umne.create_epochs_from_raw(sdata.raw, sdata.stimulus_events, meg_channels=meg_channels, tmin=tmin, tmax=tmax, decim=decim,
                                        reject=dict(grad=4000e-13, mag=4e-12))

        dm = umne.rsa_old.gen_observed_dissimilarity(epochs, event_ids=event_ids, n_pca=n_pca,
                                                     riemann=riemann, corr_metric=corr_metric, riemann_metric=riemann_metric,
                                                     sliding_window_size=sliding_window_size,
                                                     sliding_window_step=sliding_window_step,
                                                     sliding_window_min_size=sliding_window_min_size,
                                                     zscore=zscore, include_diagonal=include_diagonal)

        _save_dissimilarity(per_subj_dir + os.sep + out_filename_previx + '_' + subj_id + '.npy',
                            [subj_id], dm)

        dm_all_subjects.append(dm)

    #-- Average over all subjects
    average_dm = []
    for i in range(len(dm_all_subjects[0])):
        a = umne.rsa_old.average_matrices([m[i] for m in dm_all_subjects], averaging_method=averaging_method)
        average_dm.append(a)

    _save_dissimilarity(out_dir+os.sep+out_filename_previx+'_all_subj.npy',
                        subj_ids, average_dm)


#============================================================================================
#           Regressions
#============================================================================================

#------------------------------------------------------------------------------------------------------
def regress_one_subject(subj_id, subj_data, dissimilarity_predictor_funcs, dissimilarity_predictor_desc,
                        tmin, tmax, decim, meg_channels,
                        target_numbers, target_positions, riemann, window_size, metric,
                        n_pca=30, results_file_name='rsa_results', delete_prev_results=False):

    epochs = \
        umne.create_epochs_from_raw(subj_data.raw, subj_data.stimulus_events, meg_channels=meg_channels, tmin=tmin, tmax=tmax, decim=decim,
                                    reject=dict(grad=4000e-13, mag=4e-12))

    #-- Get results file name, load it if needed
    results_file = dpm.consts.results_dir + 'rsa/' + subj_id + "_" + results_file_name + ".npy"
    append_to_existing_results = os.path.exists(results_file) and not delete_prev_results
    if append_to_existing_results:
        print('Appending to previous results')
    else:
        print('WARNING: any previous results will be deleted')

    event_ids = [target*10+pos for target in target_numbers for pos in target_positions]
    target_pos_pairs = [(target, pos) for target in target_numbers for pos in target_positions]

    new_results = {}

    start_times = np.arange(tmin, tmax-window_size, window_size) if riemann else [0]
    if not riemann:
        # Ignore window size: only one window
        window_size = tmax - tmin

    for start_time in start_times:

        epochs_cropped = epochs.copy().crop(tmin=start_time, tmax=start_time+window_size) if riemann else epochs

        for curr_metric in metric:

            print('Processing time window {:.3f}-{:.3f}, {:}riemann method, metric={:}'.format(
                start_time, start_time + window_size, '' if riemann else 'non-', curr_metric))

            if riemann:
                observed_dissimilarity = umne.rsa_old.gen_observed_dissimilarity(epochs_cropped, event_ids, riemann=True,
                                                                                 n_pca=n_pca, riemann_metric=curr_metric)

            else:
                observed_dissimilarity = umne.rsa_old.gen_observed_dissimilarity(epochs, event_ids, riemann=False,
                                                                                 n_pca=n_pca, corr_metric=curr_metric)

            rr = umne.rsa_old.regress_dissimilarity(observed_dissimilarity, dissimilarity_predictor_funcs, target_pos_pairs)
            rr = np.array(rr)

            key = ('//riemann={:}//metric={:}//decim={:}//target_numbers={:}//target_pos={:}//' +
                   'n_pca={:}//predictors={:}//meg_channels={:}//')\
                .format(riemann, curr_metric, decim, target_numbers, target_positions, n_pca, dissimilarity_predictor_desc,
                        meg_channels)

            result_data = dict(subj_id=subj_id,
                               decim=decim,
                               meg_channels=meg_channels,
                               riemann=riemann,
                               metric=curr_metric,
                               target_numbers=target_numbers,
                               target_positions=target_positions,
                               n_pca=n_pca,
                               predictors=dissimilarity_predictor_desc,
                               tmin=start_time,
                               tmax=start_time + window_size,
                               result=rr)

            if key not in new_results:
                new_results[key] = []

            new_results[key].append(result_data)


    #-- Save results
    print('Saving all results')
    if append_to_existing_results:
        all_results = dict(np.load(results_file).item())
        for k, v in new_results.items():
            all_results[k] = v
        np.save(results_file, all_results)

    else:
        np.save(results_file, new_results)

    print('The results were saved to {:}'.format(results_file))

