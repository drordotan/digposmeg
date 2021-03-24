"""
Module for plotting decoding-related stuff
"""
import math
import glob
import pickle
import numpy as np
import jr
import os

import matplotlib.pyplot as plt

import umne
from umne import stats


#------------------------------------------------
def load_and_plot(score_files_mask, out_dir, chance_level=None, fig_id=1, min_time=-math.inf, max_time=math.inf,
                  tmin_stats=None, tmax_stats=None, plot_subjects=False):

    scores, times, filenames = load_score_files(score_files_mask, False, return_file_names=True)

    if plot_subjects:
        print('Plotting and saving data for each of the {:} subjects'.format(scores.shape[0]))
        for ii in range(scores.shape[0]):
            filename = os.path.basename(filenames[ii])
            plot_scores(scores[ii, :], times, chance_level=chance_level, min_time=min_time, max_time=max_time)
            plt.gcf().savefig(str(out_dir) + filename[:-4] + '_diag.png')
    else:
        print('Plotting data of {:} subjects'.format(scores.shape[0]))
        plot_scores(scores, times, chance_level=chance_level, fig_id=fig_id, min_time=min_time, max_time=max_time,
                    tmin_stats=tmin_stats, tmax_stats=tmax_stats)


#------------------------------------------------
def load_and_plot_gat(score_files_mask, out_dir, chance_level=None, fig_id=2, min_time=-math.inf, max_time=math.inf, tmin_stats=None,
                      tmax_stats=None, plot_subjects=False):

    scores, times, filenames = load_score_files(score_files_mask, True, return_file_names=True)
    if plot_subjects:
        print('Plotting and saving data for each of the {:} subjects'.format(scores.shape[0]))
        for ii in range(scores.shape[0]):
            plot_gat_scores(scores[np.newaxis, ii, :, :], times, chance_level=chance_level, min_time=min_time, max_time=max_time)
            filename = os.path.basename(filenames[ii])
            plt.gcf().savefig(str(out_dir) + filename[:-4] + '_GAT.png')
            plt.close('all')
    else:
        print('Plotting data of {:} subjects'.format(scores.shape[0]))
        plot_gat_scores(scores, times, chance_level=chance_level, fig_id=fig_id, min_time=min_time, max_time=max_time,
                        tmin_stats=tmin_stats, tmax_stats=tmax_stats)


#------------------------------------------------
def load_and_plot_both(score_files_mask, out_dir, chance_level=None, fig_ids=(1, 2), min_time=-math.inf, max_time=math.inf,
                       tmin_stats=None, tmax_stats=None, plot_subjects=False):

    scores_diag, times = load_score_files(score_files_mask, False)
    scores_gat, times, filenames = load_score_files(score_files_mask, True, return_file_names=True)

    if plot_subjects:
        print('Plotting and saving data for each of the {:} subjects'.format(scores_diag.shape[0]))
        for ii in range(scores_diag.shape[0]):
            filename = os.path.basename(filenames[ii])
            plot_gat_scores(scores_gat[np.newaxis, ii, :, :], times, chance_level=chance_level, min_time=min_time, max_time=max_time)
            plt.gcf().savefig(str(out_dir) + filename[:-4] + '_GAT.png')
            plt.close('all')
            plot_scores(scores_diag[ii, :], times, chance_level=chance_level, min_time=min_time, max_time=max_time)
            plt.gcf().savefig(str(out_dir) + filename[:-4] + '_diag.png')
            plt.close('all')
    else:
        print('Plotting data of {:} subjects'.format(scores_diag.shape[0]))
        plot_scores(scores_diag, times, chance_level=chance_level, fig_id=fig_ids[0], min_time=min_time, max_time=max_time)
        plot_gat_scores(scores_gat, times, chance_level=chance_level, fig_id=fig_ids[1], min_time=min_time, max_time=max_time,
                        tmin_stats=tmin_stats, tmax_stats=tmax_stats)


#------------------------------------------------------------------
def plot_scores(scores, times, chance_level=None, fig_id=1, ax=None, min_time=-math.inf, max_time=math.inf, tmin_stats=None, tmax_stats=None):
    """
    Plot the decoding scores per time - load data from files created by the dpm.decoding package

    :param chance_level: Chance-level decoding score.
                         If a umne.util.TimeRange() object is provided, use the average over this range as the chance level
    :param fig_id: The ID of the plotted figure
    :param min_time: Miminal time for plotting
    :param max_time: Maximal time for plotting
    """

    plt_scores, plt_times = filter_scores(scores, times, min_time, max_time)

    #-- Set chance level to a baseline level
    if isinstance(chance_level, umne.util.TimeRange):
        relevant_times = np.logical_and(times > chance_level.min_time, times < chance_level.max_time)
        chance_level = scores[:, relevant_times].mean()
        print('Baseline decodability = {:}'.format(chance_level))

    signif = None
    if tmin_stats is not None:
        # ====== then we run the cluster based permutation tests ===========
        filter_times = np.logical_and(times > tmin_stats, times < tmax_stats)
        inds = np.where(filter_times)
        scores_to_avg_filter = scores[:, inds[0]]
        signif = np.zeros((scores.shape[-1]))
        p_vals = stats.stats_cluster_based_permutation_test(scores_to_avg_filter - chance_level)
        sig = (p_vals < 0.05)
        for l, ll in enumerate(inds[0]):
                signif[ll] = sig[l]

    if ax is None:
        plt.close(fig_id)
        plt.figure(fig_id)
        ax = None

    jr.plot.pretty_decod(plt_scores, plt_times, chance_level, ax=ax, sig=signif, fill=True)


#------------------------------------------------------------------
def plot_gat_scores(scores, times, chance_level=None, ax=None, fig_id=2, min_time=-math.inf, max_time=math.inf,
                    tmin_stats=None, tmax_stats=None):
    """
    Plot the decoding scores per time - load data from files created by the dpm.decoding package

    :param chance_level: Chance-level decoding score. If a umne.util.TimeRange() object is provided, use this range
    """

    plt_scores, plt_times = filter_scores(scores, times, min_time, max_time)

    #-- Set chance level to a baseline level
    if isinstance(chance_level, umne.util.TimeRange):
        relevant_times = np.logical_and(times > chance_level.min_time, times < chance_level.max_time)
        scores_for_chance_level = scores[:, relevant_times, :]
        scores_for_chance_level = scores_for_chance_level[:, :, relevant_times]
        chance_level = scores_for_chance_level.mean()
        print('Baseline decodability = {:}'.format(chance_level))

    signif = None
    if tmin_stats is not None:
        # ====== then we run the cluster based permutation tests ===========
        filter_times = np.logical_and(times > tmin_stats, times < tmax_stats)
        inds = np.where(filter_times)
        scores_to_avg_filter = scores[:, inds[0], :]
        scores_to_avg_filter = scores_to_avg_filter[:, :, inds[0]]
        signif = np.zeros((scores.shape[-1], scores.shape[-1]))
        p_vals = stats.stats_cluster_based_permutation_test(scores_to_avg_filter - chance_level)
        sig = p_vals < 0.05
        for l, ll in enumerate(inds[0]):
            for m, mm in enumerate(inds[0]):
                signif[ll, mm] = sig[l, m]

    if ax is None:
        plt.close(fig_id)
        plt.figure(fig_id)
        ax = None

    jr.plot.pretty_gat(np.mean(plt_scores, axis=0), plt_times, chance_level, ax=ax, sig=signif)

    return scores


#------------------------------------------------------------------
def load_score_files(files, for_gat, return_file_names=False):
    """
    Load the score files and return a matrix of scores

    Returns scores and times:

    - "scores" is a matrix of scores: files x times when for_gat=False, and files x train_times x test_times when for_gat=True

    - "times" is an array of times

    :param files: Either a list of file names, or a mask in in glob format (i.e. with the * and ? wildcards)
    :param for_gat: Whether the score files are GAT
    """

    if isinstance(files, str):
        filenames = glob.glob(files)
        if len(filenames) == 0:
            raise Exception('File not found: {:}'.format(files))

    elif umne.util.is_collection(files):
        filenames = files

    else:
        raise Exception('Invalid "files" argument: {:}'.format(files))

    times = None
    scores = None

    n_files_with_kfold = 0

    for i_subj, filename in enumerate(filenames):
        with open(filename, 'rb') as fp:

            data = pickle.load(fp)

            if times is None:
                times = np.unique(data['times'])
                if for_gat:
                    scores = np.zeros([len(filenames), len(times), len(times)])
                else:
                    scores = np.zeros([len(filenames), len(times)])

            else:
                assert len(times) == len(np.unique(data['times'])), "Mismatch time window size when reaching file {:}".format(filename)

            curr_scores = np.asarray(data['scores'])

            #-- If the loaded data contained K-folds - average over it
            if len(curr_scores.shape) == 3:
                curr_scores = curr_scores.mean(axis=0)
                n_files_with_kfold += 1

            if for_gat:
                scores[i_subj, :, :] = curr_scores
            else:
                scores[i_subj, :] = curr_scores.diagonal()

    if n_files_with_kfold > 0:
        print('{:}/{:} files had k-folded data, which was averaged'.format(n_files_with_kfold, len(filenames)))

    if return_file_names:
        return scores, times, filenames

    return scores, times


#------------------------------------------------------------------
def filter_scores(scores, times, min_time=-math.inf, max_time=math.inf):
    """
    Filter scores for specific times
    """

    if min_time == -math.inf and max_time == math.inf:
        return scores, times

    relevant_times = np.logical_and(times > min_time, times < max_time)

    times = times[relevant_times]

    if len(scores.shape) == 3:    # subjects x times x times
        scores = scores[:, relevant_times, :]
        scores = scores[:, :, relevant_times]


    elif len(scores.shape) == 2:  # subjects x times
        scores = scores[:, relevant_times]

    else:
        raise Exception('Invalid scores format')

    return scores, times
