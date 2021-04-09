"""
Analyses for behavioral datajc
"""
import math
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import dpm


#------------------------------------------------------------------------
def load_behavioral_data(subj_ids, raw_file_names, target_filter=None):
    """
    Return all behavioral data for the given subjects.

    :param subj_ids: List of subject IDs
    :param raw_file_names: List of file names to load
    :return: pandas.DataFrame
    """

    bdata = None

    for subj_id in subj_ids:
        chunk = dpm.files.load_bresults_multiple_files(raw_file_names, subj_id=subj_id)
        chunk['subject'] = [subj_id] * len(chunk['target'])
        chunk = pd.DataFrame(data=chunk)

        if bdata is None:
            bdata = chunk
        else:
            bdata = pd.concat([bdata, chunk])

    if target_filter is not None:
        if target_filter in _target_filters:
            target_filter = _target_filters[target_filter]
        else:
            assert '__call__' in dir(target_filter)

        bdata = bdata[[target_filter(t) for t in bdata.target]]

    bdata['decade'] = bdata['target'] // 10
    bdata['unit'] = bdata['target'] % 10

    return bdata


_target_filters = {
    '1digit': lambda target: target < 10,
    '2digit': lambda target: target >= 10,
}

#------------------------------------------------------------------------
def summary_per_subj(bdata):
    """
    Get accuracy and RT for each subject and target.

    Return a data frame with mean & standard deviation of RT and accuracy for each subject and target
    """

    subj_ids = np.unique(bdata['subject'])

    result = pd.DataFrame(columns=('subject', 'accuracy', 'rt', 'rt_sd'))

    for subj in subj_ids:
        lines = bdata[bdata.subject == subj]

        correct = np.array([c for c in lines.correct if c is not None])  # The case of "None" can happen in RSVP files
        rts = np.array([rt for rt in lines.rt if rt is not None])        # The case of "None" can happen in RSVP files

        data = dict(subject = subj,
                    accuracy = correct.mean(),
                    rt = math.nan if len(rts) == 0 else rts.mean(),
                    rt_sd = math.nan if len(rts) == 0 else rts.std())

        result = result.append(data, ignore_index=True)

    return result


#------------------------------------------------------------------------
def get_outlier_subjects(bdata, do_print=False):
    """
    Find subjects with outlier error rate or RT
    """
    sum_per_subj = summary_per_subj(bdata)

    outlier_accuracy = dpm.util.outliers(sum_per_subj.accuracy, direction='low')
    outlier_rt = dpm.util.outliers(sum_per_subj.rt, direction='both')
    outlier = np.logical_or(outlier_accuracy, outlier_rt)

    if do_print:
        _print_outlier_info(sum_per_subj, outlier_accuracy, outlier_rt)

    return sorted(sum_per_subj.subject[outlier])


#-----------------------------------
def _print_outlier_info(sum_per_subj, outlier_accuracy, outlier_rt):

    sum_per_subj = [s[1] for s in sum_per_subj.iterrows()]

    info = []
    for (subj, acc, rt, rt_sd), o_acc, o_rt in zip(sum_per_subj, outlier_accuracy, outlier_rt):
        i = dict(Subject=subj,
                 Accuracy="{} {:.1f}%".format("*" if o_acc else " ", acc*100),
                 RT="{} {:.0f} (SD={:03.0f})".format("*" if o_rt else " ", rt, rt_sd))

        info.append(i)

    print(pd.DataFrame(data=info))
    print(" * = outliers")


#------------------------------------------------------------------------
def print_average_accuracy_and_rt(bdata):
    sum_per_subj = summary_per_subj(bdata)
    print('\nStats for {} subjects:'.format(sum_per_subj.shape[0]))
    print('  Average accuracy = {:.1f}%, SD = {:.1f}%'.format(sum_per_subj.accuracy.mean() * 100, sum_per_subj.accuracy.std() * 100))
    print('  Average RT = {:.0f} ms, SD = {:.0f} ms'.format(sum_per_subj.rt.mean(), sum_per_subj.rt.std()))


#------------------------------------------------------------------------
def print_accuracy_per_block(bdata):

    subj_and_file = sorted({(s, f) for s, f in zip(bdata.subject, bdata._filename_)})
    for s, f in subj_and_file:
        df = bdata[(bdata.subject == s) & (bdata._filename_ == f)]
        correct = np.array([c for c in df.correct if c is not None])
        print('Subject {}, {}: {:.1f}%'.format(s, os.path.basename(f), correct.mean() * 100))


#------------------------------------------------------------------------
def print_stats_rsvp(subj_ids, filenames):

    for subj in subj_ids:

        rts = []

        for filename in filenames:
            t = dpm.files.load_rsvp_responses_file(dpm.files.rsvp_responses_filename(filename))
            rts.extend(t)


#------------------------------------------------------------------------
# noinspection PyUnresolvedReferences
def rt_distrib_per_subject(bdata, include_errors=True, smooth_sd=20, plot_medians=False, save_as=None, cmap=cm.hsv, xlim=(0, 1500)):
    """
    Plot the distribution of RTs for each subject

    :param include_errors: Whether to include error trials
    :param smooth_sd: Standard deviation for Gaussian smoothing, in milliseconds
    :param plot_medians: Whether to plot the median RT of each subject as a vertical line
    :param cmap: Color map
    """

    if not include_errors:
        bdata = bdata[bdata.correct.astype(bool)]

    subjects = sorted(np.unique(bdata.subject))
    colors = {subj_id: cmap(i_subj / len(subjects)) for i_subj, subj_id in enumerate(subjects)}

    plt.clf()

    x = np.array(range(xlim[0], xlim[1]))
    rt_data = {}

    for subj_id in subjects:

        rts = np.array(bdata.rt[bdata.subject == subj_id])

        rt_distrib = [sum(rts == rt) for rt in x]
        rt_distrib_smoothed = dpm.util.smooth_gaussian(rt_distrib, smooth_sd)
        rt_distrib_smoothed = rt_distrib_smoothed / sum(rt_distrib_smoothed) * 100
        rt_data[subj_id] = rt_distrib_smoothed

        plt.plot(x, rt_distrib_smoothed, color=colors[subj_id], linewidth=0.5)

        print('.', end='')

    plt.legend(subjects, fontsize=8)

    if plot_medians:
        y1 = plt.ylim()[1] * 1.05
        for subj_id in subjects:
            mean_rt = bdata.rt[bdata.subject == subj_id].median()
            y0 = rt_data[subj_id][x == round(mean_rt)]
            plt.plot([mean_rt, mean_rt], (y0, y1), color=colors[subj_id], linewidth=0.5)

    plt.grid(linewidth=0.5, linestyle=':')
    plt.xlabel('Response time', fontsize=8)
    plt.ylabel('Probabiliy density (%/ms)', fontsize=8)

    if save_as is not None:
        fig = plt.gcf()
        fig.savefig(save_as)


#------------------------------------------------------------------------
def get_median_rt(bdata, subj_ids, as_dict=False):
    """
    Get the median RT for each subject.
    Return a dict/array
    """
    if as_dict:
        result = {subj_id: bdata.rt[bdata.subject == subj_id].median() for subj_id in subj_ids}
    else:
        result = [bdata.rt[bdata.subject == subj_id].median() for subj_id in subj_ids]

    return result


#------------------------------------------------------------------------
def plot_per_target(bdata, y_var, include_errors=True, save_as=None, ylim=None):
    """
    Plot the something-by-target graph
    """

    if not include_errors:
        bdata = bdata[bdata.correct.astype(bool)]

    column = 'correct' if y_var == 'errors' else y_var

    n_subjects = len(np.unique(bdata.subject))

    #-- Average all trials per subject
    bdata = bdata[['subject', 'target', column]]
    bdata = bdata.groupby(['subject', 'target']).aggregate('mean')

    #-- Average over subjects for each target
    per_target = bdata.groupby('target')
    targets = per_target.aggregate('mean').index.values
    y = per_target.aggregate('mean')[column]
    y_sd = per_target.aggregate('std')[column]

    y, y_sd, y_label = _get_y(y, y_sd, y_var)

    #-- Plot!
    plt.clf()
    plt.gca().tick_params(labelsize=16)
    plt.errorbar(targets, y, y_sd / math.sqrt(n_subjects), color='black')

    _format_per_target_plot(y_label, ylim)

    if save_as is not None:
        fig = plt.gcf()
        fig.savefig(save_as)


#---------------------------------------------------
def _format_per_target_plot(y_label, ylim):

    if ylim is not None:
        plt.gca().set_ylim(ylim)

    # -- Show the target number (44)
    ylim = plt.ylim()
    plt.plot([44, 44], ylim, color='grey', linestyle='--', linewidth=0.5)
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel('Target', fontsize=18)
    plt.grid(linewidth=0.5, linestyle=':')


#------------------------------------------------------------------------
def plot_per_target_and_pos(bdata, y_var, include_errors=True, save_as=None, ylim=None, colors=None, plot_type='line'):
    """
    Plot the something-by-target graph
    """

    if colors is None:
        colors = [np.array([1, 1, 1]) * c for c in (0, 0.2, 0.4, 0.6, 0.8, 0.9)]

    column = 'correct' if y_var == 'errors' else y_var

    if not include_errors:
        bdata = bdata[bdata.correct.astype(bool)]

    n_subjects = len(np.unique(bdata.subject))
    targets = np.array(sorted(np.unique(bdata.target)))
    positions = sorted(np.unique(bdata.position))
    npos = len(positions)

    assert len(colors) >= len(positions), "Not enough colors"

    #-- Average all trials per subject
    bdata = bdata[['subject', 'target', 'position', column]]
    bdata = bdata.groupby(['subject', 'target', 'position']).aggregate('mean')

    #-- Average over subjects for each target
    y = np.zeros([len(targets), len(positions)])
    y_sd = np.zeros([len(targets), len(positions)])
    means = bdata.groupby(['position', 'target']).aggregate('mean')
    stdevs = bdata.groupby(['position', 'target']).aggregate('std')
    for i_target, target in enumerate(targets):
        for i_pos, pos in enumerate(positions):
            y[i_target, i_pos] = means.loc[(pos, target)]
            y_sd[i_target, i_pos] = stdevs.loc[(pos, target)]

    y, y_sd, y_label = _get_y(y, y_sd, y_var)

    #-- Plot!
    plt.clf()
    for i_pos, pos in enumerate(positions):
        if plot_type == 'line':
            plt.plot(targets, y[:, i_pos], color=colors[i_pos], linewidth=1, marker='.', markersize=2)
        elif plot_type == 'errorbar':
            plt.errorbar(targets + (i_pos - npos/2) / 10, y[:, i_pos], y_sd[:, i_pos] / math.sqrt(n_subjects), color=colors[i_pos], linewidth=1)
        else:
            raise Exception('Unsupported plot type ({})'.format(plot_type))

    _set_position_labels(positions)
    _format_per_target_plot(y_label, ylim)

    if save_as is not None:
        fig = plt.gcf()
        fig.savefig(save_as)


#------------------------------------------------------
def _get_y(y, y_sd, y_var):

    if y_var == 'rt':
        y_label = 'Response time (ms)'

    elif y_var == 'correct':
        y_label = 'Correct (%)'
        y *= 100
        y_sd *= 100

    elif y_var == 'errors':
        y_label = 'Errors (%)'
        y = 100 - 100 * y
        y_sd *= 100

    else:
        raise Exception('Unsupported y variable ({})'.format(y_var))

    return y, y_sd, y_label


#--------------------------------------
def _set_position_labels(positions):

    if len(positions) == 5:
        pos_labels = 'Left', '', 'Middle', '', 'Right'
    elif len(positions) == 6:
        pos_labels = 'Left', '', '', '', '', 'Right'
    else:
        pos_labels = ['Position {}'.format(p+1) for p in positions]

    plt.legend(pos_labels)


#------------------------------------------------------------------------
def plot_per_group(bdata, y_var, grouping_var, include_errors=True, save_as=None, plot_type='box', ylim=None):
    """
    Plot RT/accuracy for each of several groups, for different possible grouping variables
    """

    if not include_errors:
        bdata = bdata[bdata.correct.astype(bool)]

    column = 'correct' if y_var == 'errors' else y_var

    n_subjects = len(np.unique(bdata.subject))

    #-- Average all trials per subject
    bdata = bdata[['subject', grouping_var, column]]
    bdata = bdata.groupby([grouping_var, 'subject']).aggregate('mean')

    #-- Average over subjects for each group
    per_group = bdata.groupby(grouping_var)
    groups = per_group.aggregate('mean').index.values
    y = per_group.aggregate('mean')[column]
    y_sd = per_group.aggregate('std')[column]

    y, y_sd, y_label = _get_y(y, y_sd, y_var)

    plt.clf()

    if plot_type == 'line':
        plt.errorbar(groups, y, y_sd / math.sqrt(n_subjects), color='grey', linestyle='none', elinewidth=2,
                     marker='o', markersize=10, markerfacecolor='black', markeredgewidth=0)

    elif plot_type == 'box':
        plt.boxplot([bdata.loc[pos].rt for pos in range(5)])

    else:
        raise Exception('Unsupported plot type')

    if ylim is not None:
        plt.gca().set_ylim(ylim)

    plt.ylabel(y_label)

    _set_grouping_labels(groups, grouping_var)

    plt.grid(linewidth=0.5, linestyle=':', axis='y')

    if save_as is not None:
        fig = plt.gcf()
        fig.savefig(save_as)


#------------------------------------------
def _set_grouping_labels(groups, grouping_var):

    if grouping_var == 'position':
        if len(groups) == 5:
            plt.xticks([0, 2, 4])
            plt.gca().set_xticklabels(['Left', 'Middle', 'Right'])

        elif len(groups) == 6:
            plt.xticks([0, 2.5, 5])
            plt.gca().set_xticklabels(['Left', 'Middle', 'Right'])

        else:
            plt.xlabel('Position')

    elif grouping_var == 'handmapping':

        plt.xticks([0, 1])
        plt.gca().set_xticklabels(['Incongruent', 'Congruent'])

    else:
        plt.xlabel(grouping_var)
