"""
Older decoding scripts - without separation into training set and test set
"""
import sys

import mne

sys.path.append('/git/digposmeg/analyze')
sys.path.append('/git/jr-tools')
sys.path.append('/git/pyRiemann')

import dpm
import dpm.plots


#------------------------------------------------------------------------------

mne.set_log_level('info')
env_per_subj=False

subj_ids = dpm.consts.subj_ids
# Still subject CD has bad data

decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'

tmin = -0.4

#------------------------------------------------------------------------------
def plot_gat(directory, file_suffix, chance_level):
    import matplotlib.pyplot as plt
    plt.close('all')
    dpm.plots.plot_decoding_score(dpm.consts.results_dir + directory + '/scores_*' + file_suffix + '*.pkl', chance_level=chance_level)


#-- Redefine plot functions - so it does nothing when we run the whole file
# noinspection PyRedeclaration
def plot_gat(directory, file_suffix, chance_level):
    pass

#------------------------------------------------------------------------------
# SANITY CHECK: (for each of the 2 datasets)
# Decode hemifield

dpm.decoding.run_fit_multi_subjects(dpm.classifiers.hemifield(), subj_ids, out_dir=dpm.decoding_dir + 'hemifield',
                                    data_filenames=dpm.rsvp_raw_files, filename_suffix='rsvp', env_per_subj=env_per_subj, tmin=tmin)

dpm.decoding.run_fit_multi_subjects(dpm.classifiers.hemifield(), subj_ids, out_dir=dpm.decoding_dir + 'hemifield',
                                    data_filenames=dpm.comparison_raw_files, filename_suffix='comp', env_per_subj=env_per_subj, tmin=tmin)

plot_gat('decoding/hemifield', 'rsvp', .5)
plot_gat('decoding/hemifield', 'comp', .5)

#------------------------------------------------------------------------------
# SANITY CHECK: (for each of the 2 datasets)
# Decode exact location within hemifield. Run only on trials from a single hemifield.


dpm.decoding.run_fit_multi_subjects(dpm.classifiers.location(min_location=0, max_location=1), subj_ids, out_dir=dpm.decoding_dir + 'location',
                                    filename_suffix='comp_left_hemifield', data_filenames=dpm.comparison_raw_files, tmax=1, env_per_subj=env_per_subj, tmin=tmin)
dpm.decoding.run_fit_multi_subjects(dpm.classifiers.location(min_location=0, max_location=1), subj_ids, out_dir=dpm.decoding_dir + 'location',
                                    filename_suffix='rsvp_left_hemifield', data_filenames=dpm.rsvp_raw_files, tmax=.6, env_per_subj=env_per_subj, tmin=tmin)

dpm.decoding.run_fit_multi_subjects(dpm.classifiers.location(min_location=3, max_location=4), subj_ids, out_dir=dpm.decoding_dir + 'location',
                                    filename_suffix='comp_right_hemifield', data_filenames=dpm.comparison_raw_files, tmax=1, env_per_subj=env_per_subj, tmin=tmin)
dpm.decoding.run_fit_multi_subjects(dpm.classifiers.location(min_location=3, max_location=4), subj_ids, out_dir=dpm.decoding_dir + 'location',
                                    filename_suffix='rsvp_right_hemifield', data_filenames=dpm.rsvp_raw_files, tmax=.6, env_per_subj=env_per_subj, tmin=tmin)

plot_gat('decoding/location', 'rsvp', .5)
plot_gat('decoding/location', 'comp', .5)


#------------------------------------------------------------------------------
# retinotopic: decode the digit in a given position on screen

for i in range(6):
    dpm.decoding.run_fit_multi_subjects(dpm.classifiers.digit_in_position(i), subj_ids, out_dir=dpm.decoding_dir + 'retinotopic', filename_suffix='comp_position_%i' % i, data_filenames=dpm.comparison_raw_files, tmax=1, env_per_subj=env_per_subj)
    dpm.decoding.run_fit_multi_subjects(dpm.classifiers.digit_in_position(i), subj_ids, out_dir=dpm.decoding_dir + 'retinotopic', filename_suffix='rsvp_position_%i' % i, data_filenames=dpm.rsvp_raw_files, tmax=.6, env_per_subj=env_per_subj)

#------------------------------------------------------------------------------
# Decode decade digit

dpm.decoding.run_fit_multi_subjects(dpm.classifiers.decade((2,3,5,8)), subj_ids, out_dir=dpm.decoding_dir + 'decade', filename_suffix='comp', data_filenames=dpm.comparison_raw_files, tmax=1, env_per_subj=env_per_subj)
dpm.decoding.run_fit_multi_subjects(dpm.classifiers.decade((2,3,5,8)), subj_ids, out_dir=dpm.decoding_dir + 'decade', filename_suffix='rsvp', data_filenames=dpm.rsvp_raw_files, tmax=.6, env_per_subj=env_per_subj)


#------------------------------------------------------------------------------
# Decode unit digit

dpm.decoding.run_fit_multi_subjects(dpm.classifiers.unit((2,3,5,8)), subj_ids, out_dir=dpm.decoding_dir + 'unit', filename_suffix='comp', data_filenames=dpm.comparison_raw_files, tmax=1, env_per_subj=env_per_subj)
dpm.decoding.run_fit_multi_subjects(dpm.classifiers.unit((2,3,5,8)), subj_ids, out_dir=dpm.decoding_dir + 'unit', filename_suffix='rsvp', data_filenames=dpm.rsvp_raw_files, tmax=.6, env_per_subj=env_per_subj)


#------------------------------------------------------------------------------
# Decode 2-digit quantity (regression)

dpm.decoding.run_fit_multi_subjects(dpm.classifiers.target(), subj_ids, decoding_method='standard_reg', out_dir=dpm.decoding_dir + 'whole_number', filename_suffix='comp', data_filenames=dpm.comparison_raw_files, tmax=1, env_per_subj=env_per_subj)
dpm.decoding.run_fit_multi_subjects(dpm.classifiers.target(), subj_ids, decoding_method='standard_reg', out_dir=dpm.decoding_dir + 'whole_number', filename_suffix='rsvp', data_filenames=dpm.rsvp_raw_files, tmax=.6, env_per_subj=env_per_subj)

#------------------------------------------------------------------------------
# Time analysis: show different effects in different time windows

