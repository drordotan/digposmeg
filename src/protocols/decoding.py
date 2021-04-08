import sys

import mne

sys.path.append('/Users/dror/git/digposmeg/analyze')
sys.path.append('/Users/dror/git/jr-tools')
sys.path.append('/Users/dror/git/pyRiemann')

import dpm
import umne
import pmne.decplt
import dpm.plots


#------------------------------------------------------------------------------

mne.set_log_level('info')
env_per_subj=False

subj_ids = dpm.consts.subj_ids

subj_ids = tuple(['ag', 'at', 'bl', 'bo', 'cb', 'cc', 'eb', 'en', 'hr', 'jm0', 'lj', 'mn', 'mp'])
# Still subject CD has bad data

decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'

tmin = -0.4

#------------------------------------------------------------------------------
def plot_gat(directory, file_suffix, chance_level,min_time=-0.1):
    import matplotlib.pyplot as plt
    plt.close('all')
    pmne.decplt.load_and_plot(dpm.consts.results_dir+directory+'/scores_*'+file_suffix+'*.pkl', chance_level=chance_level, min_time=min_time)


#-- Redefine plot functions - so it does nothing when we run the whole file
# noinspection PyRedeclaration
def plot_gat(directory, file_suffix, chance_level):
    pass


#==========================================================================================================
#==========================================================================================================
#  Split-half (generalization across trials): train on 1/2 of the trials, score on the other half
#==========================================================================================================
#==========================================================================================================

#---------------------------------------------------------------
#-- Hemifield classifier
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.hemifield(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'hemifield',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.hemifield()],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.hemifield(), data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'hemifield',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='rsvp',
                                                  classifier_specs_test=[dpm.classifiers.hemifield()],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj)

plot_gat('decoding-sh/hemifield', 'comp', .5)
plot_gat('decoding-sh/hemifield', 'rsvp', .5)

#---------------------------------------------------------------
#-- Location classifier

loc_classifier_left = dpm.classifiers.location(min_location=0, max_location=1)
loc_classifier_right = dpm.classifiers.location(min_location=3, max_location=4)

# RSVP task

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, loc_classifier_left, data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'location',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='rsvp_left_hemifield',
                                                  classifier_specs_test=[loc_classifier_left],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, loc_classifier_right, data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'location',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='rsvp_right_hemifield',
                                                  classifier_specs_test=[loc_classifier_right],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj)

# Comparison task

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, loc_classifier_left, data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'location',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp_left_hemifield',
                                                  classifier_specs_test=[loc_classifier_left],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, loc_classifier_right, data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'location',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp_right_hemifield',
                                                  classifier_specs_test=[loc_classifier_right],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)


plot_gat('decoding-sh/location', 'comp', .5)
plot_gat('decoding-sh/location', 'rsvp', .5)

#------------------------------------------------------------------------------
# retinotopic: decode the digit in a given position on screen

for i in range(6):
    cspec = dpm.classifiers.digit_in_position(i)
    dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, cspec, data_filenames=dpm.comparison_raw_files,
                                                      out_dir=decoding_sh_dir + 'retinotopic',
                                                      epoch_filter='target > 10 and decade != 4 and unit != 4',
                                                      grouping_metadata_fields=['location', 'target'],
                                                      filename_suffix_fit='comp_position_%i' % i,
                                                      classifier_specs_test=[cspec],
                                                      tmin=tmin, tmax=1, env_per_subj=env_per_subj)

    dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, cspec, data_filenames=dpm.rsvp_raw_files,
                                                      out_dir=decoding_sh_dir + 'retinotopic',
                                                      epoch_filter='target > 10 and decade != 4 and unit != 4',
                                                      grouping_metadata_fields=['location', 'target'],
                                                      filename_suffix_fit='rsvp_position_%i' % i,
                                                      classifier_specs_test=[cspec],
                                                      tmin=tmin, tmax=.6, env_per_subj=env_per_subj)


#---------------------------------------------------------------
#-- Decade classifier. Score using decade/unit labels.

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)

# ========== score decade with logistic regression =======================
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/regression/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard_reg')

# ====== train decade decoder on given positions and test it on the other positions ========

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/train_loc04_test_loc2/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard',
                                                  train_epoch_filter = 'location == 0 or location == 4',test_epoch_filter = 'location == 2')

# ====== decade locked on RT comparison task ========

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True, load_error_trials=False)


# ====== decade locked on RT RSVP task ========

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='rsvp',
                                                  filename_suffix_scores=['rsvp_score_as_decade', 'rsvp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj)
#todo: group also by response-hand-mapping?

plot_gat('decoding-sh/decade', 'comp_score_as_decade', .25)
plot_gat('decoding-sh/decade', 'comp_score_as_unit', .25)
plot_gat('decoding-sh/decade', 'rsvp_score_as_decade', .25)
plot_gat('decoding-sh/decade', 'rsvp_score_as_unit', .25)

#---------------------------------------------------------------
#-- Unit classifier. Score using decade/unit labels.

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)


# ========== score regression =============
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/regression/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)





dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='rsvp',
                                                  filename_suffix_scores=['rsvp_score_as_decade', 'rsvp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard_reg')


plot_gat('decoding-sh/unit', 'comp_score_as_decade', .25)
plot_gat('decoding-sh/unit', 'comp_score_as_unit', .25)
plot_gat('decoding-sh/unit', 'rsvp_score_as_decade', .25)
plot_gat('decoding-sh/unit', 'rsvp_score_as_unit', .25)


#------------------------------------------------------------------------------
# Decode 2-digit quantity (regression)
tmin = -0.1
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj,decoding_method='standard_reg',load_error_trials=False,cv = 5)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='rsvp',
                                                  classifier_specs_test=[dpm.classifiers.target()],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj,decoding_method='standard_reg',load_error_trials=False,cv = 5)


plot_gat('decoding-sh/whole_number', 'comp', chance_level=umne.util.TimeRange(max_time = 0))
plot_gat('decoding-sh/whole_number', 'rsvp', chance_level=umne.util.TimeRange(max_time = 0))

#------------------------------------------------------------------------------
# Decode 2-digit quantity (regression) LOCKED ON THE RESPONSE

tmax = 0
tmin = -1.5

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/RT-locked_r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()], decoding_method = 'standard_reg_r2',
                                                  tmin=tmin, tmax=tmax, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,cv=5)

#==========================================================================================================
#  Generalization across conditions/trials/labels
#==========================================================================================================

#------------------------------------------------------------------------------
# Train on RSVP, generalize to comparison

# retinotopic

for i in range(6):
    classifier_spec = dpm.classifiers.digit_in_position(i)
    dpm.decoding.run_fit_decoder(subj_ids, classifier_spec, data_filenames=dpm.rsvp_raw_files, out_dir=dpm.decoding_dir + 'retinotopic',
                                 filename_suffix='rsvp_position_{:}_train'.format(i), tmin=tmin, tmax=.6)

    fit_results_filename = 'decoder_{:}' + '_standard_-100_600_rsvp_position_{:}_train.pkl'.format(i)
    dpm.decoding.run_score_existing_decoder(subj_ids, fit_results_filename=fit_results_filename, data_filenames=dpm.comparison_raw_files,
                                            out_dir=dpm.decoding_dir + 'retinotopic', get_y_label_func=classifier_spec['y_label_func'],
                                            filename_suffix='rsvp_position_{:}_test_on_comp'.format(i))


# Decade digit
# Unit digit
# Quantity (probably won't work)


'''

import pickle
import matplotlib.pyplot as plt
from dpm import plots

path = '/Volumes/COUCOU_CFC/digitpos/exp/results/decoding/unit-gen/scores_ag_standard_-100_1000_comp_score_as_decade.pkl'

with open(path,'rb') as fid:
    scores = pickle.load(fid)

plt.figure(1)
plt.imshow(scores['scores'])

plots.plot_decoding_score(path,chance_level=0.25)

'''

#------------------------------------------------------------------------------
# ??? Train on RSVP with decade labels, test on comparison with decade labels vs. with unit labels


#------------------------------------------------------------------------------
# ??? Train on RSVP with unit labels, test on comparison with decade labels vs. with unit labels

