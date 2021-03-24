import sys

import mne
sys.path.append('/Users/dror/git/digposmeg/analyze')
sys.path.append('/Users/dror/git/jr-tools')
sys.path.append('/Users/dror/git/pyRiemann')

sys.path.append('/neurospin/meg/meg_tmp/DPEM_Dror_Fosca_2017/scripts/digposmeg/src/')


from dpm.util import create_folder

import dpm.plots
import pmne.decplt

#------------------------------------------------------------------------------

mne.set_log_level('info')
decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'
subj_ids = dpm.consts.subj_ids_clean
env_per_subj = False

decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'

tmin = -0.4


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
#-- Decade decoders. Score using decade/unit labels.
#---------------------------------------------------------------

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj)

# score decade with logistic regression
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/regression/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard_reg')

# train decade decoder on given positions and test it on the other positions

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/train_loc04_test_loc2/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard',
                                                  train_epoch_filter = 'location == 0 or location == 4',test_epoch_filter = 'location == 2')

# decade locked on RT comparison task

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True, load_error_trials=False)


# decade locked on RT RSVP task

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.rsvp_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='rsvp',
                                                  filename_suffix_scores=['rsvp_score_as_decade', 'rsvp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=.6, env_per_subj=env_per_subj)


#---------------------------------------------------------------
#-- Unit decoders. Score using unit labels.
#---------------------------------------------------------------

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj,cv=4)


dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='decade==4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_4X'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj,cv=4)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target<10',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_single_digits'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin, tmax=1, env_per_subj=env_per_subj,cv=4)


dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True,cv=4)


dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='decade==4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_4X'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True,cv=4)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target<10',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_single_digits'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True,cv=4)

#---------------------------------------------------------------
#------------ Plotting the units results -----------------------
#---------------------------------------------------------------


# dpm.consts.results_dir = "/Users/fosca/Desktop/"
# dpm.consts.figures_path = "/Users/fosca/Desktop/figures/"
# pmne.decplt.load_and_plot_gat(dpm.consts.results_dir + '/scores_*' + "comp_score_as_unit_on_single_digits" + '*.pkl',
#                           chance_level=0.25,tmin_stats=0,tmax_stats=1,plot_subjects=True)


pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit'+'/scores_*'+"comp_score_as_unit"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0.25, tmin_stats=0, tmax_stats=1, plot_subjects=True)
pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/RT-locked'+'/scores_*'+'comp_score_as_unit'+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0.25, tmin_stats=-1.5, tmax_stats=0)

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit'+'/scores_*'+"comp_score_as_unit_on_4X"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0.25, tmin_stats=0, tmax_stats=1)
pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/RT-locked'+'/scores_*'+"comp_score_as_unit_on_4X"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0.25, tmin_stats=-1.5, tmax_stats=0)

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/'+'/scores_*'+"comp_score_as_unit_on_single_digits"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0.25, tmin_stats=0, tmax_stats=1)
pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/RT-locked'+'/scores_*'+"comp_score_as_unit_on_single_digits"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0.25, tmin_stats=-1.5, tmax_stats=0)

#---------------------------------------------------------------
#-- Quantity decoders
#---------------------------------------------------------------
#------------------------------------------------------------------------------
#-------------------------  quantity classifier  ------------------------------
#------------------------------------------------------------------------------

create_folder(decoding_sh_dir + 'whole_number/regression/')
create_folder(decoding_sh_dir + 'whole_number/regression/standard/')

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/regression/standard/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],
                                                  tmin=-0.4, tmax=1, env_per_subj=env_per_subj,decoding_method='standard_reg',load_error_trials=False,
                                                  cv=4, reject='auto_global')

#------------------------------------------------------------------------------
# Decode 2-digit quantity (regression) LOCKED ON THE RESPONSE
create_folder(decoding_sh_dir + 'whole_number/regression/RT-locked/r2_score/')

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/regression/RT-locked/r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],baseline=None, decoding_method = 'standard_reg_r2',
                                                  tmin=0, tmax=-1.5, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,
                                                  cv=4, reject='auto_global')

create_folder(decoding_sh_dir + 'whole_number/regression/RT-locked/mean_squared_distance/')

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/regression/RT-locked/mean_squared_distance/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],baseline=None, decoding_method = 'standard_reg',
                                                  tmin=0, tmax=-1.5, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,
                                                  cv=4, reject='auto_global')
