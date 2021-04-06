import sys

import mne

sys.path.append('/Users/dror/git/digposmeg/analyze')
sys.path.append('/Users/dror/git/jr-tools')
sys.path.append('/Users/dror/git/pyRiemann')
from dpm.util import create_folder

import dpm.plots
import pmne.decplt

#------------------- set parameters -----------------------------------

mne.set_log_level('info')
decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'
subj_ids = dpm.consts.subj_ids_clean
env_per_subj = False

tmin_stim = -0.4
tmax_stim = 1
tmin_resp = -1.5
tmax_resp = 0
baseline_stim = (None,0)
baseline_resp = None


#==========================================================================================================
#==========================================================================================================
#                               LOW-LEVEL SPATIAL INFORMATION DECODING
#==========================================================================================================
#==========================================================================================================

#---------------------------------------------------------------
#-- Hemifield classifier
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.hemifield(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'hemifield',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.hemifield()],
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj)
#---------------------------------------------------------------
#-- Location classifier

loc_classifier = dpm.classifiers.location(min_location=0, max_location=4)
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, loc_classifier, data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'location',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=None,
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj, cv = 4)

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
                                                      tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj,cv=4)


#------------------------------------------------------------------------------
#-------------------------  quantity decoders  ------------------------------
#------------------------------------------------------------------------------

create_folder(decoding_sh_dir + 'whole_number/regression/')
create_folder(decoding_sh_dir + 'whole_number/regression/standard/')

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/regression/standard/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj,decoding_method='standard_reg',load_error_trials=False,
                                                  cv=4)

#------------------------------------------------------------------------------
# Decode 2-digit quantity (regression) LOCKED ON THE RESPONSE
create_folder(decoding_sh_dir + 'whole_number/regression/RT-locked/r2_score/')

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/regression/RT-locked/r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],baseline=baseline_resp, decoding_method = 'standard_reg_r2',
                                                  tmin=tmin_resp, tmax=tmax_resp, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,
                                                  cv=4)

create_folder(decoding_sh_dir + 'whole_number/regression/RT-locked/mean_squared_distance/')

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'whole_number/regression/RT-locked/mean_squared_distance/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  filename_suffix_fit='comp',
                                                  classifier_specs_test=[dpm.classifiers.target()],baseline=baseline_resp, decoding_method = 'standard_reg',
                                                  tmin=tmin_resp, tmax=tmax_resp, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,
                                                  cv=4)


#---------------------------------------------------------------
#-- Decade decoders. Score using decade/unit labels.
#---------------------------------------------------------------

# STIM LOCKED
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj,cv=4, load_error_trials=False)

# train decade decoder on given positions and test it on the other positions
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/train_loc04_test_loc2/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard',
                                                  train_epoch_filter = 'location == 0 or location == 4',test_epoch_filter = 'location == 2')

# RESPONSE LOCKED
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_resp, tmax=tmax_resp, env_per_subj=env_per_subj, on_response=True, load_error_trials=False,baseline=baseline_resp,cv=4)


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
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj,cv=4)


dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='decade==4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_4X'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj,cv=4)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target<10',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_single_digits'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_stim, tmax=tmax_stim, env_per_subj=env_per_subj,cv=4)

# RESPONSE LOCKED
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_resp, tmax=tmax_resp, env_per_subj=env_per_subj, on_response=True,cv=4,baseline=baseline_resp)


dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='decade==4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_4X'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_resp, tmax=tmax_resp, env_per_subj=env_per_subj, on_response=True,cv=4,baseline=baseline_resp)

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target<10',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_unit_on_single_digits'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_resp, tmax=tmax_resp, env_per_subj=env_per_subj, on_response=True,cv=4,baseline=baseline_resp)


