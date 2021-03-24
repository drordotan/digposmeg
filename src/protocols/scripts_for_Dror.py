import sys
import mne
import dpm
import umne
import dpm.plots

"""
sys.path.append('/git/digposmeg/analyze')
sys.path.append('/git/jr-tools')
sys.path.append('/git/pyRiemann')
"""
#------------------------------------------------------------------------------
mne.set_log_level('info')
env_per_subj=False
decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'
subj_ids = tuple(['ag', 'at', 'bl', 'bo', 'cb', 'cc', 'eb', 'en', 'hr', 'jm0', 'lj', 'mn', 'mp'])

decim=1
reject = None
baseline=None
tmax_RT = 0
tmin_RT = -1.5
tmin = -0.2
tmax = 1
# ______________________________________________________________________________
# ---------------------- Quantity regression locked on the RT  -----------------

# ============ R2 score =================
# dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
#                                                   out_dir=decoding_sh_dir + 'whole_number/RT-locked_r2_score/',
#                                                   grouping_metadata_fields=['location', 'target'],
#                                                   filename_suffix_fit='comp',
#                                                   classifier_specs_test=[dpm.classifiers.target()], decoding_method = 'standard_reg_r2',
#                                                   tmin=tmin_RT, tmax=tmax_RT, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,cv=5,
#                                                   baseline=baseline,decim=decim,reject=reject)
#
# # ============ MeanSquaredError =================
# dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.target(), data_filenames=dpm.comparison_raw_files,
#                                                   out_dir=decoding_sh_dir + 'whole_number/RT-locked_MSE/',
#                                                   grouping_metadata_fields=['location', 'target'],
#                                                   filename_suffix_fit='comp',
#                                                   classifier_specs_test=[dpm.classifiers.target()], decoding_method = 'standard_reg',
#                                                   tmin=tmin_RT, tmax=tmax_RT, env_per_subj=env_per_subj,on_response=True,load_error_trials=False,cv=5,
#                                                   baseline=baseline,decim=decim,reject=reject)

# ______________________________________________________________________________
# ---------------------- Decade and unit decoding locked on RT  ----------------

# decade
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_decade'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None)],
                                                  tmin=tmin_RT, tmax=tmax_RT, env_per_subj=env_per_subj, on_response=True, load_error_trials=False,
                                                  baseline=baseline,decim=decim, reject =reject)
# unit
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  tmin=tmin_RT, tmax=tmax_RT, env_per_subj=env_per_subj, on_response=True, load_error_trials=False,
                                                  baseline=baseline,decim=decim, reject =reject)
# ______________________________________________________________________________
# ----------- Decade and unit regression locked on RT and not locked on RT -----

# decade
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked_r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_decade'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None)],
                                                  decoding_method = 'standard_reg_r2',tmin=tmin_RT, tmax=tmax_RT, env_per_subj=env_per_subj, on_response=True,
                                                  load_error_trials=False,
                                                  baseline=baseline,decim=decim, reject =reject)
# unit

dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/RT-locked_r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  decoding_method = 'standard_reg_r2',tmin=tmin_RT, tmax=tmax_RT, env_per_subj=env_per_subj, on_response=True,
                                                  load_error_trials=False,
                                                  baseline=baseline,decim=decim, reject =reject)




# decade
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_decade'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None)],
                                                  decoding_method = 'standard_reg_r2',tmin=tmin, tmax=tmax, env_per_subj=env_per_subj, on_response=False,
                                                  load_error_trials=False,
                                                  baseline=baseline,decim=decim, reject =reject)



# unit
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.unit(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade_and_unit/r2_score/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_unit'],
                                                  classifier_specs_test=[dpm.classifiers.unit(filter=None)],
                                                  decoding_method = 'standard_reg_r2',tmin=tmin, tmax=tmax, env_per_subj=env_per_subj, on_response=False,
                                                  load_error_trials=False,
                                                  baseline=baseline,decim=decim, reject =reject)


# ------------------------------------------------------------------------------
# Decade classifier trained for certain positions tested on the others.
# Scored in terms of decade or unit label. (train_loc04_test_loc2)

# dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
#                                                   out_dir=decoding_sh_dir + 'decade_and_unit/train_loc04_test_loc2/',
#                                                   grouping_metadata_fields=['location', 'target'],
#                                                   epoch_filter='target>10 and decade!=4 and unit!=4',
#                                                   filename_suffix_fit='comp',
#                                                   filename_suffix_scores=['comp_decade'],
#                                                   classifier_specs_test=[dpm.classifiers.decade(filter=None)],
#                                                   tmin=tmin, tmax=tmax, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard',
#                                                   train_epoch_filter = 'location == 0 or location == 4',test_epoch_filter = 'location == 2',
#                                                   baseline = baseline, decim = decim, reject = reject)