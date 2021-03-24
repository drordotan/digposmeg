import sys
import mne
sys.path.append('/git/digposmeg/analyze')
sys.path.append('/git/jr-tools')
sys.path.append('/git/pyRiemann')

import dpm.plots
from dpm.util import create_folder

#------------------------------------------------------------------------------

mne.set_log_level('info')
decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'
#subj_ids = tuple(['ag', 'at', 'bl', 'bo', 'cb', 'cc', 'eb', 'en', 'hr', 'jm0', 'lj', 'mn', 'mp'])
#subj_ids = 'bl',
subj_ids = dpm.consts.subj_ids_clean
env_per_subj = False


#------------------------------------------------------------------------------
#-------  Decade classifier. Score using decade/unit labels.   ----------------
#------------------------------------------------------------------------------

create_folder(decoding_sh_dir + 'decade/classification/')
create_folder(decoding_sh_dir + 'decade/classification/standard/')
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/classification/standard/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=-0.4, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, cv=4, reject='auto_global')

# ========== score with logistic regression =======================

create_folder(decoding_sh_dir + 'decade/regression/')
create_folder(decoding_sh_dir + 'decade/regression/standard/')
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/regression/standard/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None)],
                                                  tmin=-0.4, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard_reg',
                                                  cv=4, reject='auto_global')

# ====== train decade decoder on given positions and test it on the other positions ========

create_folder(decoding_sh_dir + 'decade/classification/train_loc04_test_loc2/')
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/classification/train_loc04_test_loc2/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=-0.4, tmax=1, env_per_subj=env_per_subj, load_error_trials=False, decoding_method='standard',
                                                  train_epoch_filter = 'location == 0 or location == 4',test_epoch_filter = 'location == 2',
                                                  cv=4, reject='auto_global')

# ------------------------------------------------------------------------------
# ====== decode decade locked on RT ========

create_folder(decoding_sh_dir + 'decade/classification/RT-locked/')
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/classification/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True, load_error_trials=False,
                                                  cv=4, reject='auto_global')

create_folder(decoding_sh_dir + 'decade/regression/RT-locked/')
dpm.decoding.run_fit_and_score_on_separate_trials(subj_ids, dpm.classifiers.decade(filter=None), data_filenames=dpm.comparison_raw_files,
                                                  out_dir=decoding_sh_dir + 'decade/regression/RT-locked/',
                                                  grouping_metadata_fields=['location', 'target'],
                                                  epoch_filter='target>10 and decade!=4 and unit!=4',
                                                  filename_suffix_fit='comp',
                                                  filename_suffix_scores=['comp_score_as_decade', 'comp_score_as_unit'],
                                                  classifier_specs_test=[dpm.classifiers.decade(filter=None), dpm.classifiers.unit(filter=None)],
                                                  tmin=-1.5, tmax=0, env_per_subj=env_per_subj, on_response=True, load_error_trials=False,
                                                  decoding_method='standard_reg', cv = 4, reject='auto_global')


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
