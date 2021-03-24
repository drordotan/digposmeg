import re
import socket
import sys

sys.path.append('/git/digposmeg/analyze')
sys.path.append('/git/jr-tools')

import matplotlib.pyplot as plt
import mne
import dpm
import pickle

mne.set_log_level('info')


if re.match(".*dror.*", socket.gethostname(), re.IGNORECASE):
    ica_ecg_eog_dir = '/meg/digit-position/exp/ICA results/'

elif re.match(".*fosca.*", socket.gethostname(), re.IGNORECASE):
    ica_ecg_eog_dir = '/Volumes/COUCOU_CFC/digitpos/results/preprocessing/ica_components/'
else:
    ica_ecg_eog_dir = '/home/ubuntu/data/'



########### Important: use dpm.preprocess.rename_behavioral_files() to init the names of CSV files


#============ Subject GA =======================================

# Create Index of the subject's data files - MEG and behavioral.
# dpm.preprocess.create_index(dpm.subj_path.ga)

# Validate that the triggers match between the MEG file and the behavioral file
# dpm.preprocess.validate_triggers(dpm.subj_path.ga)

#-- Detect ECG/EoG ICA components

subj = 'hr'

raw_rsvp = mne.io.read_raw_fif(dpm.subj_path[subj] + "/sss/" + 'rsvp1_2_raw_sss.fif', preload=False)
raw_com = mne.io.read_raw_fif(dpm.subj_path[subj] + "/sss/" + 'comp1c_2_raw_sss.fif', preload=False)
raw_rsvp_words = mne.io.read_raw_fif(dpm.subj_path[subj] + "/sss/" + 'rsvp_word_1_raw_sss.fif', preload=False)

raw_for_ica = mne.concatenate_raws([raw_rsvp,raw_com,raw_rsvp_words])
dpm.filtering.find_ecg_eog_components(raw_for_ica, plot=True,apply_lohipass=False, save_epochs=False, save_fn=dpm.subj_path[subj] + "/sss/"+'ica_filter.pkl')




#----- Comparison task

#-- detect components on all blocks together (this must be run on a strongserver)
sdata = dpm.files.load_raw(dpm.subj_path.ga, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'ga_comp_ica_ecg_eog.pkl')
pickle.load(ica_ecg_eog_dir+'ga_comp_ica_ecg_eog.pkl')
plt.close('all')


#-------- RSVP task

#-- detect components on all blocks together (this can run on my mac)
sdata = dpm.files.load_raw(dpm.subj_path.ga, dpm.consts.rsvp_raw_files[:3])
dpm.filtering.find_ecg_eog_components(sdata.raw, save_epochs=False, save_fn=ica_ecg_eog_dir+'ga_rsvp_ica.pkl')
plt.close('all')


#--------- RSVP-words task

#-- This cleaning is OK. Finished.
ga_raw = mne.io.read_raw_fif(dpm.subj_path.ga + "/sss/" + 'rsvp_word_1_raw_sss.fif')
dpm.filtering.find_ecg_eog_components(ga_raw)
plt.close('all')



#============ Subject AM =======================================

dpm.preprocess.create_index(dpm.subj_path.am, ignore=['rsvp_word*'])

sdata = dpm.files.load_raw(dpm.subj_path.am, dpm.comparison_raw_files)




#============ Subject AT =======================================

dpm.preprocess.rename_behavioral_files(dpm.subj_path.at + "/behavior")

# Create Index of the subject's data files - MEG and behavioral.
dpm.preprocess.create_index(dpm.subj_path.at, ignore=['rsvp_word*'])

# Validate that the triggers match between the MEG file and the behavioral file
dpm.preprocess.validate_triggers(dpm.subj_path.at)


#-- Detect ECG/EoG ICA components
sdata = dpm.files.load_raw(dpm.subj_path.at, dpm.comparison_raw_files)
ica_filter = dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'at_comp_ica_ecg_eog.pkl')


dpm.filtering.plot_ecg_eog_components(ica_filter, sdata.raw)



#============ Subject BO =======================================

dpm.preprocess.create_index(dpm.subj_path.bo, ignore=['rsvp_word*'])

dpm.preprocess.validate_triggers(dpm.subj_path.bo)

sdata = dpm.files.load_raw(dpm.subj_path.bo, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'bo_comp_ica_ecg_eog.pkl')


#============ Subject CD =======================================

dpm.preprocess.create_index(dpm.subj_path.cd, ignore=['rsvp_word*'])

dpm.preprocess.validate_triggers(dpm.subj_path.cd)

sdata = dpm.files.load_raw(dpm.subj_path.cd, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'cd_comp_ica_ecg_eog.pkl')



#============ Subject EN =======================================

### Note: this subject has many bad epochs!


dpm.preprocess.create_index(dpm.subj_path.en, ignore=['rsvp_word*'])

dpm.preprocess.validate_triggers(dpm.subj_path.en)

sdata = dpm.files.load_raw(dpm.subj_path.en, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'en_comp_ica_ecg_eog.pkl')



#============ Subject HR =======================================

dpm.preprocess.create_index(dpm.subj_path.hr, ignore=['rsvp_word*'])

dpm.preprocess.validate_triggers(dpm.subj_path.hr)

sdata = dpm.files.load_raw(dpm.subj_path.hr, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'hr_comp_ica_ecg_eog.pkl')


#todo

#============ Subject JM-0 =======================================

dpm.preprocess.create_index(dpm.subj_path.jm0, ignore=['rsvp_word*'])

dpm.preprocess.validate_triggers(dpm.subj_path.jm0)

sdata = dpm.files.load_raw(dpm.subj_path.jm0, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'jm0_comp_ica_ecg_eog.pkl')


#todo

#============ Subject JM-1 =======================================

dpm.preprocess.create_index(dpm.subj_path.jm1)

dpm.preprocess.validate_triggers(dpm.subj_path.jm1)

sdata = dpm.files.load_raw(dpm.subj_path.jm1, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'jm1_comp_ica_ecg_eog.pkl')



#============ Subject MP =======================================

dpm.preprocess.create_index(dpm.subj_path.mp, ignore=['rsvp_word*'])

dpm.preprocess.validate_triggers(dpm.subj_path.mp)

sdata = dpm.files.load_raw(dpm.subj_path.mp, dpm.comparison_raw_files)
dpm.filtering.find_ecg_eog_components(sdata.raw, plot=False, save_epochs=False, save_fn=ica_ecg_eog_dir+'mp_comp_ica_ecg_eog.pkl')
