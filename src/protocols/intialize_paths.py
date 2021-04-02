import sys

base_path = '/neurospin/meg/meg_tmp/DPEM_Dror_Fosca_2017/'
sys.path.append(base_path + '/scripts/digposmeg/src/')

import dpm
import mne
fif_path = 'neurospin/meg/meg_tmp/DPEM_Dror_Fosca_2017/exp-data/ag_170045/sss/comp1c_1_raw_sss.fif'
mne.io.read_raw_fif(fif_path)


results_dir = base_path + 'results/'
raw_data_path = base_path + 'exp-data/'
sys.path.append(base_path + '/scripts/')
sys.path.append(base_path + '/scripts/digposmeg/')
sys.path.append(base_path + '/scripts/umne/src/')
sys.path.append('/neurospin/meg/meg_tmp/Geom_Seq_Fosca_2017/Geom_Seq_scripts/Analysis_scripts/packages/pyRiemann/')

import dpm.consts
