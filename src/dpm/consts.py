import socket
import re
import sys

if re.match(".*dror.*", socket.gethostname(), re.IGNORECASE):
    base_path = '/Users/dror/meg/digit-position/exp/'
    results_dir = base_path + 'results/'
    raw_data_path = base_path + 'exp-data/'
    figures_path = '/Users/dror/data/acad-proj/2-InProgress/DigitPositionMEG/figures'
elif re.match(".*calmar.*", socket.gethostname(), re.IGNORECASE) or re.match(".*alambic.*", socket.gethostname(), re.IGNORECASE):
    base_path = '/neurospin/meg/meg_tmp/DPEM_Dror_Fosca_2017/'
    results_dir = base_path + 'results/'
    raw_data_path = base_path + 'exp-data/'
    sys.path.append(base_path+'/scripts/')
    sys.path.append(base_path+'/scripts/digposmeg/')
    sys.path.append(base_path+'/scripts/umne/')
else:
    raise Exception('Unrecognized computer')

decoding_dir = results_dir + 'decoding/'
rsa_data_path = base_path + 'rsa-data/'

_subj_path = {
    'ag': raw_data_path + 'ag_170045',
    'at': raw_data_path + 'at_140305',
    'am': raw_data_path + 'am_150105',
    'bl': raw_data_path + 'bl_170454',
    'bo': raw_data_path + 'bo_160176',
    'cb': raw_data_path + 'cb_140229',
    'cc': raw_data_path + 'cc_150418',
    'cd': raw_data_path + 'cd_130323',
    'eb': raw_data_path + 'eb_170163',
    'en': raw_data_path + 'en_170221',
    'ga': raw_data_path + 'ga_130053',
    'hr': raw_data_path + 'hr_140096',
    'jm0': raw_data_path + 'jm_100042',
    'jm1': raw_data_path + 'jm_100109',
    'lb': raw_data_path + 'lb_170081',
    'lj': raw_data_path + 'lj_150477',
    'mn': raw_data_path + 'mn_170263',
    'mp': raw_data_path + 'mp_150285',
}

#-- IDs of subjects with good data
subj_ids_all = tuple(_subj_path.keys())
subj_ids = tuple([s for s in _subj_path.keys() if s != 'cd'])
subj_ids_clean = 'ag', 'at', 'bl', 'bo', 'cb', 'cc', 'eb', 'en', 'hr', 'jm0', 'lj', 'mn', 'mp'

class SubjPath(object):
    def __getattr__(self, name):
        return _subj_path[name]

    def __getitem__(self, item):
        return _subj_path[item]

subj_path = SubjPath()

#-- Files for comparison experiment
comp_prefix = ['comp1c_1', 'comp1c_2', 'comp2c_1', 'comp2c_2', 'comp1n_1', 'comp1n_2', 'comp2n_1', 'comp2n_2']
comparison_raw_files = ['%s_raw_sss.fif' % pref for pref in comp_prefix]

#-- Files for RSVP experiment
rsvp_prefix = ['rsvp1_1', 'rsvp1_2', 'rsvp2_1']
rsvp_raw_files = ['%s_raw_sss.fif' % pref for pref in rsvp_prefix]

#-- File of RSVP-words experiment
rsvp_words_prefix = ['rsvp_word_1']
rsvp_words_raw_files = ['%s_raw_sss.fif' % pref for pref in rsvp_words_prefix]
