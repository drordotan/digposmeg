"""
Create the dissimmilarity matrices for RSA
"""

"""
import sys

sys.path.append('/Users/dror/git/digposmeg/analyze')
sys.path.append('/Users/dror/git/jr-tools')
sys.path.append('/Users/dror/git/pyRiemann')
"""

import mne
mne.set_log_level('info')

import dpm


metrics = 'spearmanr', 'euclidean', 'mahalanobis'


for subj_id in dpm.consts.subj_ids_clean:
    print('================================== processing {:} ==================================='.format(subj_id))
    #dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True)
    #dpm.rsa.preprocess_and_compute_dissimilarity_response(subj_id, metrics=metrics, rejection=None, include_4=True)

    dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True,
                                                      grouping_metadata_fields=('decade', 'location'))

    dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True,
                                                      grouping_metadata_fields=('unit', 'location'))

    dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True,
                                                      grouping_metadata_fields=('decade', 'unit'))

    dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True,
                                                      grouping_metadata_fields=['decade'])

    dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True,
                                                      grouping_metadata_fields=['unit'])

    dpm.rsa.preprocess_and_compute_dissimilarity_stim(subj_id, metrics=metrics, rejection=None, include_4=True,
                                                      grouping_metadata_fields=['location'])
