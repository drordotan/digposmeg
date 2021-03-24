import sys
import numpy as np
import math

sys.path.append('/Users/dror/git/digposmeg/analyze')
sys.path.append('/Users/dror/git/jr-tools')
sys.path.append('/Users/dror/git/pyRiemann')

import mne
mne.set_log_level('info')

import umne
import dpm
dis = dpm.rsa.dissimilarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm



#----------------------------------------------------------
#-------------------- plot predictors ---------------------
#----------------------------------------------------------

umne.rsa.plot_dissimilarity(dpm.rsa.gen_predicted_dissimilarity_2digit(dis.location), tick_filter=lambda md: md['location'] == 0, get_label=lambda md: md['target'])
umne.rsa.plot_dissimilarity(dpm.rsa.gen_predicted_dissimilarity_2digit(dis.retinotopic_id), tick_filter=lambda md: md['location'] == 0, get_label=lambda md: md['target'])
umne.rsa.plot_dissimilarity(dpm.rsa.gen_predicted_dissimilarity_2digit(dis.decade_id), tick_filter=lambda md: md['location'] == 0, get_label=lambda md: md['target'])
umne.rsa.plot_dissimilarity(dpm.rsa.gen_predicted_dissimilarity_2digit(dis.unit_id), tick_filter=lambda md: md['location'] == 0, get_label=lambda md: md['target'])
umne.rsa.plot_dissimilarity(dpm.rsa.gen_predicted_dissimilarity_2digit(dis.numerical_distance), tick_filter=lambda md: md['location'] == 0, get_label=lambda md: md['target'])

# ============= correlate predictors ======================

dissim_location = dpm.rsa.gen_predicted_dissimilarity_2digit(dis.location)
dissim_retinotopic = dpm.rsa.gen_predicted_dissimilarity_2digit(dis.retinotopic_id)
dissim_decades = dpm.rsa.gen_predicted_dissimilarity_2digit(dis.decade_id)
dissim_units = dpm.rsa.gen_predicted_dissimilarity_2digit(dis.unit_id)
dissim_quantity = dpm.rsa.gen_predicted_dissimilarity_2digit(dis.numerical_distance)

dissim_matrix = [dissim_location, dissim_retinotopic,dissim_decades,dissim_units,dissim_quantity]
correlation_matrix = np.zeros((5, 5))

for k in range(5):
    for l in range(5):
        r = np.corrcoef([np.reshape(dissim_matrix[k].data, dissim_matrix[k].data.size),
                         np.reshape(dissim_matrix[l].data, dissim_matrix[l].data.size)])
        correlation_matrix[k,l]=r[0,1]

plt.imshow(correlation_matrix, cmap=cm.viridis)
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(5),['location','retinotopic','decade','unit','quantity'])
fig = plt.gcf()
fig.savefig('correlation_regressors.png')

#----------------------------------------------------------------------------------------------
# Compute dissimilarity matrix; separating trials only by target (not by location)

du_dissim_filename = dpm.consts.rsa_data_path + "dissim-matrix/{:}/decade_unit_{:}_{:}.dmat"

def create_decade_unit_epochs(epochs):
    if 'decade' not in epochs.metadata:
        epochs.metadata['decade'] = np.array([int(math.floor(x)) for x in np.array(epochs.metadata['target']) / 10])
        epochs.metadata['unit'] = epochs.metadata['target'] % 10

    epochs = epochs['target > 10 and decade != 4']

    ep_dec = umne.epochs.average_epochs_by_metadata(epochs, ['decade'])
    ep_dec.metadata['digit'] = ep_dec.metadata['decade']
    ep_dec.metadata['role'] = ['decade'] * len(ep_dec)
    ep_dec.metadata['unit'] = [None] * len(ep_dec)

    ep_un = umne.epochs.average_epochs_by_metadata(epochs, ['unit'])
    ep_un.metadata['digit'] = ep_un.metadata['unit']
    ep_un.metadata['role'] = ['unit'] * len(ep_un)
    ep_un.metadata['decade'] = [None] * len(ep_un)

    return mne.concatenate_epochs([ep_dec, ep_un])


for subj_id in dpm.consts.subj_ids:

    epochs1, epochs2 = dpm.rsa.load_average_epochs_splithalf(dpm.rsa.splithalf.format('comp', subj_id))
    epochs1 = create_decade_unit_epochs(epochs1)
    epochs2 = create_decade_unit_epochs(epochs2)

    for metric in ('spearmanr', 'euclidean'):
        dissim = umne.rsa.gen_observed_dissimilarity(epochs1, epochs2, metric=metric, sliding_window_size=100, sliding_window_step=10)
        dissim.save(du_dissim_filename.format('comp', metric, subj_id))
