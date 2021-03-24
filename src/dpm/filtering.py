"""
Filter with ICA
"""

import multiprocessing
import pickle
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import matplotlib.pyplot as plt


#-------------------------------------------------------------------
def find_ecg_eog_components(raw, apply_lohipass=True, plot=True, save_epochs=True, save_fn=None, ask_user=True):
    """
    Try filtering the raw data file with ICA: identify components suspicious as ECG/EoG components.
    
    :type raw: mne.io.fiff.raw.Raw
    :param apply_lohipass: Whether to apply a low-pass & hi-pass filter
    :param plot: Whether to plot figures
    :param save_epochs: Whether to store epochs in the returned value (set to False in order to save space)
    :param save_fn: Save the returned dict in the specified file

    :return: dict
    """

    #-- 1Hz high pass is often helpful for fitting ICA
    print('======== Loading data =========')
    raw.load_data()

    if apply_lohipass:
        print('\n======== Filtering data =========')
        raw.filter(1, 40, n_jobs=multiprocessing.cpu_count())

    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')

    #-- Create ICA and fit it to the data
    reject = dict(mag=5e-12, grad=4000e-13)
    print('\n======== ICA: identifying components =========')
    ica = ICA(n_components=25, method='fastica', random_state=23)
    ica.fit(raw, picks=picks_meg, decim=3, reject=reject)

    #----------- Filter EoG data  ---------------

    print('\n======== Detecting EoG components =========')

    eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
    eog_average = create_eog_epochs(raw, reject=reject, picks=picks_meg).average()

    #-- Find components that correlate with the EoG data
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)  # find via correlation

    if len(eog_inds) > 0:
        eog_inds_desc = ",".join([str(i) for i in eog_inds])
        print('>>> EoG components: [%s]' % eog_inds_desc)

    else:
        print(">>> No EoG components were found.")

    #----------- Filter ECG data  ---------------

    print('\n======== Detecting ECG components =========')

    ecg_average = create_ecg_epochs(raw, reject=reject, picks=picks_meg).average()
    ecg_epochs = create_ecg_epochs(raw, reject=reject)

    #-- Find components that correlate with the ECG data
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)

    if len(ecg_inds) > 0:
        ecg_inds_desc = ",".join([str(i) for i in ecg_inds])
        print('>>> ECG components: [%s]' % ecg_inds_desc)

    else:
        print(">>> No ECG components were found.")


    res = dict(ica=ica, ecg_inds=ecg_inds, eog_inds=eog_inds,
               ecg_epochs=ecg_epochs, eog_epochs=eog_epochs,
               ecg_scores=ecg_scores, eog_scores=eog_scores,
               ecg_average=ecg_average, eog_average=eog_average,
               bad_components=list(set(eog_inds + ecg_inds)),
               reject=reject)

    if plot:
        plot_ecg_eog_components(res)
        plt.show()
        plt.pause(1)

    if ask_user:
        print('---------------------------------------- \n')
        print('Do you want to update the ECG components ?')
        print('---------------------------------------- \n')

        ecg_comp = input()
        if ecg_comp=='no':
            print('>>> ECG components: [%s]' % ecg_inds)
        else:
            ecg_inds = ecg_comp

        print('---------------------------------------- \n')
        print('Do you want to update the EOG components ?')
        print('---------------------------------------- \n')

        eog_comp = input()
        if eog_comp=='no':
            print('>>> ECG components: [%s]' % ecg_inds)
        else:
            eog_inds = eog_comp


    ica.exclude.extend(eog_inds)
    ica.exclude.extend(ecg_inds)

    res = dict(ica=ica, ecg_inds=ecg_inds, eog_inds=eog_inds,
               ecg_epochs=ecg_epochs, eog_epochs=eog_epochs,
               ecg_scores=ecg_scores, eog_scores=eog_scores,
               ecg_average=ecg_average, eog_average=eog_average,
               bad_components=list(set(eog_inds + ecg_inds)),
               reject=reject)



    if not save_epochs:
        del res['ecg_epochs']
        del res['eog_epochs']

    if save_fn is not None:
        print('Saving results in %s' % save_fn)
        with open(save_fn, 'wb') as fp:
            pickle.dump(res, fp)
            ica.save(save_fn[:-4]+'-ica.fif')
        print('Saved.')

    return res

#-------------------------------------------------------------------
def plot_ecg_eog_components(res, raw=None):
    """
    Plot the results of find_ecg_eog_components()

    :param res: Return value from find_ecg_eog_components()
    :param raw: Needed when "res" does not contain ecg_epochs and eog_epochs
    :type res: dict
    """

    #----------- EoG components ---------------

    if 'eog_epochs' in res:
        eog_epochs = res['eog_epochs']
    elif raw is None:
        raise Exception('When eog_epochs is not provided as part of "res", you must specify "raw"')
    else:
        eog_epochs = create_eog_epochs(raw, reject=res['reject'])


    fig = res['ica'].plot_scores(res['eog_scores'], exclude=res['eog_inds'])  # look at r scores of components
    fig.canvas.set_window_title('component-EoG correlations')

    if len(res['eog_inds']) > 0:

        eog_inds_desc = ",".join([str(i) for i in res['eog_inds']])
        print('>>> EoG components: [%s]' % eog_inds_desc)

        figs = res['ica'].plot_properties(eog_epochs, picks=res['eog_inds'], psd_args={'fmax': 35.}, image_args={'sigma': 1.})
        for fig in figs:
            fig.canvas.set_window_title('Properties of EoG component')

        fig = res['ica'].plot_overlay(res['eog_average'], exclude=res['eog_inds'], show=False)
        fig.canvas.set_window_title('EoG average with/without components %s' % eog_inds_desc)

    else:
        print(">>> No EoG components were found.")

    #----------- ECG components ---------------

    if 'ecg_epochs' in res:
        ecg_epochs = res['ecg_epochs']
    elif raw is None:
        raise Exception('When ecg_epochs is not provided as part of "res", you must specify "raw"')
    else:
        ecg_epochs = create_ecg_epochs(raw, reject=res['reject'])

    fig = res['ica'].plot_scores(res['ecg_scores'], exclude=res['ecg_inds'])
    fig.canvas.set_window_title('component-ECG correlations')

    if len(res['ecg_inds']) > 0:

        ecg_inds_desc = ",".join([str(i) for i in res['ecg_inds']])
        print('>>> ECG components: [%s]' % ecg_inds_desc)

        figs = res['ica'].plot_properties(ecg_epochs, picks=res['ecg_inds'], psd_args={'fmax': 35.}, image_args={'sigma': 1.})
        for fig in figs:
            fig.canvas.set_window_title('Properties of ECG component')

        fig = res['ica'].plot_overlay(res['ecg_average'], exclude=res['ecg_inds'], show=False)
        fig.canvas.set_window_title('ECG average with/without components %s' % ecg_inds_desc)

    else:
        print(">>> No ECG components were found.")

    fig = res['ica'].plot_components()
