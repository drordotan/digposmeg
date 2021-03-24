"""
Visualization methods
"""

import numpy as np

import mne

#-------------------------------------------------------
def plot_target_by_time(data_meg, events_stim, events_resp=None, show_locations=False):
    """
    Plot the target stimulus presented at each time

    :param data_meg: The MEG data
    :param events_stim: The stimulus event matrix
    :param events_resp: The response event matrix
    :param show_locations: If False, show the target number. If True, show the stimulus location.
    """
    events = events_stim.copy()

    if show_locations:
        events[:,2] = events[:,2] % 10
    else:
        events[:, 2] = np.floor(events[:, 2] / 10)

    if events_resp is not None:
        events_resp = events_resp.copy()
        e = events_resp[:, 2]
        events_resp[e < 1000, 2] = 100
        events_resp[e > 1000, 2] = 101
        events = np.vstack((events, events_resp))

    mne.viz.plot_events(events, data_meg.info['sfreq'])

