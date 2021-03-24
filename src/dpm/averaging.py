
import mne

def average_stimulus_epochs(raw, events):

    epochs = mne.Epochs(raw, events=events)