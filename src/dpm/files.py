"""
 Basic handling of files
"""

import os
import re
import math
import csv
import numpy as np
import mne
import pandas as pd
import dpm
import umne

#-- Minimal/maximal response delay for RSVP trials
min_response_delay = 50
max_response_delay = 2000


#-----------------------------------------------------------
class Data(object):
    """
    The data (MEG and behavioral) from one or more files
    """

    def __init__(self, filenames, raw, stimulus_events, response_events, stim_metadata=None, response_metadata=None):

        self.filenames = [filenames] if type(filenames) == str else filenames
        self.raw = raw
        self.stimulus_events = stimulus_events
        self.response_events = response_events
        self.stimulus_metadata = stim_metadata
        self.response_metadata = response_metadata


    def __str__(self):
        return "Data ({:} files, {:} stimulus events, {:} response events".format(
            len(self.filenames), len(self.stimulus_events), len(self.response_events))


#-----------------------------------------------------------
def load_bresults(filename):
    """
    Read the behavioral results file.
    In the returned dict, there is one entry per column in the file

    :return: dict
    """
    mandatory_cols = ['target', 'position', 'trigger', 'trialTime', 'duration']
    optional_cols = ['correct', 'response', 'rt', 'numeric_target']

    hand_mapping = _get_hand_mapping_from_filename(filename)

    fp = open(filename, 'r')
    reader = csv.DictReader(fp)

    result = {}

    try:

        for field in mandatory_cols:
            if field not in reader.fieldnames:
                raise Exception('Column "{:}" missing in file {:}'.format(field, filename))
        optional_cols = [field for field in optional_cols if field in reader.fieldnames]

        is_string_targets = 'numeric_target' in optional_cols

        for row in reader:

            if is_string_targets:
                target = int(row['numeric_target'])
                stimulus = row['target']
            else:
                stimulus = row['target']
                target = int(stimulus) if stimulus.isdigit() else -1

            result.setdefault('stimulus', []).append(stimulus)
            result.setdefault('target', []).append(target)

            for field in mandatory_cols[1:] + optional_cols:
                result.setdefault(field, []).append(int(float(row[field])))

            if hand_mapping is not None:
                result.setdefault('handmapping', []).append(hand_mapping)

    finally:
        fp.close()

    #-- Convert all arrays to np.array's
    result = {k: np.array(result[k]) for k in result.keys()}

    return result


#---------------------------------
def _get_hand_mapping_from_filename(filename):
    m = re.match('comp\d(\w).*', os.path.basename(filename))
    if m is None:
        return None
    else:
        mode = m.group(1)
        if mode not in ('c', 'n'):
            raise Exception('Invalid filename for behavioral data ({}) - expecting either "c" or "n", not "{}"'.format(filename, mode))
        return mode == 'c'


#-----------------------------------------------------------------------
def load_bresults_multiple_files(filenames, subj_id=None, rsvp=None):
    """
    Load behavioral results from multiple files

    :param subj_id: if specified, 'filenames' should not be a full path but the basenames of fif files
    :param rsvp: If True, data will be loaded also for the RSVP response-times file. Default: auto-detect according to filenames.
    """

    if rsvp is None:
        rsvp = filenames[0].startswith('rsvp')

    if subj_id is not None:
        subj_dir = dpm.subj_path[subj_id]
        index = dpm.Index(subj_dir)
        filenames = [index.sss_fn_to_behavior_file_path(filename) for filename in filenames]

    result = None
    ifile = 0
    for fn in filenames:
        curr_file_data = load_bresults(fn)
        any_field = tuple(curr_file_data.keys())[0]
        n_items = len(curr_file_data[any_field])

        curr_file_data['_filename_'] = [fn] * n_items
        curr_file_data['_filenum_'] = [ifile] * n_items
        curr_file_data['_linenum_'] = list(range(n_items))
        ifile += 1

        if rsvp:
            _add_rsvp_responses(fn, curr_file_data)

        if result is None:
            result = curr_file_data

        else:
            for key in result.keys():
                result[key] = np.concatenate([result[key], curr_file_data[key]])

    return result


#-----------------------------------------------------------------------
def load_csv(filename):

    fp = open(filename)
    reader = csv.DictReader(fp)
    csv_data = list(reader)
    fp.close()

    return csv_data, reader.fieldnames


#-----------------------------------------------------------------------
def load_rsvp_responses_file(filename):
    """
    Load the file containing the response times in the RSVP task
    """
    with open(filename, 'r') as fp:
        reader = csv.DictReader(fp)
        times = [int(float(row['time'])) for row in reader]
        return times


#-----------------------------------------------------------------------
def _add_rsvp_responses(data_filename, data, read_response_files_func=load_rsvp_responses_file):
    """
    For RSVP tasks: add the info from the response-times file onto 'data'
    :param data_filename: The name of the base data file (NOT the response times file)
    :param read_response_files_func: This is only for debugging
    """

    response_times = read_response_files_func(rsvp_responses_filename(data_filename))
    #data['all_response_times'] = response_times   todo: can't add this because it disrupts the creation of a DataFramme

    catch_trial_inds = np.where(np.array(data['trigger']) == 1)[0]

    n_items = len(data['trigger'])
    data['correct'] = np.array([None] * n_items)
    data['rt'] = [None] * n_items

    for i, trial_ind in enumerate(catch_trial_inds):
        next_catch_trial_ind = None if (i >= len(catch_trial_inds) - 1) else catch_trial_inds[i+1]
        det, rt, response_time = _is_rsvp_detected(data, trial_ind, next_catch_trial_ind, np.array(response_times))
        data['correct'][trial_ind] = det
        data['rt'][trial_ind] = rt
        if det:
            response_times.remove(response_time)

    data['rt'] = np.array(data['rt'])

    return response_times


#-----------------------------------------------------------------------
def rsvp_responses_filename(data_filename):
    """
    Convert a results filename to the responses filename (for RSVP tasks)
    """
    m = re.match('(.*).csv', data_filename)
    assert m is not None, 'Invalid filename: {}'.format(data_filename)
    rsvp_filename = m.group(1)+'-responses.csv'
    return rsvp_filename


#-----------------------------------------------------------------------
def _is_rsvp_detected(data, trial_ind, next_catch_trial_ind, response_times):

    trial_time = data['trialTime'][trial_ind]
    min_response_time = trial_time+min_response_delay
    max_response_time = trial_time+max_response_delay
    if next_catch_trial_ind is not None:
        next_trial_min_response_time = data['trialTime'][next_catch_trial_ind] + min_response_delay
        max_response_time = min(max_response_time, next_trial_min_response_time)

    matching_responses = np.logical_and(min_response_time <= response_times, response_times <= max_response_time)

    if sum(matching_responses) == 0:
        return False, None, None

    else:
        response_time = response_times[matching_responses][0]
        time_of_response = response_time
        return True, time_of_response - trial_time, response_time


#-----------------------------------------------------------------------
def load_raw(subj_dir, filenames, ica_filter_filename='ica_filter-ica.fif', index=None, m_files=None, load_error_trials=True):
    """
    Load a raw data file
    
    :param subj_dir: Directory with the subject data
    :param filenames: Name of raw file (no path)
    :param ica_filter_filename: A name of a .pkl file (path relative to "subdir"). If file exists, it should contain
                                an ICA filter (obtained from dpm.filtering.find_ecg_eog_components), and this filter
                                will be immediately applied to the loaded raw data.
    :param index: A "dpm.Index" object - index of the relevant data files per MEG file
    :param m_files: Read up to this number of files (for debugging)
    :param load_error_trials: Load the error trials or not. This affects only the STIMULUS events, not the RESPONSE events.
    :return: dpm.files.Data
    """

    #subj_dir = dpm.subj_path['cc']
    # filenames = data_filenames
    subdir = 'sss'  # previously this was an argument to the function. For now, keep it here

    if index is None:
        index = dpm.Index(subj_dir)

    if isinstance(filenames, str):
        filenames = [filenames]

    index.load_trigger_mapping()

    for filename in filenames:
        if index.get_entry_for_sss(filename) is None:
            raise Exception("%s does not have trigger mapping defined in index.csv" % filename)
    if len(set([index.get_entry_for_sss(filename)['trigger_mapping_fn'] for filename in filenames])) > 1:
        raise Exception("The files provided do not rely on the same trigger mapping. Check out the index.csv to see that")

    if m_files is not None and m_files < len(filenames):
        filenames = filenames[:m_files]

    #-- Load data
    raws = [mne.io.read_raw_fif(subj_dir + "/" + subdir + "/" + filename, preload=True) for filename in filenames]
    raw = mne.concatenate_raws(raws)

    #-- Remove ECG/EoG channels
    ica_file = subj_dir + "/" + subdir + "/" + ica_filter_filename
    if os.path.exists(ica_file):
        ica = mne.preprocessing.read_ica(ica_file)
        raw = ica.apply(raw, exclude=ica.exclude)

    #-- Get events
    stimulus_events = mne.find_events(raw, stim_channel='STI101', consecutive='increasing', min_duration=0.002, mask=0x000000FF)

    _remap_triggers(filenames, index, stimulus_events)

    #-- Load behavioral results and create metadata accordingly
    behavioral_results = load_bresults_multiple_files([index.sss_fn_to_behavior_file_path(filename) for filename in filenames])
    stim_metadata = create_stim_metadata(stimulus_events, behavioral_results, subj_dir)

    response_events, response_metadata, stimulus_events, stim_metadata = \
        _events_by_responses(stimulus_events, stim_metadata, behavioral_results, load_error_trials)

    return Data(filenames, raw, stimulus_events, response_events, stim_metadata, response_metadata)


#----------------------------------------------------------
def _events_by_responses(stimulus_events, stim_metadata, behavioral_results, load_error_trials):
    """
    Create response events+metadata; and optionally filter stimulus events to only-correct trials
    """

    has_response = 'rt' in behavioral_results
    if has_response:
        rt = behavioral_results['rt']
        resp = behavioral_results['response']

        if not load_error_trials:
            ok_trials = list(stim_metadata.correct)
            idx = np.where(ok_trials)
            stimulus_events = stimulus_events[idx[0], :]
            stim_metadata = stim_metadata.iloc[idx[0]]
            rt = rt[idx[0]]
            resp = resp[idx[0]]

        response_events = stimulus_events.copy()
        response_events = response_events[resp != 0]
        response_events[:, 0] = [response_events[k, 0]+rt[k] for k in range(len(response_events))]
        response_metadata = stim_metadata[resp != 0]

    else:
        assert load_error_trials, "Can't exclude error trials when the metadata does not contain responses"
        response_events = None
        response_metadata = None

    return response_events, response_metadata, stimulus_events, stim_metadata


#-------------------------------------------------------------
def _remap_triggers(filenames, index, stimulus_events):
    triggers = stimulus_events[:, 2]
    mapping = index.get_entry_for_sss(filenames[0])['trigger_mapping']
    for i in range(len(triggers)):
        if triggers[i] in mapping:
            target = mapping[triggers[i]]['target']
            location = mapping[triggers[i]]['location']
            if target <= 0:
                stimulus_events[i, 2] = 1
            else:
                stimulus_events[i, 2] = target * 10 + location

        elif 1 < triggers[i] < 512:
            print('Warning: trigger %d is unknown to the mapping file' % triggers[i])


#-------------------------------------------------------------
def create_stim_metadata(events, behavioral_results, subj_dir):

    triggers = np.array(events[:, 2])
    if len(triggers) != len(behavioral_results['_linenum_']):
        msg = 'Invalid subject directory {:}: there are {:} trials in the MEG file but {:} trials in the behavioral file'.\
            format(subj_dir, len(triggers), len(behavioral_results['_linenum_']))
        print(msg)
        raise Exception(msg)

    location = triggers % 10
    targets = ((triggers - location) / 10).astype(int)


    digit_in_position = np.ones([triggers.shape[0], 6]) * -1

    for ii, loc in enumerate(location):
        tar = str(targets[ii])
        if len(tar) > 1:
            digit_in_position[ii, loc] = tar[0]
            digit_in_position[ii, loc+1] = tar[1]
        else:  # one digit number
            digit_in_position[ii, loc] = tar


    metadata = {
        'event_time':   events[:, 0],
        'target':       targets,
        'decade':       np.array([int(math.floor(x)) for x in targets/10]),
        'unit':         targets % 10,
        'location':     location,
        'ndigits':      np.array([len(str(x)) for x in targets]),
        'distanceto44': abs(targets - 44),
        'hemifield':    _get_hemifields(triggers),
        'ind_in_block': behavioral_results['_linenum_'],
        'block':        behavioral_results['_filenum_'],
        'digit_in_position_0': digit_in_position[:, 0],
        'digit_in_position_1': digit_in_position[:, 1],
        'digit_in_position_2': digit_in_position[:, 2],
        'digit_in_position_3': digit_in_position[:, 3],
        'digit_in_position_4': digit_in_position[:, 4],
        'digit_in_position_5': digit_in_position[:, 5],
    }

    #-- Response: available only in the comparison task
    if 'rt' in behavioral_results:
        metadata['rt'] = behavioral_results['rt']
        metadata['response'] = behavioral_results['response']
        metadata['correct'] = behavioral_results['correct']
        metadata['hand_mapping'] = behavioral_results['handmapping']

    return pd.DataFrame(metadata)






def _get_hemifields(triggers):

    location = triggers % 10

    hemifields = np.ones(triggers.shape)
    hemifields[location >= 3] = 2

    hemifields[np.logical_and(triggers >= 100, location == 2)] = -1   # exactly in the middle

    return hemifields


#-------------------------------------------------------------
def remap_stimulus_triggers(events, index, filename):
    """
    Change the triggers - which are meaningless in the raw files - into meaningful event IDs

    :param events: The stimulus events
    :type index: dpm.Index
    :param filename: the raw file name
    """
    triggers = events[:, 2]
    mapping = index.get_entry_for_sss(filename)['trigger_mapping']

    for i in range(len(triggers)):
        if triggers[i] in mapping:
            target = mapping[triggers[i]]['target']
            location = mapping[triggers[i]]['location']
            if target <= 0:
                events[i, 2] = 1
            else:
                events[i, 2] = target*10+location

        elif 1 < triggers[i] < 512:
            print('Warning: trigger %d is unknown to the mapping file' % triggers[i])

    return events


def correct_for_trigger_delay(stim_events, trigger_delay=50):

    print('============ correcting for the trigger delay ========= ')
    print(' Only do this for the stimuli events  ')

    stim_events[:, 2] = stim_events[:, 2] + trigger_delay

    return stim_events


#--------------------------------------------------------------------------------
def load_meg_events(subj_id, filenames):
    """
    Load event IDs from the MEG files
    """
    subj_dir = dpm.subj_path[subj_id]

    #-- Load data
    raws = [mne.io.read_raw_fif(subj_dir + "/sss/" + filename, preload=True) for filename in filenames]
    raw = mne.concatenate_raws(raws)

    stimulus_events = mne.find_events(raw, stim_channel='STI101', consecutive='increasing', min_duration=0.002, mask=0x000000FF)

    return stimulus_events[:, 2]


#--------------------------------------------------------------------------------
def load_subj_data(subj, filenames, lopass_filter, load_error_trials=True):

    print('----- %s: Loading data -----' % subj)

    sdata = load_raw(dpm.subj_path[subj], filenames, load_error_trials=load_error_trials)

    if lopass_filter is not None:
        print('----- {:}: Applying a low-pass filter ({:} Hz)-----'.format(subj, lopass_filter))
        sdata.raw.filter(None, lopass_filter)

    return sdata


#--------------------------------------------------------------------------------
def load_subj_epochs(subj_id, data_filenames, lopass_filter=None, decim=1, meg_channels=True, tmin=-0.1, tmax=1.0,
                     baseline=(None, 0), rejection='default', on_response=False, load_error_trials=True):
    """
    Load subject data and create epochs

    :param subj_id:
    :param data_filenames: List of file nammes (no directory)
    :param on_response: If True, epochs are locked on response
    """

    sdata = load_subj_data(subj_id, data_filenames, lopass_filter, load_error_trials=load_error_trials)
    if on_response:
        events = sdata.response_events
        metadata = sdata.response_metadata
    else:
        events = sdata.stimulus_events
        metadata = sdata.stimulus_metadata

    epochs = umne.epochs.create_epochs_from_raw(raw=sdata.raw, events=events, metadata=metadata, meg_channels=meg_channels,
                                                tmin=tmin, tmax=tmax, decim=decim, baseline=baseline, reject=rejection)
    return epochs
