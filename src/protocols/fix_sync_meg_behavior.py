"""
Fix the un-synchronized MEG and behavioral files
"""

import csv
import mne

import dpm


def _load_triggers_from_csv_file(filename):
    with open(filename, 'r') as fp:
        reader = csv.DictReader(fp)
        triggers = [int(row['trigger']) for row in reader]

    return triggers


def check_subj(subj_id, filenames, file_desc):

    subj_id_desc = '{:} ({:})'.format(subj_id, file_desc)
    subj_dir = dpm.subj_path[subj_id]

    print('=================================================================================')
    print('                        Subject {:}'.format(subj_id_desc))
    print('=================================================================================')

    all_ok = True
    messages = []

    for fif_filename in filenames:
        raw = mne.io.read_raw_fif(subj_dir + "/sss/" + fif_filename, preload=True)
        events = mne.find_events(raw, stim_channel='STI101', consecutive='increasing', min_duration=0.002, mask=0x000000FF)
        meg_events = events[:, 2]

        behavior_filename = fif_filename.replace('_raw_sss.fif', '.csv')
        behavioral_events = _load_triggers_from_csv_file(subj_dir + '/behavior/' + behavior_filename)

        messages.append('File {:} -'.format(fif_filename))
        file_ok = True

        if len(meg_events) == len(behavioral_events):
            if (meg_events == behavioral_events).all():
                messages.append('   OK')
            else:
                messages.append('   Mismatch, but same length'.format(subj_id_desc))
                file_ok = False

        else:
            if len(meg_events) < len(behavioral_events):
                shorter, longer = meg_events, behavioral_events
                shorter_name, longer_name = 'MEG', 'behavior'
            else:
                shorter, longer = behavioral_events, meg_events
                shorter_name, longer_name = 'behavior', 'MEG'

            nshort = len(shorter)
            nlong = len(longer)
            found_shift = False
            for i in range(nlong - nshort + 1):
                if (longer[i:i+nshort] == shorter).all():
                    messages.append('   Mismatch. To fix, remove {:} events from start of {:} and {:} events from its end'.
                          format(i, longer_name, nlong - i - nshort))
                    found_shift = True

            if not found_shift:
                messages.append('   Mismatch, different #events')
                file_ok = False


        if not file_ok:
            messages.append('  {:} MEG events     : {:}'.format(len(meg_events), ",".join([str(e) for e in meg_events])))
            messages.append('  {:} behavior events: {:}'.format(len(behavioral_events), ",".join([str(e) for e in behavioral_events])))
            all_ok = False

    if all_ok:
        messages.append('All files OK/fixable')

    print('')
    [print(m) for m in messages]

'''
for subj in ['ag', 'am', 'at', 'bl', 'bo', 'cb', 'cc', 'cd', 'eb', 'en', 'ga', 'hr', 'jm0', 'jm1', 'lb', 'lj', 'mn', 'mp']:
    check_subj(subj, dpm.comparison_raw_files, 'comparison')
    check_subj(subj, dpm.rsvp_raw_files, 'RSVP')
'''

check_subj('cd', dpm.comparison_raw_files, 'comparison')
