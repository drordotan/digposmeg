"""
 Pre-processing functions
"""

import os
import numpy as np
import re
import csv
from operator import itemgetter

import dpm.files


#-----------------------------------------------------------------------
def create_index(subject_path, ignore=()):
    """
    Create a CSV index of the data files of this subject
    """

    sss_path = subject_path + "/sss/"
    behavioral_path = subject_path + "/behavior/"

    if not os.path.isdir(sss_path):
        raise Exception('SSS path was not found: ' + sss_path)

    if not os.path.isdir(behavioral_path):
        raise Exception('Path for behavioral files was not found: ' + sss_path)

    index_data = []

    for sss_fn in os.listdir(sss_path):
        if sss_fn.startswith('ica') or not sss_fn.endswith(".fif"):
            continue

        if sum([re.match(pattern, sss_fn) is not None for pattern in ignore]) > 0:
            print("{:} skipped".format(sss_fn))
            continue

        m = re.search("^(.*)_raw_sss.fif$", sss_fn)
        if m is None:
            raise Exception("Invalid SSS filename: " + sss_fn)

        behavior_fn = m.group(1) + ".csv"
        if not os.path.isfile(behavioral_path + behavior_fn):
            raise Exception("The SSS file {:} has no corresponding behavioral-data file {:}".format(
                sss_fn, behavior_fn))

        if re.search("rsvp", sss_fn.lower()) is not None:
            # RSVP blocks should also have a response file
            response_fn = m.group(1) + "-responses.csv"
            if not os.path.isfile(behavioral_path + response_fn):
                print("WARNING: the SSS file {:} has no corresponding responses file {:}".format(sss_fn, response_fn))
                response_fn = ""
        else:
            response_fn = ""

        is_words = re.search("word", sss_fn.lower())
        mapping_file = 'trigger_mapping_words.csv' if is_words else 'trigger_mapping_numbers.csv'

        index_data.append({'sss': sss_fn,
                           'behavior': behavior_fn,
                           'responses': response_fn,
                           'trigger_mapping_fn': mapping_file})

    #-- Write the index
    index_path = subject_path + "/index.csv"
    fp = open(index_path, 'w')
    writer = csv.DictWriter(fp, ['sss', 'behavior', 'responses', 'trigger_mapping_fn'])
    writer.writeheader()
    for row in index_data:
        writer.writerow(row)
    fp.close()

    print("Created {:} for {:} data files".format(index_path, len(index_data)))


#-----------------------------------------------------------
def concat_bresults(in_data):
    """
    Concatenate behavioral results from several files
    """

    output = {}
    for k in in_data[0].keys():
        output[k] = np.concatenate([d[k] for d in in_data])

    return output

#-----------------------------------------------------------------------
def validate_triggers(subj_dir, sss_filename=None, index=None):

    if index is None:
        index = dpm.Index(subj_dir)

    if sss_filename is None:
        for row in index.rows:
            validate_triggers(subj_dir, row['sss'], index)
        return

    data = dpm.files.load_raw(subj_dir, sss_filename)
    meg_triggers = data.stimulus_events[:, 2]
    meg_triggers = meg_triggers[meg_triggers<1000]  # exclude responses

    behavior_fn = index.get_entry_for_sss(sss_filename)['behavior']
    bev_data = dpm.files.load_bresults(subj_dir + "/behavior/" + behavior_fn)
    expected_triggers = np.array(bev_data['target'] * 10 + bev_data['position'])
    expected_triggers[bev_data['target'] <= 0] = 1

    if meg_triggers.shape[0] != expected_triggers.shape[0]:
        print('The SSS file %s has %d triggers, but the behavioral file has %d entries' % (sss_filename, len(meg_triggers), len(expected_triggers)))
        return

    if sum(meg_triggers != expected_triggers) > 0:
        print('The SSS file %s and the corresponding behavioral file have different triggers' % sss_filename)
        return

    print("The triggers in " + sss_filename + " match the behavioral file")


#-----------------------------------------------------------------------
def rename_behavioral_files(bpath, go=False, ignore_missing_files=False):

    renames = []

    files = _get_files_matching("subj.*comp.*1c.*.csv", bpath, 2, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'comp1c_1.csv'))
        renames.append((files[1], 'comp1c_2.csv'))

    files = _get_files_matching("subj.*comp.*2c.*.csv", bpath, 2, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'comp2c_1.csv'))
        renames.append((files[1], 'comp2c_2.csv'))

    files = _get_files_matching("subj.*comp.*1n.*.csv", bpath, 2, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'comp1n_1.csv'))
        renames.append((files[1], 'comp1n_2.csv'))

    files = _get_files_matching("subj.*comp.*2n.*.csv", bpath, 2, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'comp2n_1.csv'))
        renames.append((files[1], 'comp2n_2.csv'))

    files = _get_files_matching("subj.*rsvp1.*.csv", bpath, 2, ignore_missing_files, exclude_pattern='.*resp.*')
    if len(files) > 0:
        renames.append((files[0], 'rsvp1_1.csv'))
        renames.append((files[1], 'rsvp1_2.csv'))

    files = _get_files_matching("subj.*rsvp2.*.csv", bpath, 1, ignore_missing_files, exclude_pattern='.*resp.*')
    if len(files) > 0:
        renames.append((files[0], 'rsvp2_1.csv'))

    files = _get_files_matching("subj.*words.*.csv", bpath, 1, ignore_missing_files, exclude_pattern='.*resp.*')
    if len(files) > 0:
        renames.append((files[0], 'rsvp_words.csv'))

    files = _get_files_matching("subj.*responses.*rsvp1.*.csv", bpath, 2, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'rsvp1_1-responses.csv'))
        renames.append((files[1], 'rsvp1_2-responses.csv'))

    files = _get_files_matching("subj.*responses.*rsvp2.*.csv", bpath, 1, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'rsvp2_1-responses.csv'))

    files = _get_files_matching("subj.*responses.*words.*.csv", bpath, 1, ignore_missing_files)
    if len(files) > 0:
        renames.append((files[0], 'rsvp_words-responses.csv'))

    for source, target in renames:
        print('{:} -> {:}'.format(source, target))
        if go:
            os.rename(bpath + os.sep + source, bpath + os.sep + target)

    if not go:
        print("\nNothing done. Set go=True to actually execute these commands")

#---------------------------------------------
def _get_files_matching(pattern, bpath, expected_n_match, ignore_missing_files, exclude_pattern=None):

    filenames = os.listdir(bpath)

    matching_files = [file for file in filenames if re.match(pattern, file, re.IGNORECASE) is not None]
    if exclude_pattern is not None:
        matching_files = [file for file in matching_files if re.match(exclude_pattern, file, re.IGNORECASE) is None]

    if len(matching_files) != expected_n_match:
        msg = "Invalid file names: found {:} files named {:}, expected {:}". \
                            format(len(matching_files), pattern, expected_n_match)
        if ignore_missing_files:
            print(msg)
        else:
            raise Exception(msg)

    if len(matching_files) == 1:
        return matching_files

    else:
        tmp = [(file, _get_file_time(file)) for file in matching_files]
        tmp.sort(key=itemgetter(1))
        return [file for file, t in tmp]


def _get_file_time(filename):
    m = re.match('.*-(\d+).csv', filename)
    if m is None:
        raise Exception('Invalid file name ({:}) - there is no time stamp on this file')
    return int(m.group(1))
