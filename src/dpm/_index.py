
import dpm.files


class Index(object):

    def __init__(self, index_dir, load_trigger_mapping=True):
        """
        Constructor

        :param index_dir: The directory where the index file is
        :param load_trigger_mapping: If True, the trigger mapping file will be loaded for each index entry
                        that defines trigger mapping
        """
        self._index_dir = index_dir
        self._trigger_mapping_loaded = False

        self._load(index_dir)

        if load_trigger_mapping:
            self.load_trigger_mapping()


    #-----------------------------------
    def _load(self, index_dir):
        """
        Load a subject's index file
        :param index_dir: The base directory of the subject
        :return: array of dict's
        """

        index_fn = index_dir + '/index.csv'
        index_data, fieldnames = dpm.files.load_csv(index_fn)

        for field in ['sss', 'behavior', 'responses', 'trigger_mapping_fn']:
            if field not in fieldnames:
                raise Exception('Invalid index file {:}: column "{:}" is missing'.format(index_fn, field))

        for row in index_data:
            if row['responses'] == '':
                row['responses'] = None

        self._data = index_data

    #-----------------------------------
    def load_trigger_mapping(self):

        if self._trigger_mapping_loaded:
            return

        mapping_files = set([row['trigger_mapping_fn'] for row in self._data if row['trigger_mapping_fn'] != ''])
        mappings = dict([(fn, load_trigger_mapping_file(self._index_dir + '/' + fn)) for fn in mapping_files])

        for row in self._data:
            if row['trigger_mapping_fn'] != '':
                row['trigger_mapping'] = mappings[row['trigger_mapping_fn']]

        self._trigger_mapping_loaded = True


    #-----------------------------------
    def get_entry_for_sss(self, sss_fn):
        """
        Get the index entry with the given SSS filename
        """
        for entry in self._data:
            if entry['sss'] == sss_fn:
                return entry

        return None


    #-----------------------------------
    def sss_fn_to_behavior_file_path(self, sss_fn):
        return self.index_dir + '/behavior/' + self.get_entry_for_sss(sss_fn)['behavior']


    #-----------------------------------
    @property
    def rows(self):
        """
        Iterate through the index entries
        """
        rownum = 0
        while rownum < len(self._data):
            yield self._data[rownum]
            rownum += 1

    #-----------------------------------
    @property
    def index_dir(self):
        return self._index_dir


#-----------------------------------------------------------------------
def load_trigger_mapping_file(filename):
    """
    Load a trigger-to-stimulus mapping file.

    :return: a dict with the trigger as a key
    """
    mapping, map_fieldnames = dpm.files.load_csv(filename)

    for field in ['Stimulus', 'Location', 'Group', 'Trigger']:
        if field not in map_fieldnames:
            raise Exception('Invalid mapping file {:}: column "{:}" is missing'.format(filename, field))

    has_number_col = 'Number' in map_fieldnames

    triggers = [row['Trigger'] for row in mapping]
    if len(triggers) != len(set(triggers)):
        raise Exception('Invalid mapping file {:}: some triggers are defined more than once'.format(filename))

    result = {}
    for row in mapping:
        trigger = int(row['Trigger'])
        target = int(row['Number' if has_number_col else 'Stimulus'])
        result[trigger] = {'stimulus': row['Stimulus'],
                           'target':   target,
                           'location': int(row['Location']),
                           'group':    int(row['Group']),
                           'trigger':  trigger,
                           }

    return result

