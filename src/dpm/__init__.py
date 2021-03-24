"""
Package for the digit-position-MEG project
"""

#------------------------------------------------------------------
def write_to_log(log_fn, msg, append=True):
    if log_fn is not None:
        with open(log_fn, 'a' if append else 'w') as fp:
            fp.write(msg)


from .consts import subj_path, comparison_raw_files, decoding_dir, rsvp_raw_files

from . import consts
from . import util
from . import averaging as avg
from . import decoding
from . import classifiers
from . import files
from . import filtering
from . import plots
from . import preprocess
from . import vis
from . import stimuli
from . import rsa
from . import behavior
from . import visualsimilarity

from dpm._index import Index
