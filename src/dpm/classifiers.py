"""
Extract specific features from trials
"""
import numpy as np
from enum import Enum
import math

import dpm

__default = "__default__"

#-------------------------------------------------------------------------
class DecimalRole(Enum):
    Unit = 1
    Decade = 2


#-------------------------------------------------------------------------
def _get_name(obj):
    return obj.__module__ + "." + type(obj).__name__


#------------------------------------------------------------------
class GetYLabelFromMetadataField(object):

    def __init__(self, md_field):
        self.md_field = md_field


    def __call__(self, epochs):
        return epochs.metadata[self.md_field].values

    def __str__(self):
        return "{:}('{:}')".format(_get_name(self), self.md_field)


#-------------------------------------------------------------------------
# Return a classifier-spec for hemifield classifiers
def hemifield():
    return dict(y_label_func=GetYLabelFromMetadataField('hemifield'),
                epoch_filter="hemifield != -1",
                code_writer=__name__ + ".hemifield()")


#-------------------------------------------------------------------------
# Return a classifier-spec for location classifiers
def location(include_2digit=True, include_1digit=False, min_location=None, max_location=None, hemifield=None):

    assert include_1digit or include_2digit

    filter = ""

    if not include_1digit:
        filter += " and target >= 10"

    if not include_2digit:
        filter += " and target < 10"

    if min_location is not None:
        filter += " and location >= {:}".format(min_location)

    if max_location is not None:
        filter += " and location <= {:}".format(max_location)

    if hemifield is not None:
        filter += " and hemifield == {:}".format(hemifield)

    filter = None if filter == '' else filter[4:]

    func_call = "location(include_2digit={:}, include_1digit={:}, min_location={:}, max_location={:}, hemifield={:})".\
        format(include_2digit, include_1digit, min_location, max_location, hemifield)

    return dict(y_label_func=GetYLabelFromMetadataField('location'),
                epoch_filter=filter,
                code_writer=__name__ + "." + func_call)


#-------------------------------------------------------------------------
# Return a classifier-spec for decade digit classifiers
def decade(digits=(2, 3, 5, 8), use_quantity=False, filter=__default):

    if filter == __default:
        filter = "decade in {:}".format(digits)

    return dict(y_label_func=GetYLabelFromMetadataField('decade'),
                epoch_filter=filter,
                code_writer=__name__ + ".decade(digits={:}, use_quantity={:})".format(digits, use_quantity))


#-------------------------------------------------------------------------
# Return a classifier-spec for unit digit classifiers
def unit(digits=(2, 3, 5, 8), use_quantity=False, filter=__default):

    if filter == __default:
        filter = "unit in {:}".format(digits)

    return dict(y_label_func=GetYLabelFromMetadataField('unit'),
                epoch_filter=filter,
                code_writer=__name__ + ".unit(digits={:}, use_quantity={:})".format(digits, use_quantity))


#-------------------------------------------------------------------------
# Return a classifier-spec for whole-target classifier
def target(filter=None):
    return dict(y_label_func=GetYLabelFromMetadataField('target'),
                epoch_filter=filter,
                code_writer=__name__ + ".target()")

#-------------------------------------------------------------------------
# Return a classifier-spec for a classifier that detects the digit in a given position
def digit_in_position(position):
    return dict(y_label_func=GetYLabelFromMetadataField('digit_in_position_%i' % position),
                epoch_filter="digit_in_position_%i != -1" % position,
                code_writer=__name__ + ".digit_in_position(position={:})".format(position))
