#
# This script renames the behavioral result files -
# it eliminates the trailing sequence of numbers from the filename
#

import os, re

base_path = '/meg/DecRole2D/exp/'
subj_dir = 'GA-2017-03-07'


def do_rename(change_name=False):
    dirname = base_path + subj_dir + "/behavior/"
    filenames = os.listdir(dirname)

    uniq_names = {}

    for fn in filenames:
        m = re.search("^subj-(.*)-[0-9]+(.csv)$", fn)
        if m is None:
            print "{:}: ignored".format(fn)
        else:
            new_name_prefix = m.group(1) + m.group(2)
            i = uniq_names[new_name_prefix]+1 if new_name_prefix in uniq_names else 0
            uniq_names[new_name_prefix] = i

            new_name = "{:}-{:}{:}".format(m.group(1), chr(ord('a') + i), m.group(2))

            while True:
                m = re.search("^(.*)--(.*)$", new_name)
                if m is None:
                    break
                new_name = m.group(1) + "-" + m.group(2)

            print "{:} --> {:}".format(fn, new_name)

            if change_name:
                os.rename(dirname+fn, dirname+new_name)


do_rename(False)


