import umne.rsaplot
import dpm.rsa_old

#------------------------------------------------------------------------------
def plot_dissim_by_target(matrix, stimuli, title=None):
    """ Plot a dissimilarity matrix, sorted by the target number """
    umne.rsaplot.plot_matrix(matrix, stimuli, title=title,
                             stim_to_label_func=lambda s: s['target'],
                             stim_filter_func=lambda s: s['location'] == 2)

#------------------------------------------------------------------------------
def plot_dissim_by_location(matrix, stimuli, title=None):
    """ Plot a dissimilarity matrix, sorted by the location """
    umne.rsaplot.plot_matrix(matrix, stimuli, title=title,
                             stim_to_label_func=lambda s: s['location'],
                             stim_filter_func=lambda s: s['target'] == 32,
                             stim_sort_func=dpm.rsa_old.sortmatrix.bylocation())

#------------------------------------------------------------------------------
def plot_dissim_retinotopic(matrix, stimuli, title=None):
    """ Plot a dissimilarity matrix, sorted such that stimuli with same digit in same location are nearby """
    newmat, newstim = dpm.rsa_old.remapmatrix.to_retinotopic(matrix, stimuli)
    umne.rsaplot.plot_matrix(newmat, newstim, title=title,
                             stim_to_label_func=lambda s: '{:}@{:}'.format(s['digit'], s['location']),
                             stim_to_x_label_func=lambda s: s['digit'],
                             stim_filter_func=lambda s: s['ind_in_group'] == 1 if (s['location'] in (0, 5) or s['digit'] == 4) else s['ind_in_group'] == 4)
