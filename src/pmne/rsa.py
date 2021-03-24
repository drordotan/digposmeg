
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------
def plot_matrix(matrix, labels=None, stim_filter_func=None, stim_to_label_func=None,
                stim_to_x_label_func=None, stim_sort_func=None, cmap=None, title=None,
                color_scale_max_std=2):
    """
    Plot a dissimilarity matrix

    :param matrix: The matrix
    :param stimuli: The stimuli that correspond with this matrix (same order as the matrix).
    :param labels: Labels for the stimuli
    :param stim_filter_func: A function that determines which stimuli to print in the axes.
           The function gets a stimulus and returns True/False (True = print this one)
    :param stim_to_label_func: A function that generates the axis label for each stimulus.
           Ignored if the "labels" parameter is provided.
    :param stim_to_x_label_func: specify when x labels are different
    :param stim_sort_func: A function that determines the order of stimuli in the matrix.
           The function compares 2 stimuli (compatible with Python's sorted())
    :param cmap: Color map for imshow()
    :param color_scale_max_std: set the color scale's min/max values to mean +/- std*color_scale_max_std
     """

    #-- Sort
    if stim_sort_func is not None:
        matrix = matrix.copy().sort(stim_sort_func)

    #-- Prepare labels
    y_labels = _prepare_stim_labels(stimuli=stimuli, labels=labels, stim_filter_func=stim_filter_func,
                                  stim_to_label_func=stim_to_label_func)

    if stim_to_x_label_func is None:
        x_labels = labels
    else:
        x_labels = _prepare_stim_labels(stimuli=stimuli, labels=labels, stim_filter_func=stim_filter_func,
                                        stim_to_label_func=stim_to_x_label_func)

    if color_scale_max_std is None:
        min_val = None
        max_val = None
    else:
        min_val = np.mean(matrix) - color_scale_max_std*np.std(matrix)
        max_val = np.mean(matrix) + color_scale_max_std*np.std(matrix)

    plt.imshow(matrix, interpolation='none', cmap=cmap, origin='lower',vmin=min_val, vmax=max_val)

    if y_labels is not None:
        used_inds = np.where([l is not None for l in y_labels])[0]
        used_ylabels = y_labels[used_inds]
        used_xlabels = x_labels[used_inds]
        print(used_inds)
        plt.gca().set_xticks(used_inds)
        plt.gca().set_xticklabels(used_xlabels)
        plt.gca().set_yticks(used_inds)
        plt.gca().set_yticklabels(used_ylabels)

    if title is not None:
        plt.title(title)

    plt.colorbar()


#----------------------------------------------------------------------------------------------
def _prepare_stim_labels(stimuli, labels, stim_filter_func, stim_to_label_func):

    if stimuli is None:
        return None

    else:
        if labels is None and stim_to_label_func is not None:
            labels = [stim_to_label_func(stim) for stim in stimuli]

        if labels is not None:
            if stim_filter_func is not None:
                for i in range(len(labels)):
                    if not stim_filter_func(stimuli[i]):
                        labels[i] = None

        return np.array(labels)
