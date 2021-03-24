import numpy as np
from sklearn.model_selection import StratifiedKFold


#-------------------------------------------------------------------------
def split_train_test_cv(epochs, metadata_fields, cv=5):
    """
    This function will output a set of length cv of training and testing epochs balanced according to the metadata_fields
    """
    # build the labels that will be provided to the stratified k folds
    y = []
    for ii in range(len(epochs)):
        this_epoch = epochs[ii]
        this_y = 0
        for k in range(len(metadata_fields)):
            this_y += 10**(3*k)*this_epoch.metadata[metadata_fields[k]].values
        y.append(this_y)

    y = np.concatenate(y)

    skf = StratifiedKFold(n_splits=cv)

    epochs_train = []
    epochs_test = []
    for train_index, test_index in skf.split(epochs.get_data(), y):
        print("The ratio of testing/training epochs is %.2f"%(len(test_index)/len(train_index)))
        epochs_train.append(epochs[train_index])
        epochs_test.append(epochs[test_index])

    return epochs_train, epochs_test