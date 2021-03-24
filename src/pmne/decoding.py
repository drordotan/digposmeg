
import numpy as np
import multiprocessing

from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from mne.decoding import SlidingEstimator, UnsupervisedSpatialFilter, cross_val_multiscore

from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import ERPCovariances, XdawnCovariances, HankelCovariances

import umne
from mne.decoding import GeneralizingEstimator


#------------------------------------------------------------------
# noinspection PyIncorrectDocstring
def decode(epochs, get_y_label_func, epoch_filter=None, decoding_method='standard', sliding_window_size=None, sliding_window_step=None,
           n_jobs=multiprocessing.cpu_count(), equalize_event_counts=True, only_fit=False, generalize_across_time=True):
    """
    Basic flow for decoding
    """

    config = dict(equalize_event_counts=equalize_event_counts,
                  only_fit=only_fit,
                  sliding_window_size=sliding_window_size,
                  sliding_window_step=sliding_window_step,
                  decoding_method=decoding_method,
                  generalize_across_time=generalize_across_time,
                  epoch_filter=str(epoch_filter))

    if epoch_filter is not None:
        epochs = epochs[epoch_filter]

    #-- Classify epochs into groups (training epochs)
    y_labels = get_y_label_func(epochs)

    if equalize_event_counts:
        epochs.events[:, 2] = y_labels
        epochs.event_id = {str(label): label for label in np.unique(y_labels)}
        min_n_items_per_y_label = min([len(epochs[cond]) for cond in epochs.event_id.keys()])
        print("\nEqualizing the number of epochs to %d per condition..." % min_n_items_per_y_label)
        epochs.equalize_event_counts(epochs.event_id.keys())
        y_labels = epochs.events[:, 2]

    print("The epochs were classified into %d groups:" % len(set(y_labels)))
    for g in set(y_labels):
        print("Group {:}: {:} epochs".format(g, sum(np.array(y_labels) == g)))

    #-- Create the decoding pipeline
    print("Creating the classification pipeline...")

    epochs_data = epochs.get_data()

    preprocess_pipeline = None

    if decoding_method.startswith('standard'):

        if 'reg' in decoding_method:
            clf = make_pipeline(StandardScaler(),
                                Ridge())
        else:
            clf = make_pipeline(StandardScaler(),
                                svm.SVC(C=1, kernel='linear', class_weight='balanced'))

        if 'raw' not in decoding_method:
            assert sliding_window_size is not None
            assert sliding_window_step is not None
            preprocess_pipeline = \
                make_pipeline(umne.transformers.SlidingWindow(window_size=sliding_window_size, step=sliding_window_step, average=True))

    elif decoding_method == 'ERP_cov':
        clf = make_pipeline(
            UnsupervisedSpatialFilter(PCA(20), average=False),
            ERPCovariances(estimator='lwf'),  # todo how to apply sliding window?
            CSP(30, log=False),
            TangentSpace('logeuclid'),
            LogisticRegression('l2'))  # todo why logistic regression?

    elif decoding_method == 'Xdawn_cov':
        clf = make_pipeline(
            UnsupervisedSpatialFilter(PCA(50), average=False),
            XdawnCovariances(12, estimator='lwf', xdawn_estimator='lwf'),
            TangentSpace('logeuclid'),
            LogisticRegression('l2'))

    elif decoding_method == 'Hankel_cov':
        clf = make_pipeline(
            UnsupervisedSpatialFilter(PCA(70), average=False),
            HankelCovariances(delays=[1, 8, 12, 64], estimator='oas'),
            CSP(15, log=False),
            TangentSpace('logeuclid'),
            LogisticRegression('l2'))

    else:
        raise Exception('Unknown decoding method: {:}'.format(decoding_method))

    print('\nDecoding pipeline:')
    for i in range(len(clf.steps)):
        print('Step #{:}: {:}'.format(i + 1, clf.steps[i][1]))

    if preprocess_pipeline is not None:
        print('\nApplying the pre-processing pipeline:')
        for i in range(len(preprocess_pipeline.steps)):
            print('Step #{:}: {:}'.format(i + 1, preprocess_pipeline.steps[i][1]))
        epochs_data = preprocess_pipeline.fit_transform(epochs_data)

    if only_fit:

        #-- Only fit the decoders

        procedure = 'only_fit'
        scores = None
        cv = None

        if decoding_method.startswith('standard'):
            if 'reg' in decoding_method:
                if 'r2' in decoding_method:
                    scoring = metrics.make_scorer(metrics.r2_score)
                else:
                    scoring = metrics.make_scorer(metrics.mean_squared_error)
            else:
                scoring = 'accuracy'
            if generalize_across_time:
                estimator = GeneralizingEstimator(clf, scoring=scoring, n_jobs=n_jobs)
            else:
                estimator = SlidingEstimator(clf, scoring=scoring, n_jobs=n_jobs)
        else:
            estimator = clf

        estimator.fit(X=epochs_data, y=y_labels)

    else:

        #-- Classify & score -- cross-validation

        procedure = 'fit_and_score'
        print("\nCreating a classifier and calculating accuracy scores (this may take some time)...")

        cv = StratifiedKFold(n_splits=5)
        if decoding_method.startswith('standard'):
            if 'reg' in decoding_method:
                if 'r2' in decoding_method:
                    scoring = metrics.make_scorer(metrics.r2_score)
                else:
                    scoring = metrics.make_scorer(metrics.mean_squared_error)

            else:
                scoring = 'accuracy'
            if generalize_across_time:
                estimator = GeneralizingEstimator(clf, scoring=scoring, n_jobs=n_jobs)
            else:
                estimator = SlidingEstimator(clf, scoring=scoring, n_jobs=n_jobs)

            scores = cross_val_multiscore(estimator=estimator, X=epochs_data, y=np.array(y_labels), cv=cv)
        else:
            scores = _run_cross_validation(X=epochs_data, y=np.array(y_labels), clf=clf, cv=cv)
            estimator = 'None'  # Estimator is not defined in the case of Riemannian decoding

    times = np.linspace(epochs.tmin, epochs.tmax, epochs_data.shape[2])

    return dict(procedure=procedure, estimator=estimator, scores=scores, pipeline=clf,
                preprocess=preprocess_pipeline, cv=cv, times=times, config=config)


#------------------------------------------------------------------
def _run_cross_validation(X, y, clf, cv):

    y = np.asarray(y)
    scores = []
    for train_index, test_index in cv.split(X, y):
        print(train_index)
        print(test_index)
        print('we are in the CV loop')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train on X_train, y_train
        clf.fit(X_train, y_train)
        # Predict the category on X_test
        y_pred = clf.predict(X_test)

        scores.append(metrics.accuracy_score(y_true=y_test, y_pred=y_pred))

    return np.asarray(scores)
