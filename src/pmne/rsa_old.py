import numpy as np
from mne.decoding import UnsupervisedSpatialFilter
from pyriemann.utils.distance import distance as riemann_distance
from scipy.stats import spearmanr
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from statsmodels import api as sm

from umne import transformers



#----------------------------------------------------------------------------------------------
def correlate_dissimilarity_matrices(dissimilarity_funcs, stimuli, corrmethod='pearsonr', include_diagonal=True):
    """
    Compute the correlations between pairs of similarity matrices

    If provided with N dissimilarity functions, this function returns a N*N matrix of correlation coefficients

    :param dissimilarity_funcs: A list of dissimilarity functions. Each will be used to generate a dissimilarity matrix.
    :param stimuli: Stimuli for which the matrices will be generated
    :param corrmethod: The method to use for computing the correlation between 2 matrices
    :param include_diagonal: Whether the diagonal should be included when computing the correlation
    """

    matrices = [gen_predicted_dissimilarity(f, stimuli).matrix for f in dissimilarity_funcs]

    if include_diagonal:
        def matrix_to_array(m):
            return [x for row in m for x in row]
    else:
        def matrix_to_array(m):
            n = range(len(m))
            return [m[i,j] for i in n for j in n if i != j]

    matrices_as_arrays = [matrix_to_array(m) for m in matrices]

    if corrmethod == 'pearsonr':
        return np.corrcoef(matrices_as_arrays)

    if corrmethod == 'spearmanr':
        # noinspection PyTypeChecker
        return spearmanr(matrices_as_arrays, axis=1)

    else:
        raise Exception('Unsupported correlation method: {:}'.format(corrmethod))



#=========================================================================================================
#               Transformers
#=========================================================================================================

#-------------------------------------------------------------------------------------------------
class DissimilarityByCorrelation(TransformerMixin):
    """
    Create a dissimilarity matrix by computing the correlation in each

    Input matrix: Stimuli x Channels/components x TimePoints
    Output matrix: TimePoints x Stimuli x Stimuli
    """

    # --------------------------------------------------
    def __init__(self, metric, debug=False):
        self._debug = debug
        if metric == 'spearmanr':
            self._metric = DissimilarityByCorrelation._spearmanr
        else:
            self._metric = metric

    # --------------------------------------------------
    @staticmethod
    def _spearmanr(a, b):
        # noinspection PyUnresolvedReferences
        return 1-spearmanr(a, b).correlation


    # --------------------------------------------------
    def transform(self, x):

        print(
        'DissimilarityByCorrelation: computing a {:}*{:} dissimilarity matrix using correlations for each of {:} time points...'.
        format(x.shape[0], x.shape[0], x.shape[2]))

        dissim_matrices = np.asarray([pairwise_distances(x[:, :, t], metric=self._metric) for t in range(x.shape[2])])

        if self._debug:
            print('DissimilarityByCorrelation: transformed from shape={:} to shape={:}'.format(x.shape,
                                                                                               dissim_matrices.shape))

        return dissim_matrices


    # --------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self


#-------------------------------------------------------------------------------------------------
class RiemannDissimilarity(TransformerMixin):
    """
    Compute Riemann dissimilarity

    Alternative #1:
        Input matrix: Stimuli x Channels/components x TimePoints
        Output matrix: Stimuli x Stimuli

    Alternative #1:
        Input matrix: N-Outupt-TimePoints x Stimuli x Channels/components x TimePoints-of-one-matrix
        Output matrix: N-Outupt-TimePoints x Stimuli x Stimuli
    """

    def __init__(self, metric='riemann', debug=False):
        self._metric = metric
        self._debug = debug


    # --------------------------------------------------
    def _create_one_dissimilarity_matrix(self, x):
        """
        :param x: Stimuli x Channels/components x TimePoints
        """
        averaged_epochs_cov = np.asarray([np.cov(stim) for stim in x])
        dissim_matrix = np.array([[riemann_distance(k, l, metric=self._metric) for l in averaged_epochs_cov]
                                  for k in averaged_epochs_cov])
        return dissim_matrix


    # --------------------------------------------------
    def transform(self, x):

        if len(np.array(x[0]).shape) == 2:
            # -- x has 3 dimensions: Stimuli x Channels/components x TimePoints
            print('RiemannDissimilarity: computing a {:}*{:} dissimilarity matrix...'.format(len(x), len(x)))
            dissim_matrices = [self._create_one_dissimilarity_matrix(x)]

        elif len(np.array(x[0]).shape) == 3:
            # -- x has 4 dimensions: Windows x Stimuli x Channels/components x TimePoints
            print('RiemannDissimilarity: computing {:} {:}*{:} dissimilarity matrices...'.
                  format(len(x), len(x[0]), len(x[0])))
            dissim_matrices = [self._create_one_dissimilarity_matrix(m) for m in x]

        else:
            raise Exception('Invalid input')

        dissim_matrices = np.array(dissim_matrices)

        if self._debug:
            print(
            'RiemannDissimilarity: transformed from shape={:} to shape={:}'.format(x.shape, dissim_matrices.shape))

        return dissim_matrices


    # --------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

