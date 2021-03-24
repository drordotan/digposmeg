import unittest
import numpy as np

from pmne import rsa_old


class RSA_old_Tests(unittest.TestCase):

    #=============================================================================
    #    test exclude_diagonal()
    #=============================================================================

    #---------------------------------------------------------------
    def test_exclude_diagonal(self):
        matrix = [[10, 20, 30], [100, 200, 300], [1000, 2000, 3000]]
        matrix = rsa_old.exclude_diagonal(matrix)
        self.assertEqual(matrix.tolist(), [[0, 20, 30], [100, 0, 300], [1000, 2000, 0]])


    # =============================================================================
    #    test zscore_matrix()
    # =============================================================================

    #---------------------------------------------------------------
    def test_zscore(self):
        matrix = [[1, 1], [0, 0]]
        matrix = rsa_old.zscore_matrix(matrix)
        self.assertEqual(matrix.tolist(), [[1, 1], [-1, -1]])

    def test_zscore_exclude(self):
        matrix = [[1, 1], [0, 0]]
        matrix = rsa_old.zscore_matrix(matrix, include_diagonal=False)
        self.assertEqual(matrix.tolist(), [[0, 1], [-1, 0]])


    # =============================================================================
    #    test sort_matrix()
    # =============================================================================

    #---------------------------------------------------------------
    def test_sort_matrix(self):
        matrix = [[10, 20, 30], [100, 200, 300], [1000, 2000, 3000]]
        stimuli = [7, 8, 9]

        m, s = rsa_old.sort_matrix(matrix, stimuli, lambda a, b: b-a)  # reverse the order
        self.assertEqual(s.tolist(), [9, 8, 7])
        self.assertEqual(m.tolist(), [[3000, 2000, 1000], [300, 200, 100], [30, 20, 10]])


    # =============================================================================
    #    test average_matrices()
    # =============================================================================

    #---------------------------------------------------------------
    def test_average_linear(self):
        matrices = [[[1, 2], [2, 1]],
                    [[11, 12], [12, 11]]]
        self.assertEqual(rsa_old.average_matrices(matrices, 'linear').tolist(), [[6, 7], [7, 6]])


    #---------------------------------------------------------------
    def test_average_square(self):
        matrices = [[[2, 2], [2, 2]],
                    [[3, 3], [3, 3]]]
        x = np.sqrt(6.5)  # mean of 2^2 and 3^2
        self.assertEqual(rsa_old.average_matrices(matrices, 'square').tolist(), [[x, x], [x, x]])


    #---------------------------------------------------------------
    def test_average_square_negative_1(self):
        matrices = [[[-2]],
                    [[-3]]]
        x = np.sqrt(6.5)  # mean of 2^2 and 3^2
        self.assertEqual(rsa_old.average_matrices(matrices, 'square').tolist(), [[-x]])

    #---------------------------------------------------------------
    def test_average_square_negative_2(self):
        matrices = [[[-2]],
                    [[3]]]
        x = np.sqrt(2.5)  # mean of -(2^2) and 3^2
        self.assertEqual(rsa_old.average_matrices(matrices, 'square').tolist(), [[x]])

    # =============================================================================
    #    test remap_stimuli()
    # =============================================================================

    def test_remap_1(self):

        old_matrix = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        old_stimuli = [1, 2, 3]

        def filter_func(stim, group):
            if group == 'le2':
                return stim <= 2
            elif group == 'ge2':
                return stim >= 2

        def merge_func(stim, group):
            return '{:}{:}'.format(stim, group)

        new_mat, new_stim = rsa_old.remap_stimuli(old_matrix, old_stimuli, ['le2', 'ge2'], filter_func, lambda a, b: b-a, merge_func)

        self.assertEqual(new_stim, ['2le2', '1le2', '3ge2', '2ge2'])
        self.assertEqual(new_mat, [[22, 21, 23, 22], [12, 11, 13, 12], [32, 31, 33, 32], [22, 21, 23, 22]])



if __name__ == '__main__':
    unittest.main()
