import unittest

import numpy as np
import pandas as pd

from umne import rsa


#-----------------------------------------------------------------------------
class DummyDissimMatrix(rsa.DissimilarityMatrix):

    def __init__(self, n1=None, n2=None, data=None, metadata1=None, metadata2=None, timepoint_multiply_factor=(1, )):

        if data is None:
            if n2 is None:
                n2 = n1
            data = np.array([ [i1 * 10 + i2 + 11 for i2 in range(n2)] for i1 in range(n1)])
            data = np.array([data * f for f in timepoint_multiply_factor])

        else:
            data = np.array(data)
            assert data.ndim == 3

            if n1 is None:
                n1 = data.shape[1]
            else:
                assert n1 == data.shape[1]

            if n2 is None:
                n2 = data.shape[2]
            else:
                assert n2 == data.shape[2]

        metadata1 = create_dummy_metadata(n1) if metadata1 is None else pd.DataFrame(metadata1)
        metadata2 = create_dummy_metadata(n2) if metadata2 is None else pd.DataFrame(metadata2)

        super(DummyDissimMatrix, self).__init__(data, metadata1, metadata2)


#-----------------------------------------------------------------------------
def create_dummy_metadata(len, nfields=3):

    field_names = [chr(ord('a') + i) for i in range(nfields)]
    metadata = {}
    for fld in field_names:
        metadata[fld] = [fld + str(i+1) for i in range(len)]

    return pd.DataFrame(metadata)



class RSATests(unittest.TestCase):

    #=============================================================================
    #    test reorder_matrix()
    #=============================================================================

    #---------------------------------------------------------------
    def test_reorder_matrix(self):
        m = DummyDissimMatrix(4, metadata1={'a': [1, 3, 2, 4]}, metadata2={'b': [4, 3, 2, 1]})
        ms = rsa.reorder_matrix(m, ['a'], ['b'])
        self.assertEqual(ms.data[0, 0, 0], 14)
        self.assertEqual(ms.data[0, 0, 1], 13)
        self.assertEqual(ms.data[0, 1, 0], 34)
        self.assertEqual(ms.data[0, 1, 3], 31)


    #---------------------------------------------------------------
    def test_reorder_matrix_2dim(self):
        m = DummyDissimMatrix(4, metadata1={'a': [1, 3, 2, 4]}, metadata2={'b': [4, 3, 2, 1]}, timepoint_multiply_factor=[1, 2])
        ms = rsa.reorder_matrix(m, ['a'], ['b'])
        self.assertEqual(ms.data[0, 0, 0], 14)
        self.assertEqual(ms.data[1, 0, 0], 28)


    #=============================================================================
    #    test average_matrices()
    #=============================================================================

    #---------------------------------------------------------------
    def test_average_matrices(self):

        m1 = DummyDissimMatrix(data=[[[1, 2], [3, 4]], [[10, 20], [30, 40]]])
        m2 = DummyDissimMatrix(data=[[[5, 6], [7, 8]], [[50, 60], [70, 80]]])
        mavg = rsa.average_matrices([m1, m2])

        self.assertEqual(mavg.n_timepoints, 2)
        self.assertEqual(mavg.size, (2, 2))

        data_as_tuple = tuple(map(tuple, mavg.data[0]))
        self.assertEqual(data_as_tuple, ((3, 4), (5, 6)))

        data_as_tuple = tuple(map(tuple, mavg.data[1]))
        self.assertEqual(data_as_tuple, ((30, 40), (50, 60)))


    #=============================================================================
    #    test aggregate_matrix()
    #=============================================================================

    #---------------------------------------------------------------
    def test_aggregate_matrix(self):

        dissim = DummyDissimMatrix(n1=6, n2=6, metadata1={'a': [1, 2, 3] * 2}, metadata2={'b': [1, 1, 1, 2, 2, 2]},
                                   timepoint_multiply_factor=(1, 2))
        grouped = rsa.aggregate_matrix(dissim, ['a'], ['b'])

        self.assertEqual((3, 2), grouped.size)
        self.assertEqual(2, grouped.n_timepoints)
        self.assertEqual(3, len(grouped.md0))
        self.assertEqual(2, len(grouped.md1))

        ind_x1 = np.where(grouped.md0['a'] == 1)[0][0]
        ind_y1 = np.where(grouped.md1['b'] == 1)[0][0]
        self.assertEqual(27, grouped.data[0, ind_x1, ind_y1])
        self.assertEqual(54, grouped.data[1, ind_x1, ind_y1])

        ind_x1 = np.where(grouped.md0['a'] == 1)[0][0]
        ind_y1 = np.where(grouped.md1['b'] == 2)[0][0]
        self.assertEqual(30, grouped.data[0, ind_x1, ind_y1])
        self.assertEqual(60, grouped.data[1, ind_x1, ind_y1])


if __name__ == '__main__':
    unittest.main()
