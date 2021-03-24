import os
import numpy as np
import pandas as pd


#--------------------------------------------------------------------
class DigitSimilarity(object):

    def __init__(self, identical_digits_default=None):
        self._similarity = {}
        self.identical_digits_default = identical_digits_default

    def _key(self, digits):
        return tuple(sorted(digits))

    def __getitem__(self, digits):
        assert len(digits) == 2
        key = self._key(digits)
        if key in self._similarity:
            return self._similarity[key]
        elif digits[0] == digits[1] and self.identical_digits_default is not None:
            return self.identical_digits_default
        else:
            return None

    def __setitem__(self, digits, sim):
        assert len(digits) == 2
        self._similarity[self._key(digits)] = sim

    @property
    def fullmatrix(self):
        m = np.array([[self.identical_digits_default if i == j else self[i, j] for i in range(10)] for j in range(10)])
        assert not (m == None).any(), "Some values are missing in the dissimilarity matrix"
        return m


#--------------------------------------------------------------------
def load_similarity(filename, negate=True, identical_digits_default=None):
    """
    Load the matrix of dissimilarity between digits from an excel file with 3 columns
    """

    result = DigitSimilarity(identical_digits_default=identical_digits_default)

    if not os.path.exists(filename):
        raise Exception('The similarity file ({}) does not exist'.format(filename))

    df = pd.read_csv(filename)
    nrows = df.shape[0]

    factor = -1 if negate else 1

    for i in range(nrows):
        d1 = int(df['digit1'][i])
        d2 = int(df['digit2'][i])
        assert 0 <= d1 <= 9
        assert 0 <= d2 <= 9
        result[d1, d2] = float(df['distance'][i]) * factor

    return result


