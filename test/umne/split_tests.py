import unittest

import numpy as np

from umne.split import _split_ids_into_groups


class SplitTests(unittest.TestCase):

    def _run_split(self, group_per_epoch, group1_prop, round_toward):
        group_per_epoch = np.array(group_per_epoch)
        ids = _split_ids_into_groups(group_per_epoch, set1_proportion=group1_prop, round_toward=round_toward)
        ids = np.array(ids)

        n_group1_half1 = sum(np.logical_and(group_per_epoch == 1, ids == 1))
        n_group2_half1 = sum(np.logical_and(group_per_epoch == 2, ids == 1))

        return n_group1_half1, n_group2_half1


    def test_split_ids_into_groups__set1_half__round_to_1(self):
        n_group1_half1, n_group2_half1 = self._run_split([1, 1, 1, 1, 1, 2, 2, 2, 2], 0.5, 1)
        self.assertEqual(n_group1_half1, 3)
        self.assertEqual(n_group2_half1, 2)

    def test_split_ids_into_groups__set1_half__round_to_2(self):
        n_group1_half1, n_group2_half1 = self._run_split([1, 1, 1, 1, 1, 2, 2, 2, 2], 0.5, 2)
        self.assertEqual(n_group1_half1, 2)
        self.assertEqual(n_group2_half1, 2)

    def test_split_ids_into_groups__set1_quarter__round_to_1(self):
        n_group1_half1, n_group2_half1 = self._run_split([1, 1, 1, 1, 1, 2, 2, 2, 2], 0.25, 1)
        self.assertEqual(n_group1_half1, 2)
        self.assertEqual(n_group2_half1, 1)

    def test_split_ids_into_groups__set1_quarter__round_to_2(self):
        n_group1_half1, n_group2_half1 = self._run_split([1, 1, 1, 1, 1, 2, 2, 2, 2], 0.25, 2)
        self.assertEqual(n_group1_half1, 1)
        self.assertEqual(n_group2_half1, 1)

    def test_split_ids_into_groups__set1_half__round_random(self):
        n_group1_half1 = 0
        n_group1_half2 = 0
        n_attempts = 10000
        for i in range(n_attempts):
            n1, n2 = self._run_split([1, 1, 1, 1, 1, 2, 2, 2, 2], 0.5, 0)
            n_group1_half1 += n1
            n_group1_half2 += n2

        self.assertAlmostEqual(n_group1_half1 / n_attempts, 2.5, 1)
        self.assertEqual(n_group1_half2 / n_attempts, 2)

    def test_split_ids_into_groups__set1_quarter__round_random(self):
        n_group1_half1 = 0
        n_group1_half2 = 0
        n_attempts = 10000
        for i in range(n_attempts):
            n1, n2 = self._run_split([1, 1, 1, 1, 1, 2, 2, 2, 2], 0.25, 0)
            n_group1_half1 += n1
            n_group1_half2 += n2

        self.assertAlmostEqual(n_group1_half1 / n_attempts, 5/4, 1)
        self.assertEqual(n_group1_half2 / n_attempts, 1)


if __name__ == '__main__':
    unittest.main()
