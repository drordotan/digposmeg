import unittest

import numpy as np

from umne.transformers import AveragePerEvent


class TransformerAveragePerEventTests(unittest.TestCase):

    #-----------------------------------------------------------
    def test_average(self):

        #-- A 4 x 2 matrix
        data = [
            [0, 10],
            [1, 11],
            [20, 110],
            [21, 111]]

        y = [1, 2, 1, 2]  # labels

        expected = [
            [10, 60],
            [11, 61]
        ]

        xf = AveragePerEvent()
        result = xf.fit_transform(data, y)

        self.assertEqual(result.tolist(), expected)


    #-----------------------------------------------------------
    def test_average_with_event_ids(self):

        #-- A 4 x 2 matrix
        data = [
            [0, 10],
            [1, 11],
            [20, 110],
            [21, 111]]

        y = [1, 2, 1, 2]  # labels

        expected = [
            [10, 60],
        ]

        xf = AveragePerEvent(event_ids=[1])
        result = xf.fit_transform(data, y)

        self.assertEqual(result.tolist(), expected)


    #-----------------------------------------------------------
    def test_average_with_event_ids_2_results_per_event(self):

        #-- A 4 x 2 matrix
        data = [
            [0, 10],
            [20, 110],
            [1, 11],
            [21, 111]]

        y = [1, 1, 2, 2]  # labels

        xf = AveragePerEvent(event_ids=[1], n_results_per_event=2)
        result = xf.fit_transform(data, y).tolist()

        if result[0] == data[0]:
            self.assertEqual(result, data[:2])
        else:
            self.assertEqual(result, [data[1], data[0]])


    #========================= test the _aggregate() function ============================

    #-----------------------------------------------------------
    def test_aggregate_1result_1epoch(self):
        xf = AveragePerEvent(event_ids=[1], n_results_per_event=1)
        self.assertEqual(xf._aggregate([1]), [[1]])

    #-----------------------------------------------------------
    def test_aggregate_1result_n_epochs(self):
        xf = AveragePerEvent(event_ids=[1], n_results_per_event=1)
        self.assertEqual(xf._aggregate([1, 2, 3]), [[1, 2, 3]])

    #-----------------------------------------------------------
    def test_aggregate_2result_1epoch(self):
        xf = AveragePerEvent(event_ids=[1], n_results_per_event=2, max_events_with_missing_epochs=1)
        self.assertEqual(sort_1int_lists(xf._aggregate([1])), [[1], [1]])

    #-----------------------------------------------------------
    def test_aggregate_2result_2epochs(self):
        xf = AveragePerEvent(event_ids=[1], n_results_per_event=2)
        self.assertEqual(sort_1int_lists(xf._aggregate([1, 2])), [[1], [2]])

    #-----------------------------------------------------------
    def test_aggregate_2result_3epochs(self):
        xf = AveragePerEvent(event_ids=[1], n_results_per_event=2)
        result = xf._aggregate([1, 2, 3])
        self.assertEqual(len(result), 2, 'Invalid result: '.format(result))
        self.assertEqual({len(result[0]), len(result[1])}, {1, 2}, 'Invalid result: '.format(result))
        self.assertEqual(sorted([x for a in result for x in a]), [1, 2, 3], 'Invalid result: '.format(result))

    #-----------------------------------------------------------
    def test_aggregate_3result_2epochs(self):
        xf = AveragePerEvent(event_ids=[1], n_results_per_event=3)
        result = xf._aggregate([1, 2])
        self.assertEqual(len(result), 3, 'Invalid result: '.format(result))
        self.assertEqual([len(result[0]), len(result[1]), len(result[1])], [1, 1, 1], 'Invalid result: '.format(result))
        self.assertEqual(set([x for a in result for x in a]), {1, 2}, 'Invalid result: '.format(result))


def sort_1int_lists(lists):
    return sorted(lists, cmp = lambda a, b: a[0] - b[0])


if __name__ == '__main__':
    unittest.main()
