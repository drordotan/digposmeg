import unittest

from umne.transformers import SlidingWindow


class SlidingWindowTests(unittest.TestCase):

    #-------------------------------------------------------------------
    def test_multiple_timeseries(self):
        x = [[[1, 3, 5], [10, 30, 50]], [[100, 300, 500], [1, 3, 5]], [[100, 300, 500], [1, 3, 5]]]
        xform = SlidingWindow(window_size=2, step=1, debug=True)
        self.assertEqual(xform.fit_transform(x).tolist(), [[[2, 4], [20, 40]], [[200, 400], [2, 4]], [[200, 400], [2, 4]]])

    #-------------------------------------------------------------------
    def test_step2(self):
        x = [[[1, 2, 3, 4, 5, 6]]]
        xform = SlidingWindow(window_size=3, step=2, debug=True)
        self.assertEqual(xform.fit_transform(x).tolist(), [[[2, 4]]])

    #-------------------------------------------------------------------
    def test_step2_last_window_small(self):
        x = [[[1, 2, 3, 4, 5, 6]]]
        xform = SlidingWindow(window_size=3, step=2, min_window_size=2, debug=True)
        self.assertEqual(xform.fit_transform(x).tolist(), [[[2, 4, 5.5]]])

    #-------------------------------------------------------------------
    def test_not_enough_time_points(self):
        x = [[[1, 2, 3]]]
        xform = SlidingWindow(window_size=4, step=1, debug=True)
        self.assertRaises(Exception, lambda: xform.fit_transform(x).tolist())

    #-------------------------------------------------------------------
    def test_not_one_small_window(self):
        x = [[[1, 2, 3]]]
        xform = SlidingWindow(window_size=4, step=1, min_window_size=3, debug=True)
        self.assertEqual(xform.fit_transform(x).tolist(), [[[2]]])

    #-------------------------------------------------------------------
    def test_dont_average(self):
        x = [[[1, 2, 3, 4, 5, 6, 7]]]
        xform = SlidingWindow(window_size=3, step=3, average=False, debug=True)
        result = [m.tolist() for m in xform.fit_transform(x)]
        self.assertEqual(result, [[[[1, 2, 3]]], [[[4, 5, 6]]]])

    #-------------------------------------------------------------------
    def test_dont_average_last_smaller(self):
        x = [[[1, 2, 3, 4, 5]]]
        xform = SlidingWindow(window_size=2, step=2, min_window_size=1, average=False, debug=True)
        result = [m.tolist() for m in xform.fit_transform(x)]
        self.assertEqual(result, [[[[1, 2]]], [[[3, 4]]], [[[5]]]])


if __name__ == '__main__':
    unittest.main()
