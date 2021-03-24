import unittest

from dpm.rsa_old import dissimilarity as d

def stim(target, location):
    return dict(target=target, location=location)


class RSADissimilarityTests(unittest.TestCase):

    #------------------------- location ---------------------------------------------

    def test_location_same_loc_2digit(self):
        self.assertEqual(d.location(stim(10, 0), stim(22, 0)), -2)

    def test_location_same_loc_1digit(self):
        self.assertEqual(d.location([5, 0], [8, 0]), -1)

    def test_location_same_loc_1_2_digits(self):
        self.assertEqual(d.location([5, 4], [23, 4]), -1)
        self.assertEqual(d.location([23, 4], [5, 4]), -1)

    def test_location_same_loc_reversed_1_2_digits(self):
        self.assertEqual(d.location([5, 4], [23, 3]), -1)
        self.assertEqual(d.location([23, 3], [5, 4]), -1)

    def test_location_overlap_2digit(self):
        self.assertEqual(d.location([10, 0], [22, 1]), -1)
        self.assertEqual(d.location([10, 1], [22, 0]), -1)

    def test_location_no_overlap_2digit(self):
        self.assertEqual(d.location([10, 0], [22, 2]), 0)
        self.assertEqual(d.location([10, 2], [22, 0]), 0)

    def test_location_no_overlap_1digit(self):
        self.assertEqual(d.location([9, 0], [7, 1]), 0)
        self.assertEqual(d.location([9, 0], [7, 2]), 0)
        self.assertEqual(d.location([5, 1], [8, 0]), 0)
        self.assertEqual(d.location([5, 2], [8, 0]), 0)


    #------------------------- n_same_per_location ---------------------------------------------

    def test_nsl_1digit_all_same(self):
        self.assertEqual(d.retinotopic_id([5, 0], [5, 0]), -1)

    def test_nsl_1digit_different_locations(self):
        self.assertEqual(d.retinotopic_id([5, 0], [5, 1]), 0)
        self.assertEqual(d.retinotopic_id([5, 0], [5, 2]), 0)
        self.assertEqual(d.retinotopic_id([5, 1], [5, 0]), 0)

    def test_nsl_1digit_same_loc_different_digit(self):
        self.assertEqual(d.retinotopic_id([5, 0], [6, 0]), 0)

    def test_nsl_2digit_all_same(self):
        self.assertEqual(d.retinotopic_id([24, 0], [24, 0]), -2)

    def test_nsl_2digit_same_number_different_loc(self):
        self.assertEqual(d.retinotopic_id([24, 0], [24, 1]), 0)
        self.assertEqual(d.retinotopic_id([24, 0], [24, 2]), 0)
        self.assertEqual(d.retinotopic_id([24, 1], [24, 0]), 0)
        self.assertEqual(d.retinotopic_id([24, 2], [24, 0]), 0)

    def test_nsl_2digit_same_loc_different_number(self):
        self.assertEqual(d.retinotopic_id([24, 0], [56, 0]), 0)
        self.assertEqual(d.retinotopic_id([24, 0], [26, 0]), -1)
        self.assertEqual(d.retinotopic_id([24, 0], [64, 0]), -1)

    def test_nsl_2digit_overlap_location(self):
        self.assertEqual(d.retinotopic_id([24, 0], [42, 1]), -1)
        self.assertEqual(d.retinotopic_id([24, 1], [42, 0]), -1)
        self.assertEqual(d.retinotopic_id([24, 0], [43, 1]), -1)
        self.assertEqual(d.retinotopic_id([43, 1], [24, 0]), -1)


    #------------------------- decade_identity ---------------------------------------------

    def test_decadeid_2digit_same(self):
        self.assertEqual(d.decade_id([26, 0], [26, 0]), 0)
        self.assertEqual(d.decade_id([26, 0], [28, 0]), 0)  # Different unit digit - irrelevant
        self.assertEqual(d.decade_id([26, 0], [28, 4]), 0)  # Different location - irrelevant
        self.assertEqual(d.decade_id([26, 3], [28, 0]), 0)  # Different location - irrelevant

    def test_decadeid_2digit_different_decade(self):
        self.assertEqual(d.decade_id([26, 0], [36, 0]), 1)

    def test_decadeid_2digit_different_decade_and_unit(self):
        self.assertEqual(d.decade_id([26, 0], [38, 0]), 1)

    def test_decadeid_2digit_different_decade_same_loc(self):
        self.assertEqual(d.decade_id([26, 1], [32, 0]), 1)

    #------------------------- unit_identity ---------------------------------------------

    def test_unitid_2digit_same(self):
        self.assertEqual(d.unit_id([26, 0], [26, 0]), 0)
        self.assertEqual(d.unit_id([26, 0], [76, 0]), 0)  # Different unit digit - irrelevant
        self.assertEqual(d.unit_id([26, 0], [76, 4]), 0)  # Different location - irrelevant
        self.assertEqual(d.unit_id([26, 3], [76, 0]), 0)  # Different location - irrelevant

    def test_unitid_2digit_different_unit(self):
        self.assertEqual(d.unit_id([26, 0], [28, 0]), 1)

    def test_unitid_2digit_different_decade_and_unit(self):
        self.assertEqual(d.unit_id([26, 0], [38, 0]), 1)

    def test_unitid_2digit_different_unit_same_loc(self):
        self.assertEqual(d.unit_id([23, 1], [22, 0]), 1)





if __name__ == '__main__':
    unittest.main()
