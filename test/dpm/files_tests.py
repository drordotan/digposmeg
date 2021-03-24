import unittest

import dpm.files


def _get_dummy_data(catch_trial_inds):
    data = dict(trigger=[10]*5, trialTime=[0, 1000, 2000, 3000, 4000])
    for i in catch_trial_inds:
        data['trigger'][i] = 1
    return data



class RSVPTests(unittest.TestCase):

    #------------------------------------------------------------------------------
    def test_response_detected(self):

        def _get_response_times(x):
            return [2500]

        data = _get_dummy_data([2, 4])
        dpm.files._add_rsvp_responses('stam.csv', data, _get_response_times)
        self.assertEqual([None, None, True, None, False], list(data['correct']))

        rts = list(data['rt'])
        self.assertEqual([None, None, 500, None, None], rts)


    #------------------------------------------------------------------------------
    def test_response_too_early(self):

        def _get_response_times(x):
            return [2020]

        data = _get_dummy_data([2, 4])
        dpm.files._add_rsvp_responses('stam.csv', data, _get_response_times)
        self.assertEqual([None, None, False, None, False], list(data['correct']))
        self.assertEqual([None, None, None, None, None], list(data['rt']))


    #------------------------------------------------------------------------------
    def test_response_too_late(self):

        def _get_response_times(x):
            return [4010]

        data = _get_dummy_data([2, 4])
        dpm.files._add_rsvp_responses('stam.csv', data, _get_response_times)
        self.assertEqual([None, None, False, None, False], list(data['correct']))
        self.assertEqual([None, None, None, None, None], list(data['rt']))


    #------------------------------------------------------------------------------
    def test_response_after_next_catch_trial(self):

        def _get_response_times(x):
            return [3100]

        data = _get_dummy_data([2, 3])
        dpm.files._add_rsvp_responses('stam.csv', data, _get_response_times)
        self.assertEqual([None, None, False, True, None], list(data['correct']))


    #------------------------------------------------------------------------------
    def test_response_detected_in_last_trial(self):

        def _get_response_times(x):
            return [4500]

        data = _get_dummy_data([4])
        dpm.files._add_rsvp_responses('stam.csv', data, _get_response_times)
        self.assertEqual([None, None, None, None, True], list(data['correct']))

        rts = list(data['rt'])
        self.assertEqual([None, None, None, None, 500], rts)




if __name__ == '__main__':
    unittest.main()
