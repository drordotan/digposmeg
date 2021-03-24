
import numpy as np
import scipy.signal
import scipy.stats
import os
import math


#----------------------------------------------------------------------
def outliers(values, direction='high'):

    assert direction in ('high', 'low', 'both')

    p25, p75 = np.percentile(values, [25, 75])

    d = 1.5 * (p75 - p25)
    threshold_high = p75 + d
    threshold_low = p25 - d

    if direction == 'high':
        return np.array([v > threshold_high for v in values])

    elif direction == 'low':
        return np.array([v < threshold_low for v in values])

    else:
        return np.array([v < threshold_low or v > threshold_high for v in values])


#----------------------------------------------------------------------
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


#----------------------------------------------------------------------
def _gaussian(gap, max_sd=3):
    x = np.array(range(math.floor(max_sd/gap)+1))*gap
    weights = scipy.stats.norm.pdf(x, 0, 1)
#    print('weights={}'.format({k:v for k,v in zip(x, weights)}))
    return weights


#----------------------------------------------------------------------
def smooth_gaussian(data, stdev, max_sd=3, end_value_policy='extrapolate'):
    weights = _gaussian(1/stdev, max_sd)
    return smooth_by_weights(data, weights, end_value_policy)


#----------------------------------------------------------------------
def smooth_by_weights(data, weights, end_value_policy='extrapolate'):

    if end_value_policy == 'simple':
        #-- Simple smoothing, no restrictions
        def do_shift(data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright):
            return data_shifted_left[1:], data_shifted_right[:-1], missing_data_sleft[1:], missing_data_sright[:-1]

    elif end_value_policy == 'extrapolate':

        #-- Use the end-of-vector value as extrapolation
        def do_shift(data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright):
            data_shifted_left = data_shifted_left[1:]
            data_shifted_left.append(data_shifted_left[-1])

            missing_data_sleft = missing_data_sleft[1:]
            missing_data_sleft.append(missing_data_sleft[-1])

            data_shifted_right = data_shifted_right[:-1]
            data_shifted_right.insert(0, data_shifted_right[0])

            missing_data_sright = missing_data_sright[:-1]
            missing_data_sright.insert(0, missing_data_sright[0])

            return data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright

    elif end_value_policy == 'symmetric':
        #-- Keep symmetry: When a value is missing on one side, ignore the corresponding value on the other side.
        def do_shift(data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright):
            data_shifted_left = data_shifted_left[1:]
            data_shifted_left.append(None)

            data_shifted_right = data_shifted_right[:-1]
            data_shifted_right.insert(0, None)

            missing_data_sleft = [l is None or r is None for l, r in zip(data_shifted_left, data_shifted_right)]
            missing_data_sright = missing_data_sleft

            return data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright

    else:
        raise Exception('Invalid end_value_policy ({})'.format(end_value_policy))

    data = np.array(data)

    result = np.array(data) * weights[0]

    #-- The loop iterates through all weights (except weights(1), which was already accumulated) and accumulates them one at a time.

    data_shifted_left = list(data)
    data_shifted_right = list(data)

    missing_data_sleft = [x is None for x in data_shifted_left]
    missing_data_sright = list(missing_data_sleft   )

    total_weights_per_point = [weights[0]] * len(data)

    for i in range(1, min(len(data), len(weights))):
        data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright = \
            do_shift(data_shifted_left, data_shifted_right, missing_data_sleft, missing_data_sright)

        weighted_left = np.array(data_shifted_left) * weights[i]
        weighted_left[missing_data_sleft] = 0
        result += weighted_left
        total_weights_per_point += np.logical_not(missing_data_sleft) * weights[i]

        weighted_right = np.array(data_shifted_right) * weights[i]
        weighted_right[missing_data_sright] = 0
        result += weighted_right
        total_weights_per_point += np.logical_not(missing_data_sright) * weights[i]

    result /= total_weights_per_point

    return result
