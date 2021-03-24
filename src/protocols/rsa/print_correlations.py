
import dpm
import itertools
import scipy.stats
from operator import itemgetter

dis = dpm.rsa.dissimilarity


#--------------------------------------------------------------
def generate_all_2digit_stim():

    result = []

    for d in (2, 3, 5, 8):
        for u in (2, 3, 5, 8):
            for loc in range(5):
                result.append(dict(decade=d, unit=u, location=loc, target=10*d+u))

    return result



#-------------------------------------------------------------
def print_correlations(predictors, stimuli):

    stim_pairs = [pair for pair in itertools.combinations(stimuli, 2)]

    info = []

    for pred1, pred2 in itertools.combinations(predictors, 2):
        dissim1 = [pred1(a, b) for a, b in stim_pairs]
        dissim2 = [pred2(a, b) for a, b in stim_pairs]
        name1 = pred1.__name__ if type(pred1) == type(lambda x:0) else type(pred1).__name__
        name2 = pred2.__name__ if type(pred2) == type(lambda x:0) else type(pred2).__name__
        # r = np.corrcoef(dissim1, dissim2)[0]
        r, p = scipy.stats.pearsonr(dissim1, dissim2)
        info.append((abs(r), '{} vs {}: r={:.3f}, p={:.3f}'.format(name1, name2, r, p)))

    info.sort(key=itemgetter(0))
    for r, i in info:
        print(i)


#-------------------------------------------------------------
print_correlations([dis.location,
                    dis.cmp_result(),
                    dis.retinotopic_id,
                    dis.decade_id,
                    dis.decade_group,
                    dis.unit_id,
                    dis.numerical_distance,
                    dis.distance_to_ref()],
                   generate_all_2digit_stim())
