from umne import stats, util
from pmne import decplt
import numpy as np
import glob
import pickle
from jr.plot import pretty_gat, pretty_decod
import matplotlib.pyplot as plt

fig_path = '/Users/fosca/Downloads/scores/figures_updated/'
base_path = '/Users/fosca/Downloads/scores/'

conditions = [['/decade/scores_*_standard_-400_600_rsvp_score_as_decade.pkl'],['/decade/scores_*_standard_-400_1000_comp_score_as_decade.pkl'],
              ['/decade/scores_*_standard_-400_600_rsvp_score_as_unit.pkl'],['/decade/scores_*_standard_-400_1000_comp_score_as_unit.pkl'],
              ['/hemifield/scores_*_standard_-400_600_rsvp.pkl'],
              ['/hemifield/scores_*_standard_-400_1000_comp.pkl'],
              ['/location/scores_*_standard_-400_600_rsvp_left_hemifield.pkl',
               '/location/scores_*_standard_-400_600_rsvp_right_hemifield.pkl'],
              ['/location/scores_*_standard_-400_1000_comp_left_hemifield.pkl',
              '/location/scores_*_standard_-400_1000_comp_right_hemifield.pkl'],
              ['/retinotopic/scores_*_standard_-400_600_rsvp_position_0.pkl',
              '/retinotopic/scores_*_standard_-400_600_rsvp_position_1.pkl',
              '/retinotopic/scores_*_standard_-400_600_rsvp_position_2.pkl',
              '/retinotopic/scores_*_standard_-400_600_rsvp_position_3.pkl',
              '/retinotopic/scores_*_standard_-400_600_rsvp_position_4.pkl',
              '/retinotopic/scores_*_standard_-400_600_rsvp_position_5.pkl'],
              ['/retinotopic/scores_*_standard_-400_1000_comp_position_0.pkl',
              '/retinotopic/scores_*_standard_-400_1000_comp_position_1.pkl',
              '/retinotopic/scores_*_standard_-400_1000_comp_position_2.pkl',
              '/retinotopic/scores_*_standard_-400_1000_comp_position_3.pkl',
              '/retinotopic/scores_*_standard_-400_1000_comp_position_4.pkl',
              '/retinotopic/scores_*_standard_-400_1000_comp_position_5.pkl'],
              ['/unit/scores_*_standard_-400_600_rsvp_score_as_decade.pkl'],
              ['/unit/scores_*_standard_-400_600_rsvp_score_as_unit.pkl'],
              ['/unit/scores_*_standard_-400_1000_comp_score_as_decade.pkl'],
              ['/unit/scores_*_standard_-400_1000_comp_score_as_unit.pkl'],
              ['/whole_number/scores_*_standard_-400_600_rsvp.pkl'],
              ['/whole_number/scores_*_standard_-400_1000_comp.pkl']]


names = ['decade_rsvp_score_decade.png','decade_comp_score_decade.png','decade_rsvp_score_unit.png','decade_comp_score_unit.png',
         'hemifield_rsvp.png','hemifield_comp.png','location_within_left_right_hemi_rsvp.png','location_within_left_right_hemi_comp.png',
         'retinotopic_rsvp_position.png','retinotopic_comp_position.png','unit_rsvp_score_decade.png','unit_rsvp_score_unit.png','unit_comp_score_decade.png','unit_comp_score_unit.png',
         'whole_number_rsvp.png','whole_number_comp.png']

chance = [0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.2,0.2,0.25,0.25,0.25,0.25,0,0]



def load_scores(file_names_list):

    scores_to_avg = []
    for score_name in file_names_list:
        fid = open(score_name,"rb")
        d = pickle.load(fid)
        scores_to_avg.append(d['scores'])

    return np.asarray(scores_to_avg),d['times']






# ======== try to integrate with Dror's function ========


for k in range(len(conditions)):
    print(k)
    # determine if there are only one filenames in conditions[k]:
    chance_level = chance[k]
    scores_all_conditions = []
    for ll in range(len(conditions[k])):
        file_names = glob.glob(base_path + conditions[k][ll])
        scores_ll, times = load_scores(file_names)
        scores_all_conditions.append(scores_ll)

    scores = np.asarray(scores_all_conditions)
    scores = np.mean(scores,axis=0)

    if 'rsvp' in conditions[k][0]:
        tmin_stats = 0
        tmax_stats = 0.5
    elif 'comp' in conditions[k][0]:
        tmin_stats = 0
        tmax_stats = 0.7
    else:
        print('There is something wrong with the filename')
    if 'whole_number' in conditions[k][0]:
        chance_level = util.TimeRange(max_time = 0)

    decplt.plot_scores(scores.diagonal(axis1=1, axis2=2), times, chance_level=chance_level, tmin_stats=0, tmax_stats=0.5)
    fig1 = plt.gcf()
    fig1.savefig(fig_path+'diagonal_'+names[k])

    decplt.plot_gat_scores(scores, times, chance_level=chance_level, tmin_stats=0, tmax_stats=0.5)
    fig2 = plt.gcf()
    fig2.savefig(fig_path + names[k])
    plt.close('all')



# ================= subplots ===============











for k in range(len(conditions)):
    print(k)
    # determine if there are only one filenames in conditions[k]:

    scores_all_conditions = []
    for ll in range(len(conditions[k])):
        file_names = glob.glob(base_path + conditions[k][ll])
        scores_ll, time_list = load_scores(file_names)
        scores_all_conditions.append(scores_ll)

    scores_to_avg = np.asarray(scores_all_conditions)
    scores_to_avg = np.mean(scores_to_avg,axis=0)

    if 'whole_number' in conditions[k][0]:
        #  baseline the scores
        filter_baseline = time_list < 0
        inds_baseline = np.where(filter_baseline)[0]
        baseline_scores = scores_to_avg[:, inds_baseline, :]
        baseline_scores = baseline_scores[:, :, inds_baseline]
        baseline_scores = np.mean(baseline_scores, axis=-1)
        baseline_scores = np.mean(baseline_scores, axis=-1)
        for sub_num in range(scores_to_avg.shape[0]):
            scores_to_avg[sub_num,:,:] = scores_to_avg[sub_num,:,:]-baseline_scores[sub_num]

    # ==== permutation test from 0 to 500 ms for RSVP and from 0 to 700ms for Comparison ======
    if 'rsvp' in conditions[k][0]:
        filter_times = np.logical_and(time_list>0,time_list<0.5)
    elif 'comp' in conditions[k][0]:
        filter_times = np.logical_and(time_list>0,time_list<0.7)
    else:
        print('There is something wrong with the filename')

    inds = np.where(filter_times)
    scores_to_avg_filter = scores_to_avg[:,inds[0],:]
    scores_to_avg_filter = scores_to_avg_filter[:,:,inds[0]]

    signif = np.zeros((scores_to_avg.shape[1],scores_to_avg.shape[2]))
    p_vals = stats.stats_cluster_based_permutation_test(scores_to_avg_filter-chance[k])
    sig = (p_vals<0.05)

    for l, ll in enumerate(inds[0]):
        for m, mm in enumerate(inds[0]):
            signif[ll,mm]=sig[l,m]

    pretty_gat(np.mean(scores_to_avg,axis=0), times=time_list, chance=chance[k], sfreq=100,sig=signif)
    fig = plt.gcf()
    fig.savefig(fig_path+names[k])
    plt.close('all')

# ============== cluster based permutation test for the diagonal =================


for k in range(len(conditions)):
    print(k)

    scores_all_conditions = []
    for ll in range(len(conditions[k])):
        file_names = glob.glob(base_path + conditions[k][ll])
        scores_ll, time_list = load_scores(file_names)
        scores_all_conditions.append(scores_ll)

    scores_to_avg = np.asarray(scores_all_conditions)
    scores_to_avg = np.mean(scores_to_avg,axis=0)

    if 'whole_number' in conditions[k][0]:
        #  baseline the scores
        filter_baseline = time_list < 0
        inds_baseline = np.where(filter_baseline)[0]
        baseline_scores = scores_to_avg[:, inds_baseline, :]
        baseline_scores = baseline_scores[:, :, inds_baseline]
        baseline_scores = np.mean(baseline_scores, axis=-1)
        baseline_scores = np.mean(baseline_scores, axis=-1)
        for sub_num in range(scores_to_avg.shape[0]):
            scores_to_avg[sub_num,:,:] = scores_to_avg[sub_num,:,:]-baseline_scores[sub_num]

    if 'rsvp' in conditions[k][0]:
        filter_times = np.logical_and(time_list>0,time_list<0.5)
    elif 'comp' in conditions[k][0]:
        filter_times = np.logical_and(time_list>0,time_list<0.7)
    else:
        print('There is something wrong with the filename')


    inds = np.where(filter_times)
    scores_to_avg_filter = scores_to_avg[:,inds[0],inds[0]]

    signif = np.zeros((scores_to_avg.shape[1],1))
    p_vals = stats.stats_cluster_based_permutation_test(scores_to_avg_filter-chance[k])
    sig = (p_vals<0.05)

    for l, ll in enumerate(inds[0]):
            signif[ll]=sig[l]

    pretty_decod(np.mean(scores_to_avg,axis=0), times=time_list, chance=chance[k], sfreq=100,sig=signif,fill=True)
    fig = plt.gcf()
    fig.savefig(fig_path+'diagonal_'+names[k])
    plt.close('all')


# ============= fancy plot ============

# COMP
# hemifield, location-within-hemifield, retinotopic, decade symbolic, unit symbolic, quantity

conditions = [['/hemifield/scores_*_standard_-400_1000_comp.pkl'],
              ['/location/scores_*_standard_-400_1000_comp_left_hemifield.pkl',
              '/location/scores_*_standard_-400_1000_comp_right_hemifield.pkl'],
              ['/decade/scores_*_standard_-400_1000_comp_score_as_decade.pkl'],
              ['/unit/scores_*_standard_-400_1000_comp_score_as_unit.pkl'],
              ['/whole_number/scores_*_standard_-400_1000_comp.pkl']
              ]

chance = [0.5,0.5,0.25,0.25,0]

fig,ax = plt.subplots(5,1,figsize=(15,15))


for k in range(len(conditions)):
    print(k)
    # determine if there are only one filenames in conditions[k]:
    chance_level = chance[k]
    scores_all_conditions = []
    for ll in range(len(conditions[k])):
        file_names = glob.glob(base_path + conditions[k][ll])
        scores_ll, times = load_scores(file_names)
        scores_all_conditions.append(scores_ll)

    scores = np.asarray(scores_all_conditions)
    scores = np.mean(scores,axis=0)

    if 'rsvp' in conditions[k][0]:
        tmin_stats = 0
        tmax_stats = 0.5
    elif 'comp' in conditions[k][0]:
        tmin_stats = 0
        tmax_stats = 0.7
    else:
        print('There is something wrong with the filename')
    if 'whole_number' in conditions[k][0]:
        chance_level = util.TimeRange(max_time = 0)

    decplt.plot_scores(scores.diagonal(axis1=1, axis2=2), times, chance_level=chance_level, tmin_stats=0, tmax_stats=0.5, ax=ax[k])


fig = plt.gcf()
fig.savefig(fig_path+'_all_together_comp.png')







conditions = [['/hemifield/scores_*_standard_-400_600_rsvp.pkl'],
              ['/location/scores_*_standard_-400_600_rsvp_left_hemifield.pkl',
              '/location/scores_*_standard_-400_600_rsvp_right_hemifield.pkl'],
              ['/decade/scores_*_standard_-400_600_rsvp_score_as_decade.pkl'],
              ['/unit/scores_*_standard_-400_600_rsvp_score_as_unit.pkl'],
              ['/whole_number/scores_*_standard_-400_600_rsvp.pkl']
              ]

chance = [0.5,0.5,0.25,0.25,0]

fig,ax = plt.subplots(5,1,figsize=(15,15))


for k in range(len(conditions)):
    print(k)
    # determine if there are only one filenames in conditions[k]:
    chance_level = chance[k]
    scores_all_conditions = []
    for ll in range(len(conditions[k])):
        file_names = glob.glob(base_path + conditions[k][ll])
        scores_ll, times = load_scores(file_names)
        scores_all_conditions.append(scores_ll)

    scores = np.asarray(scores_all_conditions)
    scores = np.mean(scores,axis=0)

    if 'rsvp' in conditions[k][0]:
        tmin_stats = 0
        tmax_stats = 0.5
    elif 'comp' in conditions[k][0]:
        tmin_stats = 0
        tmax_stats = 0.7
    else:
        print('There is something wrong with the filename')
    if 'whole_number' in conditions[k][0]:
        chance_level = util.TimeRange(max_time = 0)

    decplt.plot_scores(scores.diagonal(axis1=1, axis2=2), times, chance_level=chance_level, tmin_stats=0, tmax_stats=0.5, ax=ax[k])


fig = plt.gcf()
fig.savefig(fig_path+'_all_together_rsvp.png')

