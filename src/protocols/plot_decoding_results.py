import sys
import dpm
import umne
import dpm.plots
import mne
import pickle

sys.path.append('/git/digposmeg/analyze')
sys.path.append('/git/jr-tools')
sys.path.append('/git/pyRiemann')


#------------------------------------------------------------------------------

mne.set_log_level('info')
env_per_subj=False
subj_ids = tuple(['ag', 'at', 'bl', 'bo', 'cb', 'cc', 'eb', 'en', 'hr', 'jm0', 'lj', 'mn', 'mp'])

decoding_sh_dir = dpm.consts.results_dir + 'decoding-sh/'


# #------------------------------------------------------------------------------
def plot_gat(directory, file_suffix, chance_level,min_time=-0.1):
    import matplotlib.pyplot as plt
    plt.close('all')
    pmne.decplt.load_and_plot_both(dpm.consts.results_dir+directory+'/scores_*'+file_suffix+'*.pkl', dpm.consts.figures_path,
                                   chance_level=chance_level, min_time=min_time)


#
# # ================= now the plots ====================================
# # # Decode 2-digit quantity (regression) LOCKED ON THE RESPONSE

directory = '/decoding-sh/whole_number/RT-locked_r2_score/'
file_suffix = 'comp'
chance_level = umne.util.TimeRange(max_time =0)
min_time = -1.5
max_time = 0

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+directory+'/scores_*'+file_suffix+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=0)

pmne.decplt.load_and_plot(dpm.consts.results_dir+directory+'/scores_*'+file_suffix+'*.pkl',
                          chance_level=chance_level)


path = '/Volumes/COUCOU_CFC/digitpos/exp/results/decoding-sh/decade_and_unit/RT-locked_r2_score/decade/decoder_ag_standard_reg_r2_-1500_0_comp_.pkl'
with open(path,'rb') as fid:
    results = pickle.load(fid)


#
#
# #------------------------------------------------------------------------------
#
# directory = 'decoding-sh/whole_number'
# file_suffix = 'comp'
# chance_level=pmne.util.TimeRange(max_time = 0)
# min_time = -0.1
# max_time = 0.6
# tmin_stats = 0.1
# tmax_stats=0.6
#
# pmne.decplt.load_and_plot(dpm.consts.results_dir + directory + '/scores_*' + file_suffix + '*.pkl',
#                           chance_level=chance_level, min_time=min_time,max_time=max_time,tmin_stats = 0.1,tmax_stats=0.6)
#
#
#
# #------------------------------------------------------------------------------
# # question: how does it know that it is a regression ?
directory = 'decoding-sh/decade/RT-locked/'
# file_suffix = 'comp_score_as_decade'
file_suffix = 'comp_score_as_unit'
#
chance_level = umne.util.TimeRange(min_time = -1.5,max_time = 0)
#
pmne.decplt.load_and_plot(dpm.consts.results_dir+directory+'/scores_*'+file_suffix+'*.pkl',
                          chance_level=chance_level)
pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+directory+'/scores_*'+file_suffix+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_level)
#------------------------------------------------------------------------------

