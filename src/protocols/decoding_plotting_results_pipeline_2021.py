import sys

import mne

sys.path.append('/Users/dror/git/digposmeg/analyze')
sys.path.append('/Users/dror/git/jr-tools')
sys.path.append('/Users/dror/git/pyRiemann')
from dpm.util import create_folder

import dpm.plots
import pmne.decplt

#------------------- set parameters -----------------------------------
tmin_stats_stim = 0
tmax_stats_stim = 1

tmin_stats_resp = -1.5
tmax_stats_resp = 0

#---------------------------------------------------------------
#------------ Plotting visual results -----------------------
#---------------------------------------------------------------

# hemifield
chance_hemifield = 0.5

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/hemifield'+'/scores_*'+"comp"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_hemifield, tmin_stats=tmin_stats_stim, tmax_stats=tmax_stats_stim, plot_subjects=True)



# location
chance_location = 0.2

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/location'+'/scores_*'+"comp"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_location, tmin_stats=tmin_stats_stim, tmax_stats=tmax_stats_stim, plot_subjects=True)

# retinotopic
chance_location = 0.5
for i in range(6):
    pmne.decplt.load_and_plot_gat(dpm.consts.results_dir + 'decoding-sh/retinotopic' + '/scores_*' +'comp_position_%i' % i + '*.pkl',
                                  dpm.consts.figures_path,
                                  chance_level=chance_location, tmin_stats=tmin_stats_stim, tmax_stats=tmax_stats_stim,
                                  plot_subjects=True)

#---------------------------------------------------------------
#------------ Plotting the units results -----------------------
#---------------------------------------------------------------
chance_unit = 0.25

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit'+'/scores_*'+"comp_score_as_unit"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_unit, tmin_stats=tmin_stats_stim, tmax_stats=tmax_stats_stim, plot_subjects=True)

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/RT-locked'+'/scores_*'+'comp_score_as_unit'+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_unit, tmin_stats=tmin_stats_resp, tmax_stats=tmax_stats_resp)

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit'+'/scores_*'+"comp_score_as_unit_on_4X"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_unit, tmin_stats=tmin_stats_stim, tmax_stats=tmax_stats_stim)


pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/RT-locked'+'/scores_*'+"comp_score_as_unit_on_4X"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_unit, tmin_stats=tmin_stats_resp, tmax_stats=tmax_stats_resp)

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/'+'/scores_*'+"comp_score_as_unit_on_single_digits"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_unit, tmin_stats=tmin_stats_stim, tmax_stats=tmax_stats_stim)

pmne.decplt.load_and_plot_gat(dpm.consts.results_dir+'decoding-sh/unit/RT-locked'+'/scores_*'+"comp_score_as_unit_on_single_digits"+'*.pkl',
                              dpm.consts.figures_path,
                              chance_level=chance_unit, tmin_stats=tmin_stats_resp, tmax_stats=tmax_stats_resp)

