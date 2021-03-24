"""

#------------------------------------------------------
# -------- Regress the dissimilarity matrix -----------
#------------------------------------------------------

"""

import mne
mne.set_log_level('info')

import umne
import dpm
from umne.rsa import load_and_regress_dissimilarity

dis = dpm.rsa.dissimilarity


metric = 'spearman'
#metric = 'euclidean'
#metric = 'mahalanobis'

fig_dir = '/Users/dror/data/acad-proj/2-InProgress/DigitPositionMEG/figures/rsa/{}/'.format(metric)
exp_dir = '/Users/dror/meg/digit-position/exp/'

noloc_colors = umne.rsa.default_colors[1:]

#-- Whether the dissimilarity matrices were created with the split-half method
split_half = True


print('>>>>>>>>> Metric={}, split-half={}'.format(metric, split_half))


#---------------------------------------------------
# ------- Predictor codes in filenames:    ---------
# (L)ocation
# (C)omparison results (i.e. larger than or smaller than 44)
# (R)etinotopic: same digit in same location
# (D)ecade
# (U)nit
# (Q)uantity (2-digit)
# (T): Reaction time
# (4): Distance to 44 (the reference number)
#---------------------------------------------------

included_cells_getter = dpm.rsa.IncludedCellsDef(exclude_digits=[4], ndigits=2, only_top_half=not split_half)
included_cells_getter_adjacent = dpm.rsa.IncludedCellsDef(exclude_digits=[4], ndigits=2, only_top_half=not split_half, only_adjacent_locations=True)

comparison_dissim_files = exp_dir+'rsa-data/dissim/full-comp/*{}*stimulus*.dmat'.format(metric)
comparison_dissim_files_resp = exp_dir+'rsa-data/dissim/full-comp/*{}*response*.dmat'.format(metric)
rsvp_dissim_files = exp_dir+'rsa-data/dissim/full-RSVP/*{}*.dmat'.format(metric)
inter_task_dissim_files = exp_dir+'rsa-data/dissim/inter-task/*{}*.dmat'.format(metric)
visual_dissim_fn = '/Users/dror/data/acad-proj/2-InProgress/DigitPositionMEG/data/visual-similarity.csv'


comp2d_bhvr = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids_clean, dpm.consts.comparison_raw_files, target_filter='2digit')


#----------------- Comparison task: retinotopic ----------------------

rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files,
                                                 [dis.retinotopic_visual_similarity(visual_dissim_fn)],
                                                 included_cells_getter=included_cells_getter_adjacent)

umne.rsa.plot_regression_results(rrc[:, :, :-1], times,
                                 legend=['Visual dissimilarity'],
                                 save_as=fig_dir + 'comparison_LRvs.pdf', show_significance=True)


#----------------- Comparison task ----------------------

#--------------- 4 predictors (loc, retin, decade, unit)

rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files,
                                                 [dis.location, dis.retinotopic_id, dis.decade_id, dis.unit_id, dis.distance_to_ref()],
                                                 included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrc[:, :, :-1], times,
                                 legend=('Location', 'Retinotopic', 'Decade identical', 'Unit identical', 'Distance to 44'),
                                 save_as=fig_dir + 'comparison_LRDU4.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrc[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Retinotopic', 'Decade identical', 'Unit identical', 'Distance to 44'),
                                 save_as=fig_dir+'comparison_LRDU4_nice.pdf', show_significance=True)

#--------------- 6 predictors (4 basic + location, quantity)

rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files,
                                                   [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_id, dis.unit_id,
                                                    dis.numerical_distance],
                                                   included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrc[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade identical', 'Unit identical', 'Quantity'),
                                 save_as=fig_dir + 'comparison_LCRDUQ.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrc[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade identical', 'Unit identical', 'Quantity'),
                                 save_as=fig_dir+'comparison_LCRDUQ_nice.pdf', show_significance=True)

#--------------- 5 predictors (4 basic + location)

rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files,
                                                   [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_id, dis.unit_id],
                                                   included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrc[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade identical', 'Unit identical'),
                                 save_as=fig_dir + 'comparison_LCRDU.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrc[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade identical', 'Unit identical'),
                                 save_as=fig_dir + 'comparison_LCRDU_nice.pdf', show_significance=True)

#---------------  location, comparison result, retin., quantity
rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files,
                                                   [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.numerical_distance],
                                                   included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrc[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Quantity'),
                                 save_as=fig_dir + 'comparison_LCRQ.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrc[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Quantity'),
                                 save_as=fig_dir+'comparison_LCRQ_nice.pdf', show_significance=True)

#---------------
rrcdist, sid, times = load_and_regress_dissimilarity(comparison_dissim_files,
                                                      [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_distance, dis.unit_distance],
                                                      included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrcdist[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade distance', 'Unit distance'),
                                 save_as=fig_dir + 'comparison_LCRDUdist.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrcdist[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade distance', 'Unit distance'),
                                 save_as=fig_dir + 'comparison_LCRDUdist_nice.pdf', show_significance=True)


#--------------- Only for numbers with decade=4
rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files, [dis.location, dis.retinotopic_id, dis.unit_id],
                                                 included_cells_getter=dpm.rsa.IncludedCellsDef(include_digits=[4], ndigits=2, only_top_half=not split_half))

umne.rsa.plot_regression_results(rrc[:, :, :-1], times, legend=('Location', 'Retinotopic', 'Unit identical'),
                                 save_as=fig_dir + 'comparison_d=4_LRU.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrc[:, :, 1:-1], times, colors=noloc_colors, legend=('Retinotopic', 'Unit identical'),
                                 save_as=fig_dir+'comparison_d=4_LRU_nice.pdf', show_significance=True)


#--------------- Locked on response

rrc, sid, times = load_and_regress_dissimilarity(comparison_dissim_files_resp,
                                                 [dis.location, dis.retinotopic_id, dis.decade_id, dis.unit_id, dis.distance_to_ref()],
                                                 included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrc[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Retinotopic', 'Decade identical', 'Unit identical', 'Distance to 44'),
                                 save_as=fig_dir+'comparison_resp_LRDU4_nice.pdf', show_significance=True)


#---------------------- RSVP task ---------------------------------

#---------------
rrr5, sid, times = load_and_regress_dissimilarity(rsvp_dissim_files,
                                                  [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_id, dis.unit_id,
                                                   dis.numerical_distance],
                                                  included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrr5[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade identical', 'Unit identical', 'Quantity'),
                                 save_as=fig_dir + 'rsvp_LCRDUQ.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrr5[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade identical', 'Unit identical', 'Quantity'),
                                 save_as=fig_dir + 'rsvp_LCRDUQ_nice.pdf', show_significance=True)

#---------------
rrr4, sid, times = load_and_regress_dissimilarity(rsvp_dissim_files,
                                                  [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_id, dis.unit_id],
                                                  included_cells_getter=included_cells_getter)

umne.rsa.plot_regression_results(rrr4[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade identical', 'Unit identical'),
                                 save_as=fig_dir + 'rsvp_LCRDU.pdf', show_significance=True)

umne.rsa.plot_regression_results(rrr4[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade identical', 'Unit identical'),
                                 save_as=fig_dir + 'rsvp_LCRDU_nice.pdf', show_significance=True)


#----------------------- RSVP -> Comparison task ----------------------------

#---------------
rri5, sid, times = load_and_regress_dissimilarity(inter_task_dissim_files,
                                                  [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_id, dis.unit_id,
                                                   dis.numerical_distance])

umne.rsa.plot_regression_results(rri5[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade identical', 'Unit identical', 'Quantity'),
                                 save_as=fig_dir + 'intertask_LCRDUQ.pdf', show_significance=True)

umne.rsa.plot_regression_results(rri5[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade identical', 'Unit identical', 'Quantity'),
                                 save_as=fig_dir + 'intertask_LCRDUQ_nice.pdf', show_significance=True)

#---------------
rri4, sid, times = load_and_regress_dissimilarity(inter_task_dissim_files,
                                                  [dis.location, dis.cmp_result(), dis.retinotopic_id, dis.decade_id, dis.unit_id])

umne.rsa.plot_regression_results(rri4[:, :, :-1], times,
                                 legend=('Location', 'Comparison', 'Retinotopic', 'Decade identical', 'Unit identical'),
                                 save_as=fig_dir + 'intertask_LCRDU.pdf', show_significance=True)

umne.rsa.plot_regression_results(rri4[:, :, 1:-1], times, colors=noloc_colors,
                                 legend=('Comparison', 'Retinotopic', 'Decade identical', 'Unit identical'),
                                 save_as=fig_dir + 'intertask_LRDU_nice.pdf', show_significance=True)
