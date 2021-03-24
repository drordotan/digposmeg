import dpm

fig_dir = '/Users/dror/data/acad-proj/2-InProgress/DigitPositionMEG/figures/'


print('=======================================================================================')
print('             Comparison task ')
print('=======================================================================================')

comp = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids_all, dpm.consts.comparison_raw_files)
comp2d = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids_all, dpm.consts.comparison_raw_files, target_filter='2digit')
comp2d_good = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids_clean, dpm.consts.comparison_raw_files, target_filter='2digit')

print('Analysis of outliers:')
comp_outliers = dpm.behavior.get_outlier_subjects(comp2d, True)
assert comp_outliers == ['am', 'ga', 'jm1', 'lb']  # Just making sure we know who they are

comp = comp[[s not in comp_outliers for s in comp.subject]]
comp2d = comp2d[[s not in comp_outliers for s in comp2d.subject]]

dpm.behavior.print_average_accuracy_and_rt(comp2d)

#-- plot RT/errors per target
dpm.behavior.plot_per_target(comp, 'rt', include_errors=False, save_as=fig_dir+'comp_rt_per_target.pdf', ylim=(500, 850))
dpm.behavior.plot_per_target(comp, 'errors', save_as=fig_dir+'comp_acc_per_target.pdf', ylim=(0, 15))

#-- Plot RT/errors per target, one line per position
dpm.behavior.plot_per_target_and_pos(comp2d, 'rt', include_errors=False, save_as=fig_dir+'comp_rt_per_target_pos.pdf', ylim=(500, 850))
dpm.behavior.plot_per_target_and_pos(comp2d, 'errors', save_as=fig_dir+'comp_acc_per_target_pos.pdf', ylim=(0, 18))

#-- Plot RT/errors per position
dpm.behavior.plot_per_group(comp2d, 'rt', 'position', include_errors=False, save_as=fig_dir+'comp_rt_per_pos.pdf', plot_type='line', ylim=(500, 850))
dpm.behavior.plot_per_group(comp2d, 'errors', 'position', save_as=fig_dir+'comp_acc_per_pos.pdf', plot_type='line', ylim=(0, 11))

#-- Plot RT/errors by hand-mapping
dpm.behavior.plot_per_group(comp2d, 'rt', 'handmapping', include_errors=False, save_as=fig_dir+'comp_rt_per_hand.pdf', plot_type='line', ylim=(500, 850))
dpm.behavior.plot_per_group(comp2d, 'errors', 'handmapping', save_as=fig_dir+'comp_acc_per_hand.pdf', plot_type='line', ylim=(0, 11))

#-- Plot RT distribution for each subject
dpm.behavior.rt_distrib_per_subject(comp2d_good, save_as=fig_dir+'rt_distrib_per_subject.pdf', smooth_sd=20, plot_medians=True, include_errors=False)


print('=======================================================================================')
print('             Digits RSVP task ')
print('=======================================================================================')

rsvp = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids_all, dpm.consts.rsvp_raw_files)
rsvp2d = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids_all, dpm.consts.rsvp_raw_files, target_filter='2digit')

print('Analysis of outliers:')
rsvp_outliers = dpm.behavior.get_outlier_subjects(rsvp, True)
#assert comp_outliers == ['am', 'ga', 'jm1', 'lb']  # Just making sure we know who they are
dpm.behavior.print_average_accuracy_and_rt(rsvp)


print('=======================================================================================')
print('             Words RSVP task ')
print('=======================================================================================')

wrsvp = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids, dpm.consts.rsvp_words_raw_files)
wrsvp2d = dpm.behavior.load_behavioral_data(dpm.consts.subj_ids, dpm.consts.rsvp_words_raw_files, target_filter='2digit')

print('Analysis of outliers:')
wrsvp_outliers = dpm.behavior.get_outlier_subjects(wrsvp, True)
#assert comp_outliers == ['am', 'ga', 'jm1', 'lb']  # Just making sure we know who they are
dpm.behavior.print_average_accuracy_and_rt(wrsvp)
