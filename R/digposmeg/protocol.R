library(lme4)
library(car)
library(zeallot)
library(ggplot2)

source('/Users/dror/git/digposmeg/R/digposmeg/dpm-funcs.R')

base_dir = '/Users/dror/meg/digit-position/exp/exp-data'
subj_dirs = c('ag_170045', 'at_140305', 'bl_170454', 'bo_160176', 'cb_140229', 'cc_150418', 'eb_170163', 'en_170221', 'hr_140096', 'jm_100042', 'lj_150477', 'mn_170263', 'mp_150285')


data12 = load_subjects(base_dir, subj_dirs, comparison_files, only_2digits = FALSE)
data12_correct = data12[data12$correct > 0,]

data = load_subjects(base_dir, subj_dirs, comparison_files, only_2digits = TRUE)
data_correct = data[data$correct > 0,]

#------- Analysis of basic factors -------

#-- Test each of the basic factors that may affect RT/accuracy
test_additive_effects(data_correct, 'rt')
test_additive_effects(data, 'correct')

# Replicate with continuous target
test_additive_effects(data_correct, 'rt', target_categorical = FALSE)

test_location_effect(data_correct, 'rt')
test_location_effect(data, 'correct')

#------- Show that unit digit is considered -------

test_unit_distance_effect(data_correct, 'rt')
test_unit_distance_effect(data, 'correct')

#------- Analysis of interactions -------

#-- Target*position ineraction: exists?
test_target_position_interaction(data_correct, 'rt', 'target', 'position')
test_target_position_interaction(data_correct, 'rt', 'gt44', 'right_side')

# this worked but not sure it's important
test_target_position_interaction(data_correct[data_correct$hand_mapping == 1,], 'rt', 'gt44')

test_simon_effect(data_correct, 'rt')
test_simon_effect(data, 'correct')

test_snarc_effect(data_correct, 'rt')
test_snarc_effect(data_correct, 'rt', target_categorical=FALSE)
test_snarc_effect(data, 'correct')

test_compatibility_effect(data_correct, 'rt')
test_compatibility_effect(data, 'correct')
