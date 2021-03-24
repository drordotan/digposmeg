
comparison_files = c('comp1c_1.csv', 'comp1c_2.csv', 'comp1n_1.csv', 'comp1n_2.csv', 'comp2c_1.csv', 'comp2c_2.csv', 'comp2n_1.csv', 'comp2n_2.csv')


#--------------------------------------------------------------------------------
load_subjects <- function(base_dir, subj_dirs, filenames, only_2digits=TRUE) {
  result = c()
  for (subj_dir in subj_dirs) {
    subj_data = load_subject(base_dir, subj_dir, filenames, only_2digits)
    if (length(result) == 0) {
      result = subj_data
    } else {
      result = rbind(result, subj_data)
    }
  }
  return(result)
}


#--------------------------------------------------------------------------------
load_subject <- function(base_dir, subj_dir, filenames, only_2digits=TRUE) {
  
  full_path = paste(base_dir, subj_dir, 'behavior', sep='/')
  
  result = c()
  for (filename in filenames) {
    file_path = paste(full_path, filename, sep='/')
    file_data = read.csv(file_path)
    
    file_data = file_data[file_data$response != 0,]
    
    if (only_2digits) {
      file_data = file_data[file_data$target >= 10, ]
    }
    
    file_data$subject = substr(subj_dir, 1, 2)
    
    if (substr(filename, 1, 4) == 'comp') {
      cong_flag = substr(filename, 6, 6)
      if (cong_flag == 'c') {
        file_data$hand_mapping = 1
      } else if (cong_flag == 'n') {
        file_data$hand_mapping = 0
      } else {
        stop('Unexpected file name %s', filename)
      }
    }
    
    if (length(result) == 0) {
      result = file_data
    } else {
      result = rbind(result, file_data)
    }
  }
  
  result$target = as.numeric(result$target)
  result$gt44 = result$target > 44
  result$distance = as.numeric(abs(44 - result$target))
  result$hand_mapping = as.factor(result$hand_mapping)
  result$position = as.numeric(result$position)
  result$response = as.factor(result$response)
  result$right_side = NA
  result[result$position > 2, 'right_side'] = TRUE
  result[result$target >= 10 & result$position < 2, 'right_side'] = FALSE
  result[result$target < 10 & result$position <= 2, 'right_side'] = FALSE
  
  result$position = factor(result$position)
  
  result$decade = floor(result$target / 10)
  result$unit = result$target %% 10
  result$u_compatible = result$gt44 == (result$unit > 4)
  
  result$correct = result$correct * 100  # Transform into percentages
  
  return(result)
}


#------------------------------------------------------------
compare_models <- function(model0, model1, effect_name, trace=FALSE) {
  
  model_diff = anova(model0, model1)
  if (trace) {
    print(model_diff)
  }
  
  likelihood0 = model_diff[1, 4]
  likelihood1 = model_diff[2, 4]
  
  if (likelihood1 < likelihood0) {
    
    print(sprintf('(Model comparison) effect of %s: strange, the model without the additional factor was better', effect_name))
    print(model_diff)
    
  } else if (likelihood1 == likelihood0) {
    
    print(sprintf('(Model comparison) no effect of %s: removing the factor did not change the model\'s likelihood', effect_name))
    
  } else {
    chi2 = model_diff[2,6]
    chi2df = model_diff[2,7]
    p = model_diff[2,8] / 2
    
    print(sprintf('(Model comparison) effect of %s: chi2(%d)=%.2f, 1-tailed p=%s', effect_name, chi2df, chi2, p_str(p)))
  }
  
  if (trace) {
    print('')
  }
}


#--------------------------------------------------------------------------------------------------
print_model_significance <- function(mdl, dependent_var, factor_names, factor_est_levels, coeff_precision=1) {
  sm = summary(mdl)
  sig = Anova(mdl)
  
  coeff_precision = ifelse(dependent_var == 'rt', 1, 3)
  coeff_units = ifelse(dependent_var == 'rt', ' ms', '%')
  
  for (factor in factor_names) {
    c(est_name, desc) %<-% add_level(factor, factor_est_levels)
    
    if (! (est_name %in% row.names(sm$coefficients))) {
      print(sm)
      stop(sprintf('Factor "%s" not found in model summary', est_name))
    }
    if (desc == '') {
      coeff_suffix = ''
    } else {
      coeff_suffix = sprintf(' (for %s)', desc)
    }
    
    part1 = sprintf('(LMM) Factor %s: coeff%s is %.*f%s, SE = %.*f%s', factor, coeff_suffix,
                    coeff_precision, sm$coefficients[est_name, 'Estimate'], coeff_units,
                    coeff_precision, sm$coefficients[est_name, 'Std. Error'], coeff_units)
    part2 = sprintf('; chi2(%d) = %.2f, p=%s', sig[factor, 'Df'], sig[factor, 'Chisq'], p_str(sig[factor, 'Pr(>Chisq)'])) 
    print(paste(part1, part2, sep=''))
  }
}

#-------------------------------------------------------------------
# Convert a list of factors ("factor1:factor2" string) to the same factors with levels: "factor1A:factor2B", where A and B are the levels defined for factors 1 and 2
# If a factor does not have a level, don't add it
add_level <- function(factor, factor_est_levels) {
  
  factors = strsplit(factor, ':')[[1]]
  
  factor_with_level = c()
  desc = c()
  
  for (f in factors) {
    
    if (f %in% names(factor_est_levels)) {
      factor_and_level = paste(f, factor_est_levels[f], sep='')
      desc = c(desc, sprintf('%s=%s', f, factor_est_levels[f]))
    } else {
      factor_and_level = f
    }
    
    factor_with_level = c(factor_with_level, factor_and_level)
  }
  
  factor_with_level = paste(factor_with_level, collapse=':')
  desc = paste(desc, collapse=', ')
  
  return (c(factor_with_level, desc))
}

#--------------------------------------------------------------------------------------------------

test_additive_effects <- function(bdata, dependent_var, log_distance=TRUE, compatibility_effect=TRUE, target_categorical=TRUE, trace=FALSE) {
  
  validate_dependent_var(dependent_var)
  
  if (compatibility_effect) {
    bdata = bdata[bdata$decade != 4,]
  }
  
  if (log_distance) {
    bdata$distance = log(bdata$distance)
  }
  
  target_factor = ifelse(target_categorical, 'gt44', 'target')
  target_factor_name = ifelse(target_categorical, 'target>44', 'target')
  
  compatibility_factor = ifelse(compatibility_effect, '+ u_compatible ', '')
  formula.full.int = sprintf('%s ~ %s * distance * position * hand_mapping %s+ (1|subject)', dependent_var, target_factor, ifelse(compatibility_effect, '* u_compatible ', ''))
  formula.full = sprintf('%s ~ %s + distance + position + hand_mapping %s+ (1|subject)', dependent_var, target_factor, compatibility_factor)
  formula.no.target = sprintf('%s ~ distance + position + hand_mapping %s+ (1|subject)', dependent_var, compatibility_factor)
  formula.no.distance = sprintf('%s ~ %s + position + hand_mapping %s+ (1|subject)', dependent_var, target_factor, compatibility_factor)
  formula.no.pos = sprintf('%s ~ %s + distance + hand_mapping %s+ (1|subject)', dependent_var, target_factor, compatibility_factor)
  formula.no.hand = sprintf('%s ~ %s + distance + position %s+ (1|subject)', dependent_var, target_factor, compatibility_factor)

  mdl.int = lmer(as.formula(formula.full.int), data = bdata, REML = FALSE)
  mdl = lmer(as.formula(formula.full), data = bdata, REML = FALSE)
  mdl.no.target = lmer(as.formula(formula.no.target), data = bdata, REML = FALSE)
  mdl.no.distance = lmer(as.formula(formula.no.distance), data = bdata, REML = FALSE)
  mdl.no.pos = lmer(as.formula(formula.no.pos), data = bdata, REML = FALSE)
  mdl.no.hand = lmer(as.formula(formula.no.hand), data = bdata, REML = FALSE)
  
  if (trace) {
    print('The full model:')
    print(summary(mdl))
  }
  
  check_sig_factors = c(target_factor, 'distance', 'hand_mapping')
  if (compatibility_effect) {
    check_sig_factors = c(check_sig_factors, 'u_compatible')
  }
  print_model_significance(mdl, dependent_var, factor_names=check_sig_factors, factor_est_levels = list(gt44='TRUE', hand_mapping='1', u_compatible='TRUE'))

  compare_models(mdl.no.target, mdl, target_factor_name, trace=FALSE)
  compare_models(mdl.no.distance, mdl, 'numerical distance', trace=FALSE)
  compare_models(mdl.no.pos, mdl, "screen location", trace=FALSE)
  #print_gain(bdata, dependent_var, 'hand_mapping', 'Congruent hand mapping')
  compare_models(mdl.no.hand, mdl, "hand mapping", trace=FALSE)
  
  if (compatibility_effect) {
    formula.no.compat = sprintf('%s ~ %s + distance + position + hand_mapping + (1|subject)', dependent_var, target_factor)
    mdl.no.compat = lmer(as.formula(formula.no.compat), data = bdata, REML = FALSE)
    compare_models(mdl.no.compat, mdl, "decade-unit compatibility", trace=FALSE)
  }
  
  compare_models(mdl, mdl.int, "all interactions", trace=FALSE)
}


#--------------------------------------------------------------------------------------------------
test_unit_distance_effect <- function(bdata, dependent_var) {
  
  validate_dependent_var(dependent_var)
  
  bdata = bdata[bdata$decade != 4,]
  
  bdata$decade_distance = abs(bdata$decade - 4)
  
  formula1 = sprintf('%s ~ gt44 + decade_distance + distance + position + hand_mapping + (1|subject)', dependent_var)
  formula0 = sprintf('%s ~ gt44 + decade_distance + position + hand_mapping + (1|subject)', dependent_var)
  
  mdl = lmer(as.formula(formula1), data = bdata, REML = FALSE)
  mdl0 = lmer(as.formula(formula0), data = bdata, REML = FALSE)
  
  compare_models(mdl0, mdl, "whole-number distance on top of decade distance", trace=FALSE)
  
}


#--------------------------------------------------------------------------------------------------
test_location_effect <- function(bdata, dependent_var) {
  
  validate_dependent_var(dependent_var)
  
  bdata = bdata[bdata$position != 2,]
  bdata$close_to_mid = as.factor(2 - abs(bdata$position - 2))  # 0 or 1

  formula.full = sprintf('%s ~ gt44 + distance + close_to_mid + right_side + hand_mapping + (1|subject)', dependent_var)
  formula.no.side = sprintf('%s ~ gt44 + distance + close_to_mid + hand_mapping + (1|subject)', dependent_var)
  formula.no.distance = sprintf('%s ~ gt44 + distance + right_side + hand_mapping + (1|subject)', dependent_var)
  
  print_gain(bdata, dependent_var, 'right_side', 'Right side', 'better when target appeared on the right side', 'better when target appeared on the left side')
  mdl.full = lmer(as.formula(formula.full), data = bdata, REML = FALSE)
  mdl0 = lmer(as.formula(formula.no.side), data = bdata, REML = FALSE)
  
  print_model_significance(mdl.full, dependent_var,
                           factor_names = c('right_side', 'close_to_mid'),
                           factor_est_levels = list(right_side='TRUE', close_to_mid=1))
  
  compare_models(mdl0, mdl.full, "side (left/right)", trace=FALSE)

  print_gain(bdata, dependent_var, 'close_to_mid', 'Close to mid-screen')
  mdl0 = lmer(as.formula(formula.no.distance), data = bdata, REML = FALSE)
  compare_models(mdl0, mdl.full, "distance from middle", trace=FALSE)
}


#--------------------------------------------------------------------------------------------------
test_target_position_interaction <- function(bdata, dependent_var, target_factor, position_factor) {
  
  validate_dependent_var(dependent_var)
  
  if (position_factor == 'right_side') {
    bdata = bdata[bdata$position != 2,]
  }
  
  if (target_factor == 'gt44' && position_factor == 'right_side') {
    bdata$congruent = bdata$gt44 == bdata$right_side
    print_gain(bdata, dependent_var, 'congruent', 'Target size-side congruency')
  }
  
  formula1 = sprintf('%s ~ hand_mapping + distance + %s * %s + (1|subject)', dependent_var, target_factor, position_factor)
  formula0 = sprintf('%s ~ hand_mapping + distance + %s + %s + (1|subject)', dependent_var, target_factor, position_factor)
  
  mdl = lmer(as.formula(formula1), data = bdata, REML = FALSE)
  mdl0 = lmer(as.formula(formula0), data = bdata, REML = FALSE)
  
  compare_models(mdl0, mdl, sprintf("%s*%s interaction", target_factor, position_factor), trace=FALSE)
}


#--------------------------------------------------------------------------------------------------
test_simon_effect <- function(bdata, dependent_var) {
  validate_dependent_var(dependent_var)
  
  bdata = bdata[bdata$position != 2,]
  
  bdata$right_field = bdata$position > 2
  bdata$simon = bdata$right_field == (bdata$response == 2)
  
  gain = mean(bdata[bdata$simon == 0, dependent_var]) - mean(bdata[bdata$simon == 1, dependent_var])
  if (dependent_var == 'rt') {
    print(sprintf('Simon effect gain: %d ms (positive=as predicted)', round(gain)))
  } else {
    print(sprintf('Simon effect gain: %.1f%% (positive=as predicted)', -gain))
  }

  formula1 = sprintf("%s ~ distance + hand_mapping + gt44 + simon + (1|subject)", dependent_var)
  formula0 = sprintf("%s ~ distance + hand_mapping + gt44  + (1|subject)", dependent_var)
  
  mdl = lmer(as.formula(formula1), data = data, REML = FALSE)
  mdl0 = lmer(as.formula(formula0), data = bdata, REML = FALSE)
  
  compare_models(mdl0, mdl, "simon effect (congruency of stimulus side & response side)", trace=FALSE)
}


#--------------------------------------------------------------------------------------------------
test_snarc_effect <- function(bdata, dependent_var, target_categorical=TRUE) {
  
  validate_dependent_var(dependent_var)
  
  bdata = bdata[bdata$position != 2 | bdata$target < 10,]
  
  target_factor = ifelse(target_categorical, 'gt44', 'target')
  
  formula1 = sprintf('%s ~ position + distance + %s * response + (1|subject)', dependent_var, target_factor)
  formula0 = sprintf('%s ~ position + distance + %s + response + (1|subject)', dependent_var, target_factor)
  
  mdl = lmer(as.formula(formula1), data = bdata, REML = FALSE)
  mdl0 = lmer(as.formula(formula0), data = bdata, REML = FALSE)
  
  print_model_significance(mdl, dependent_var,
                           factor_names = c('distance', target_factor, 'response',  sprintf('%s:response', target_factor)), 
                           factor_est_levels = list(gt44='TRUE', response='2'))
  
  compare_models(mdl0, mdl, "large/small - response side congruency (SNARC effect)", trace=FALSE)
}


#--------------------------------------------------------------------------------------------------
plot_snarc <- function(bdata) {
  
  bdata_left = bdata[bdata$response == 1,]
  bdata_right = bdata[bdata$response == 2,]
  
  targets = sort(unique(bdata$target))
  rt_left = c()
  rt_right = c()
  
  for (target in targets) {
    rt_left = c(rt_left, mean(bdata_left$rt[bdata_left$target == target]))
    rt_right = c(rt_right, mean(bdata_right$rt[bdata_right$target == target]))
  }
  
  result = data.frame(targets=targets, rt_left=rt_left, rt_right=rt_right)
  
  ggplot(s, mapping = aes(targets, rt_right - rt_left)) + geom_point() + geom_line()
  
  return(result)
}

#--------------------------------------------------------------------------------------------------
test_compatibility_effect <- function(bdata, dependent_var) {
  
  validate_dependent_var(dependent_var)
  
  bdata = bdata[bdata$decade != 4,]
  
  bdata$compatible = (bdata$decade > 4) == (bdata$unit > 4)
  
  print_gain(bdata, dependent_var, 'compatible', 'Decade-unit compatibility')
  
  formula.with.interaction = sprintf('%s ~ distance + gt44 + position + hand_mapping + compatible + (1|subject)', dependent_var)
  formula.no.interaction = sprintf('%s ~ distance + gt44 + position + hand_mapping + (1|subject)', dependent_var)
  
  mdl = lmer(as.formula(formula.with.interaction), data = bdata, REML = FALSE)
  mdl0 = lmer(as.formula(formula.no.interaction), data = bdata, REML = FALSE)
  
  compare_models(mdl0, mdl, "decade-unit compatibility", trace=FALSE)
}



#--------------------------------------------------------------------------------------------------
validate_dependent_var <- function(dependent_var) {
  if (! (dependent_var %in% c('correct', 'rt'))) {
    stop(sprintf('Unknown dependent_var variable: %s',  dependent_var))
  }
}

#--------------------------------------------------------------------------------------------------
print_gain <- function(bdata, dependent_var, effect_field, effect_name, good_desc='as predicted', bad_desc='contrary to prediction') {
  
  gain = mean(bdata[bdata[,effect_field] == 0, dependent_var]) - mean(bdata[bdata[,effect_field] == 1, dependent_var])
  if (is.nan(gain)) {
    print(sprintf('%s gain cannot be computed', effect_name))
    return()
  }
  
  if (dependent_var == 'correct') {
    print(sprintf('%s accuracy gain: %.1f%% (%s)', effect_name, -gain, ifelse(gain<0, good_desc, bad_desc)))
  } else {
    print(sprintf('%s RT gain: %d ms (%s)', effect_name, round(gain), ifelse(gain>0, good_desc, bad_desc)))
  }
}

#--------------------------------------------------------------------------------------------------
p_str <- function(p) {
  
  if (p > .1) {
    return(sprintf("%.02f", p))
  } else if (p > .001) {
    return(sprintf("%.03f", p))
  } else if (p > .0001) {
    return(sprintf("%.04f", p))
  } else {
    return(sprintf("1e%d", ceiling(log(p) / log(10))))
  }
}
