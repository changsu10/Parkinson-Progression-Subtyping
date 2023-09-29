# Load info table
label_df = read.csv('[your directory]/cluster_label.csv', check.names = FALSE)

dk_label = read.csv('[your directory]/dk.csv', check.names = FALSE) # brain region dictionary
dk_label = dk_label[complete.cases(dk_label), ]
dk_label$label = gsub('lh_', '', dk_label$label)
dk_label$label = gsub('rh_', '', dk_label$label)
dk_label = unique(dk_label)

# Load data
wmv_df = read.csv('[your directory]/white_matter_volume_BL-1Year.csv', check.names = FALSE)

for(i in 2:length(colnames(wmv_df))){
  temp = colnames(wmv_df)[i]
  colnames(wmv_df)[i] = dk_label[dk_label[, 'label'] == temp, ]$region
}

var_list = colnames(wmv_df)[2:length(colnames(wmv_df))]

wmv_df = merge(label_df, wmv_df, on='PATNO')
wmv_df$label = factor(wmv_df$label)

wmv_0 = wmv_df[wmv_df[, 'label'] == 0 | wmv_df[, 'label'] == 'HC', ]
wmv_0$label[wmv_0$label == 0] = 1
wmv_0$label[wmv_0$label == 'HC'] = 0

wmv_1 = wmv_df[wmv_df[, 'label'] == 1 | wmv_df[, 'label'] == 'HC', ]
wmv_1$label[wmv_1$label == 'HC'] = 0

wmv_2 = wmv_df[wmv_df[, 'label'] == 2 | wmv_df[, 'label'] == 'HC', ]
wmv_2$label[wmv_2$label == 2] = 1
wmv_2$label[wmv_2$label == 'HC'] = 0

wmv_0_1 = wmv_df[wmv_df[, 'label'] == 0 | wmv_df[, 'label'] == 1, ]
wmv_0_1$label[wmv_0_1$label == 0] = 'HC'
wmv_0_1$label[wmv_0_1$label == 1] = 0
wmv_0_1$label[wmv_0_1$label == 'HC'] = 1

wmv_0_2 = wmv_df[wmv_df[, 'label'] == 0 | wmv_df[, 'label'] == 2, ]
wmv_0_2$label[wmv_0_2$label == 2] = 1

wmv_1_2 = wmv_df[wmv_df[, 'label'] == 1 | wmv_df[, 'label'] == 2, ]
wmv_1_2$label[wmv_1_2$label == 1] = 0
wmv_1_2$label[wmv_1_2$label == 2] = 1

wmv_PD = wmv_df[wmv_df[, 'label'] == 0 | wmv_df[, 'label'] == 1 | wmv_df[, 'label'] == 2, ]

wmv_0R = wmv_PD
wmv_0R$label[wmv_0R$label == 0] = 'HC'
wmv_0R$label[wmv_0R$label == 1] = 0
wmv_0R$label[wmv_0R$label == 2] = 0
wmv_0R$label[wmv_0R$label == 'HC'] = 1

wmv_1R = wmv_PD
wmv_1R$label[wmv_1R$label == 2] = 0

wmv_2R = wmv_PD
wmv_2R$label[wmv_2R$label == 1] = 0
wmv_2R$label[wmv_2R$label == 2] = 1

# res_0 = data.frame()
# res_1 = data.frame()
# res_2 = data.frame()
res_0_1 = data.frame()
res_0_2 = data.frame()
res_1_2 = data.frame()
# res_0R = data.frame()
# res_1R = data.frame()
# res_2R = data.frame()
for(var_name in var_list){
  # ttest_res_0 = t.test(wmv_0[[var_name]] ~ wmv_0$label)
  # ttest_res_1 = t.test(wmv_1[[var_name]] ~ wmv_1$label)
  # ttest_res_2 = t.test(wmv_2[[var_name]] ~ wmv_2$label)
  
  ttest_res_0_1 = t.test(wmv_0_1[[var_name]] ~ wmv_0_1$label)
  ttest_res_0_2 = t.test(wmv_0_2[[var_name]] ~ wmv_0_2$label)
  ttest_res_1_2 = t.test(wmv_1_2[[var_name]] ~ wmv_1_2$label)
  
  # ttest_res_0R = t.test(wmv_0R[[var_name]] ~ wmv_0R$label)
  # ttest_res_1R = t.test(wmv_1R[[var_name]] ~ wmv_1R$label)
  # ttest_res_2R = t.test(wmv_2R[[var_name]] ~ wmv_2R$label)
  # 
  # row_0 = c(var_name, ttest_res_0[[3]], ttest_res_0[[5]][[2]] > ttest_res_0[[5]][[1]])
  # row_1 = c(var_name, ttest_res_1[[3]], ttest_res_1[[5]][[2]] > ttest_res_1[[5]][[1]])
  # row_2 = c(var_name, ttest_res_2[[3]], ttest_res_2[[5]][[2]] > ttest_res_2[[5]][[1]])
  # 
  row_0_1 = c(var_name, ttest_res_0_1[[3]], ttest_res_0_1[[5]][[2]] > ttest_res_0_1[[5]][[1]])
  row_0_2 = c(var_name, ttest_res_0_2[[3]], ttest_res_0_2[[5]][[2]] > ttest_res_0_2[[5]][[1]])
  row_1_2 = c(var_name, ttest_res_1_2[[3]], ttest_res_1_2[[5]][[2]] > ttest_res_1_2[[5]][[1]])
  
  # row_0R = c(var_name, ttest_res_0R[[3]], ttest_res_0R[[5]][[2]] > ttest_res_0R[[5]][[1]])
  # row_1R = c(var_name, ttest_res_1R[[3]], ttest_res_1R[[5]][[2]] > ttest_res_1R[[5]][[1]])
  # row_2R = c(var_name, ttest_res_2R[[3]], ttest_res_2R[[5]][[2]] > ttest_res_2R[[5]][[1]])
  # 
  # res_0 = rbind(res_0, row_0)
  # res_1 = rbind(res_1, row_1)
  # res_2 = rbind(res_2, row_2)
  
  res_0_1 = rbind(res_0_1, row_0_1)
  res_0_2 = rbind(res_0_2, row_0_2)
  res_1_2 = rbind(res_1_2, row_1_2)
  
  # res_0R = rbind(res_0R, row_0R)
  # res_1R = rbind(res_1R, row_1R)
  # res_2R = rbind(res_2R, row_2R)
}

colnames(res_0) = colnames(res_1) = colnames(res_2) = colnames(res_0_1) = colnames(res_0_2) = colnames(res_1_2) = colnames(res_0R) = colnames(res_1R) = colnames(res_2R) = c('region', 'p-value', 'normal')

# write.csv(res_0, '[your directory]/wmv_0_HC_t.csv', row.names = FALSE)
# write.csv(res_1, '[your directory]/wmv_1_HC_t.csv', row.names = FALSE)
# write.csv(res_2, '[your directory]/wmv_2_HC_t.csv', row.names = FALSE)
write.csv(res_0_1, '[your directory]/wmv_0_1_t.csv', row.names = FALSE)
write.csv(res_0_2, '[your directory]/wmv_0_2_t.csv', row.names = FALSE)
write.csv(res_1_2, '[your directory]/wmv_1_2_t.csv', row.names = FALSE)
# write.csv(res_0R, '[your directory]/wmv_0_Rest_t.csv', row.names = FALSE)
# write.csv(res_1R, '[your directory]/wmv_1_Rest_t.csv', row.names = FALSE)
# write.csv(res_2R, '[your directory]/wmv_2_Rest_t.csv', row.names = FALSE)


