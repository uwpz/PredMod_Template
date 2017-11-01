


#######################################################################################################################-
#|||| Variable Importance ||||----
#######################################################################################################################-

#--- Default Variable Importance: uses gain sum of all trees ---------------------------------------------------------

# Default plot 
plot(varImp(fit)) 



#--- Variable Importance by permuation argument -------------------------------------------------------------------

## Importance for "total" fit (on test data!)
df.varimp = get_varimp_by_permutation(df.test, fit, vars = predictors, metric = metric)

# Visual check how many variables needed 
ggplot(df.varimp) + 
  geom_bar(aes(x = reorder(variable, importance), y = importance), stat = "identity") +
  coord_flip() 
topn = 10
topn_vars = df.varimp[1:topn, "variable"]

# Add other information (e.g. special coloring): color variable is needed -> fill with "dummy" if it should be ommited
df.varimp %<>% mutate(color = cut(importance, c(-1,10,50,100), labels = c("low","middle","high")))



## Importance for bootstrapped models (and bootstrapped data)
# Get boostrap values
df.varimp_boot = c()
for (i in 1:n.boot) {
  set.seed(i*1234)
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.varimp_boot %<>% 
    bind_rows(get_varimp_by_permutation(df.test_boot, l.boot[[paste0("fit_",i)]], predictors, metric = metric) %>% 
                mutate(run = i))
}



## Plot
plot = get_plot_varimp(df.varimp, topn_vars, df.plot_boot = df.varimp_boot)
ggsave(paste0(plotloc, "variable_importance.pdf"), plot, w = 8, h = 6)





#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-


## Partial depdendance for "total" fit 
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.05)))
df.partialdep = get_partialdep(df.test, fit, vars = topn_vars, levs = levs, quantiles = quantiles)

if (type == "class") {
  # Rescale
  df.partialdep$yhat = prob_samp2full(df.partialdep$yhat, b_sample, b_all)
} else {
}

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(0,4))
ggsave(paste0(plotloc, "partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)



## Partial Dependance for bootstrapped models (and bootstrapped data) 
# Get boostrap values
df.partialdep_boot = c()
for (i in 1:n.boot) {
  set.seed(i*1234)
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.partialdep_boot %<>% 
    bind_rows(get_partialdep(df.test_boot, l.boot[[paste0("fit_",i)]], 
                             vars = topn_vars, levs = levs, quantiles = quantiles) %>% 
                mutate(run = i))
}
if (type == "class") {
  # Rescale
  df.partialdep_boot$yhat = prob_samp2full(df.partialdep_boot$yhat, b_sample, b_all)
} else {
}



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(1,3))
ggsave(paste0(plotloc, "partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)





#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-

## Derive betas for all test cases

# Get model matrix and DMatrix
m.model_train = model.matrix(formula, data = df.train[c("target",predictors)], contrasts = NULL)[,-1]
m.train = xgb.DMatrix(m.model_train)
m.model_test = model.matrix(formula, data = df.train[c("target",predictors)], contrasts = NULL)[,-1]
m.test = xgb.DMatrix(m.model_test)

# Create explainer data table from train data
if (type == "class") {
  # Rescale
  df.explainer = buildExplainer(fit$finalModel, m.train, type = "binary")
  
  # Switch coefficients (as explainer takes "N" as target = 1)
  cols = setdiff(colnames(df.explainer), c("leaf","tree"))
  df.explainer[, (cols) := lapply(.SD, function(x) -x), .SDcols = cols]
} else {
  df.explainer = buildExplainer(fit$finalModel, m.train, type = "regression")
}

# Get predictions for all test data
df.predictions = explainPredictions(fit$finalModel, df.explainer, m.test)
df.predictions$id = 1:nrow(df.predictions)

# Get value data frame
df.model_test = as.data.frame(m.model_test)
df.model_test$id = 1:nrow(df.model_test)

## Plot
type = "regr"
if (type == "class") {
  plots = get_plot_explainer(df.plot = df.predictions[1:12,], df.values = df.model_test[1:12,], type = "class")
} else {
  plots = get_plot_explainer(df.plot = df.predictions[1:12,], df.values = df.model_test[1:12,], type = "regr", 
                             ylim = c(1,3))
}
ggsave(paste0(plotloc, "explanations.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)



#save.image("3_interpret.rdata")



#######################################################################################################################-
# Interactions
#######################################################################################################################-

# 
# ## Test for interaction (only for gbm)
# (intervars = rownames(varImp(fit.gbm)$importance)[order(varImp(fit.gbm)$importance, decreasing = TRUE)][1:10])
# plot_interactiontest("./output/interactiontest_gbm.pdf", vars = intervars, l.boot = l.boot)
# 
# 
# ## -> Relvant interactions
# inter1 = c("TT4","TSH_LOG_")
# inter2 = c("sex","referral_source_OTHER_") 
# inter3 = c("TT4","referral_source_OTHER_")
# plot_inter("./output/inter1.pdf", inter1, ylim = c(0,.4))
# plot_inter("./output/inter2.pdf", inter2, ylim = c(0,.4))
# plot_inter("./output/inter3.pdf", inter3, ylim = c(0,.4))
# 
# plot_inter_active("./output/anim", vars = inter1, duration = 3)

