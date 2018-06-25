
skip = function() {
  # Set target type -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
  rm(list = ls())
  TYPE = "CLASS"
  TYPE = "REGR"
  TYPE = "MULTICLASS"
}

#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load(paste0(TYPE,"_1_explore.rdata"))

# Load libraries and functions
source("./code/0_init.R")

# Adapt some default parameter different for target types -> probably also different for a new use-case
type = switch(TYPE, "CLASS" = "prob", "REGR" = "raw", "MULTICLASS" = "prob") #do not change this one
color = switch(TYPE, "CLASS" = twocol, "REGR" = twocol, "MULTICLASS" = threecol) #probably need to change multiclass opt
ylim1 = switch(TYPE, "CLASS" = c(0,1), "REGR"  = c(-5e4,5e4), "MULTICLASS" = c(0,1))
ylim2 = switch(TYPE, "CLASS" = c(0.2,0.7), "REGR"  = c(1.5e5,2.5e5), "MULTICLASS" = c(0,1)) #need to adapt
ylim3 = switch(TYPE, "CLASS" = c(0,1), "REGR"  = c(0,5e5), "MULTICLASS" = c(0,1))
topn = switch(TYPE, "CLASS" = 10, "REGR" = 20, "MULTICLASS" = 20) #remove here and set hard below
b_all = b_sample = NULL #do not change this one (as it is default in regression case)



## Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster



## Tuning parameter to use
if (TYPE == "CLASS") {
  tunepar = expand.grid(nrounds = 500, max_depth = 3, 
                        eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
                        min_child_weight = 2, subsample = 0.7)
}
if (TYPE %in% c("REGR","MULTICLASS")) {
  tunepar = expand.grid(nrounds = 500, max_depth = 3, 
                        eta = 0.05, gamma = 0, colsample_bytree = 0.3, 
                        min_child_weight = 5, subsample = 0.7)
}




# Sample data --------------------------------------------------------------------------------------------

# Training data
if (TYPE %in% c("CLASS","MULTICLASS")) {
  # Just take data from train fold (take all but n_maxpersample at most)
  summary(df[df$fold == "train", "target"])
  c(df.train, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_maxpersample = 500)) 
  summary(df.train$target); b_sample; b_all
  
  # Set metric for peformance comparison
  metric = "AUC"
}
if (TYPE == "REGR") {
  # Just take data from train fold
  df.train = df %>% filter(fold == "train") #%>% sample_n(1000)

  # Set metric for peformance comparison
  metric = "spearman"
}

# Define test data
df.test = df %>% filter(fold == "test")




#######################################################################################################################-
#|||| Performance ||||----
#######################################################################################################################-

#---- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
tmp = Sys.time()
fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[features])), df.train$target,
#fit = train(formula, data = df.train[c("target",features)],
            trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
            method = "xgbTree", 
            tuneGrid = tunepar)
Sys.time() - tmp

# Predict
yhat_test_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.test[features])),
                             type = type)
summary(yhat_test_unscaled)
# # Scoring in chunks in parallel in case of high memory consumption of xgboost
# l.split = split(df.test[features], (1:nrow(df.test)) %/% 50000)
# yhat_test_unscaled = foreach(df.split = l.split, .combine = bind_rows) %dopar% {
#   predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.split)), type = type)
# }

# Rescale 
yhat_test = scale_pred(yhat_test_unscaled, b_sample, b_all)
y_test = df.test$target

# Plot performance
mysummary(data.frame(y = y_test, yhat = yhat_test))
plots = get_plot_performance(yhat = yhat_test, y = y_test, reduce_factor = NULL, colors = color)
ggsave(paste0(plotloc,TYPE,"_performance.pdf"), marrangeGrob(plots, ncol = length(plots)/2, nrow = 2, top = NULL), 
       w = 18, h = 12)




#---- Check residuals ----------------------------------------------------------------------------------

## Residuals
if (TYPE %in% c("CLASS","MULTICLASS")) {
  # Decide for a reference member
  #levels(y_test);  k = 2;  df.test$residual = ifelse(y_test == levels(y_test)[k], 1, 0) - yhat_test[,k]
  # Preferred: dynamic refernce member per obs, i.e. the true label
  df.test$residual = 1 - rowSums(yhat_test * model.matrix(~ -1 + y_test, data.frame(y_test)))
}
if (TYPE == "REGR") df.test$residual = y_test - yhat_test
df.test$abs_residual = abs(df.test$residual)
summary(df.test$residual)

# For non-regr tasks one might want to plot the following for each target level (df.test %>% filter(target == "level"))
plots = c(suppressMessages(get_plot_distr_metr(df.test, metr, target_name = "residual", ylim = ylim1, 
                                               missinfo = misspct, color = hexcol)), 
          suppressMessages(get_plot_distr_nomi(df.test, nomi, target_name = "residual", ylim = ylim1,
                                               color = color)))
ggsave(paste0(plotloc,TYPE,"_diagnosis_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), width = 18, height = 12)



## Absolute residuals
if (TYPE == "REGR") {
  summary(df.test$abs_residual)
  plots = c(suppressMessages(get_plot_distr_metr(df.test, metr, target_name = "abs_residual", ylim = c(0,ylim1[2]), 
                                                 missinfo = misspct, color = hexcol)), 
            suppressMessages(get_plot_distr_nomi(df.test, nomi, target_name = "abs_residual", ylim = c(0,ylim1[2]),
                                                 color = color)))
  ggsave(paste0(plotloc,TYPE,"_diagnosis_absolute_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
         width = 18, height = 12)
}




#---- Check performance on some bootstrapped fits ---------------------------------------------------------------------

## Fit
n.boot = 5
l.boot = foreach(i = 1:n.boot, .combine = c, .packages = c("caret","ROCR","xgboost","Matrix")) %dopar% { 
  
  # Bootstrap
  set.seed(i*1234)
  i.boot = sample(1:nrow(df.train), replace = TRUE) #resampled bootstrap observations
  i.oob = (1:nrow(df.train))[-unique(i.boot)] #Out-of-bag observations
  df.boot = df.train[i.boot,]
  
  # Fit
  fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.boot[features])), df.boot$target,
              #fit = train(formula, data = df.train[c("target",features)],
              trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
              method = "xgbTree", 
              tuneGrid = tunepar)
  yhat = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[i.oob,features])), 
                 type = type) #do NOT rescale on OOB data
  perf = as.numeric(mysummary(data.frame(yhat = yhat, y = df.train$target[i.oob]))[metric])
  return(setNames(list(fit, perf), c(paste0("fit_",i), c(paste0(metric,"_",i)))))
}



## Is it stable?

# Performance on Out-of-bag data
map_dbl(1:n.boot, ~ l.boot[[paste0(metric,"_",.)]]) 

# Performance on test data
map_df(1:n.boot, ~ {
  yhat_unscaled = predict(l.boot[[paste0("fit_",.)]], 
                          xgb.DMatrix(sparse.model.matrix(formula_rightside, df.test[features])),
                          type = type)
  yhat = scale_pred(yhat_unscaled, b_sample, b_all)
  data.frame(t(mysummary(data.frame(yhat = yhat, y = df.test$target)))) 
})




#--- Most important variables (importance > 10) model fit -------------------------------------------------------------------

# Variable importance (on train data!)
df.varimp_train = get_varimp_by_permutation(df.train, fit, dmatrix = TRUE,
                                            b_sample = b_sample, b_all = b_all, vars = features, metric = metric)
features_top = df.varimp_train %>% filter(importance > 10) %>% .$variable
formula_top = as.formula(paste("target", "~", paste(features_top, collapse = " + ")))
formula_top_rightside = as.formula(paste("~", paste(features_top, collapse = " + ")))

# Fit again -> possibly tune nrounds again
fit_top = train(xgb.DMatrix(sparse.model.matrix(formula_top_rightside, df.train[features_top])), df.train$target,
                trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
                method = "xgbTree", 
                tuneGrid = tunepar)

# Plot performance
yhat_top_unscaled = predict(fit_top, xgb.DMatrix(sparse.model.matrix(formula_top_rightside, df.test[features_top])),
                       type = type)
yhat_top = scale_pred(yhat_top_unscaled, b_sample, b_all)
plots = get_plot_performance(yhat = yhat_top, y = df.test$target, reduce_factor = NULL, colors = color)
plots[1]




#######################################################################################################################-
#|||| Variable Importance ||||----
#######################################################################################################################-

#--- Default Variable Importance: uses gain sum of all trees ---------------------------------------------------------

# Default plot 
plot(varImp(fit)) 



#--- Variable Importance by permuation argument -------------------------------------------------------------------

## Importance for "total" fit (on test data!)
df.varimp = get_varimp_by_permutation(df.test, fit, dmatrix = TRUE, feature_names = features,
                                      b_sample = b_sample, b_all = b_all, vars = features, metric = metric)

# Visual check how many variables needed 
ggplot(df.varimp) + 
  geom_bar(aes(x = reorder(variable, importance), y = importance), stat = "identity") +
  coord_flip() 
topn = topn
topn_vars = df.varimp[1:topn, "variable"]

# Add other information (e.g. special coloring): color variable is needed -> fill with "dummy" if it should be ommited
df.varimp %<>% mutate(color = cut(importance, c(-1,10,50,101), labels = c("low","middle","high")))



## Importance for bootstrapped models (and bootstrapped data): ONLY for topn_vars
# Get boostrap values
df.varimp_boot = c()
for (i in 1:n.boot) {
  set.seed(i*1234)
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.varimp_boot %<>% 
    bind_rows(get_varimp_by_permutation(df.test_boot, l.boot[[paste0("fit_",i)]], dmatrix = TRUE, 
                                        feature_names = features,
                                        b_sample = b_sample, b_all = b_all,
                                        vars = topn_vars, metric = metric) %>% 
                mutate(run = i))
}

# Plot
plots = get_plot_varimp(df.varimp, topn_vars, df.plot_boot = df.varimp_boot)
ggsave(paste0(plotloc,TYPE,"_variable_importance.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1), w = 12, h = 8)




#--- Compare variable importance for train and test (hints to variables prone to overfitting) -------------------------

for (impvar in c("importance","importance_sumnormed")) {
  #impvar = "importance" #impvar = "importance_sumnormed"
  df.tmp = df.varimp %>% select_("variable",impvar) %>% mutate(type = "test") %>% 
    bind_rows(df.varimp_train %>% select_("variable",impvar) %>% mutate(type = "train")) %>% 
    filter(variable %in% topn_vars)
  ggplot(df.tmp, aes_string("variable", impvar)) +
    geom_bar(aes(fill = type), position = "dodge", stat = "identity") +
    #scale_x_discrete(limits = rev(df.varimp$variable)) +
    scale_fill_discrete(limits = c("train","test")) +
    coord_flip() +
    labs(title = impvar)
}




#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-

## Partial depdendance for "total" fit 
# Get "x-axis" points
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.05)))
df.partialdep = get_partialdep(df.test, fit, b_sample = b_sample, b_all = b_all,
                               vars = topn_vars, l.levs = levs, l.quantiles = quantiles)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = ylim2, colors = color)
ggsave(paste0(plotloc,TYPE,"_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       w = 18, h = 12)



## Partial dependance for bootstrapped models (and bootstrapped data) 
# Get boostrap values
df.partialdep_boot = c()
for (i in 1:n.boot) {
  set.seed(i*1234)
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.partialdep_boot %<>% 
    bind_rows(get_partialdep(df.test_boot, l.boot[[paste0("fit_",i)]], b_sample = b_sample, b_all = b_all,
                             vars = topn_vars, l.levs = levs, l.quantiles = quantiles) %>% 
                mutate(run = i))
}



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = ylim2, colors = color)
ggsave(paste0(plotloc,TYPE,"_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       w = 18, h = 12)




#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-

if (TYPE %in% c("CLASS","REGR")) {
  
  # Make yhat_test a vector
  if (TYPE == "CLASS") yhat_explain = yhat_test[,2] else yhat_explain = yhat_test
  
  # Derive id
  df.test$id = 1:nrow(df.test)
  
  # Subset test data (explanations are only calcualted for this subset)
  i.top = order(yhat_explain, decreasing = TRUE)[1:20]
  i.bottom = order(yhat_explain)[1:20]
  i.random = sample(1:length(yhat_explain), 20)
  i.explain = sample(unique(c(i.top, i.bottom, i.random)))
  df.test_explain = df.test[i.explain, c("id", features)]
  
  # Get explanations
  df.explanations = get_explanations(b_sample = b_sample, b_all = b_all, type = tolower(TYPE))
  
  
  ## Plot
  df.values = df.test_explain
  df.values[metr] = map(df.values[metr], ~ round(.,2))
  plots = get_plot_explanations(df.plot = df.explanations, df.values = df.values, type = tolower(TYPE), topn = 10, 
                                ylim = ylim3)
  ggsave(paste0(plotloc,TYPE,"_explanations.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2), 
         w = 18, h = 12)

}

save.image(paste0(TYPE,"_3_interpret.rdata"))
#load(paste0(TYPE,"_3_interpret.rdata"))



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

