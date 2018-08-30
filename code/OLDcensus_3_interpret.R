
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load(paste0("census_1_explore.rdata"))

# Load libraries and functions
source("./code/0_init.R")

# Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(6)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster



## Tuning parameter to use
tunepar = expand.grid(nrounds = 1500, max_depth = 6, 
                      eta = c(0.01), gamma = 0, colsample_bytree = c(0.7), 
                      min_child_weight = c(2), subsample = c(0.7))




# Sample data --------------------------------------------------------------------------------------------

# Training data: just take data from train fold (take all but n_maxpersample at most)
summary(df[df$fold == "train", "target"])
c(df.train, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_maxpersample = 10000)) 
summary(df.train$target); b_sample; b_all

# Set metric for peformance comparison
metric = "AUC"

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
                             type = "prob")
summary(yhat_test_unscaled)
# # Scoring in chunks in parallel in case of high memory consumption of xgboost
# l.split = split(df.test[features], (1:nrow(df.test)) %/% 50000)
# yhat_test_unscaled = foreach(df.split = l.split, .combine = bind_rows) %dopar% {
#   predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.split)), type = "prob")
# }

# Rescale 
yhat_test = scale_pred(yhat_test_unscaled, b_sample, b_all)
y_test = df.test$target

# Plot performance
mysummary(data.frame(y = y_test, yhat = yhat_test))
plots = get_plot_performance(yhat = yhat_test, y = y_test, reduce_factor = NULL, colors = twocol)
ggsave(paste0(plotloc,"census_performance.pdf"), marrangeGrob(plots, ncol = length(plots)/2, nrow = 2, top = NULL), 
       w = 18, h = 12)




#---- Check residuals ----------------------------------------------------------------------------------

## Residuals
df.test$residual = ifelse(y_test == levels(y_test)[2], 1, 0) - yhat_test[,2]
df.test$abs_residual = abs(df.test$residual)
summary(df.test$residual)
plots = c(suppressMessages(get_plot_distr_metr(df.test, metr, target_name = "residual", ylim = c(-0.1,0.1), 
                                               missinfo = misspct, color = hexcol)), 
          suppressMessages(get_plot_distr_nomi(df.test, nomi, target_name = "residual", ylim = c(-0.1,0.1),
                                               color = twocol)))
ggsave(paste0(plotloc,"census_diagnosis_residual.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2), width = 18, height = 12)



## Absolute residuals
summary(df.test$abs_residual)
plots = c(suppressMessages(get_plot_distr_metr(df.test, metr, target_name = "abs_residual", ylim = c(0,0.2), 
                                               missinfo = misspct, color = hexcol)), 
          suppressMessages(get_plot_distr_nomi(df.test, nomi, target_name = "abs_residual", ylim = c(0,0.2),
                                               color = twocol)))
ggsave(paste0(plotloc,"census_diagnosis_absolute_residual.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2), 
       width = 18, height = 12)





#---- Check performance on some bootstrapped fits ---------------------------------------------------------------------

## Fit
n.boot = 10
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
                 type = "prob") #do NOT rescale on OOB data
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
                          type = "prob")
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
                       type = "prob")
yhat_top = scale_pred(yhat_top_unscaled, b_sample, b_all)
plots = get_plot_performance(yhat = yhat_top, y = df.test$target, reduce_factor = NULL, colors = twocol)
plots[1]




#######################################################################################################################-
#|||| Variable Importance ||||----
#######################################################################################################################-

#--- Default Variable Importance: uses gain sum of all trees ---------------------------------------------------------

# Default plot 
plot(varImp(fit)) 



#--- Variable Importance by permuation argument -------------------------------------------------------------------

## Importance for "total" fit (on test data!)
df.varimp = get_varimp_by_permutation(df.test, fit, dmatrix = TRUE,
                                      b_sample = b_sample, b_all = b_all, vars = features, metric = metric)

# Visual check how many variables needed 
ggplot(df.varimp) + 
  geom_bar(aes(x = reorder(variable, importance), y = importance), stat = "identity") +
  coord_flip() 
topn = 30
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
                                        b_sample = b_sample, b_all = b_all,
                                        vars = topn_vars, metric = metric) %>% 
                mutate(run = i))
}

# Plot
plots = get_plot_varimp(df.varimp, topn_vars, df.plot_boot = df.varimp_boot)
ggsave(paste0(plotloc,"census_variable_importance.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1), w = 12, h = 8)




#--- Compare variable importance for train and test (hints to variables prone to overfitting) -------------------------

df.tmp = df.varimp[c("variable","importance")] %>% rename("train" = "importance") %>% 
  left_join(df.varimp_train[c("variable","importance")] %>% rename("test" = "importance"), by = "variable") %>% 
  gather(key = type, value = importance, train, test) %>% 
  filter(variable %in% topn_vars)
  
ggplot(df.tmp, aes(variable, importance)) +
  geom_bar(aes(fill = type), position = "dodge", stat = "identity") +
  #scale_x_discrete(limits = rev(df.varimp$variable)) +
  scale_fill_discrete(limits = c("train","test")) +
  coord_flip()



#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-


## Partial depdendance for "total" fit 
# Get "x-axis" points
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.05)))
df.partialdep = get_partialdep(df.test, fit, b_sample = b_sample, b_all = b_all,
                               vars = topn_vars, levs = levs, quantiles = quantiles)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(0,0.3), colors = twocol)
ggsave(paste0(plotloc,"census_partial_dependence.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2), 
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
                             vars = topn_vars, levs = levs, quantiles = quantiles) %>% 
                mutate(run = i))
}



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(0,0.2), colors = twocol)
ggsave(paste0(plotloc,"census_partial_dependence.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2), 
       w = 18, h = 12)




#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-

# Make yhat_test a vector
yhat_explain = yhat_test[,2]

# Derive id
df.test$id = 1:nrow(df.test)

# Subset test data (explanations are only calcualted for this subset)
i.top = order(yhat_explain, decreasing = TRUE)[1:20]
i.bottom = order(yhat_explain)[1:20]
i.random = sample(1:length(yhat_explain), 20)
i.explain = sample(unique(c(i.top, i.bottom, i.random)))
df.test_explain = df.test[i.explain, c("id", features)]

# Get explanations
tmp = Sys.time()
df.explanations = get_explanations(type = "class", b_sample = b_sample, b_all = b_all)
Sys.time() - tmp


## Plot
df.values = df.test_explain
df.values[metr] = map(df.values[metr], ~ round(.,2))
plots = get_plot_explanations(df.plot = df.explanations, df.values = df.values, type = "class", topn = 10, 
                              ylim = c(0,1))
ggsave(paste0(plotloc,"census_explanations.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2), 
       w = 18, h = 12)


  
save.image(paste0("census_3_interpret.rdata"))
#load(paste0("census_3_interpret.rdata"))



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

