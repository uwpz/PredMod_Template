
# Set target type -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
rm(list = ls())
TYPE = "CLASS"
#TYPE = "REGR"
TYPE = "MULTICLASS"




#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load(paste0(TYPE,"_1_explore.rdata"))

# Load libraries and functions
source("./code/0_init.R")

# Adapt some default parameter different for target types -> probably also different for a new use-case
pred_type = switch(TYPE, "CLASS" = "prob", "REGR" = "raw", "MULTICLASS" = "prob") #do not change this one
color = switch(TYPE, "CLASS" = twocol, "REGR" = twocol, "MULTICLASS" = threecol) #probably need to change multiclass opt
ylim_res = switch(TYPE, "CLASS" = c(0,1), "REGR"  = c(-5e4,5e4), "MULTICLASS" = c(0,1))
ylim_pd = switch(TYPE, "CLASS" = c(0.2,0.7), "REGR"  = c(1.5e5,2.5e5), "MULTICLASS" = c(0,1)) #need to adapt
ylim_expl = switch(TYPE, "CLASS" = c(0,1), "REGR"  = c(0,6e5), "MULTICLASS" = c(0,1))
topn = switch(TYPE, "CLASS" = 8, "REGR" = 20, "MULTICLASS" = 20) #remove here and set hard below
b_all = b_sample = NULL #do not change this one (as it is default in regression case)
id_name = switch(TYPE, "CLASS" = "id", "REGR"  = "PID", "MULTICLASS" = "PID")



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
  c(df.train, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_maxpersample = 3e6)) 
  summary(df.train$target); b_sample; b_all
  #df.train = bind_rows(df.train, df.train %>% filter(target == "Cat_2"))
  #b_sample = df.train$target %>% (function(.) summary(.)/length(.))
  
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
m.train = sparse.model.matrix(formula, data = df.train[c("target",features)])
fit = train(xgb.DMatrix(m.train), df.train$target,
#fit = train(formula, data = df.train[c("target",features)],
            trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
            method = "xgbTree", 
            tuneGrid = tunepar)
Sys.time() - tmp

# Predict
m.test = sparse.model.matrix(formula, df.test[c("target",features)])
yhat_test = predict(fit, xgb.DMatrix(m.test), type = pred_type) %>% 
  scale_pred(b_sample, b_all) #rescale
# # Scoring in chunks in parallel in case of high memory consumption of xgboost
# l.split = split(df.test[c("target",features)], (1:nrow(df.test)) %/% 50)
# yhat_test = foreach(df.split = l.split, .combine = bind_rows, .packages = c("Matrix","xgboost")) %dopar% {
#   predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.split)), type = pred_type)
# } %>% 
# scale_pred(b_sample, b_all)
y_test = df.test$target

# Plot performance
mysummary(data.frame(y = y_test, yhat = yhat_test))
plots = get_plot_performance(yhat = yhat_test, y = y_test, reduce_factor = NULL, colors = color)
ggsave(paste0(plotloc,TYPE,"_performance.pdf"), marrangeGrob(plots, ncol = length(plots)/2, nrow = 2, top = NULL), 
       w = 18, h = 12)

# Training performance (to estimate generalization gap)
yhat_train = predict(fit, xgb.DMatrix(m.train), type = pred_type) %>% 
  scale_pred(b_sample, b_all) #rescale
mysummary(data.frame(y = df.train$target, yhat = yhat_train))



#---- Check residuals ----------------------------------------------------------------------------------

## Residuals 
if (TYPE == "CLASS") {
  df.test$yhat = yhat_test[,2]
  df.test$residual = df.test$target_num - df.test$yhat
}
if (TYPE == "MULTICLASS")  {
  # Dynamic reference member per obs, i.e. the true label gets a "1" and the residual is 1-predicted_prob ...
  # ... in case of binary classifiaction this is equal to the absolute residual
  df.test$yhat = rowSums(yhat_test * model.matrix(~ -1 + y_test, data.frame(y_test)))
  df.test$residual = 1 - df.test$yhat
}
if (TYPE == "REGR") {
  df.test$yhat = yhat_test
  df.test$residual = df.test$target - df.test$yhat
}
summary(df.test$residual)
df.test$abs_residual = abs(df.test$residual)


# For non-regr tasks one might want to plot the following for each target level (df.test %>% filter(target == "level"))
plots = c(suppressMessages(get_plot_distr_metr(df.test, metr, target_name = "residual", ylim = ylim_res, 
                                               missinfo = misspct, color = hexcol)), 
          suppressMessages(get_plot_distr_nomi(df.test, nomi, target_name = "residual", ylim = ylim_res,
                                               color = color)))
ggsave(paste0(plotloc,TYPE,"_diagnosis_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), width = 18, height = 12)



## Absolute residuals
if (TYPE %in% c("CLASS","REGR")) {
  summary(df.test$abs_residual)
  plots = c(suppressMessages(get_plot_distr_metr(df.test, metr, target_name = "abs_residual", ylim = c(0,ylim_res[2]), 
                                                 missinfo = misspct, color = hexcol)), 
            suppressMessages(get_plot_distr_nomi(df.test, nomi, target_name = "abs_residual", ylim = c(0,ylim_res[2]),
                                                 color = color)))
  ggsave(paste0(plotloc,TYPE,"_diagnosis_absolute_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
         width = 18, height = 12)
}




#---- Explain bad predictions ----------------------------------------------------------------------------------

## Get explainer data
df.explainer = get_explainer()

## Get n_worst most false predicted ids
n_worst = 30
df.test_explain = df.test %>% arrange(desc(residual)) %>% top_n(n_worst, abs_residual) 
yhat_explain = scale_pred(predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.test_explain[c("target",features)])),
                                  type = pred_type),
                          b_sample, b_all)
if (TYPE == "CLASS") yhat_explain = yhat_explain[,2]



## Get explanations
df.explanations = get_explanations(fit.for_explain = fit, feature_names = features,
                                   df.test_explain = df.test_explain, id_name = id_name, yhat_explain = yhat_explain,
                                   df.explainer = df.explainer,
                                   b_sample = b_sample, b_all = b_all)

# For Multiclass target take only explanations for target (and not other levels)
if (TYPE == "multiclass") {
  df.explanations = df.explanations %>% inner_join(df.test_explain %>% select_("target", id_name, "yhat"))
}



## Plot
# Create titles
df.title = df.test_explain %>% select_("target", id_name, "yhat") %>% 
  mutate(title = paste0(" (y = ",target,", ","y_hat = ", round(yhat, 2), ")"))

plots = get_plot_explanations(df.plot = df.explanations,
                              df.title = df.title,
                              id_name = id_name,
                              scale_target = tolower(TYPE), topn = topn, ylim = ylim_expl)
ggsave(paste0(plotloc,TYPE,"_fails.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2),
       w = 18, h = 12)



#---- Check performance for crossvalidated fits ---------------------------------------------------------------------

# CV on all or only on train data?
df.cv = df

# CV strategy
n.folds = 5
n.repeat = 1
l.folds = createMultiFolds(1:nrow(df.cv), k = n.folds, times = n.repeat)

## Fit
l.cv = foreach(i = 1:length(l.folds), .combine = c, .packages = c("caret","ROCR","xgboost","Matrix","purrr")) %dopar% { 

  # Fit
  fit = train(xgb.DMatrix(sparse.model.matrix(formula, df.cv[l.folds[[i]],c("target",features)])), 
              df.cv$target[l.folds[[i]]],
              #fit = train(formula, data = df.cv[c("target",features)],
              trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
              method = "xgbTree", 
              tuneGrid = tunepar)
  yhat = predict(fit, 
                 xgb.DMatrix(sparse.model.matrix(formula, df.cv[-l.folds[[i]],c("target",features)])), 
                 type = pred_type) #do NOT rescale on OOB data
  perf = as.numeric(mysummary(data.frame(yhat = yhat, y = df.cv$target[-l.folds[[i]]]))[metric])
  return(setNames(list(fit, perf), c(paste0("fit_",i), c(paste0(metric,"_",i)))))
}
# Performance 
unlist(l.cv[grep("AUC",names(l.cv))])

# Copy for later usage
l.fits = l.cv[grep("fit",names(l.cv))]





#--- Most important variables (importance > 10) model fit -------------------------------------------------------------------

# Variable importance (on train data!)
df.varimp_train = get_varimp_by_permutation(df.train, fit, sparse = TRUE,
                                            b_sample = b_sample, b_all = b_all, vars = features, metric = metric)
features_top = df.varimp_train %>% filter(importance_cum < 90) %>% .$variable
formula_top = as.formula(paste("target", "~ -1 + ", paste(features_top, collapse = " + ")))


# Fit again -> possibly tune nrounds again
fit_top = train(xgb.DMatrix(sparse.model.matrix(formula_top, df.train[c("target",features_top)])), df.train$target,
                trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
                method = "xgbTree", 
                tuneGrid = tunepar)

# Plot performance
yhat_top = predict(fit_top, 
                   xgb.DMatrix(sparse.model.matrix(formula_top, df.test[c("target",features_top)])),
                   type = pred_type) %>% 
  scale_pred(b_sample, b_all)
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
df.varimp = get_varimp_by_permutation(df.test, fit, sparse = TRUE, feature_names = features,
                                      b_sample = b_sample, b_all = b_all, vars = features, metric = metric)

# Visual check how many variables needed 
ggplot(df.varimp) + 
  geom_bar(aes(x = reorder(variable, importance), y = importance), stat = "identity") +
  coord_flip() 
topn = topn
topn_vars = df.varimp[1:topn, "variable"]

# Add other information (e.g. special coloring): color variable is needed -> fill with "dummy" if it should be ommited
df.varimp %<>% mutate(color = cut(importance, c(-1,10,50,101), labels = c("low","middle","high")))



## Crossvalidate Importance: ONLY for topn_vars
# Get boostrap values
df.varimp_cv = c()
for (i in 1:length(l.folds)) {
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.varimp_cv %<>% 
    bind_rows(get_varimp_by_permutation(df.cv[-l.folds[[i]],], l.fits[[i]], sparse = TRUE, 
                                        feature_names = features,
                                        b_sample = b_sample, b_all = b_all,
                                        vars = topn_vars, metric = metric) %>% 
                mutate(fold = i))
}

# Plot
plots = get_plot_varimp(df.varimp, topn_vars, df.plot_cv = df.varimp_cv, run_name = "fold")
ggsave(paste0(plotloc,TYPE,"_variable_importance.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1), w = 12, h = 8)




#--- Compare variable importance for train and test (hints to variables prone to overfitting) -------------------------

plots = map(c("importance","importance_sumnormed"), ~ {
  df.tmp = df.varimp %>% select_("variable",.x) %>% mutate(type = "test") %>% 
    bind_rows(df.varimp_train %>% select_("variable",.x) %>% mutate(type = "train")) %>% 
    filter(variable %in% topn_vars)
  ggplot(df.tmp, aes_string("variable", .x)) +
          geom_bar(aes(fill = type), position = "dodge", stat = "identity") +
          scale_fill_discrete(limits = c("train","test")) +
          coord_flip() +
          labs(title = .x)
})
ggsave(paste0(plotloc,TYPE,"_variable_importance_comparison.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1), w = 12, h = 8)





#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-

## Partial depdendance for "total" fit 
# Get "x-axis" points
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.1)))
df.partialdep = get_partialdep(df.test, fit, b_sample = b_sample, b_all = b_all,
                               vars = topn_vars, l.levs = levs, l.quantiles = quantiles)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = ylim_pd, colors = color)
ggsave(paste0(plotloc,TYPE,"_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       w = 18, h = 12)



## Partial dependance cv models  
# Get boostrap values
df.partialdep_cv = c()
for (i in 1:length(l.folds)) {
  df.partialdep_cv %<>% 
    bind_rows(get_partialdep(df.cv[-l.folds[[i]],], l.fits[[i]], b_sample = b_sample, b_all = b_all,
                             vars = topn_vars, l.levs = levs, l.quantiles = quantiles) %>% 
                mutate(run = i))
}



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_cv = df.partialdep_cv, ylim = ylim_pd, colors = color)
ggsave(paste0(plotloc,TYPE,"_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       w = 18, h = 12)




#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-

# Subset test data (explanations are only calculated for this subset)
i.top = order(df.test$yhat, decreasing = TRUE)[1:20]
i.bottom = order(df.test$yhat)[1:20]
i.random = sample(1:length(df.test$yhat), 20)
i.explain = sample(unique(c(i.top, i.bottom, i.random)))
df.test_explain = df.test[i.explain,]
yhat_explain = scale_pred(predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.test_explain[c("target",features)])),
                                  type = pred_type),
                          b_sample, b_all)
if (TYPE == "CLASS") yhat_explain = yhat_explain[,2]

# Get explanations
df.explanations = get_explanations(fit.for_explain = fit, feature_names = features,
                                   df.test_explain = df.test_explain, id_name = id_name, yhat_explain = yhat_explain,
                                   df.explainer = df.explainer,
                                   b_sample = b_sample, b_all = b_all)

# For Multiclass target take only explanations for target (and not other levels)
if (TYPE == "multiclass") {
  df.explanations = df.explanations %>% inner_join(df.test_explain %>% select_("target", id_name, "yhat"))
}

# Create titles
df.title = df.test_explain %>% select_("target", id_name, "yhat") %>% 
  mutate(title = paste0(" (y = ",target,", ","y_hat = ", round(yhat, 2), ")"))

# Plot
plots = get_plot_explanations(df.plot = df.explanations,
                              df.title = df.title,
                              id_name = id_name,
                              scale_target = tolower(TYPE), topn = topn, ylim = ylim_expl)
ggsave(paste0(plotloc,TYPE,"_explanations.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2), 
       w = 18, h = 12)





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

