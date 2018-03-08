
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load("regr_1_explore.rdata")

# Load libraries and functions
source("./code/0_init.R")

# Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster


## Tuning parameter to use
tunepar = expand.grid(nrounds = 100, max_depth = 6, 
                      eta = 0.1, gamma = 0, colsample_bytree = 0.7, 
                      min_child_weight = 5, subsample = 0.7)



# Sample data --------------------------------------------------------------------------------------------

df.train = df %>% filter(fold == "train") #%>% sample_n(1000)

# Set metric for peformance comparison
metric = "spearman"

# Define test data
df.test = df %>% filter(fold == "test")




#######################################################################################################################-
#|||| Performance ||||----
#######################################################################################################################-

#---- Do the full fit and predict on test data -------------------------------------------------------------------

# Fit
tmp = Sys.time()
fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[predictors])), df.train$target,
            #fit = train(formula, data = df.train[c("target",predictors)],
            trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
            method = "xgbTree", 
            tuneGrid = tunepar)
Sys.time() - tmp

# Predict
yhat_test = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.test[predictors])))
summary(yhat_test)
# # Scoring in chunks in parallel in case of high memory consumption of xgboost
# l.split = split(df.test[predictors], (1:nrow(df.test)) %/% 50000)
# yhat_test = foreach(df.split = l.split, .combine = c) %dopar% {
#   predict(fit, df.split)
# }
y_test = df.test$target

# Plot performance
mysummary_regr(data.frame(yhat = yhat_test, y = y_test))
plots = get_plot_performance_regr(yhat = yhat_test, y = y_test, quantiles = seq(0, 1, 0.05))
ggsave(paste0(plotloc, "regr_performance.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2, top = NULL), 
       w = 18, h = 12)




#---- Check residuals ----------------------------------------------------------------------------------

# Residuals
df.test$residual = y_test - yhat_test
df.test$abs_residual = abs(df.test$residual)
summary(df.test$residual)
plots = c(suppressMessages(get_plot_distr_metr_regr(df.test, metr, target_name = "residual", ylim = NULL, missinfo = misspct)), 
          get_plot_distr_nomi_regr(df.test, nomi, target_name = "residual", ylim = NULL))
ggsave(paste0(plotloc, "regr_diagnosis_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), width = 18, height = 12)

# Absolute residuals
summary(df.test$abs_residual)
plots = c(suppressMessages(get_plot_distr_metr_regr(df.test, metr, target_name = "abs_residual", ylim = NULL, missinfo = misspct)), 
          get_plot_distr_nomi_regr(df.test, nomi, target_name = "abs_residual", ylim = NULL))
ggsave(paste0(plotloc, "regr_diagnosis_absolute_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
       width = 18, height = 12)





#---- Do some bootstrapped fits ----------------------------------------------------------------------------------
n.boot = 5
l.boot = foreach(i = 1:n.boot, .combine = c, .packages = c("caret","ROCR","xgboost","Matrix")) %dopar% { 
  
  # Bootstrap
  set.seed(i*1234)
  i.boot = sample(1:nrow(df.train), replace = TRUE) #resampled bootstrap observations
  i.oob = (1:nrow(df.train))[-unique(i.boot)] #Out-of-bag observations
  df.boot = df.train[i.boot,]
  
  # Fit
  fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.boot[predictors])), df.boot$target,
              #fit = train(formula, data = df.train[c("target",predictors)],
              trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
              method = "xgbTree", 
              tuneGrid = tunepar)
  yhat = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[i.oob,predictors])))
  perf = as.numeric(mysummary_regr(data.frame(yhat = yhat, y = df.train$target[i.oob]))[metric])
  return(setNames(list(fit, perf), c(paste0("fit_",i), c(paste0(metric,"_",i)))))
}

# Is it stable?
map_dbl(1:n.boot, ~ l.boot[[paste0(metric,"_",.)]]) #on Out-of-bag data
map_df(1:n.boot, ~ {
  yhat = predict(l.boot[[paste0("fit_",.)]], 
                 xgb.DMatrix(sparse.model.matrix(formula_rightside, df.test[predictors])))
  data.frame(t(mysummary_regr(data.frame(yhat = yhat, y = df.test$target)))) 
})




#--- Top variable importance model fit -------------------------------------------------------------------

# Variable importance (on train data!)
df.varimp_train = get_varimp_by_permutation(df.train, fit, vars = predictors, metric = metric)
predictors_top = df.varimp_train %>% filter(importance > 10) %>% .$variable
formula_top = as.formula(paste("target", "~", paste(predictors_top, collapse = " + ")))
formula_top_rightside = as.formula(paste("~", paste(predictors_top, collapse = " + ")))

# Fit again -> possibly tune nrounds again
fit_top = train(xgb.DMatrix(sparse.model.matrix(formula_top_rightside, df.train[predictors_top])), df.train$target,
                trControl = trainControl(method = "none", returnData = FALSE, allowParallel = FALSE), 
                method = "xgbTree", 
                tuneGrid = tunepar)

# Plot performance
tmp = predict(fit_top, 
              xgb.DMatrix(sparse.model.matrix(formula_top_rightside, df.test[predictors_top])))
plots = get_plot_performance_regr(yhat = tmp, y = df.test$target, quantiles = seq(0, 1, 0.05))
plots[1]




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
topn = 20
topn_vars = df.varimp[1:topn, "variable"]

# Add other information (e.g. special coloring): color variable is needed -> fill with "dummy" if it should be ommited
df.varimp %<>% mutate(color = cut(importance, c(-1,10,50,101), labels = c("low","middle","high")))



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

# Plot
plots = get_plot_varimp(df.varimp, topn_vars, df.plot_boot = df.varimp_boot)
ggsave(paste0(plotloc, "regr_variable_importance.pdf"), marrangeGrob(plots, ncol = 2, nrow = 1), w = 12, h = 8)




#--- Compare variable importance for train and test -------------------------------------------------------------------

df.tmp = df.varimp[c("variable","importance")] %>% rename("train" = "importance") %>% 
  left_join(df.varimp_train[c("variable","importance")] %>% rename("test" = "importance"), by = "variable") %>% 
  gather(key = type, value = importance, train, test) 

ggplot(df.tmp, aes(variable, importance)) +
  geom_bar(aes(fill = type), position = "dodge", stat="identity") +
  scale_x_discrete(limits = rev(df.varimp$variable)) +
  scale_fill_discrete(limits = c("train","test")) +
  coord_flip()




#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-

## Partial depdendance for "total" fit 
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.05)))
df.partialdep = get_partialdep(df.test, fit, vars = topn_vars, levs = levs, quantiles = quantiles)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(1.5e5,2.5e5), 
                            ref = mean(df.train$target))
ggsave(paste0(plotloc, "regr_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
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



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(1.5e5,2.5e5),
                            ref = mean(df.train$target))
ggsave(paste0(plotloc, "regr_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       w = 18, h = 12)





#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-

# Derive id
df.test$id = 1:nrow(df.test)


## Derive betas for all test cases

# Value data frame
i.top = order(yhat_test, decreasing = TRUE)[1:20]
i.bottom = order(yhat_test)[1:20]
i.random = sample(1:length(yhat_test), 20)
i.explain = sample(unique(c(i.top, i.bottom, i.random)))
ids = df.test$id[i.explain]

# Get model matrix and DMatrix for train and test (sample) data
m.model_train = sparse.model.matrix(formula_rightside, data = df.train[predictors])
dm.train = xgb.DMatrix(m.model_train) 
m.test_explain = sparse.model.matrix(formula_rightside, data = df.test[i.explain,predictors])
dm.test_explain = xgb.DMatrix(m.test_explain)
df.test_explain = as.data.frame(as.matrix(m.test_explain))

# Create explainer data table from train data
df.explainer = buildExplainer(fit$finalModel, dm.train, type = "regression")

# Get predictions for test data
df.predictions = explainPredictions(fit$finalModel, df.explainer, dm.test_explain)

# Aggregate predictions for all nominal variables
df.predictions = as.data.frame(df.predictions)
df.map = data.frame(varname = predictors[attr(model.matrix(formula_rightside, data = df.train[1,predictors]), "assign")],
                    levname = colnames(m.model_train)[-1])
for (i in 1:length(nomi)) {
  #i=1
  varname = nomi[i]
  levnames = as.character(df.map[df.map$varname == varname,]$levname)
  df.predictions[varname] = apply(df.predictions[levnames], 1, function(x) sum(x, na.rm = TRUE))
  df.predictions[levnames] = NULL
}

# Check
rowSums(df.predictions[,-1])
yhat_test[i.explain]


## Plot
df.test_explain$id = ids 
df.predictions$id = ids  
plots = get_plot_explainer(df.plot = df.predictions, df.values = df.test_explain, type = "regr", topn = 10, 
                           ylim = NULL)
ggsave(paste0(plotloc,"regr_explanations.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2), 
       w = 18, h = 12)



#save.image("regr_3_interpret.rdata")
#load("regr_3_interpret.rdata")



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

