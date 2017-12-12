
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load("1_explore.rdata")

# Load libraries and functions
source("./code/0_init.R")

# Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster




# Sample data --------------------------------------------------------------------------------------------

# Just take data from train fold
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
fit = train( formula, data = df.train[c("target",predictors)], 
             trControl = trainControl(method = "none", returnData = TRUE, allowParallel = FALSE), 
             method = "xgbTree",
             tuneGrid = expand.grid(nrounds = 400, max_depth = c(6),
                                    eta = c(0.02), gamma = 0, colsample_bytree = c(0.7),
                                    min_child_weight = c(10), subsample = c(0.7)))
Sys.time() - tmp

# Predict
yhat_test = predict(fit, df.test[predictors])
summary(yhat_test)
skip = function() {
  # Predict in parallel (to keep memory consumption small)
  l.split = split(1:nrow(df.test), (1:nrow(df.test)) %/% 50000)
  yhat_test = foreach(i = 1:length(l.split), .combine = bind_rows) %dopar% {
    predict(fit, df.test[l.split[[i]],predictors])
  }
}
y_test = df.test$target

# Plot performance
mysummary_regr(data.frame(yhat = yhat_test, y = y_test))
plots = get_plot_performance_regr(yhat = yhat_test, y = y_test, quantiles = seq(0, 1, 0.05))
ggsave(paste0(plotloc, "performance.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2, top = NULL), 
       w = 18, h = 12)





#---- Check residuals ----------------------------------------------------------------------------------

# Residuals
df.test$residual = y_test - yhat_test
summary(df.test$residual)
plots = c(suppressMessages(get_plot_distr_metr_regr(df.test, metr, target_name = "residual", ylim = c(-3,3))), 
          get_plot_distr_nomi_regr(df.test, nomi, target_name = "residual", ylim = c(-1,1)))
ggsave(paste0(plotloc, "diagnosis_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), width = 18, height = 12)

# Absolute residuals
df.test$abs_residual = abs(y_test - yhat_test)
summary(df.test$abs_residual)
plots = c(suppressMessages(get_plot_distr_metr_regr(df.test, metr, target_name = "abs_residual", ylim = c(0,1))), 
          get_plot_distr_nomi_regr(df.test, nomi, target_name = "abs_residual", ylim = c(0,1)))
ggsave(paste0(plotloc, "diagnosis_absolute_residual.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
       width = 18, height = 12)





#---- Do some bootstrapped fits ----------------------------------------------------------------------------------

n.boot = 5
l.boot = foreach(i = 1:n.boot, .combine = c, .packages = c("caret")) %dopar% { 
  
  # Bootstrap
  set.seed(i*1234)
  i.boot = sample(1:nrow(df.train), replace = TRUE) #resampled bootstrap observations
  i.oob = (1:nrow(df.train))[-unique(i.boot)] #Out-of-bag observations
  df.boot = df.train[i.boot,]
  
  # Fit
  fit = train(formula, data = df.boot[c("target",predictors)], 
              trControl = trainControl(method = "none", returnData = TRUE, allowParallel = FALSE), 
              method = "xgbTree",
              tuneGrid = expand.grid(nrounds = 400, max_depth = c(6),
                                     eta = c(0.02), gamma = 0, colsample_bytree = c(0.7),
                                     min_child_weight = c(10), subsample = c(0.7)))
  yhat = predict(fit, df.train[i.oob,predictors])
  spearman = cor(yhat, df.train[i.oob,]$target, method = "spearman")
  return(setNames(list(fit, spearman), c(paste0("fit_",i), c(paste0("spearman_",i)))))
}

# Is it stable?
map_dbl(1:n.boot, ~ l.boot[[paste0("spearman_",.)]]) #on Out-of-bag data
map_df(1:n.boot, ~ data.frame(t(
  mysummary_regr(data.frame(yhat = predict(l.boot[[paste0("fit_",.)]], df.test[predictors]), 
                            y = df.test$target)))
)) 




#--- Top variable importance model fit -------------------------------------------------------------------

# Variable importance (on train data!)
df.varimp_train = get_varimp_by_permutation(df.train, fit, vars = predictors, metric = metric)
predictors_top = df.varimp_train %>% filter(importance > 10) %>% .$variable
formula_top = as.formula(paste("target", "~", paste(predictors_top, collapse = " + ")))

# Fit again -> possibly tune nrounds again
fit_top = train(formula_top, data = df.train[c("target",predictors_top)], 
                trControl = trainControl(method = "none", returnData = TRUE, allowParallel = FALSE), 
                method = "xgbTree",
                tuneGrid = expand.grid(nrounds = 400, max_depth = c(6),
                                       eta = c(0.02), gamma = 0, colsample_bytree = c(0.7),
                                       min_child_weight = c(10), subsample = c(0.7)))

# Plot performance
tmp = predict(fit_top, df.test[predictors_top])
plots = get_plot_performance_regr(yhat = tmp, y = df.test$target)
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

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(1,4))
ggsave(paste0(plotloc, "partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
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
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(1,4))
ggsave(paste0(plotloc, "partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       w = 18, h = 12)





#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-

## Derive betas for all test cases

# Value data frame
i.top = order(yhat_test, decreasing = TRUE)[1:20]
df.model_test = df.test[i.top,c("target",predictors)]
df.model_test$id = 1:nrow(df.model_test)

# Get model matrix and DMatrix
m.model_train = model.matrix(formula, data = df.train[c("target",predictors)], contrasts = NULL)[,-1]
m.train = xgb.DMatrix(m.model_train)
m.model_test = model.matrix(formula, data = df.model_test, contrasts = NULL)[,-1]
m.test = xgb.DMatrix(m.model_test)

# Create explainer data table from train data
df.explainer = buildExplainer(fit$finalModel, m.train, type = "regression")

# Get predictions for all test data
df.predictions = explainPredictions(fit$finalModel, df.explainer, m.test)
df.predictions$id = 1:nrow(df.predictions)

# Aggregate predictions for all nominal variables
df.predictions = as.data.frame(df.predictions)
for (i in 1:length(fit$xlevels)) {
  #i=1
  varname = names(fit$xlevels)[i]
  levnames = paste0(varname, fit$xlevels[[i]][-1])
  df.predictions[varname] = apply(df.predictions[levnames], 1, function(x) sum(x, na.rm = TRUE))
  df.predictions[levnames] = NULL
}



## Plot
plots = get_plot_explainer(df.plot = df.predictions, df.values = df.model_test, type = "regr", topn = 10, 
                           ylim = c(0,8))
ggsave(paste0(plotloc, "explanations.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
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

