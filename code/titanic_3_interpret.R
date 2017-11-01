
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load data and functions
load("titanic_1_explore.rdata")
source("./code/0_init.R")



#### Initialize parallel processing ####
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster




# Do not undersample data --------------------------------------------------------------------------------------------

# Just take data from train fold
summary(df[df$fold == "train", "target"])
df.train = df %>% filter(fold == "train")
# Define prior base probabilities (needed to correctly switch probabilities of undersampled data)
b_all = mean(df %>% filter(fold == "train") %>% .$target_num)
b_sample = mean(df.train$target_num)

# Define test data
df.test = df %>% filter(fold == "test")

#df.train = droplevels(df.train) # Drop levels
#(onelev = which(sapply(df.train[,predictors], function(x) ifelse(is.factor(x), length(levels(x)), NA)) <= 1))
#predictors = setdiff(predictors, predictors[onelev])




#######################################################################################################################-
#|||| Performance ||||----
#######################################################################################################################-

#---- Do the full fit and predict on test data -------------------------------------------------------------------

## Fit
tmp = Sys.time()
fit = train( formula, data = df.train[c("target",predictors)], 
             trControl = trainControl(method = "none", returnData = TRUE, allowParallel = FALSE), 
             method = "xgbTree",
             tuneGrid = expand.grid(nrounds = 700, max_depth = c(6),
                                    eta = c(0.01), gamma = 0, colsample_bytree = c(0.7),
                                    min_child_weight = c(5), subsample = c(0.7)))
Sys.time() - tmp



## Predict
yhat_test_unscaled = predict(fit, df.test[predictors], type = "prob")[["Y"]]
summary(yhat_test_unscaled)
skip = function() {
  # Predict in parallel (to keep memory consumption small)
  l.split = split(1:nrow(df.test), (1:nrow(df.test)) %/% 50000)
  yhat_test_unscaled = foreach(i = 1:length(l.split), .combine = bind_rows) %dopar% {
    predict(fit, df.test[l.split[[i]],predictors], type="prob")[["Y"]]
  }
}

# Rescale to non-undersampled data
yhat_test = prob_samp2full(yhat_test_unscaled, b_sample, b_all)
y_test = df.test$target



## Plot performance
print(performance(prediction(yhat_test, y_test), "auc" )@y.values[[1]])
plots = get_plot_performance_class(yhat = yhat_test, y = y_test, reduce_factor = NULL)
ggsave(paste0(plotloc, "titanice_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)




#---- Do some bootstrapped fits ----------------------------------------------------------------------------------

n.boot = 20
l.boot = foreach(i = 1:n.boot, .combine = c, .packages = c("caret","ROCR")) %dopar% { 
  
  # Bootstrap
  set.seed(i*1234)
  i.boot = sample(1:nrow(df.train), replace = TRUE) #resampled bootstrap observations
  i.oob = (1:nrow(df.train))[-unique(i.boot)] #Out-of-bag observations
  df.boot = df.train[i.boot,]
  
  # Fit
  fit = train( formula, data = df.boot[c("target",predictors)], 
               trControl = trainControl(method = "none", returnData = TRUE, allowParallel = FALSE), 
               method = "xgbTree",
               tuneGrid = expand.grid(nrounds = 700, max_depth = c(6),
                                      eta = c(0.01), gamma = 0, colsample_bytree = c(0.7),
                                      min_child_weight = c(5), subsample = c(0.7)))
  yhat_unscaled = predict(fit, df.train[i.oob,predictors], type = "prob")[["Y"]]
  auc = performance(prediction(yhat_unscaled, df.train[i.oob, "target"][[1]]), "auc")@y.values[[1]]
  return(setNames(list(fit, auc), c(paste0("fit_",i), c(paste0("auc_",i)))))
}

# Is it stable?
map_dbl(1:n.boot, ~ l.boot[[paste0("auc_",.)]]) #on Out-of-bag data
map_df(1:n.boot, ~ data.frame(t(
  mysummary_class(data.frame(yhat = predict(l.boot[[paste0("fit_",.)]], df.test[predictors], type = "prob")[["Y"]],
                             y = df.test[["target"]]))))) #on test data




#######################################################################################################################-
#|||| Variable Importance ||||----
#######################################################################################################################-

#--- Default Variable Importance: uses gain sum of all trees ---------------------------------------------------------

# Default plot 
plot(varImp(fit)) 




#--- Variable Importance by permuation argument -------------------------------------------------------------------

## Importance for "total" fit
df.varimp = get_varimp_by_permutation(df.test, fit, vars = predictors)

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
    bind_rows(get_varimp_by_permutation(df.test_boot, l.boot[[paste0("fit_",i)]], predictors) %>% 
                mutate(run = i))
}



## Plot
plot = get_plot_varimp(df.varimp, topn_vars, df.plot_boot = df.varimp_boot)
ggsave(paste0(plotloc, "titanic_variable_importance.pdf"), plot, w = 8, h = 6)





#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-


## Partial depdendance for "total" fit 
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.05)))
df.partialdep = get_partialdep(df.test, fit, vars = topn_vars, levs = levs, quantiles = quantiles)

# Rescale
df.partialdep$yhat = prob_samp2full(df.partialdep$yhat, b_sample, b_all)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(0,0.7))
ggsave(paste0(plotloc, "titanic_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)



## Partial dependance for bootstrapped models (and bootstrapped data) 
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

# Rescale
df.partialdep_boot$yhat = prob_samp2full(df.partialdep_boot$yhat, b_sample, b_all)



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(0,0.7))
ggsave(paste0(plotloc, "titanic_partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)


#save.image("3_interpret.rdata")



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
df.explainer = buildExplainer(fit$finalModel, m.train, type = "binary")

# Switch coefficients (as explainer takes "N" as target = 1)
cols = setdiff(colnames(df.explainer), c("leaf","tree"))
df.explainer[, (cols) := lapply(.SD, function(x) -x), .SDcols = cols]

# Get predictions for all test data
df.predictions = explainPredictions(fit$finalModel, df.explainer, m.test)
df.predictions$id = 1:nrow(df.predictions)

# Get value data frame
df.model_test = as.data.frame(m.model_test)
df.model_test$id = 1:nrow(df.model_test)

## Plot
plots = get_plot_explainer(df.plot = df.predictions[1:12,], df.values = df.model_test[1:12,])
ggsave(paste0(plotloc, "titanic_explanations.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)




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

