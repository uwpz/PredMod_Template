
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load data and functions
load("1_explore.rdata")
source("./code/0_init.R")



#### Initialize parallel processing ####
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster




# Undersample data --------------------------------------------------------------------------------------------

# Just take data from train fold
summary(df[df$fold == "train", "target"])
df.train = c()
for (i in 1:2) {
  i.samp = which(df$fold == "train" & df$target == levels(df$target)[i])
  set.seed(i*123)
  df.train = bind_rows(df.train, df[sample(i.samp, min(1000, length(i.samp))),]) #take all but 1000 at most
}
summary(df.train$target)

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
             tuneGrid = expand.grid(nrounds = 400, max_depth = c(6),
                                    eta = c(0.02), gamma = 0, colsample_bytree = c(0.7),
                                    min_child_weight = c(10), subsample = c(0.7)))
Sys.time() - tmp



## Predict
yhat_test_unscaled = predict(fit, df.test[predictors], type = "prob")[["Y"]]
summary(yhat_test_unscaled)
skip = function() {
  # Predict in parallel (to keep memory consumption small)
  l.split = split(1:nrow(df.test), (1:nrow(df.test)) %/% 50000)
  yhat_test_unscaled = foreach(i = 1:length(l.split), .combine = bind_rows) %dopar% {
    predict(fit, df.test[l.split[[i]],predictors], type="prob")
  }
}

# Rescale to non-undersampled data
yhat_test = prob_samp2full(yhat_test_unscaled, b_sample, b_all)
y_test = df.test$target



## Plot performance
print(performance(prediction(yhat_test, y_test), "auc" )@y.values[[1]])
plots = get_plot_performance(yhat = yhat_test, y = y_test, reduce = NULL)
ggsave(paste0(plotloc, "performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)




#---- Do some bootstrapped fits ----------------------------------------------------------------------------------

n.boot = 5
l.boot = foreach(i = 1:n.boot, .combine = c, .packages = c("caret","ROCR")) %dopar% { 
  
  # Bootstrap
  set.seed(i*1234)
  i.boot = sample(1:nrow(df.train), replace = TRUE) #resampled bootstrap observations
  i.oob = (1:nrow(df.train))[-unique(i.boot)] #Out-of-bag observations
  df.boot = df.train[i.boot,]
  
  # Fit
  fit = train( formula, data = df.boot[c("target",predictors)], 
               trControl = ctrl_none, 
               method = "xgbTree",
               tuneGrid = expand.grid(nrounds = 400, max_depth = c(6),
                                      eta = c(0.02), gamma = 0, colsample_bytree = c(0.7),
                                      min_child_weight = c(10), subsample = c(0.7)))
  yhat_unscaled = predict(fit, df.train[i.oob,predictors], type = "prob")[["Y"]]
  auc = performance(prediction(yhat_unscaled, df.train[i.oob, "target"][[1]]), "auc")@y.values[[1]]
  return(setNames(list(fit, auc), c(paste0("fit_",i), c(paste0("auc_",i)))))
}

# Is it stable?
map_dbl(1:n.boot, ~ l.boot[[paste0("auc_",.)]]) #on Out-of-bag data
map_dbl(1:n.boot, ~ performance(prediction(predict( l.boot[[paste0("fit_",.)]], df.test[predictors], type = "prob")[["Y"]],
                                          df.test[["target"]]), "auc")@y.values[[1]]) #on test data
       




#######################################################################################################################-
#|||| Variable Importance ||||----
#######################################################################################################################-

#--- Default Variable Importance: uses gain sum of all trees ---------------------------------------------------------

# Default plot 
plot(varImp(fit)) 




#--- Variable Importance by permuation argument -------------------------------------------------------------------

## Importance for "total" fit
df.varimp = get_varimp_by_permutation(df.test, fit, predictors, vars = predictors)

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
    bind_rows(get_varimp_by_permutation(df.test_boot, l.boot[[paste0("fit_",i)]], predictors, topn_vars) %>% 
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
df.partialdep = get_partialdep(df.test, fit, predictors, topn_vars, levs = levs, quantiles = quantiles)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(0,0.2))
ggsave(paste0(plotloc, "partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)



## Importance for bootstrapped models (and bootstrapped data) 
# Get boostrap values
df.partialdep_boot = c()
for (i in 1:n.boot) {
  set.seed(i*1234)
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.partialdep_boot %<>% 
    bind_rows(get_partialdep(df.test_boot, l.boot[[paste0("fit_",i)]], predictors, topn_vars, levs, quantiles) %>% 
                mutate(run = i))
}



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(0,0.2))
ggsave(paste0(plotloc, "partial_dependence.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
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

