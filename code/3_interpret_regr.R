
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load data and functions
load("1_explore_regr.rdata")
source("./code/0_init.R")



#### Initialize parallel processing ####
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster




# Sample data --------------------------------------------------------------------------------------------

# Just take data from train fold
summary(df[df$fold == "train", "target"])
df.train = df %>% filter(fold == "train")
summary(df.train$target)

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



## Plot performance
cor(yhat_test, y_test, method = "spearman")
plots = get_plot_performance_regr(yhat = yhat_test, y = y_test, quantiles = seq(0, 1, 0.05))
ggsave(paste0(plotloc, "performance_regr.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2, top = NULL), 
       w = 12, h = 8)




#---- Do some bootstrapped fits ----------------------------------------------------------------------------------

n.boot = 5
l.boot = foreach(i = 1:n.boot, .combine = c, .packages = c("caret")) %dopar% { 
  
  # Bootstrap
  set.seed(i*1234)
  i.boot = sample(1:nrow(df.train), replace = TRUE) #resampled bootstrap observations
  i.oob = (1:nrow(df.train))[-unique(i.boot)] #Out-of-bag observations
  df.boot = df.train[i.boot,]
  
  # Fit
  fit = train( formula, data = df.boot[c("target",predictors)], 
               trControl = trainControl(method = "none", returnData = TRUE, allowParallel = FALSE), 
               method = "xgbTree",
               tuneGrid = expand.grid(nrounds = 400, max_depth = c(6),
                                      eta = c(0.02), gamma = 0, colsample_bytree = c(0.7),
                                      min_child_weight = c(10), subsample = c(0.7)))
  yhat = predict(fit, df.train[i.oob,predictors])
  spearman = cor(yhat, df.train[i.oob, "target"][[1]], method = "spearman")
  return(setNames(list(fit, spearman), c(paste0("fit_",i), c(paste0("spearman_",i)))))
}

# Is it stable?
map_dbl(1:n.boot, ~ l.boot[[paste0("spearman_",.)]]) #on Out-of-bag data
map_dbl(1:n.boot, ~ cor(predict(l.boot[[paste0("fit_",.)]], df.test[predictors]), df.test[["target"]], 
        method = "spearman")) #on test data
       




#######################################################################################################################-
#|||| Variable Importance ||||----
#######################################################################################################################-

#--- Default Variable Importance: uses gain sum of all trees ---------------------------------------------------------

# Default plot 
plot(varImp(fit)) 




#--- Variable Importance by permuation argument -------------------------------------------------------------------

## Importance for "total" fit
df.varimp = get_varimp_by_permutation_regr(df.test, fit, predictors, target_name = "target", vars = predictors)

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
    bind_rows(get_varimp_by_permutation_regr(df.test_boot, l.boot[[paste0("fit_",i)]], predictors, 
                                             target_name = "target", vars = topn_vars) %>% 
                mutate(run = i))
}



## Plot
plot = get_plot_varimp(df.varimp, topn_vars, df.plot_boot = df.varimp_boot)
ggsave(paste0(plotloc, "variable_importance_regr.pdf"), plot, w = 8, h = 6)





#######################################################################################################################-
#|||| Partial Dependance ||||----
#######################################################################################################################-


## Partial depdendance for "total" fit 
levs = map(df.test[nomi], ~ levels(.))
quantiles = map(df.test[metr], ~ quantile(., na.rm = TRUE, probs = seq(0,1,0.05)))
df.partialdep = get_partialdep(df.test, fit, predictors, "target", topn_vars, levs = levs, quantiles = quantiles)

# Visual check whether all fits 
plots = get_plot_partialdep(df.partialdep, topn_vars, df.for_partialdep = df.test, ylim = c(1,3))
ggsave(paste0(plotloc, "partial_dependence_regr.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)



## Importance for bootstrapped models (and bootstrapped data) 
# Get boostrap values
df.partialdep_boot = c()
for (i in 1:n.boot) {
  set.seed(i*1234)
  df.test_boot = df.test[sample(1:nrow(df.test), replace = TRUE),]  
  #df.test_boot = df.test
  df.partialdep_boot %<>% 
    bind_rows(get_partialdep(df.test_boot, l.boot[[paste0("fit_",i)]], predictors, "target", 
                             topn_vars, levs, quantiles) %>% 
                mutate(run = i))
}



## Plot
plots = get_plot_partialdep(df.partialdep, topn_vars, df.plot_boot = df.partialdep_boot, ylim = c(1,3))
ggsave(paste0(plotloc, "partial_dependence_regr.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)





#######################################################################################################################-
#|||| xgboost Explainer ||||----
#######################################################################################################################-


library(xgboostExplainer)
## Explainer
m.tmp = model.matrix(formula, data = df.train[c("target",predictors)], contrasts = NULL)[, -1]
m.train = xgb.DMatrix(m.tmp)

my_buildExplainer = function (xgb.model, trainingData, type = "binary", base_score = 0.5) 
{
  col_names = attr(trainingData, ".Dimnames")[[2]]
  best_ntreelimit <- xgb.model$best_ntreelimit
  if (!is.null(xgb.model$best_ntreelimit)) {
    best_ntreelimit <- xgb.model$best_ntreelimit - 1
  }
  cat("\nCreating the trees of the xgboost model...")
  #trees = xgb.model.dt.tree(col_names, model = xgb.model, trees = best_ntreelimit)
  trees = xgb.model.dt.tree(col_names, model = xgb.model, n_first_tree = best_ntreelimit)
  cat("\nGetting the leaf nodes for the training set observations...")
  nodes.train = predict(xgb.model, trainingData, predleaf = TRUE)
  cat("\nBuilding the Explainer...")
  cat("\nSTEP 1 of 2")
  tree_list = xgboostExplainer:::getStatsForTrees(trees, nodes.train, type = type, 
                                                  base_score = base_score)
  cat("\n\nSTEP 2 of 2")
  explainer = xgboostExplainer:::buildExplainerFromTreeList(tree_list, col_names)
  cat("\n\nDONE!\n")
  return(explainer)
}
df.explainer = my_buildExplainer(fit$finalModel, m.train, type="binary")
cols = setdiff(colnames(df.explainer), c("leaf","tree"))
df.explainer = as.data.frame(df.explainer)
df.explainer[cols] = map_df(df.explainer[cols], ~ {-1*.})
df.explainer = as.data.table(df.explainer)
df.predictions = explainPredictions(fit$finalModel, df.explainer, m.train)

weights = rowSums(df.predictions)
pred.xgb = 1/(1+exp(-weights))
#cat(max(xgb.preds-pred.xgb),'\n')

#idx_to_get = as.integer(802)
#test[idx_to_get,-"left"]
a = showWaterfall(fit$finalModel, df.explainer, m.train, m.tmp, 5, type = "binary")

predict(fit, df.train, type = "prob")[1:10,2]
1-predict(fit$finalModel, m.train)[1:10]



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

