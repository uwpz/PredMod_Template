
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load(paste0("census_1_explore.rdata"))

# Load libraries and functions
source("./code/0_init.R")

# Adapt some parameter differnt for target types -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
splitrule = switch(TYPE, "class" = "gini", "regr" = "variance", "multiclass" = "gini")
type = switch(TYPE, "class" = "prob", "regr" = "raw", "multiclass" = "prob")
plotloc = paste0(plotloc,TYPE,"/")

# Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster

# Set metric for peformance comparison
metric = "AUC"

  


#######################################################################################################################-
#|||| Test an algorithm (and determine parameter grid) ||||----
#######################################################################################################################-

# Sample data --------------------------------------------------------------------------------------------

# Just take data from train fold (take all but n_maxpersample at most)
summary(df[df$fold == "train", "target"])
c(df.train, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_maxpersample = 1e5)) 
summary(df.train$target); b_sample; b_all




# Define some controls --------------------------------------------------------------------------------------------

set.seed(998)
l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
ctrl_idx_fff = trainControl(method = "cv", number = 1, index = l.index, 
                            returnResamp = "final", returnData = FALSE,
                            summaryFunction = mysummary, classProbs = TRUE,
                            indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
ctrl_idx_nopar_fff = trainControl(method = "cv", number = 1, index = l.index, 
                                  returnResamp = "final", returnData = FALSE,
                                  allowParallel = FALSE, #no parallel e.g. for xgboost on big data or with DMatrix
                                  summaryFunction = mysummary, classProbs = TRUE, 
                                  indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
ctrl_cv_nopar_fff = trainControl(method = "cv", number = 5, 
                                 returnResamp = "final", returnData = FALSE,
                                 allowParallel = FALSE, #no parallel e.g. for xgboost on big data or with DMatrix
                                 summaryFunction = mysummary, classProbs = TRUE,
                                 indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!


# Fits --------------------------------------------------------------------------------------------

## Lasso / Elastic Net
fit = train(sparse.model.matrix(formula_binned_rightside, df.train[features_binned]), df.train$target,
#fit = train(formula_binned, data = df.train[c("target",features_binned)], 
            trControl = ctrl_idx_fff, metric = metric, 
            method = "glmnet", 
            tuneGrid = expand.grid(alpha = 1, lambda = 2^(seq(-3, -15, -2))))
            #preProc = c("center","scale")) #no scaling needed due to dummy coding of all variables 
plot(fit, ylim = c(0.94,0.95))
# -> keep alpha=1 to have a full Lasso



## DeepLearning
fit = train(formula, df.train[c("target",features)],
            #fit = train(formula_binned, data = df.train[c("target",features_binned)], 
            trControl = ctrl_idx_nopar_fff, metric = metric, 
            method = "mlpKerasDecay", 
            tuneGrid = expand.grid(size = c(100), lambda = c(0,0.01),
                                   batch_size = c(10), lr = c(1e-4), 
                                   rho = 0.9, decay = 0, activation = "relu"),
            preProc = c("center","scale"))#,
            #verbose = 0) 
plot(fit)
# -> xxx



## Boosted Trees
tmp = Sys.time()
fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[features])), df.train$target,
#fit = train(formula, data = df.train[c("target",features)],
            trControl = ctrl_idx_nopar_fff, metric = metric, #no parallel for DMatrix
            method = "xgbTree", 
            # tuneGrid = expand.grid(nrounds = seq(100,2100,200), max_depth = c(3,6,12), 
            #                        eta = c(0.05,0.01), gamma = 0, colsample_bytree = c(0.3,0.7), 
            #                        min_child_weight = c(2,10), subsample = c(0.3,0.7)))
            tuneGrid = expand.grid(nrounds = seq(100,2100,200), max_depth = c(6), 
                                               eta = c(0.01), gamma = 0, colsample_bytree = c(0.7), 
                                               min_child_weight = c(2), subsample = c(0.7)))
plot(fit)
print(Sys.time() - tmp)

tmp = Sys.time()
fit = train(df.train[features], df.train$target,
            trControl = ctrl_idx_fff, metric = metric,
            method = ms_boosttree,
            # tuneGrid = expand.grid(numTrees = seq(100,2100,500), numLeaves = c(2,20,128),
            #                        learningRate = c(0.05,0.01), featureFraction = c(0.3,0.7),
            #                        minSplit = c(2,10), exampleFraction = c(0.3,0.7)),
            tuneGrid = expand.grid(numTrees = seq(100,1100,500), numLeaves = 20,
                                   learningRate = c(0.01), featureFraction = c(0.3),
                                   minSplit = c(10), exampleFraction = c(0.7)),
            verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
plot(fit)
print(Sys.time() - tmp)

tmp = Sys.time()
fit = train(sparse.model.matrix(formula_rightside, df.train[features]), df.train$target,
            #fit = train(formula_binned, data = as.data.frame(df.train[c("target",features_binned)]),
            trControl = ctrl_idx_fff, metric = metric,
            method = lgbm,
            # tuneGrid = expand.grid(nrounds = seq(100,2100,200), num_leaves = c(2,20,128),
            #                        learning_rate = c(0.05, 0.01), feature_fraction = c(0.3,0.7),
            #                        min_data_in_leaf = c(2,10), bagging_fraction = c(0.3,0.7)),
            tuneGrid = expand.grid(nrounds = seq(100,1100,200), num_leaves = 20,
                                   learning_rate = c(0.01), feature_fraction = c(0.7),
                                   min_data_in_leaf = c(2), bagging_fraction = c(0.7)),
            #max_depth = 3,
            verbose = 0)
plot(fit)
print(Sys.time() - tmp)
# -> max_depth = 3 / numLeaves = 20, shrinkage = 0.01, colsample_bytree = subsample = 0.7, min_xxx = 5



# DeepLearning
#TODO



## Special plotting
fit
plot(fit)
varImp(fit) 
skip = function() {
  y = metric
  
  # xgboost
  x = "nrounds"; color = "as.factor(max_depth)"; linetype = "as.factor(eta)"; 
  shape = "as.factor(min_child_weight)"; facet = "min_child_weight ~ subsample + colsample_bytree"
  
  # ms_boosttree
  x = "numTrees"; color = "as.factor(numLeaves)"; linetype = "as.factor(learningRate)";
  shape = "as.factor(minSplit)"; facet = "minSplit ~ exampleFraction + featureFraction"
  
  # lgbm
  x = "nrounds"; color = "as.factor(num_leaves)"; linetype = "as.factor(learning_rate)";  
  shape = "as.factor(min_data_in_leaf)"; facet = "min_data_in_leaf ~ bagging_fraction + feature_fraction"
  
  # Plot tuning result with ggplot
  fit$results %>% 
    ggplot(aes_string(x = x, y = y, colour = color, shape = shape)) +
    geom_line(aes_string(linetype = linetype)) +
    geom_point() +
    #geom_errorbar(mapping = aes_string(ymin = "auc - aucSD", ymax = "auc + aucSD", linetype = linetype, width = 100)) +
    facet_grid(as.formula(paste0("~",facet)), labeller = label_both) +
    ylim(c(0.94,0.95))
  
}




#######################################################################################################################-
#|||| Simulation: compare algorithms ||||----
#######################################################################################################################-

# Basic data sampling
df.sim = df #%>% sample_n(1000)

# Tunegrid
tunepar = expand.grid(nrounds = 1500, max_depth = 6, 
                      eta = c(0.01), gamma = 0, colsample_bytree = c(0.7), 
                      min_child_weight = c(2), subsample = c(0.7))




#---- Simulation function ---------------------------------------------------------------------------------------

perfcomp = function(method, nsim = 5) { 
  
  result = foreach(sim = 1:nsim, .combine = bind_rows, .packages = c("caret","Matrix","xgboost"), 
                   .export = c("df.sim","mysummary","metric",
                               "features_binned","features",
                               "formula","formula_binned","formula_rightside","formula_binned_rightside",
                               "tunepar")) %dopar% 
  {
    
    # Hold out a k*100% set
    set.seed(sim*999)
    k = 0.2
    i.test = sample(1:nrow(df.sim), floor(k*nrow(df.sim)))
    df.test = df.sim[i.test,]
    df.train = df.sim[-i.test,]    
    
    # Control for train
    set.seed(999)
    l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
    ctrl_idx = trainControl(method = "cv", number = 1, index = l.index, 
                            returnResamp = "final", returnData = FALSE,
                            summaryFunction = mysummary, classProbs = TRUE)
    ctrl_idx_nopar = trainControl(method = "cv", number = 1, index = l.index, 
                          returnResamp = "final", returnData = FALSE,
                          allowParallel = FALSE,
                          summaryFunction = mysummary, classProbs = TRUE)
    
    
    ## Fit data
    fit = NULL
    
    if (method == "glmnet") {  
      fit = train(sparse.model.matrix(formula_binned_rightside, df.train[features_binned]), df.train$target,
      #fit = train(formula_binned, data = df.train[c("target",features_binned)], 
                  trControl = ctrl_idx, 
                  metric = metric, 
                  method = "glmnet", 
                  tuneGrid = expand.grid(alpha = 0.2, lambda = 2^(seq(-5, -10, -1))))
    }     
    

    if (method == "xgbTree") { 
      fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[features])), df.train$target,
      #fit = train(formula, data = df.train[c("target",features)],
                  trControl = ctrl_idx_nopar, 
                  metric = metric, 
                  method = "xgbTree", 
                  tuneGrid = tunepar)
    }

    
    
    ## Get metrics
    
    # Calculate test performance
    if (method %in% c("glmnet")) {
      yhat_test = predict(fit, sparse.model.matrix(formula_binned_rightside, df.test[features_binned]),
                             type = "prob") 
    } else if (method == "xgbTree") {
      yhat_test = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.test[features])),
                             type = "prob")
    } else  {
      yhat_test = predict(fit, df.test[features], type = "prob") 
    }
    yhat = yhat_test #needed for multiclass
    perf_test = mysummary(data.frame(y = df.test$target, yhat))

    # Return result
    data.frame(sim = sim, method = method, t(perf_test))
  }   
  result
}




#---- Simulate --------------------------------------------------------------------------------------------

df.sim_result = as.data.frame(c())
nsim = 5
df.sim_result = bind_rows(df.sim_result, perfcomp(method = "glmnet", nsim = nsim) )   
df.sim_result = bind_rows(df.sim_result, perfcomp(method = "xgbTree", nsim = nsim))       
#df.sim_result = bind_rows(df.sim_result, perfcomp(method = "deepLearning", nsim = nsim))       
df.sim_result$sim = as.factor(df.sim_result$sim)
df.sim_result$method = factor(df.sim_result$method, levels = unique(df.sim_result$method))




#---- Plot simulation --------------------------------------------------------------------------------------------

p = ggplot(df.sim_result, aes_string(x = "method", y = metric)) + 
  geom_boxplot() + 
  geom_point(aes(color = sim), shape = 15) +
  geom_line(aes(color = sim, group = sim), linetype = 2) +
  scale_x_discrete(limits = rev(levels(df.sim_result$method))) +
  coord_flip() +
  labs(title = "Model Comparison") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5))
p  
ggsave(paste0(plotloc,"census_model_comparison.pdf"), p, width = 12, height = 8)




#######################################################################################################################-
#|||| Learning curve for winner algorithm ||||----
#######################################################################################################################-

# Basic data sampling
df.lc = df #%>% sample_n(1000)

# Tunegrid
tunepar = expand.grid(nrounds = seq(100,2100,200), max_depth = 3, 
                      eta = c(0.01), gamma = 0, colsample_bytree = c(0.7), 
                      min_child_weight = c(2), subsample = c(0.7))



#---- Loop over training chunks --------------------------------------------------------------------------------------

chunks_pct = c(seq(10,10,1), seq(20,100,10))
to = length(chunks_pct)

df.lc_result = foreach(i = 1:to, .combine = bind_rows, 
                       .packages = c("zeallot","dplyr","caret","Matrix","xgboost")) %do%  #NO dopar for xgboost!
{ 
  #i = 1
  
  ## Sample
  set.seed(chunks_pct[i])
  #df.train = df.lc %>% filter(fold == "train") %>% sample_frac(chunks_pct[i]/100)
  #b_sample = b_all = df.train$target %>% (function(.) {summary(.)/length(.)})
  # Balanced ("as long as possible")
  c(df.train, b_sample, b_all) %<-%
      undersample_n(df.lc %>% filter(fold == "train"),
                    n_maxpersample = chunks_pct[i]/100 * max(summary(df.lc[df.lc$fold == "train",][["target"]])))
  df.test = df.lc %>% filter(fold == "test") #%>% sample_n(500)
  
  
  ## Fit on chunk
  set.seed(1234)
  l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
  ctrl_idx_nopar = trainControl(method = "cv", number = 1, index = l.index, 
                                returnResamp = "final", returnData = FALSE,
                                allowParallel = FALSE,
                                summaryFunction = mysummary, classProbs = TRUE)
  ctrl_none = trainControl(method = "none", returnData = FALSE, classProbs = TRUE)
  tmp = Sys.time()
  fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[features])), 
              df.train$target,
              trControl = ctrl_idx_nopar, 
              metric = metric, #no parallel for DMatrix
              method = "xgbTree", 
              tuneGrid = tunepar)
  print(Sys.time() - tmp)
  
  
  
  ## Score (needs rescale to prior probs)
  # Train data 
  y_train = df.train$target
  yhat_train_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[features])),
                                type = "prob")
  yhat_train = scale_pred(yhat_train_unscaled, b_sample, b_all)

  # Test data 
  y_test = df.test$target
  yhat_test_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.test)), type = "prob")
  # # Scoring in chunks in parallel in case of high memory consumption of xgboost
  # l.split = split(df.test[features], (1:nrow(df.test)) %/% 50000)
  # yhat_test_unscaled = foreach(df.split = l.split, .combine = bind_rows) %dopar% {
  #   predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.split)), type = "prob")
  # }
  yhat_test = scale_pred(yhat_test_unscaled, b_sample, b_all)

  # Bind together
  res = rbind(cbind(data.frame("fold" = "train", "numtrainobs" = nrow(df.train)), bestTune = fit$bestTune$nrounds,
                    t(mysummary(data.frame(y = y_train, yhat = yhat_train)))),
              cbind(data.frame("fold" = "test", "numtrainobs" = nrow(df.train)), bestTune = fit$bestTune$nrounds,
                    t(mysummary(data.frame(y = y_test, yhat = yhat_test)))))
  
  
  ## Garbage collection and output
  gc()
  res
}
#save(df.lc_result, file = "df.lc_result.RData")




#---- Plot results --------------------------------------------------------------------------------------

p = ggplot(df.lc_result, aes_string("numtrainobs", metric, color = "fold")) +
  geom_line() +
  geom_point() +
  scale_color_manual(values = c("#F8766D", "#00BFC4")) 
p
ggsave(paste0(plotloc,"census_learningCurve.pdf"), p, width = 8, height = 6)




