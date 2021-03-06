
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load("class_1_explore.rdata")

# Load libraries and functions
source("./code/0_init.R")

# Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster




# Undersample data --------------------------------------------------------------------------------------------

# Just take data from train fold
n_maxpersample = 1000 #Take all but n_maxpersample at most
summary(df[df$fold == "train", "target"])
df.train = c()
for (i in 1:2) {
  i.samp = which(df$fold == "train" & df$target == levels(df$target)[i])
  set.seed(i*123)
  df.train = bind_rows(df.train, df[sample(i.samp, min(n_maxpersample, length(i.samp))),]) 
}
summary(df.train$target)

# Define prior base probabilities (needed to correctly switch probabilities of undersampled data)
b_all = mean(df %>% filter(fold == "train") %>% .$target_num)
b_sample = mean(df.train$target_num)

# Set metric for peformance comparison
metric = "auc"

# Define test data
df.test = df %>% filter(fold == "test")




#######################################################################################################################-
#|||| Test an algorithm (and determine parameter grid) ||||----
#######################################################################################################################-

# Possible controls
set.seed(999)
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


## Fits

# GLM 
fit = train(sparse.model.matrix(formula_binned_rightside, df.train[predictors_binned]), df.train$target,
#fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
            trControl = ctrl_idx_fff, metric = metric, 
            method = "glmnet", 
            tuneGrid = expand.grid(alpha = c(0,0.2,0.4,0.6,0.8,1), lambda = 2^(seq(-3, -10, -1))))
            #preProc = c("center","scale")) #no scaling needed due to dummy coding of all variables 
plot(fit, ylim = c(0.8,0.85))
# -> keep alpha=1 to have a full Lasso


# Random Forest
fit = train(df.train[predictors], df.train$target, 
            trControl = ctrl_idx_fff, metric = metric, 
            method = "ranger", 
            tuneGrid = expand.grid(mtry = seq(1,length(predictors),3),
                                   splitrule = "gini",
                                   min.node.size = c(1,5,10)), 
            num.trees = 500) #use the Dots (...) for explicitly specifiying randomForest parameter
plot(fit)
fit = train(as.data.frame(df.train[predictors]), df.train$target,
            trControl = ctrl_idx_fff, metric = metric,
            method = ms_forest,
            tuneGrid = expand.grid(numTrees = c(100,300,500), splitFraction = c(0.1,0.3,0.5)),
            verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
plot(fit)
# # -> keep around the recommended values: mtry = floor(sqrt(length(predictors))) or splitFraction = 0.3


# Boosted Trees
fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[predictors])), df.train$target,
#fit = train(formula, data = df.train[c("target",predictors)],
            trControl = ctrl_idx_nopar_fff, metric = metric, #no parallel for DMatrix
            method = "xgbTree", 
            tuneGrid = expand.grid(nrounds = seq(100,2100,200), max_depth = c(3,6,9), 
                                   eta = c(0.1,0.01), gamma = 0, colsample_bytree = c(0.5,0.7), 
                                   min_child_weight = c(5,10), subsample = c(0.5,0.7)))
plot(fit)
fit = train(df.train[predictors], df.train$target,
            trControl = ctrl_idx_fff, metric = metric,
            method = ms_boosttree,
            tuneGrid = expand.grid(numTrees = seq(100,1100,500), numLeaves = c(10,20),
                                   learningRate = c(0.1,0.01), featureFraction = c(0.5,0.7),
                                   minSplit = c(5,10), exampleFraction = c(0.5,0.7)),
            verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
plot(fit)
fit = train(sparse.model.matrix(formula_rightside, df.train[predictors]), df.train$target,
#fit = train(formula, data = as.data.frame(df.train[c("target",predictors)]),
            trControl = ctrl_idx_fff, metric = metric, 
            method = lgbm, 
            tuneGrid = expand.grid(nrounds = seq(100,1100,100), num_leaves = c(10,20),
                                   learning_rate = c(0.1,0.01), feature_fraction = c(0.5,0.7),
                                   min_data_in_leaf = c(5,10), bagging_fraction = c(0.5,0.7)),
            max_depth = 3,
            verbose = 0) 
plot(fit)
# -> max_depth = 3 / numLeaves = 20, shrinkage = 0.01, colsample_bytree = subsample = 0.7, min_xxx = 5

# DeepLearning
#xxxxxxxxxxxxxxxxxxxxxxxxxxxx



## Special plotting
fit
plot(fit, ylim = c(0.85,0.93))
varImp(fit) 
# unique(fit$results$lambda)

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
    facet_grid(as.formula(paste0("~",facet)), labeller = label_both) 
  
}




#######################################################################################################################-
#|||| Compare algorithms ||||----
#######################################################################################################################-

# Data to compare on
df.comp = df.train



#---- Simulation function ---------------------------------------------------------------------------------------

perfcomp = function(method, nsim = 5) { 
  
  result = foreach(sim = 1:nsim, .combine = bind_rows, .packages = c("caret","Matrix","xgboost"), 
                   .export = c("df.comp","mysummary_class","metric",
                               "predictors_binned","predictors",
                               "formula","formula_binned","formula_rightside","formula_binned_rightside"))
  %dopar% 
  {
    
    # Hold out a k*100% set
    set.seed(sim*999)
    k = 0.2
    i.holdout = sample(1:nrow(df.comp), floor(k*nrow(df.comp)))
    df.holdout = df.comp[i.holdout,]
    df.train = df.comp[-i.holdout,]    
    
    # Control for train
    set.seed(999)
    l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
    ctrl_idx = trainControl(method = "cv", number = 1, index = l.index, 
                            returnResamp = "final", returnData = FALSE,
                            summaryFunction = mysummary_class, classProbs = TRUE)
    ctrl_idx_nopar = trainControl(method = "cv", number = 1, index = l.index, 
                          returnResamp = "final", returnData = FALSE,
                          allowParallel = FALSE,
                          summaryFunction = mysummary_class, classProbs = TRUE)
    
    
    ## Fit data
    fit = NULL
    
    if (method == "glmnet") {  
      fit = train(sparse.model.matrix(formula_binned_rightside, df.train[predictors_binned]), df.train$target,
      #fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
                  trControl = ctrl_idx, metric = metric, 
                  method = "glmnet", 
                  tuneGrid = expand.grid(alpha = 1, lambda = 2^(seq(-3, -10, -1))))
    }     
    

    if (method == "xgbTree") { 
      fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[predictors])), df.train$target,
      #fit = train(formula, data = df.train[c("target",predictors)],
                  trControl = ctrl_idx_nopar, metric = metric, #no parallel for DMatrix
                  method = "xgbTree", 
                  tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = 6, 
                                         eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
                                         min_child_weight = 5, subsample = 0.7))
    }

    
    
    ## Get metrics
    
    # Calculate holdout performance
    if (method %in% c("glmnet","lgbm")) {
      yhat_holdout = predict(fit, sparse.model.matrix(formula_binned_rightside, df.holdout[predictors_binned]),
                             type = "prob")[[2]] 
    } else if (method == "xgbTree") {
      yhat_holdout = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.holdout[predictors])),
                             type = "prob")[[2]] 
    } else  {
      yhat_holdout = predict(fit, df.holdout[predictors], type = "prob")[[2]] 
    }
    perf_holdout = mysummary_class(data.frame(y = df.holdout$target, yhat = yhat_holdout))

    # Return result
    data.frame(sim = sim, method = method, t(perf_holdout))
  }   
  result
}




#---- Simulate --------------------------------------------------------------------------------------------

df.result = as.data.frame(c())
nsim = 2
df.result = bind_rows(df.result, perfcomp(method = "glmnet", nsim = nsim) )   
df.result = bind_rows(df.result, perfcomp(method = "xgbTree", nsim = nsim))       
#df.result = bind_rows(df.result, perfcomp(method = "deepLearning", nsim = nsim))       
df.result$sim = as.factor(df.result$sim)
df.result$method = factor(df.result$method, levels = unique(df.result$method))




#---- Plot simulation --------------------------------------------------------------------------------------------

p = ggplot(df.result, aes_string(x = "method", y = metric)) + 
  geom_boxplot() + 
  geom_point(aes(color = sim), shape = 15) +
  geom_line(aes(color = sim, group = sim), linetype = 2) +
  scale_x_discrete(limits = rev(levels(df.result$method))) +
  coord_flip() +
  labs(title = "Model Comparison") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5))
p  
ggsave(paste0(plotloc, "class_model_comparison.pdf"), p, width = 12, height = 8)




#######################################################################################################################-
#|||| Learning curve for winner algorithm ||||----
#######################################################################################################################-

skip = function() {
  # For testing on smaller data
  df.train = df.train %>% sample_n(500)
  df.test = df.test %>% sample_n(500)
}




#---- Loop over training chunks --------------------------------------------------------------------------------------

chunks_pct = c(seq(10,10,1), seq(20,100,10))
to = length(chunks_pct)

df.lc = foreach(i = 1:to, .combine = bind_rows, .packages = c("caret","xgboost")) %do% #NO dopar for xgboost!
{ 
  #i = 5
  
  ## Sample chunk
  set.seed(chunks_pct[i])
  i.samp = sample(1:nrow(df.train), floor(chunks_pct[i]/100 * nrow(df.train)))
  #i.no = which(df.train$target == "N")
  #i.yes = which(df.train$target == "Y")
  #i.samp = c(i.no, sample(i.yes,floor(chunks_pct[i]/100 * length(i.yes))))
  print(length(i.samp))
  
  
  
  ## Fit on chunk
  set.seed(1234)
  l.index = list(i = sample(1:nrow(df.train[i.samp,]), floor(0.8*nrow(df.train[i.samp,]))))
  ctrl_idx_nopar = trainControl(method = "cv", number = 1, index = l.index, 
                                returnResamp = "final", returnData = FALSE,
                                allowParallel = FALSE,
                                summaryFunction = mysummary_class, classProbs = TRUE)
  #ctrl_none = trainControl(method = "none", returnData = FALSE, classProbs = TRUE)
  tmp = Sys.time()
  fit = train(xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[i.samp,predictors])), 
              df.train[i.samp,]$target,
              trControl = ctrl_idx_nopar, metric = metric, #no parallel for DMatrix
              method = "xgbTree", 
              tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = 6, 
                                     eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
                                     min_child_weight = 5, subsample = 0.7))
  
  print(Sys.time() - tmp)
  
  
  
  ## Score (needs rescale to prior probs)
  # Train data 
  y_train = df.train[i.samp,]$target
  yhat_train_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.train[i.samp,predictors])),
                                type = "prob")[[2]]
  (b_sample = (summary(y_train) / length(y_train))[[2]]) #new b_sample
  yhat_train = prob_samp2full(yhat_train_unscaled, b_sample, b_all)
  
  # Test data 
  y_test = df.test$target
  yhat_test_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula_rightside, df.test)),
                               type = "prob")[[2]]
  # # Scoring in chunks in parallel in case of high memory consumption of xgboost
  # l.split = split(df.test[predictors], (1:nrow(df.test)) %/% 50000)
  # yhat_test_unscaled = foreach(df.split = l.split, .combine = c) %dopar% {
  #   predict(fit, df.split, type = "prob")[[2]]
  # }
  yhat_test = prob_samp2full(yhat_test_unscaled, b_sample, b_all)
  
  # Bind together
  res = rbind(cbind(data.frame("fold" = "train", "numtrainobs" = length(i.samp)), bestTune = fit$bestTune$nrounds,
                    t(mysummary_class(data.frame(y = y_train, yhat = yhat_train)))),
              cbind(data.frame("fold" = "test", "numtrainobs" = length(i.samp)), bestTune = fit$bestTune$nrounds,
                    t(mysummary_class(data.frame(y = y_test, yhat = yhat_test)))))
  
  
  ## Garbage collection and output
  gc()
  res
}
#save(df.lc, file = "df.lc.RData")




#---- Plot results --------------------------------------------------------------------------------------

p = ggplot(df.lc, aes_string("numtrainobs", metric, color = "fold")) +
  geom_line() +
  geom_point() +
  scale_color_manual(values = c("#F8766D", "#00BFC4")) 
p
ggsave(paste0(plotloc,"class_learningCurve.pdf"), p, width = 8, height = 6)




