
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
#|||| Test an algorithm (and determine parameter grid) ||||----
#######################################################################################################################-

## Validation information
metric = "spearman"

# Possible controls
set.seed(999)
ctrl_cv = trainControl(method = "repeatedcv", number = 4, repeats = 1, returnResamp = "final",
                       summaryFunction = mysummary_regr) #NOT USED
l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
ctrl_index_fff = trainControl(method = "cv", number = 1, index = l.index, returnResamp = "final",
                              summaryFunction = mysummary_regr, 
                              indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!


## Fits

fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
            trControl = ctrl_index_fff, metric = metric, 
            method = "glmnet", 
            tuneGrid = expand.grid(alpha = c(0,0.2,0.4,0.6,0.8,1), lambda = 2^(seq(-3, -10, -1))),
            #tuneLength = 20, 
            preProc = c("center","scale")) 
plot(fit, ylim = c(0.49,0.51))
# -> keep alpha=1 to have a full Lasso


fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
            trControl = ctrl_index_fff, metric = metric, 
            method = "glm", 
            tuneLength = 1,
            preProc = c("center","scale"))
# -> no tuning as it is a glm


fit = train( df.train[predictors], df.train$target, 
             trControl = ctrl_index_fff, metric = metric, 
             method = "rpart",
             tuneGrid = expand.grid(cp = 2^(seq(-20, -1, 2))) )
plot(fit)


fit = train(df.train[predictors], df.train$target, 
            trControl = ctrl_index_fff, metric = metric, 
            method = "rf", 
            tuneGrid = expand.grid(mtry = seq(1,length(predictors),3)), 
            ntree = 200) #use the Dots (...) for explicitly specifiying randomForest parameter
plot(fit)
# -> keep around the recommended values: mtry = floor(sqrt(length(predictors)))


fit = train(as.data.frame(df.train[predictors]), df.train$target, 
            trControl = ctrl_index_fff, metric = metric, 
            method = "gbm", 
            tuneGrid = expand.grid(n.trees = seq(100,1100,100), interaction.depth = c(3,6,9), 
                                   shrinkage = c(0.1,0.01), n.minobsinnode = c(5,10)), 
            verbose = FALSE)
plot(fit)
# -> keep to the recommended values: interaction.depth = 6, shrinkage = 0.01, n.minobsinnode = 10


fit = train(formula, data = df.train[c("target",predictors)],
            trControl = ctrl_index_fff, metric = metric, 
            method = "xgbTree", 
            tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = c(3,6,9), 
                                   eta = c(0.1,0.01), gamma = 0, colsample_bytree = c(0.5,0.7), 
                                   min_child_weight = c(5,10), subsample = c(0.5,0.7)))
plot(fit)
# -> max_depth = 3, shrinkage = 0.01, colsample_bytree = subsample = 0.7, n.minobsinnode = 5


fit = train(df.train[,predictors], df.train$target,
            trControl = ctrl_index_fff, metric = metric,
            method = ms_boosttree,
            tuneGrid = expand.grid(numTrees = seq(100,1100,500), numLeaves = c(10,20),
                                   learningRate = c(0.1,0.01), featureFraction = c(0.5,0.7),
                                   minSplit = c(5,10), exampleFraction = c(0.5,0.7)),
            verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
plot(fit)
# -> numLeaves = 20, learning_rate = 0.01, feature_fraction = example_fraction = 0.7, minSplit = 10


fit = train(df.train[,predictors], df.train$target,
            trControl = ctrl_index_fff, metric = metric,
            method = ms_forest,
            tuneGrid = expand.grid(numTrees = c(100,300,500), splitFraction = c(0.1,0.3,0.5)),
            verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
plot(fit)
# -> splitFraction = 0.3


fit = train(formula, data = as.data.frame(df.train[c("target",predictors)]),
            trControl = ctrl_index_fff, metric = metric, 
            method = lgbm, 
            # tuneGrid = expand.grid(num_rounds = c(50,100,200), num_leaves = 10,  
            #                        learning_rate = .1, feature_fraction = .7,  
            #                        min_data_in_leaf = 5, bagging_fraction = .7),
            tuneGrid = expand.grid(num_rounds = seq(100,1100,100), num_leaves = c(10,20),
                                   learning_rate = c(0.1,0.01), feature_fraction = c(0.5,0.7),
                                   min_data_in_leaf = c(5,10), bagging_fraction = c(0.5,0.7)),
            max_depth = 3,
            verbose = 0) 
plot(fit)
# -> numLeaves = 20, learning_rate = 0.01, feature_fraction = example_fraction = 0.7, minSplit = 10



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
  x = "num_rounds"; color = "as.factor(num_leaves)"; linetype = "as.factor(learning_rate)";  
  shape = "as.factor(min_data_in_leaf)"; facet = "min_data_in_leaf ~ bagging_fraction + feature_fraction"
  
  # Plot tuning result with ggplot
  fit$results %>% 
    ggplot(aes_string(x = x, y = y, colour = color)) +
    geom_line(aes_string(linetype = linetype, dummy = shape)) +
    geom_point(aes_string(shape = shape)) +
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
  
  result = NULL
  
  for (sim in 1:nsim) {
    
    # Hold out a k*100% set
    set.seed(sim*999)
    k = 0.2
    i.holdout = sample(1:nrow(df.comp), floor(k*nrow(df.comp)))
    df.holdout = df.comp[i.holdout,]
    df.train = df.comp[-i.holdout,]    
    
    # Control for train
    set.seed(999)
    l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
    ctrl_index = trainControl(method = "cv", number = 1, index = l.index, returnResamp = "final",
                              summaryFunction = mysummary_regr)    
    
    
    ## Fit data
    fit = NULL
    
    if (method == "glmnet") {  
      fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
                  trControl = ctrl_index, metric = metric, 
                  method = "glmnet", 
                  tuneGrid = expand.grid(alpha = 1, lambda = 2^(seq(-3, -10, -1))),
                  #tuneLength = 20, 
                  preProc = c("center","scale")) 
    }     
    
    if (method == "glm") {      
      fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
                  trControl = ctrl_index, metric = metric, 
                  method = "glm", 
                  tuneLength = 1,
                  preProc = c("center","scale"))
    }
    
    if (method == "rpart") {      
      fit = train( df.train[predictors], df.train$target, 
                   trControl = ctrl_index, metric = metric, 
                   method = "rpart",
                   tuneGrid = expand.grid(cp = 2^(seq(-20, -2, 2))) )
    }
    
    if (method == "rf") {      
      fit = train(df.train[predictors], df.train$target, 
                  trControl = ctrl_index, metric = metric, 
                  method = "rf", 
                  tuneGrid = expand.grid(mtry = c(4,5,6,7)), 
                  ntree = 500) 
    }
    
    if (method == "gbm") { 
      fit = train(as.data.frame(df.train[predictors]), df.train$target, 
                  trControl = ctrl_index, metric = metric, 
                  method = "gbm", 
                  tuneGrid = expand.grid(n.trees = seq(100,1100,100), interaction.depth = 6, 
                                         shrinkage = 0.01, n.minobsinnode = 10), 
                  verbose = FALSE)
    }
    
    if (method == "xgbTree") { 
      fit = train(formula, data = df.train[c("target",predictors)],
                  trControl = ctrl_index, metric = metric, 
                  method = "xgbTree", 
                  tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = 3, 
                                         eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
                                         min_child_weight = 5, subsample = 0.7))
    }
    
    if (method == "ms_boosttree") { 
      fit = train(df.train[,predictors], df.train$target, 
                  trControl = ctrl_index, metric = metric, 
                  method = ms_boosttree, 
                  tuneGrid = expand.grid(numTrees = seq(400,1000,200), numLeaves = 10,  
                                         learningRate = 0.01, featureFraction = 0.7,  
                                         minSplit = 10, exampleFraction = 0.7),
                  verbose = 0) 
    }
    
    if (method == "ms_forest") { 
      fit = train(df.train[,predictors], df.train$target, 
                  trControl = ctrl_index, metric = metric, 
                  method = ms_forest, 
                  tuneGrid = expand.grid(numTrees = c(100,300,500), splitFraction = 0.3),
                  verbose = 0) 
      plot(fit)
    }
    
    
    
    ## Get metrics
    
    # Calculate holdout performance
    if (method %in% c("glmnet","glm")) {
      yhat_holdout = predict(fit, df.holdout[predictors_binned]) 
    } else  {
      yhat_holdout = predict(fit, df.holdout[predictors]) 
    }
    perf_holdout = mysummary_regr(data.frame(y = df.holdout$target, yhat = yhat_holdout))
    
    # Put all together
    result = rbind(result, data.frame(sim = sim, method = method, t(perf_holdout)))
  }   
  result
}




#---- Simulate --------------------------------------------------------------------------------------------

df.result = as.data.frame(c())
nsim = 5
df.result = bind_rows(df.result, perfcomp(method = "glmnet", nsim = nsim) )   
df.result = bind_rows(df.result, perfcomp(method = "glm", nsim = nsim) )     
df.result = bind_rows(df.result, perfcomp(method = "rpart", nsim = nsim))      
df.result = bind_rows(df.result, perfcomp(method = "rf", nsim = nsim))        
df.result = bind_rows(df.result, perfcomp(method = "gbm", nsim = nsim))       
df.result = bind_rows(df.result, perfcomp(method = "xgbTree", nsim = nsim))       
df.result = bind_rows(df.result, perfcomp(method = "ms_boosttree", nsim = nsim))       
df.result = bind_rows(df.result, perfcomp(method = "ms_forest", nsim = nsim))       
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
ggsave(paste0(plotloc, "model_comparison.pdf"), p, width = 12, height = 8)




#######################################################################################################################-
#|||| Check number of trainings records needed for winner algorithm ||||----
#######################################################################################################################-

skip = function() {
  # For testing on smaller data
  df.train = df.train[sample(1:nrow(df.train), 5000),]
  df.test = df.test[sample(1:nrow(df.test), 5000),]
}




#---- Loop over training chunks --------------------------------------------------------------------------------------

chunks_pct = c(seq(10,10,1), seq(20,100,10))

df.obsneed = c()  
df.obsneed = foreach(i = 1:length(chunks_pct), .combine = bind_rows, .packages = c("caret")) %do% #NO dopar for xgboost!
{ 
  #i = 1
  
  ## Sample chunk
  set.seed(chunks_pct[i])
  i.train = sample(1:nrow(df.train), floor(chunks_pct[i]/100 * nrow(df.train)))
  print(length(i.train))
  
  
  
  ## Fit on chunk
  ctrl_none = trainControl(method = "none")
  tmp = Sys.time()
  fit = train(formula, data = df.train[i.train,c("target",predictors)],
              trControl = ctrl_none, metric = metric, 
              method = "xgbTree", 
              tuneGrid = expand.grid(nrounds = 500, max_depth = 3, 
                                     eta = 0.01, gamma = 0, colsample_bytree = 0.7, 
                                     min_child_weight = 5, subsample = 0.7))
  
  print(Sys.time() - tmp)
  
  
  
  ## Score 
  # Train data 
  y_train = df.train[i.train,]$target
  yhat_train = predict(fit, df.train[i.train,predictors])
  
  # Test data 
  y_test = df.test$target
  l.split = split(1:nrow(df.test), (1:nrow(df.test)) %/% 50000)
  yhat_test = foreach(i = 1:length(l.split), .combine = bind_rows) %do% {
    # Scoring in chunks due to high memory consumption of xgboost
    yhat = predict(fit, df.test[l.split[[i]],predictors])
    gc()
    yhat
  }
  
  # Bind together
  res = rbind(cbind(data.frame("fold" = "train", "numtrainobs" = length(i.train)),
                    t(mysummary_regr(data.frame(y = y_train, yhat = yhat_train)))),
              cbind(data.frame("fold" = "test", "numtrainobs" = length(i.train)),
                    t(mysummary_regr(data.frame(y = y_test, yhat = yhat_test)))))
  

  
  ## Garbage collection and output
  gc()
  res
}
#save(df.obsneed, file = "df.obsneed.RData")




#---- Plot results --------------------------------------------------------------------------------------

p = ggplot(df.obsneed, aes_string("numtrainobs", metric, color = "fold")) +
  geom_line() +
  geom_point() +
  scale_color_manual(values = c("#F8766D", "#00BFC4")) 
p
ggsave(paste0(plotloc,"learningCurve.pdf"), p, width = 8, height = 6)




