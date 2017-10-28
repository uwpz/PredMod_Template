
#######################################################################################################################-
# Initialize ----
#######################################################################################################################-

load("1_explore.rdata")
source("./code/0_init.R")


## Initialize parallel processing
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



## Validation information
metric = "auc"

# Possible validation controls
skip = function() {
  # Cross validation
  ctrl_cv = trainControl(method = "repeatedcv", number = 4, repeats = 1,
                         summaryFunction = mysummary_class, classProbs = TRUE, returnResamp = "final")
  ctrl_cv_fff = trainControl(method = "repeatedcv", number = 4, repeats = 1,
                             summaryFunction = mysummary_class, classProbs = TRUE, returnResamp = "final",
                             indexFinal = sample(1:nrow(df.train), 100))  #"Fast" final fit!!!
  set.seed(sim*1000) #for following train-call
  
  # Index validation
  set.seed(999)
  l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
  ctrl_index = trainControl(method = "cv", number = 1, index = l.index,
                            summaryFunction = mysummary_class, classProbs = TRUE, returnResamp = "final")
  ctrl_index_fff = trainControl(method = "cv", number = 1, index = l.index,
                                summaryFunction = mysummary_class, classProbs = TRUE, returnResamp = "final",
                                indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
}



#######################################################################################################################-
# Test an algorithm (and determine parameter grid) ----
#######################################################################################################################-

# Validation control
set.seed(999)
l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
ctrl_index_fff = trainControl(method = "cv", number = 1, index = l.index,
                              summaryFunction = mysummary_class, classProbs = TRUE, returnResamp = "final",
                              indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!


fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
            trControl = ctrl_index_fff, metric = metric, 
            method = "glmnet", 
            tuneGrid = expand.grid(alpha = c(0,0.2,0.4,0.6,0.8,1), lambda = 2^(seq(-1, -10, -1))),
            #tuneLength = 20, 
            preProc = c("center","scale")) 
plot(fit)
# -> keep alpha=1 to have a full Lasso


fit = train(formula_binned, data = df.train[c("target",predictors_binned)], 
            trControl = ctrl_index_fff, metric = metric, 
            method = "glm", 
            tuneLength = 1,
            preProc = c("center","scale"))
# -> no tuning as it is a glm


fit = train(df.train[predictors], df.train$target, 
            trControl = ctrl_index_fff, metric = metric, 
            method = "rf", 
            tuneGrid = expand.grid(mtry = seq(1,length(predictors),3)), 
            ntree = 500) #use the Dots (...) for explicitly specifiying randomForest parameter
plot(fit)
# -> keep to the recommended values: mtry = floor(sqrt(length(predictors)))


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
            verbose = 0) 
plot(fit)
# -> splitFraction = 0.3



## Special plotting
fit
plot(fit, ylim = c(0.85,0.93))
varImp(fit) 
# unique(fit$results$lambda)

skip = function() {
  
  y = "auc"
  x = "nrounds"
  color = "as.factor(eta)"
  linetype = "as.factor(max_depth)"
  shape = "as.factor(min_child_weight)"
  facet = "min_child_weight ~ subsample + colsample_bytree"
  
  # Plot tuning result with ggplot
  fit$results %>% 
    ggplot(aes_string(x = x, y = y, colour = color)) +
    geom_line(aes_string(linetype = linetype)) +
    #geom_point(aes_string(shape = shape)) +
    #geom_errorbar(mapping = aes_string(ymin = "RMSE - RMSESD", ymax = "RMSE + RMSESD", linetype = linetype)) +
    facet_grid(as.formula(paste0("~",facet)), labeller = label_both) 
  
}




########################################################################################################################
# Compare algorithms
########################################################################################################################

## Simulation function

df.comp = df.train

perfcomp = function(method, nsim = 5) { 
  
  result = NULL
  rows = nrow(df.comp)

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
    ctrl_index = trainControl(method = "cv", number = 1, index = l.index,
                              summaryFunction = mysummary_class, classProbs = TRUE, returnResamp = "final")
    
    ## fit data
    fit = NULL
    if (method == "glm") {      
      fit = train( formula, data = df.train[c("target",predictors)], trControl = ctrl, metric = "ROC", 
                   method = "glm", 
                   tuneLength = 1,
                   preProc = c("center","scale") )
    }

    
    if (method == "glmnet") {      
      fit = train( formula, data = df.train[c("target",predictors)], trControl = ctrl, metric = "ROC", 
                   method = "glmnet", 
                   tuneGrid = expand.grid(alpha = 1, lambda = 2^(seq(-2, -10, -1))),
                   preProc = c("center","scale") ) 
    }     
    

    if (method == "rpart") {      
      fit = train( df.train[predictors], df.train$target, trControl = ctrl, metric = "ROC", 
                   method = "rpart",
                   tuneGrid = expand.grid(cp = 2^(seq(-20, -1, 2))) )
    }
    
    
    if (method == "rf") {      
      fit = train( df.train[predictors], df.train$target, trControl = ctrl, metric = "ROC", 
                   method = "rf", 
                   tuneGrid = expand.grid(mtry = 5), 
                   ntree = 300 )
    }
    
    
    if (method == "gbm") { 
      fit = train( df.train[predictors], df.train$target, trControl = ctrl, metric = "ROC",
                   method = "gbm", 
                   tuneGrid = expand.grid(n.trees = seq(100,1100,100), interaction.depth = 6, 
                                          shrinkage = 0.01, n.minobsinnode = 10), 
                   verbose = FALSE )
    }


    ## Get metrics

    # Calculate holdout performance
    yhat_holdout = predict(fit, df.holdout[predictors], type = "prob")[2] 
    perf_holdout = performance( prediction(yhat_holdout, df.holdout$target), metric)@y.values[[1]]
    #TODO: take mysummary
    
    # Put all together
    result = rbind(result, cbind(sim = sim, method = method, fit$resample, perf_holdout = perf_holdout))
  }   
  result
}

df.result = as.data.frame(c())
nsim = 5
df.result = rbind.fill(df.result, perfcomp(method = "glm", nsim = nsim) )     
df.result = rbind.fill(df.result, perfcomp(method = "glmnet", nsim = nsim) )   
df.result = rbind.fill(df.result, perfcomp(method = "rpart", nsim = nsim))      
df.result = rbind.fill(df.result, perfcomp(method = "rf", nsim = nsim))        
df.result = rbind.fill(df.result, perfcomp(method = "gbm", nsim = nsim))       
df.result$sim = as.factor(df.result$sim)


## Plot results
p = ggplot(df.result, aes(method, ROC)) + 
  geom_boxplot() + 
  geom_point(aes(method, perf_holdout, color = sim), unique(df.result[c("sim","method","perf_holdout")]), shape = 15) +
  coord_flip() +
  labs(title = "Model Comparison") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5))
p  
ggsave("./output/model_comparison.pdf", p, width = 4, height = 6)





########################################################################################################################-
# Check number of trainings records needed for winner algorithm ----
########################################################################################################################-


# NOT RUN
skip = function() {
  # For testing on smaller data
  df.train = df.train[sample(1:nrow(df.train), 5000),]
  df.test = df.test[sample(1:nrow(df.test), 5000),]
}



## Loop over training chunks
chunks_pct = c(seq(1,10,1), seq(20,100,10))

df.obsneed = c()  
df.obsneed = foreach(i = 1:length(chunks_pct), .combine = rbind, .packages = c("caret")) %do% { #NO dopar for xgboost!
  #i = 10
  
  # Sample chunk
  set.seed(chunks_pct[i])
  i.train = sample(1:nrow(df.train), floor(chunks_pct[i]/100 * nrow(df.train)))
  print(length(i.train))
  
  
  
  ## Fit on chunk
  ctrl_none = trainControl(method = "none", classProbs = TRUE)
  
  print(Sys.time())
  # predictors = predictors_glmnet_all
  # fit = train( as.formula(paste("target", "~", paste(predictors, collapse = " + "))),
  #              data = as.data.frame(df.train[i.train,c("target",predictors)]), trControl = ctrl_none, metric = "Mean_AUC",
  #              method = "glmnet",
  #              tuneGrid = expand.grid(alpha = 1, lambda = 2^-11),
  #              preProc = c("center","scale") )
  
  predictors = predictors_all
  fit = train( as.formula(paste("target", "~", paste(predictors, collapse = " + "))),
               data = as.data.frame(df.train[i.train,c("target",predictors)]), trControl = ctrl_none, metric = "Mean_AUC",
               method = "xgbTree",
               tuneGrid = expand.grid(nrounds = 700, max_depth = c(12),
                                      eta = c(0.02), gamma = 0, colsample_bytree = c(0.5),
                                      min_child_weight = c(10), subsample = c(0.6)))
  
  print(Sys.time())
  
  
  
  ## Score (needs rescale to prior probs)
  
  # Train data 
  y_train = df.train[i.train,]$target
  yhat_train = predict(fit, df.train[i.train,predictors], type = "prob")
  (b_sample = summary(y_train) / length(y_train))
  for (lev in colnames(yhat_train)) {yhat_train[[lev]] = yhat_train[[lev]] * b_all[lev] / b_sample[lev]}
  yhat_train = yhat_train / rowSums(yhat_train)
  yhat_train_nonprob = factor(colnames(yhat_train)[apply(as.matrix(yhat_train), 1, which.max)], levels = levels(y_train))
  
  # Test data 
  y_test = df.test$target
  l.split = split(1:nrow(df.test), (1:nrow(df.test)) %/% 50000)
  yhat_test = foreach(i = 1:length(l.split), .combine = bind_rows) %do% {
    # Parallelize scoring due to high memory consumption of xgboost
    pred = predict(fit, df.test[l.split[[i]],predictors], type="prob")
    gc()
    pred
  }
  for (lev in colnames(yhat_test)) {yhat_test[[lev]] = yhat_test[[lev]] * b_all[lev] / b_sample[lev]}
  yhat_test = yhat_test / rowSums(yhat_test)
  yhat_test_nonprob = factor(colnames(yhat_test)[apply(as.matrix(yhat_test), 1, which.max)], levels = levels(y_test))
  
  # Bind together
  res = rbind( cbind(data.frame("fold" = "train", "numtrainobs" = length(i.train)),
                     t(my_multiClassSummary(cbind(yhat_train, "obs" = y_train, "pred" = yhat_train_nonprob)))),
               cbind(data.frame("fold" = "test", "numtrainobs" = length(i.train)),
                     t(my_multiClassSummary(cbind(yhat_test, "obs" = y_test, "pred" = yhat_test_nonprob)))) )
  
  
  
  ## Garbage collection and output
  gc()
  res
}

df.obsneed$model = "xgbTree"
df.obsneed_tmp = df.obsneed
#save(df.obsneed, file = "df.obsneed.RData")


# Plot result
p = ggplot(df.obsneed, aes(numtrainobs, Mean_AUC, color = fold)) +
  geom_line() +
  geom_point() +
  scale_color_manual(values = c("#F8766D", "#00BFC4")) 
p
ggsave("./output/learningCurve.pdf", p, width = 8, height = 6)




