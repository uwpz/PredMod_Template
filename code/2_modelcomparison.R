
# Set target type -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
rm(list = ls())
#TYPE = "CLASS"
#TYPE = "REGR"
TYPE = "MULTICLASS"



#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load result from exploration
load(paste0(TYPE,"_1_explore.rdata"))

# Load libraries and functions
source("./code/0_init.R")

# Adapt some parameter differnt for target types -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
splitrule = switch(TYPE, "CLASS" = "gini", "REGR" = "variance", "MULTICLASS" = "gini") #do not change this one
type = switch(TYPE, "CLASS" = "prob", "REGR" = "raw", "MULTICLASS" = "prob") #do not change this one

# Set metric for peformance comparison
if (TYPE %in% c("CLASS","MULTICLASS")) metric = "AUC"
if (TYPE == "REGR") metric = "spearman"



## Initialize parallel processing
closeAllConnections() #reset
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl); closeAllConnections() #stop cluster


  


#######################################################################################################################-
#|||| Test an algorithm (and determine parameter grid) ||||----
#######################################################################################################################-

# Sample data --------------------------------------------------------------------------------------------

if (TYPE %in% c("CLASS","MULTICLASS")) {
  # Just take data from train fold (take all but n_maxpersample at most)
  summary(df[df$fold == "train", "target"])
  c(df.train, b_sample, b_all) %<-%  (df %>% filter(fold == "train") %>% undersample_n(n_maxpersample = 500)) 
  summary(df.train$target); b_sample; b_all
}
if (TYPE == "REGR") {
  # Just take data from train fold
  df.train = df %>% filter(fold == "train") #%>% sample_n(1000)
}




# Define some controls --------------------------------------------------------------------------------------------

set.seed(998)
l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
ctrl_idx = trainControl(method = "cv", number = 1, index = l.index, 
                        returnResamp = "final", returnData = FALSE,
                        summaryFunction = mysummary, classProbs = TRUE) 
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
fit = train(sparse.model.matrix(formula_binned, df.train[c("target",features_binned)]), df.train$target,
#fit = train(formula_binned, data = df.train[c("target",features_binned)], 
            method = "glmnet", 
            trControl = ctrl_idx_fff, 
            metric = metric, 
            tuneGrid = expand.grid(alpha = c(0,0.2,0.4,0.6,0.8,1), 
                                   lambda = 2^(seq(-3, -10, -1))))
            #preProc = c("center","scale")) #no scaling needed due to dummy coding of all variables 
plot(fit)
# -> keep alpha=1 to have a full Lasso



## Random Forest
fit = train(df.train[features], df.train$target, 
            method = "ranger", 
            trControl = ctrl_idx_fff, 
            metric = metric, 
            tuneGrid = expand.grid(mtry = seq(1,length(features),10),
                                   splitrule = splitrule,
                                   min.node.size = c(1,5,10)), 
            num.trees = 500) #use the Dots (...) for explicitly specifiying randomForest parameter
plot(fit)
# -> keep around the recommended values: mtry(class) = sqrt(length(features), mtry(regr) = 0.3 * length(features))


if (TYPE != "MULTICLASS") {
  fit = train(as.data.frame(df.train[features]), df.train$target,
              method = ms_forest,
              trControl = ctrl_idx_fff, 
              metric = metric,
              tuneGrid = expand.grid(numTrees = c(100,300,500), 
                                     splitFraction = c(0.1,0.3,0.5)),
              verbose = 0) #!numTrees is not a sequential parameter (like in xgbTree)
  plot(fit)
  # # -> keep around the recommended values: mtry = floor(sqrt(length(features))) or splitFraction = 0.3
}



## Boosted Trees

# Default xgbTree: no parallel processing possible with DMatrix (and using sparse matrix will result in nonsparse trafo)
fit = train(xgb.DMatrix(sparse.model.matrix(formula, df.train[c("target",features)])), df.train$target,
            method = "xgbTree", 
            trControl = ctrl_idx_nopar_fff, #no parallel for DMatrix
            metric = metric, 
            tuneGrid = expand.grid(nrounds = seq(100,1100,200), eta = c(0.01),
                                   max_depth = c(3), min_child_weight = c(10),
                                   colsample_bytree = c(0.7), subsample = c(0.7),
                                   gamma = 0))
plot(fit)

# Overwritten xgbTree: additional alpha and lambda parameter. Possible to use sparse matrix and parallel processing
fit = train(sparse.model.matrix(formula, df.train[c("target",features)]), df.train$target,
            method = xgb, 
            trControl = ctrl_idx_fff, #parallel for overwritten xgb
            metric = metric, 
            tuneGrid = expand.grid(nrounds = seq(100,1100,200), eta = c(0.01),
                                   max_depth = c(3), min_child_weight = c(10),
                                   colsample_bytree = c(0.7), subsample = c(0.7),
                                   gamma = 0, alpha = 1, lambda = 0))
plot(fit)

if (TYPE != "MULTICLASS") {
  # MicrosoftML: numTrees is not a sequential parameter (like in xgbTree) !!!
  fit = train(df.train[features], df.train$target,
              trControl = ctrl_idx_fff, metric = metric,
              method = ms_boosttree,
              tuneGrid = expand.grid(numTrees = seq(100,2100,500), learningRate = c(0.1,0.01), 
                                     numLeaves = c(10,20), minSplit = c(10), 
                                     featureFraction = c(0.7), exampleFraction = c(0.7)),
              verbose = 0) 
  plot(fit)

  # Lightgbm
  fit = train(sparse.model.matrix(formula, df.train[c("target",features)]), df.train$target,
              method = lgbm,
              trControl = ctrl_idx_fff, 
              metric = metric,
              tuneGrid = expand.grid(nrounds = seq(100,1100,200), learning_rate = c(0.1, 0.01), 
                                     num_leaves = c(5), min_data_in_leaf = c(10),
                                     feature_fraction = c(0.7), bagging_fraction = c(0.7)),
              #max_depth = 3, #use for small data
              verbose = 0)
  plot(fit)
  # -> max_depth = 3 / numLeaves = 20, shrinkage = 0.01, colsample_bytree = subsample = 0.7, min_xxx = 5
}



## DeepLearning
fit = train(formula, df.train[c("target",features)],
            method = deepLearn, 
            trControl = ctrl_idx_nopar_fff, 
            metric = metric, 
            tuneGrid = expand.grid(size = c("10","10-10","10-10-10"), 
                                   lambda = c(0,2^-1), dropout = 0.5,
                                   batch_size = c(100), lr = c(1e-3), 
                                   batch_normalization = TRUE, 
                                   activation = c("relu","elu"),
                                   epochs = 10),
            preProc = c("center","scale"),
            verbose = 0) 
plot(fit)






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
    facet_grid(as.formula(paste0("~",facet)), labeller = label_both) 
  
}



#######################################################################################################################-
#|||| Evaluate generalization gap ||||----
#######################################################################################################################-

## Prepare for caret
# Duplicate training 
df.gap = bind_rows(df %>% filter(fold == "train"), df %>% filter(fold == "train"), 
                   df %>% filter(fold == "test"))

# Summary function: hard code n.train in following code if using caret in parallel mode
(n.train = sum(df.gap$fold == "train") / 2)
tmp_summary = function(data, lev = NULL, model = NULL) 
{
  #browser()
  
  if (nrow(data) == 10) {
    data.train = data
    data.test = data
  } else {
    i.train = 1:2051 ################### hard code here !!!!!!!!!!!!!!!!!!!!
    data.train = data[i.train,]
    data.test = data[-i.train,]
  }
  
  stats_all = c()
  
  for (fold in c("train","test")) {
    df.tmp = get(paste0("data.",fold))
    
    # AUC stats
    prob_stats = map(levels(data[, "pred"]), function(x) {
      AUCs <- try(ModelMetrics::auc(ifelse(df.tmp[, "obs"] == x, 1, 0), df.tmp[, x]), silent = TRUE)
      AUCs = max(AUCs, 1 - AUCs)
      return(AUCs)
    })
    roc_stats = c("AUC" = mean(unlist(prob_stats)), 
                  "Weighted_AUC" = sum(unlist(prob_stats) * table(data$obs)/nrow(data)))
    
    # Confusion matrix stats
    CM = confusionMatrix(df.tmp[, "pred"],df.tmp[, "obs"])
    class_stats = CM$byClass
    if (!is.null(dim(class_stats))) class_stats = colMeans(class_stats)
    names(class_stats) <- paste0("Mean_", names(class_stats))
    
    # Collect metrics
    stats = c(roc_stats, CM$overall[c("Accuracy","Kappa")], class_stats)
    names(stats) = gsub("[[:blank:]]+", "_", names(stats))
    
    names(stats) = paste0(names(stats),"_",fold)
  
    # Collect
    stats_all = c(stats_all, stats)
  }
  stats_all
}



## Fit 
l.index = list(i = 1:n.train)
ctrl_idx_nopar_fff = trainControl(method = "cv", number = 1, index = l.index, 
                                  returnResamp = "final", returnData = FALSE,
                                  allowParallel = FALSE, #no parallel e.g. for xgboost on big data or with DMatrix
                                  summaryFunction = tmp_summary, classProbs = TRUE, 
                                  indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
fit = train(xgb.DMatrix(sparse.model.matrix(formula, df.gap[c("target",features)])), df.gap$target,
            method = "xgbTree", 
            trControl = ctrl_idx_nopar_fff, #no parallel for DMatrix
            metric = metric, 
            tuneGrid = expand.grid(nrounds = seq(100,1100,200), eta = c(0.01),
                                   max_depth = c(1,3,6), min_child_weight = c(5,10,20),
                                   colsample_bytree = c(0.7), subsample = c(0.7),
                                   gamma = 0))


## Plot
y = "auc"; x = "nrounds"; linetype = "type";
color =  "as.factor(max_depth)"; 
shape = "as.factor(min_child_weight)"; facet = "min_child_weight ~ subsample + colsample_bytree"

fit$results %>% gather(key = "type", value = "auc", AUC_train, AUC_test) %>% 
  ggplot(aes_string(x = x, y = y, colour = color, shape = shape)) +
  geom_line(aes_string(linetype = linetype)) +
  geom_point() +
  facet_grid(as.formula(paste0("~",facet)), labeller = label_both) 




#######################################################################################################################-
#|||| Simulation: compare algorithms ||||----
#######################################################################################################################-

# Basic data sampling
df.sim = df #%>% sample_n(1000)

# Tunegrid
if (TYPE == "CLASS") {
  tunepar = expand.grid(nrounds = seq(100,500,200), eta = 0.01, 
                        max_depth = 3, min_child_weight = 2, 
                        colsample_bytree = 0.7, subsample = 0.7,
                        gamma = 0)
}
if (TYPE %in% c("REGR","MULTICLASS")) {
  tunepar = expand.grid(nrounds = seq(100,500,200), eta = 0.05,  
                        max_depth = 6, min_child_weight = 5, 
                        colsample_bytree = 0.3, subsample = 0.7, 
                        gamma = 0)
}




#---- Simulation function ---------------------------------------------------------------------------------------

perfcomp = function(method, nsim = 5) { 
  
  result = foreach(sim = 1:nsim, .combine = bind_rows, .packages = c("caret","Matrix","xgboost"), 
                   .export = c("df.sim","mysummary","metric","type",
                               "features_binned","features",
                               "formula","formula_binned",
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
      fit = train(sparse.model.matrix(formula_binned, df.train[c("target",features_binned)]), df.train$target,
      #fit = train(formula_binned, data = df.train[c("target",features_binned)], 
                  method = "glmnet", 
                  trControl = ctrl_idx, 
                  metric = metric, 
                  tuneGrid = expand.grid(alpha = 1, lambda = 2^(seq(-3, -10, -1))))
    }     
    

    if (method == "xgbTree") { 
      fit = train(xgb.DMatrix(sparse.model.matrix(formula, df.train[c("target",features)])), df.train$target,
                  method = "xgbTree", 
                  trControl = ctrl_idx_nopar, 
                  metric = metric, 
                  tuneGrid = tunepar)
    }

    
    
    ## Get metrics
    
    # Calculate test performance
    if (method %in% c("glmnet")) {
      yhat_test = predict(fit, sparse.model.matrix(formula_binned, df.test[c("target",features_binned)]),
                             type = type) 
    } else if (method == "xgbTree") {
      yhat_test = predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.test[c("target",features)])),
                             type = type)
    } else  {
      yhat_test = predict(fit, df.test[features], type = type) 
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
nsim = 10
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
ggsave(paste0(plotloc,TYPE,"_model_comparison.pdf"), p, width = 12, height = 8)




#######################################################################################################################-
#|||| Learning curve for winner algorithm ||||----
#######################################################################################################################-

# Basic data sampling
df.lc = df #%>% sample_n(1000)

# Tunegrid
if (TYPE == "CLASS") {
  tunepar = expand.grid(nrounds = seq(100,500,100), eta = 0.01, 
                        max_depth = 3, min_child_weight = 2, 
                        colsample_bytree = 0.7, subsample = 0.7,
                        gamma = 0)
}
if (TYPE %in% c("REGR","MULTICLASS")) {
  tunepar = expand.grid(nrounds = seq(100,500,100), eta = 0.05,  
                        max_depth = 6, min_child_weight = 5, 
                        colsample_bytree = 0.3, subsample = 0.7, 
                        gamma = 0)
}




#---- Loop over training chunks --------------------------------------------------------------------------------------

chunks_pct = c(seq(10,10,1), seq(20,100,10))
to = length(chunks_pct)

df.lc_result = foreach(i = 1:to, .combine = bind_rows, 
                       .packages = c("zeallot","dplyr","caret","Matrix","xgboost")) %do%  #NO dopar for xgboost!
{ 
  #i = 1
  
  ## Sample
  set.seed(chunks_pct[i])
  df.train = df.lc %>% filter(fold == "train") %>% sample_frac(chunks_pct[i]/100)
  if (TYPE %in% c("CLASS","MULTICLASS")) {
    b_sample = b_all = df.train$target %>% (function(.) {summary(.)/length(.)})
    ## Balanced ("as long as possible")
    #c(df.train, b_sample, b_all) %<-% 
    #    undersample_n(df.lc %>% filter(fold == "train"), 
    #                  n_maxpersample = chunks_pct[i]/100 * max(summary(df.lc[df.lc$fold == "train",][["target"]])))
  }
  if (TYPE == "REGR") b_sample = b_all = NULL 
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
  fit = train(xgb.DMatrix(sparse.model.matrix(formula, df.train[c("target",features)])), df.train$target,
              method = "xgbTree", 
              trControl = ctrl_idx_nopar, 
              metric = metric, 
              tuneGrid = tunepar)
  print(Sys.time() - tmp)
  
  
  
  ## Score (needs rescale to prior probs)
  # Train data 
  y_train = df.train$target
  yhat_train_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.train[c("target",features)])),
                                type = type)
  yhat_train = scale_pred(yhat_train_unscaled, b_sample, b_all)

  # Test data 
  y_test = df.test$target
  yhat_test_unscaled = predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.test)), type = type)
  # # Scoring in chunks in parallel in case of high memory consumption of xgboost
  # l.split = split(df.test[features], (1:nrow(df.test)) %/% 50000)
  # yhat_test_unscaled = foreach(df.split = l.split, .combine = bind_rows) %dopar% {
  #   predict(fit, xgb.DMatrix(sparse.model.matrix(formula, df.split)), type = type)
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
ggsave(paste0(plotloc,TYPE,"_learningCurve.pdf"), p, width = 8, height = 6)




