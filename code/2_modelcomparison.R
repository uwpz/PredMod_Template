
#######################################################################################################################-
# Initialize ----
#######################################################################################################################-

load("1_explore.rdata")
source("./code/0_init.R")


## Initialize parallel processing
Sys.getenv("NUMBER_OF_PROCESSORS") 
cl = makeCluster(4)
registerDoParallel(cl) 
# stopCluster(cl) #stop cluster



#######################################################################################################################-
# Test an algorithm (and determine parameter grid) ----
#######################################################################################################################-

df.train = as.data.frame(df)
 
ctrl = trainControl(method = "repeatedcv", number = 4, repeats = 1, 
                    summaryFunction = my_twoClassSummary, classProbs = T, returnResamp = "final", 
                    indexFinal = sample(1:nrow(df.train), 100))  #"Fast" final fit!!! 

fit = train( formula, data = df.train[c("target",predictors)], trControl = ctrl, metric = "auc", 
             method = "glmnet", 
             tuneGrid = expand.grid(alpha = c(0,0.2,0.4,0.6,0.8,1), lambda = 2^(seq(-1, -10, -1))),
             #tuneLength = 20, 
             preProc = c("center","scale") ) 
# -> keep alpha=1 to have a full Lasso


fit = train( formula, data = df.train[c("target",predictors)], trControl = ctrl, metric = "auc", 
             method = "glm", 
             tuneLength = 1,
             preProc = c("center","scale") )
# -> no tuning as it is a glm


fit = train( df.train[predictors], df.train$target, trControl = ctrl, metric = "ROC", 
             method = "rf", 
             tuneGrid = expand.grid(mtry = seq(1,11,2)), 
             #tuneLength = 2,
             ntree = 500 ) #use the Dots (...) by explicitly specifiying randomForest parameter
# -> keep to the recommended values: mtry = sqrt(length(predictors))

fit = train( df.train[,predictors], df.train$target, trControl = ctrl, metric = "ROC",
             method = "gbm", 
             tuneGrid = expand.grid(n.trees = seq(100,1100,100), interaction.depth = c(3,6,9), 
                                    shrinkage = c(0.1,0.01,0.001), n.minobsinnode = c(5,10)), 
             #tuneLength = 6,
             verbose = FALSE )
# -> keep to the recommended values: interaction.depth = 6, shrinkage = 0.01, n.minobsinnode = 10


fit
plot(fit, ylim = c(0.85,0.93))
varImp(fit) 

# unique(fit$results$lambda)
# plot(fit$finalModel, i.var = 1, type="response", col = twocol)




########################################################################################################################
# Compare algorithms
########################################################################################################################

## Simulation function

df.cv = as.data.frame(df.samp)

perfcomp = function(method, nsim = 5) { 
  
  result = NULL
  rows = nrow(df.cv)

  for (sim in 1:nsim) {

    # Hold out a k*100% set
    set.seed(sim*999)
    k = 0.2
    i.holdout = sample(1:nrow(df.cv), floor(k*nrow(df.cv)))
    df.holdout = df.cv[i.holdout,]
    df.train = df.cv[-i.holdout,]    
    
    # Control and seed for train
    ctrl = trainControl(method = "repeatedcv", number = 4, repeats = 1, 
                        summaryFunction = twoClassSummary, classProbs = T,
                        returnResamp = "final")
    set.seed(sim*1000) #for following train-call
    

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
    perf_holdout = performance( prediction(yhat_holdout, df.holdout$target), "auc" )@y.values[[1]]
    
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





