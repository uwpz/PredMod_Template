
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Locations
dataloc = "./data/"

# Libraries
library(plyr) #always load plyr before dpylr / tidyverse
library(dplyr)
library(purrr)
library(readr)
library(readxl)
library(forcats)
library(stringr)
library(xgboost)
library(caret)
library(Matrix)
library(zeallot)




## Functions
# Calculate predictions from probability of sample data and the corresponding base probabilities (classification)
scale_pred = function(yhat, b_sample = NULL, b_all = NULL) {
    as.data.frame(t(t(as.matrix(yhat)) * (b_all / b_sample))) %>% (function(x) x/rowSums(x))
}

# Custom summary function for classification performance (use by caret)
mysummary = function(data, lev = NULL, model = NULL) 
{
  #browser()
  # Adapt target observations
  if ("y" %in% colnames(data)) data$obs = data$y
  
  # Switch colnames in case of classification
  colnames(data) = gsub("yhat.","",colnames(data))
  
  ## Classification or Multiclass-Classifiction
  if (is.factor(data$obs)) {
    # Adapt prediction observations
    if (!("pred" %in% colnames(data))) data$pred = factor(levels(data$obs)[apply(data[levels(data$obs)], 1, 
                                                                                 function(x) which.max(x))], 
                                                          levels = levels(data$obs))
    if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) stop("levels of observed and predicted data do not match")
    
    # Logloss stats
    if (is.null(lev)) lev = levels(data$obs)
    lloss <- mnLogLoss(data = data, lev = lev, model = model)
    
    # AUC stats
    prob_stats <- lapply(levels(data[, "pred"]), function(x) {
      obs <- ifelse(data[, "obs"] == x, 1, 0)
      prob <- data[, x]
      AUCs <- try(ModelMetrics::auc(obs, data[, x]), silent = TRUE)
      AUCs = max(AUCs, 1 - AUCs)
      return(AUCs)
    })
    roc_stats <- c("AUC" = mean(unlist(prob_stats)), 
                   "Weighted_AUC" = sum(unlist(prob_stats) * table(data$obs)/nrow(data)))
    
    # Confusion matrix stats
    CM <- confusionMatrix(data[, "pred"], data[, "obs"])
    class_stats <- CM$byClass
    if (!is.null(dim(class_stats))) class_stats = colMeans(class_stats)
    names(class_stats) <- paste0("Mean_", names(class_stats))
    
    # Collect metrics
    stats = c(roc_stats, CM$overall[c("Accuracy","Kappa")], lloss, class_stats)
    names(stats) <- gsub("[[:blank:]]+", "_", names(stats))
  } 
  
  ## Regression
  if (is.numeric(data$obs)) {
    
    # Derive concordance
    concord = function(obs, pred, n=100000) {
      i.samp1 = sample(1:length(obs), n, replace = TRUE)
      i.samp2 = sample(1:length(obs), n, replace = TRUE)
      obs1 = obs[i.samp1]
      obs2 = obs[i.samp2]
      pred1 = pred[i.samp1]
      pred2 = pred[i.samp2]
      sum((obs1 > obs2) * (pred1 > pred2) + (obs1 < obs2) * (pred1 < pred2) + 0.5*(obs1 == obs2)) / sum(obs1 != obs2)
    }
    
    # Get y and yhat ("else" is default caret behavior)
    y = data$obs 
    if ("yhat" %in% colnames(data)) yhat = data$yhat else yhat = data$pred
    
    # Remove NA in target
    i.notna = which(!is.na(yhat))
    yhat = yhat[i.notna]
    y = y[i.notna]
    res = y - yhat
    absres = abs(res) #absolute residual
    
    # Derive stats
    spearman = cor(yhat, y, method = "spearman")
    pearson = cor(yhat, y, method = "pearson")
    IqrE = IQR(res)
    AUC = concord(yhat, y)
    MAE = mean(absres)
    MdAE = median(absres)
    MAPE = mean(absres / abs(y))
    MdAPE = median(absres / abs(y))
    sMAPE = mean(2 * absres / (abs(yhat) + abs(y)))
    sMdAPE = median(2 * absres / (abs(yhat) + abs(y)))
    MRAE = mean(absres / abs(y - mean(y)))
    MdRAE = median(absres / abs(y - mean(y)))
    
    # Collect metrics
    stats = c(spearman, pearson, IqrE, AUC, MAE, MdAE, MAPE, MdAPE, sMAPE, sMdAPE, MRAE, MdRAE)
    names(stats) = c("spearman","pearson","IqrE","AUC", "MAE", "MdAE", "MAPE", "MdAPE", "sMAPE", "sMdAPE", "MRAE", "MdRAE")
  }
  stats
}

# Undersample
undersample_n = function(df, target_name = "target", n_maxpersample) {
  #browser()
  i.samp = unlist(map(levels(df[[target_name]]), ~ {
    which(df[[target_name]] == .x) %>% sample(min(length(.), n_maxpersample))
  }))
  list(df = df[i.samp,], 
       b_sample = df[[target_name]][i.samp] %>% (function(.) summary(.)/length(.)), 
       b_all = df[[target_name]] %>% (function(.) summary(.)/length(.)))
}


#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data and transform -----------------------------------------------------------------------------------------------

# ABT
df = read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE)
df$deck = as.factor(str_sub(df$cabin, 1, 1))

# Read column metadata 
# no column metadata in this example

# Define target 
df$target = factor(ifelse(df$survived == 0, "N", "Y"), levels = c("N","Y"))
summary(df$target)



# Adapt nominal variables ----------------------------------------------------------------------------------

# Define nominal features
nomi = c("pclass","sex","sibsp","parch","deck","embarked","home.dest")

# Make them factors
df[nomi] = map(df[nomi], ~ as.factor(as.character(.)))

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))

# Remove variables with only 1 value
remove = nomi[map_lgl(df[nomi], ~ length(levels(.)) <= 1)]
nomi = setdiff(nomi, remove) 

# Add _OTHER_ to all nominal variables
df[nomi] = map(df[nomi], ~ fct_expand(.,"_OTHER_"))

# Derive "toomanys"
topn_toomany = 8
levinfo = map_int(df[nomi], ~ length(levels(.))) 
data.frame(n = levinfo[order(levinfo, decreasing = TRUE)])
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #Set exception for important variables

# Encode toomany variables (by count encoding) TODO: Arno count encoding
l.encoding = list()
for (var in toomany) {
  #var="deck"
  l.encoding[[var]] = table(df[[var]]) %>% .[order(., decreasing = TRUE)] %>% 
    {setNames(1:length(.), names(.))}
  # # Alternative: Arncoding 
  # l.encoding[[var]] = table(df[[var]]) %>% .[order(., decreasing = TRUE)] %>% 
  #   {c(setNames(rep(0, topn_toomany), names(.)[1:topn_toomany]), #encode topn_toomany levels all with 0 (at "start")
  #      setNames(1:(length(.) - topn_toomany), names(.)[length(.):(topn_toomany+1)]))} #put most freq at "end"
  df[paste0(var,"_ENCODED")] =  l.encoding[[var]][df[[var]]]
}
summary(df[paste0(toomany,"_ENCODED")])

# Overwrite toomany with just topn_toomany levels and rest in "_OTHER_"
df[toomany] = map(df[toomany], ~ fct_lump(fct_infreq(.), topn_toomany, other_level = "_OTHER_")) #collapse

# Save levels
l.levels = map(df[nomi], ~ levels(.))

# Catch information about levels (needed for scoring)
l.metanomi = list(levels = l.levels, encoding = l.encoding)




# Adapt metric variables ----------------------------------------------------------------------------------

metr = c("age","fare")
summary(df[metr])

# Impute 0
(miss = metr[map_lgl(df[metr], ~ any(is.na(.)))]) #vars with misings
(contain_0 = miss[map_lgl(df[miss], ~ between(0, min(., na.rm = TRUE), max(., na.rm = TRUE)))]) #vars with missing and 0
mins = map_dbl(df[contain_0], ~ min(., na.rm = TRUE)) #get minimum for these vars
df[contain_0] = map(df[contain_0], ~ . + min(., na.rm = TRUE) + 1) #shift
df[miss] = map(df[miss], ~ impute(., type = "zero"))
l.metametr = list(mins = mins)



# Define features
features = c(metr, nomi, paste0(toomany,"_ENCODED"))




#######################################################################################################################-
#|||| Train model ||||----
#######################################################################################################################-

# Undersample 
c(df.train, b_sample, b_all) %<-%  (df %>% undersample_n(n_maxpersample = 1e7)) #Take all but n_maxpersample at most
summary(df.train$target); b_sample; b_all

# Save Metainformation (needed for scoring)
l.metasample = list(b_all = b_all, b_sample = b_sample)




# Fit ----------------------------------------------------------------------------------

## NOT RUN: Use this snippet for tuning
skip = function() {
  
  # Initialize parallel processing
  closeAllConnections() #reset
  Sys.getenv("NUMBER_OF_PROCESSORS") 
  cl = makeCluster(4)
  registerDoParallel(cl) 
  # stopCluster(cl); closeAllConnections() #stop cluster
  
  # Fit
  set.seed(999)
  l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
  ctrl_idx_nopar_fff = trainControl(method = "cv", number = 1, index = l.index, 
                                    returnResamp = "final", returnData = FALSE,
                                    allowParallel = FALSE, #no parallel e.g. for xgboost on big data or with DMatrix
                                    summaryFunction = mysummary, classProbs = TRUE, 
                                    indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
  formula_rightside = as.formula(paste("~", paste(features, collapse = " + ")))
  
  
  df.train$age[-c(1:1)] = 0
  df.train$fare[1] = 0
  
  options(na.action = "na.pass")
  m.train = sparse.model.matrix(formula_rightside, data = df.train[features])
  options(na.action = "na.omit")
  DM.train = xgb.DMatrix(m.train)
  fit = train(DM.train, df.train$target,              
              trControl = ctrl_idx_nopar_fff, metric = "AUC", 
              method = "xgbTree", 
              tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = c(3), 
                                     eta = c(0.01), gamma = 0, colsample_bytree = c(0.9), 
                                     min_child_weight = c(5), subsample = c(0.9)))
  fit
  plot(fit)
  #summary(predict(fit, DM.train, type = "prob"))

}


# Final Fit
ctrl_none = trainControl(method = "none", returnData = FALSE, classProbs = TRUE)
formula_rightside = as.formula(paste(" ~ ", paste(features, collapse = " + ")))
options(na.action = "na.pass")
DM.train = xgb.DMatrix(sparse.model.matrix(formula_rightside, data = df.train[features]))
options(na.action = "na.omit")
Sys.time()
fit = train(DM.train, df.train$target,              
            trControl = ctrl_none, metric = "Mean_AUC", 
            method = "xgbTree", 
            tuneGrid = expand.grid(nrounds = 700, max_depth = 3, 
                                   eta = 0.01, gamma = 0, colsample_bytree = 0.9, 
                                   min_child_weight = 5, subsample = 0.9))
Sys.time()

# Check
mysummary(data.frame(y = df.train$target, 
                     yhat = scale_pred(predict(fit, DM.train, type = "prob"), b_sample, b_all)))




# Save Metadata ----------------------------------------------------------------------------------

l.metadata = list("nomi" = l.metanomi, metr = l.metametr, "features" = list("metr" = metr, "nomi" = nomi),
                  "sample" = l.metasample, "fit" = fit)
save(l.metadata, file = paste0(dataloc,"METADATA.RData"))







