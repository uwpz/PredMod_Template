
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



## Functions

# Calculate probabilty on all data from probabilt from sample data and the corresponding (prior) base probabilities 
prob_samp2full = function(p_sample, b_sample, b_all) {
  p_all = b_all * ((p_sample - p_sample*b_sample) / 
                     (b_sample - p_sample*b_sample + b_all*p_sample - b_sample*b_all))
  p_all
}

# Custom summary function for classification performance (use by caret)
mysummary_class = function(data, lev = NULL, model = NULL) 
{
  #browser()
  # Get y and yhat ("else" is default caret behavior)
  if ("y" %in% colnames(data)) y = data$y else y = data$obs 
  if ("yhat" %in% colnames(data)) yhat = data$yhat else yhat = data[[levels(y)[[2]]]]
  
  conf_obj = caret::confusionMatrix(factor(ifelse(yhat > 0.5,"Y","N"), levels = levels(y)), y)
  accuracy = as.numeric(conf_obj$overall["Accuracy"])
  missclassification = 1 - accuracy
  
  if (is.numeric(yhat)) { #Fix for change in caret and parallel processing
    pred_obj = ROCR::prediction(yhat, y)
    auc = ROCR::performance(pred_obj, "auc" )@y.values[[1]]
  } else {
    auc = 0
  }
  
  out = c("auc" = auc, "accuracy" = accuracy, "missclassification" = missclassification)
  out
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

# Remove variables with only 1 value
remove = nomi[map_lgl(df[nomi], ~ length(unique(.)) <= 1)]
nomi = setdiff(nomi, remove) 

# Make them factors
df[nomi] = map(df[nomi], ~ as.factor(as.character(.)))

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))

# Add _OTHER_ to all nominal variables
df[nomi] = map(df[nomi], ~ fct_expand(.,"_OTHER_"))

# Save levels
l.levels = map(df[nomi], ~ levels(.))



## Create count encoding for covariates wich "too many members" 
# Derive "toomanys"
topn_toomany = 9
levinfo = map_int(df[nomi], ~ length(levels(.))) 
data.frame(n = levinfo[order(levinfo, decreasing = TRUE)])
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #Set exception for important variables

# Create new variables wiht just topn_toomany levels and rest in "_OTHER_"
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(fct_infreq(.), topn_toomany, other_level = "_OTHER_")) #collapse
l.levels = c(l.levels,  map(df[paste0(toomany,"_OTHER_")], ~ levels(.)))
nomi = c(nomi, paste0(toomany,"_OTHER_"))

# Additionally encode toomany variables (by count encoding)
l.encoding = list()
for (var in toomany) {
  #var="dep_airport"
  tmp = table(df[[var]]) %>% .[order(., decreasing = TRUE)]
  l.encoding[[var]] = set_names(1:length(tmp), names(tmp))
  df[var] =  l.encoding[[var]][df[[var]]]
}
summary(df[toomany])



# Catch information about levels (needed for scoring)
l.metanomi = list(levels = l.levels, encoding = l.encoding)



# Adapt metric variables ----------------------------------------------------------------------------------

metr = c("age","fare")



#######################################################################################################################-
#|||| Train model ||||----
#######################################################################################################################-

# Undersample ----------------------------------------------------------------------------------

n_maxpersample = 10000000 #Take all but n_maxpersample at most
df.train = c()
for (i in 1:length(levels(df$target))) {
  i.samp = which(df$target == levels(df$target)[i])
  set.seed(i * 999)
  df.train = bind_rows(df.train, df[sample(i.samp, min(n_maxpersample, length(i.samp))),]) 
}
summary(df.train$target)
nrow(df.train)

# Get prior probabilities for rescaling scores ####
(b_all = summary(df$target)["Y"] / nrow(df))
(b_sample = summary(df.train$target)["Y"] / nrow(df.train))

# Save Metainformation (needed for scoring)
l.metasample = list( b_all = b_all, b_sample = b_sample)



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
  ctrl_index_fff = trainControl(method = "cv", number = 1, index = l.index, 
                                returnResamp = "final", returnData = FALSE,
                                allowParallel = FALSE, #!!! NO parallel in case of DGMatrix
                                summaryFunction = mysummary_class, classProbs = TRUE, 
                                indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
  formula_rightside = as.formula(paste("~", paste(c(metr,nomi), collapse = " + ")))
  
  
  df.train$age[-c(1:1)] = NA
  
  options(na.action = "na.pass")
  dm.train = xgb.DMatrix(sparse.model.matrix(formula_rightside, data = df.train[c(metr,nomi)]))
  options(na.action = "na.omit")
  fit = train(dm.train, df.train$target,              
              trControl = ctrl_index_fff, metric = "Mean_AUC", 
              method = "xgbTree", 
              tuneGrid = expand.grid(nrounds = seq(100,1100,200), max_depth = c(3), 
                                     eta = c(0.01), gamma = 0, colsample_bytree = c(0.9), 
                                     min_child_weight = c(5), subsample = c(0.9)))
  fit
  plot(fit)

}


# Final Fit
ctrl_none = trainControl(method = "none", returnData = FALSE, classProbs = TRUE)
formula_rightside = as.formula(paste(" ~ ", paste(c(metr,nomi), collapse = " + ")))
options(na.action = "na.pass")
dm.train = xgb.DMatrix(sparse.model.matrix(formula_rightside, data = df.train[c(metr,nomi)]))
options(na.action = "na.omit")
Sys.time()
fit = train(dm.train, df.train$target,              
            trControl = ctrl_none, metric = "Mean_AUC", 
            method = "xgbTree", 
            tuneGrid = expand.grid(nrounds = 700, max_depth = 3, 
                                   eta = 0.01, gamma = 0, colsample_bytree = 0.9, 
                                   min_child_weight = 5, subsample = 0.9))
Sys.time()

# Check
mysummary_class(data.frame(y = df.train$target, yhat = predict(fit, dm.train, type = "prob")[[2]]))




# Save Metadata ----------------------------------------------------------------------------------

l.metadata = list("nomi" = l.metanomi, "sample" = l.metasample, 
                  "predictors" = list("metr" = metr, "nomi" = nomi),
                  "fit" = fit)
save(l.metadata, file = paste0(dataloc,"METADATA.RData"))







