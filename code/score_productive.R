
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Locations
dataloc = "./data/"

# Libraries
library(plyr) #always load plyr before dpylr / tidyverse
library(ggplot2)
library(dplyr)
library(purrr)
library(readr)
library(forcats)
library(xgboost)
library(caret)
library(doParallel)
library(Matrix)



## Functions

# Calculate probabilty on all data from probabilt from sample data and the corresponding (prior) base probabilities 
prob_samp2full = function(p_sample, b_sample, b_all) {
  p_all = b_all * ((p_sample - p_sample*b_sample) / 
                     (b_sample - p_sample*b_sample + b_all*p_sample - b_sample*b_all))
  p_all
}



#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

# ABT
df = read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE) %>% filter(pclass != "3rd")
df$deck = as.factor(str_sub(df$cabin, 1, 1))
df$id = 1:nrow(df)

# Metadata
load(file = paste0(dataloc,"METADATA.RData"))




# Adapt nominal variables ----------------------------------------------------------------------------------

# Define nominal features
nomi = l.metadata$predictors$nomi

# Duplicate variables for encoding
toomany = names(l.metadata$nomi$encoding)
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ .)
                                    
# Make them character
df[nomi] = map(df[nomi], ~ as.character(.))

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ ifelse(is.na(.), "(Missing)", .))

# Map unknown content to _OTHER_
df[nomi] = map(nomi, ~ ifelse(df[[.]] %in% l.metadata$nomi$levels[[.]], df[[.]], "_OTHER_"))

# Make them factors with same levels as for training
df[nomi] = map(nomi, ~ factor(df[[.]], l.metadata$nomi$levels[[.]]))

# Map encoding
df[toomany] = map(toomany, ~ {l.metadata$nomi$encoding[[.]][df[[.]]]})




# Adapt metric variables ----------------------------------------------------------------------------------

metr = l.metadata$predictors$metr




#######################################################################################################################-
#|||| Score ||||----
#######################################################################################################################-

# Score and rescale ----------------------------------------------------------------------------------

formula_rightside = as.formula(paste("~", paste(c(metr,nomi), collapse = " + ")))
options(na.action = "na.pass")
dm = xgb.DMatrix(sparse.model.matrix(formula_rightside, data = df[c(metr,nomi)]))
options(na.action = "na.omit")
yhat_score = prob_samp2full(predict(l.metadata$fit, dm, type="prob")[[2]], 
                            l.metadata$sample$b_sample, l.metadata$sample$b_all)
# # Rescale 
# for (lev in colnames(yhat_score)) {yhat_score[[lev]] = 
#   yhat_score[[lev]] * l.metadata$sample$b_all[lev] / l.metadata$sample$b_sample[lev]}
# yhat_score = yhat_score / rowSums(yhat_score)




# Write scored data ----------------------------------------------------------------------------------

df.score = bind_cols(df[c("id")], "score" = round(yhat_score, 5))
write_delim(df.score, paste0(dataloc,"scoreddata.psv"), delim = "|")

