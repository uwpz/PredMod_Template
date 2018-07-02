
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

# Calculate predictions from probability of sample data and the corresponding base probabilities (classification)
scale_pred = function(yhat, b_sample = NULL, b_all = NULL) {
  as.data.frame(t(t(as.matrix(yhat)) * (b_all / b_sample))) %>% (function(x) x/rowSums(x))
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




# Adapt nominal variables: Order of following steps is very important! -----------------------------------------------

# Define nominal features
nomi = l.metadata$features$nomi

# Make them character
df[nomi] = map(df[nomi], ~ as.character(.))

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ ifelse(is.na(.), "(Missing)", .))

# Create encoded variables
toomany = names(l.metadata$nomi$encoding)
df[paste0(toomany,"_ENCODED")] = map(toomany, ~ ifelse(df[[.]] %in% names(l.metadata$nomi$encoding[[.]]), 
                                                       df[[.]], "_OTHER_")) #map unknown content to _OTHER_
df[paste0(toomany,"_ENCODED")] = map(toomany, ~ {l.metadata$nomi$encoding[[.]][df[[paste0(.,"_ENCODED")]]]})
#TODO: Arncoding

# Map (now for nomi) unknown content to _OTHER_
df[nomi] = map(nomi, ~ ifelse(df[[.]] %in% l.metadata$nomi$levels[[.]], df[[.]], "_OTHER_"))

# Map nomi to factors with same levels as for training
df[nomi] = map(nomi, ~ factor(df[[.]], l.metadata$nomi$levels[[.]]))




# Adapt metric variables ----------------------------------------------------------------------------------

metr = l.metadata$features$metr

# Impute
mins = l.metadata$metr$mins
if (length(mins)) {
  df[names(mins)] = map(names(mins), ~ ifelse(df[[.]] < mins[.], mins[.], df[[.]])) #set lower values to min
  df[names(mins)] = map(names(mins), ~ df[[.]] - mins[.] + 1) #shift
}
df[metr] = map(df[metr], ~ impute(., type = "zero"))



#######################################################################################################################-
#|||| Score ||||----
#######################################################################################################################-

# Define features
features = c(metr, nomi, paste0(toomany,"_ENCODED"))

# Score and rescale 
formula_rightside = as.formula(paste("~", paste(features, collapse = " + ")))
options(na.action = "na.pass")
DM.score = xgb.DMatrix(sparse.model.matrix(formula_rightside, data = df[features]))
options(na.action = "na.omit")
yhat_score = scale_pred(predict(l.metadata$fit, DM.score, type="prob"), 
                        l.metadata$sample$b_sample, l.metadata$sample$b_all)

# Write scored data 
df.score = bind_cols(df[c("id")], "score" = round(yhat_score[,2], 5))
write_delim(df.score, paste0(dataloc,"scoreddata.psv"), delim = "|")

