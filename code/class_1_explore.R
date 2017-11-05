
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load libraries and functions
source("./code/0_init.R")




#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

df.orig = read_csv(paste0(dataloc,"thyroid.csv"), col_names = TRUE)
skip = function() {
  # Check some stuff
  df.tmp = df.orig %>% mutate_if(is.character, as.factor) 
  summary(df.tmp)
  table(df.tmp$Class) / nrow(df.tmp)
}

# "Save" original data
df = df.orig 




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
df = mutate(df, target = factor(ifelse(Class == "negative", "N", "Y"), levels = c("N","Y")),
            target_num = ifelse(target == "N", 0 ,1))
summary(df[c("target","target_num")])

# Train/Test fold: usually split by time
df$fold = factor("train", levels = c("train", "test"))
set.seed(123)
df[sample(1:nrow(df), floor(0.3*nrow(df))),"fold"] = "test"
summary(df$fold)




#######################################################################################################################-
#|||| Metric variables: Explore and adapt ||||----
#######################################################################################################################-

# Define metric covariates -------------------------------------------------------------------------------------

metr = c("age","TSH","TT4","T4U","FTI","T3") # NOT USED: "TBG" -> only missings
summary(df[metr]) 




# Create nominal variables for all metric variables (for linear models) before imputing -------------------------------

metr_binned = paste0(metr,"_BINNED_")
df[metr_binned] = map(df[metr], ~ {
  cut(., unique(quantile(., seq(0,1,0.1), na.rm = TRUE)), include.lowest = TRUE)
})

# Convert missings to own level ("(Missing)")
df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))
summary(df[metr_binned],11)




# Handling missings ----------------------------------------------------------------------------------------------

# Remove covariates with too many missings from metr 
misspct = map_dbl(df[metr], ~ round(sum(is.na(.)/nrow(df)), 3)) #misssing percentage
misspct[order(misspct, decreasing = TRUE)] #view in descending order
(remove = names(misspct[misspct > 0.99])) 
metr = setdiff(metr, remove)
summary(df[metr]) 

# Create mising indicators
(miss = metr[map_lgl(df[metr], ~ any(is.na(.)))])
df[paste0("MISS_",miss)] = map(df[miss], ~ as.factor(ifelse(is.na(.x), "miss", "no_miss")))
summary(df[,paste0("MISS_",miss)])

# Impute missings with randomly sampled value (or median, see below)
df[miss] = map(df[miss], ~ {
  i.na = which(is.na(.x))
  .x[i.na] = sample(.x[-i.na], length(i.na) , replace = TRUE)
  #.x[i.na] = median(.x[-i.na], na.rm = TRUE) #median imputation
  .x }
)
summary(df[metr]) 




# Outliers + Skewness --------------------------------------------------------------------------------------------

# Check for outliers and skewness
plots = get_plot_distr_metr_class(df, metr, missinfo = NULL)
ggsave(paste0(plotloc, "distr_metr.pdf"), suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2)), 
       width = 18, height = 12)

# Winsorize
df[,metr] = map(df[metr], ~ {
  q_lower = quantile(., 0.01, na.rm = TRUE)
  q_upper = quantile(., 0.99, na.rm = TRUE)
  .[. < q_lower] = q_lower
  .[. > q_upper] = q_upper
  . }
)

# Log-Transform
tolog = c("TSH")
df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(.+1) else log(.)})
metr = map_chr(metr, ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metr and keep order
names(misspct) = metr #adapt misspct names




# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance
varimp = filterVarImp(df[metr], df$target, nonpara = TRUE) %>% 
  mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .$Y
names(varimp) = metr
varimp[order(varimp, decreasing = TRUE)]

# Plot 
plots = get_plot_distr_metr_class(df, metr, missinfo = misspct, varimpinfo = varimp)
ggsave(paste0(plotloc, "distr_metr_final.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), width = 18, height = 12)




# Removing variables -------------------------------------------------------------------------------------------

# Remove Self predictors
metr = setdiff(metr, "T3")

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = 0.2)
ggsave(paste0(plotloc, "corr_metr.pdf"), plot, width = 8, height = 8)
metr = setdiff(metr, c("xxx")) #Put at xxx the variables to remove
metr_binned = setdiff(metr_binned, c("xxx_BINNED_")) #Put at xxx the variables to remove




#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------

nomi = c("sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick","pregnant","thyroid_surgery",
         "I131_treatment","query_hypothyroid","lithium","goitre","tumor","psych","referral_source")  
nomi = union(nomi, paste0("MISS_",miss)) #Add missing indicators
df[nomi] = map(df[nomi], ~ as.factor(as.character(.)))
summary(df[nomi])




# Handling factor values ----------------------------------------------------------------------------------------------

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))
summary(df[nomi])

# Create compact covariates for "too many members" columns 
topn_toomany = 4
levinfo = map_int(df[nomi], ~ length(levels(.))) 
levinfo[order(levinfo, decreasing = TRUE)]
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #Set exception for important variables
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(fct_infreq(.), topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #Exchange name
summary(df[nomi], topn_toomany + 2)

# Univariate variable importance
varimp = filterVarImp(df[nomi], df$target, nonpara = TRUE) %>% 
  mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .$Y
names(varimp) = nomi
varimp[order(varimp, decreasing = TRUE)]


# Check
plots = get_plot_distr_nomi_class(df, nomi, varimpinfo = varimp)
ggsave(paste0(plotloc, "distr_nomi.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), width = 18, height = 12)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-predictors
nomi = setdiff(nomi, "xxx")

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = 0)
ggsave(paste0(plotloc, "corr_nomi.pdf"), plot, width = 12, height = 12)
nomi = setdiff(nomi, "MISS_FTI")




#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final predictors ----------------------------------------------------------------------------------------

predictors = c(metr, nomi)
formula = as.formula(paste("target", "~", paste(predictors, collapse = " + ")))
predictors_binned = c(metr_binned, setdiff(nomi, paste0("MISS_",miss))) #do not need indicators if binned variables
formula_binned = as.formula(paste("target", "~", paste(predictors_binned, collapse = " + ")))

# Check
summary(df[predictors])
setdiff(predictors, colnames(df))
summary(df[predictors_binned])
setdiff(predictors_binned, colnames(df))




# Save image ----------------------------------------------------------------------------------------------------------

save.image("1_explore.rdata")







