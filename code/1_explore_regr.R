
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load functions
source("./code/0_init.R")




#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

df.orig = read_csv(paste0(dataloc,"thyroid.csv"), col_names = TRUE)
skip = function() {
  # Check some stuff
  df.orig %>% mutate_if(is.character, as.factor) %>% summary()
  table(df.orig$Class) / nrow(df.orig)
}

# "Save" orignial data
df = df.orig %>% filter(!is.na(T3)) 




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
df$target = df$T3 #for regression
summary(df$target)
hist(df$target, breaks = 20)

# Train/Test fold: usually split by time
df$fold = factor("train", levels = c("train", "test"))
set.seed(123)
df[sample(1:nrow(df), floor(0.4*nrow(df))),"fold"] = "test"
summary(df$fold)




#######################################################################################################################-
#|||| Metric variables: Explore and adapt ||||----
#######################################################################################################################-

# Define metric covariates -------------------------------------------------------------------------------------

metr = c("age","TSH","TT4","T4U","FTI") #for regression
summary(df[metr]) 




# Create nominal variables for all metric variables (for glmnet) before imputing -------------------------------

metr_binned = paste0(metr,"_BINNED_")
df[metr_binned] = map(df[metr], ~ {
  cut(., unique(quantile(., seq(0,1,0.1), na.rm = TRUE)), include.lowest = TRUE)
})
# Convert missings to own level ("(Missing)")
df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))




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
# tmp = df %>% mutate_at(miss, funs("MISS" = as.factor(ifelse(is.na(.), "miss", "no_miss")))) %>% 
#   rename_(.dots = setNames(as.list(paste0(miss,"_MISS")), paste0("MISS_",miss)))
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
plots = suppressMessages(get_plot_distr_metr_regr(df, metr, missinfo = misspct, ylim = c(0,6))) 
ggsave(paste0(plotloc, "distr_metr_regr.pdf"), suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL)), 
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
# for (varname in tolog) { 
#   #adapt binned name (to keep metr and metr_BINNED in sync)
#   colnames(df)[colnames(df) == paste0(varname,"_BINNED_")] = paste0(varname,"_LOG__BINNED_")
# }
names(misspct) = metr #adapt misspct names

# Plot again
plots = suppressMessages(get_plot_distr_metr_regr(df, metr, missinfo = misspct, ylim = c(0,6))) 
ggsave(paste0(plotloc, "distr_metr_regr_final.pdf"), 
       suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL)), 
       width = 18, height = 12)



# Removing variables -------------------------------------------------------------------------------------------

# Remove Self predictors
metr = setdiff(metr, "xxx")

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = 0.2)
ggsave(paste0(plotloc, "corr_metr_regr.pdf"), plot, width = 12, height = 12)
metr = setdiff(metr, c("xxx")) #Put at xxx the variables to remove
metr_binned = setdiff(metr_binned, c("xxx_BINNED_")) #Put at xxx the variables to remove




#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------

nomi = c("sex","on_thyroxine","query_on_thyroxine","on_antithyroid_medication","sick","pregnant","thyroid_surgery",
         "I131_treatment","query_hypothyroid","query_hyperthyroid","lithium","goitre","tumor","hypopituitary",
         "psych","referral_source", "Class") 
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
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(., topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #Exchange name
summary(df[nomi], topn_toomany + 2)

# Check
plots = get_plot_distr_nomi_regr(df, nomi)
#plots = get_plot_distr_nomi_regr(df, nomi) #for regression
ggsave(paste0(plotloc, "distr_nomi_regr.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3, top = NULL), 
       width = 18, height = 12)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-predictors
nomi = setdiff(nomi, "xxx")

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = 0)
ggsave(paste0(plotloc, "corr_nomi_regr.pdf"), plot, width = 12, height = 12)
nomi = setdiff(nomi, "MISS_FTI")




#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final predictors ----------------------------------------------------------------------------------------

predictors = c(metr, nomi)
formula = as.formula(paste("target", "~", paste(predictors, collapse = " + ")))
predictors_glmnet = c(metr_binned, setdiff(nomi, paste0("MISS_",miss))) #do not need indicators if binned variables
formula_glmnet = as.formula(paste("target", "~", paste(predictors_glmnet, collapse = " + ")))

# Check
summary(df[predictors])
setdiff(predictors, colnames(df))
summary(df[predictors_glmnet])
setdiff(predictors_glmnet, colnames(df))




# Save image ----------------------------------------------------------------------------------------------------------

save.image("1_explore_regr.rdata")







