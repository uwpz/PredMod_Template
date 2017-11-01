
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load functions
source("./code/0_init.R")




#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

df.orig = read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE)
skip = function() {
  # Check some stuff
  df.tmp = df.orig %>% mutate_if(is.character, as.factor) 
  summary(df.tmp)
  table(df.orig$survived) / nrow(df.orig)
}

# "Save" orignial data
df = df.orig




# Feature engineering -------------------------------------------------------------------------------------------------------------

df$deck = as.factor(str_sub(df$cabin, 1, 1))
summary(df$deck)
# also possible: df$familysize = df$sibsp + df$parch as well as something with title from name




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
df = mutate(df, target = factor(ifelse(survived == 0, "N", "Y"), levels = c("N","Y")),
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

metr = c("age","fare") 
summary(df[metr]) 




# Create nominal variables for all metric variables (for linear models) before imputing -------------------------------

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
plots = get_plot_distr_metr_class(df, metr, missinfo = NULL)
ggsave(paste0(plotloc, "titanic_distr_metr.pdf"), suppressMessages(marrangeGrob(plots, ncol = 2, nrow = 2, top = NULL)), 
       width = 12, height = 8)

# Winsorize
df[,metr] = map(df[metr], ~ {
  q_lower = quantile(., 0.01, na.rm = TRUE)
  q_upper = quantile(., 0.99, na.rm = TRUE)
  .[. < q_lower] = q_lower
  .[. > q_upper] = q_upper
  . }
)

# Log-Transform
tolog = c("fare")
df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(.+1) else log(.)})
metr = map_chr(metr, ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metr and keep order
# for (varname in tolog) { 
#   #adapt binned name (to keep metr and metr_BINNED in sync)
#   colnames(df)[colnames(df) == paste0(varname,"_BINNED_")] = paste0(varname,"_LOG__BINNED_")
# }
names(misspct) = metr #adapt misspct names

# Plot again
plots = get_plot_distr_metr_class(df, metr, missinfo = misspct)
ggsave(paste0(plotloc, "titanic_distr_metr_final.pdf"), marrangeGrob(plots, ncol = 2, nrow = 2), 
       width = 12, height = 8)




# Removing variables -------------------------------------------------------------------------------------------

# Remove Self predictors
metr = setdiff(metr, "xxx")

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = 0)
ggsave(paste0(plotloc, "titanic_corr_metr.pdf"), plot, width = 6, height = 6)
metr = setdiff(metr, c("xxx")) #Put at xxx the variables to remove
metr_binned = setdiff(metr_binned, c("xxx_BINNED_")) #Put at xxx the variables to remove




#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------

nomi = c("pclass","sex","sibsp","parch","deck","embarked","boat","home.dest")
nomi = union(nomi, paste0("MISS_",miss)) #Add missing indicators
df[nomi] = map(df[nomi], ~ as.factor(as.character(.)))
summary(df[nomi])




# Handling factor values ----------------------------------------------------------------------------------------------

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))
summary(df[nomi])

# Create compact covariates for "too many members" columns 
topn_toomany = 30
levinfo = map_int(df[nomi], ~ length(levels(.))) 
levinfo[order(levinfo, decreasing = TRUE)]
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #Set exception for important variables
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(., topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #Exchange name
summary(df[nomi], topn_toomany + 2)

# Check
plots = get_plot_distr_nomi_class(df, nomi)
ggsave(paste0(plotloc, "titanic_distr_nomi.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2, top = NULL), 
       width = 18, height = 12)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-predictors
nomi = setdiff(nomi, "boat")

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = 0)
ggsave(paste0(plotloc, "titanic_corr_nomi.pdf"), plot, width = 12, height = 12)
nomi = setdiff(nomi, "xxx")




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

save.image("titanic_1_explore.rdata")







