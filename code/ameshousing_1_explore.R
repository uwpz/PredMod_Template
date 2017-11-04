
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load functions
source("./code/0_init.R")




#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

df.orig = read_delim(paste0(dataloc,"AmesHousing.txt"), delim = "\t", col_names = TRUE)  
colnames(df.orig)[c(22,45,46,71)] = c("Year_RemodAdd","First Flr SF", "Second Flr SF","threeSsn Porch")
colnames(df.orig) = str_replace_all(colnames(df.orig), " ", "_")
skip = function() {
  # Check some stuff
  df.tmp = df.orig %>% mutate_if(is.character, as.factor) 
  summary(df.tmp)
}

# "Save" orignial data
df = df.orig 




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
df$target = df$SalePrice
summary(df$target)

# Train/Test fold: usually split by time
df$fold = factor("train", levels = c("train", "test"))
set.seed(123)
df[sample(1:nrow(df), floor(0.3*nrow(df))),"fold"] = "test"
summary(df$fold)




#######################################################################################################################-
#|||| Metric variables: Explore and adapt ||||----
#######################################################################################################################-

# Define metric covariates -------------------------------------------------------------------------------------
metr = c("Lot_Frontage","Lot_Area","Year_Built","Year_RemodAdd","Mas_Vnr_Area","BsmtFin_SF_1","BsmtFin_SF_2",
         "Bsmt_Unf_SF","Total_Bsmt_SF","First_Flr_SF","Second_Flr_SF","Low_Qual_Fin_SF","Gr_Liv_Area",
         "Garage_Yr_Blt","Garage_Area","Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch","threeSsn_Porch","Screen_Porch",
         "Pool_Area","Misc_Val") 
summary(df[metr]) 




# Create nominal variables for all metric variables (for linear models) before imputing -------------------------------

metr_binned = paste0(metr,"_BINNED_")
df[metr_binned] = map(df[metr], ~ {
  cut(., unique(quantile(., seq(0,1,0.05), na.rm = TRUE)), include.lowest = TRUE)
})
# Convert missings to own level ("(Missing)")
df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))
summary(df[metr_binned],11)

# Remove binned variables with just 1 bin
metr_binned = setdiff(metr_binned, c("Low_Qual_Fin_SF_BINNED_","threeSsn_Porch_BINNED_",
                                     "Pool_Area_BINNED_","Misc_Val_BINNED_"))



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
plots = suppressMessages(get_plot_distr_metr_regr(df, metr, missinfo = misspct, ylim = NULL)) 
ggsave(paste0(plotloc, "ameshousing_distr_metr.pdf"), 
       suppressMessages(marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL)), 
       width = 18, height = 12)

# Winsorize
df[,metr] = map(df[metr], ~ {
  q_lower = quantile(., 0.001, na.rm = TRUE)
  q_upper = quantile(., 0.999, na.rm = TRUE)
  .[. < q_lower] = q_lower
  .[. > q_upper] = q_upper
  . }
)

# # Log-Transform
# tolog = c("TSH")
# df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(.+1) else log(.)})
# metr = map_chr(metr, ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metr and keep order
# # for (varname in tolog) { 
# #   #adapt binned name (to keep metr and metr_BINNED in sync)
# #   colnames(df)[colnames(df) == paste0(varname,"_BINNED_")] = paste0(varname,"_LOG__BINNED_")
# # }
# names(misspct) = metr #adapt misspct names




# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance
varimp = sqrt(filterVarImp(df[metr], df$target, nonpara = TRUE)) %>% .$Overall
names(varimp) = metr

# Plot 
plots = suppressMessages(get_plot_distr_metr_regr(df, metr, missinfo = misspct, varimpinfo = varimp, ylim = NULL)) 
ggsave(paste0(plotloc, "ameshousing_distr_metr_final.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       width = 18, height = 12)




# Removing variables -------------------------------------------------------------------------------------------

# Remove Self predictors
metr = setdiff(metr, "xxx")

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = 0)
ggsave(paste0(plotloc, "ameshousing_corr_metr.pdf"), plot, width = 12, height = 12)
metr = setdiff(metr, c("xxx")) #Put at xxx the variables to remove
metr_binned = setdiff(metr_binned, c("xxx_BINNED_")) #Put at xxx the variables to remove




#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------
nomi = c("MS_SubClass","MS_Zoning","Street","Alley","Lot_Shape","Land_Contour","Utilities","Lot_Config","Land_Slope",
         "Neighborhood","Condition_1","Condition_2","Bldg_Type","House_Style","Overall_Qual","Overall_Cond",
         "Roof_Style","Roof_Matl","Exterior_1st","Exterior_2nd","Mas_Vnr_Type","Exter_Qual","Exter_Cond","Foundation",
         "Bsmt_Qual","Bsmt_Cond","Bsmt_Exposure","BsmtFin_Type_1","BsmtFin_Type_2","Heating","Heating_QC","Central_Air",
         "Electrical","Bsmt_Full_Bath","Bsmt_Half_Bath","Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr",
         "Kitchen_Qual","TotRms_AbvGrd","Functional","Fireplaces","Fireplace_Qu","Garage_Type","Garage_Finish",
         "Garage_Cars","Garage_Qual","Garage_Cond","Paved_Drive","Pool_QC","Fence","Misc_Feature","Mo_Sold","Yr_Sold",
         "Sale_Type","Sale_Condition") 
nomi = union(nomi, paste0("MISS_",miss)) #Add missing indicators
df[nomi] = map(df[nomi], ~ as.factor(as.character(.)))
summary(df[nomi])




# Handling factor values ----------------------------------------------------------------------------------------------

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))
summary(df[nomi])

# Reorder some variables
tmp = c("MS_SubClass","Overall_Qual","Overall_Cond","Bedroom_AbvGr","TotRms_AbvGrd","Mo_Sold")
df[tmp] =  map(df[tmp], ~ fct_relevel(., levels(.)[order(as.numeric(levels(.)), na.last = FALSE)]))

# Create compact covariates for "too many members" columns 
topn_toomany = 30
levinfo = map_int(df[nomi], ~ length(levels(.))) 
levinfo[order(levinfo, decreasing = TRUE)]
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #Set exception for important variables
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(., topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #Exchange name
summary(df[nomi], topn_toomany + 2)

# Univariate variable importance
varimp = sqrt(filterVarImp(df[nomi], df$target, nonpara = TRUE)) %>% .$Overall
names(varimp) = nomi

# Check
plots = get_plot_distr_nomi_regr(df, nomi, varimpinfo = varimp) 
ggsave(paste0(plotloc, "ameshousing_distr_nomi.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3, top = NULL), width = 18, height = 12)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-predictors
nomi = setdiff(nomi, "xxx")

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = 0.9)
ggsave(paste0(plotloc, "ameshousing_corr_nomi.pdf"), plot, width = 12, height = 12)
nomi = setdiff(nomi, c("MISS_Garage_Yr_Blt","MISS_Garage_Area","MISS_Mas_Vnr_Area",
                       "MISS_BsmtFin_SF_1","MISS_Bsmt_Unf_SF","MISS_Total_Bsmt_SF"))




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

save.image("ameshousing_1_explore.rdata")







