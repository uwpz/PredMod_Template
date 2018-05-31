#TODO: function impute, varimp_impute

skip = function() {
  # Set target type -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
  TYPE = "class"
  TYPE = "regr"
  TYPE = "multiclass"
}



#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load libraries and functions
source("./code/0_init.R")

# Adapt some parameter differnt for target types -> REMOVE AND ADAPT AT APPROPRIATE LOCATION FOR A USE-CASE
color = switch(TYPE, "class" = twocol, "regr" = hexcol, "multiclass" = fourcol)
cutoff = switch(TYPE, "class" = 0.1, "regr"  = 0.9, "multiclass" = 0.9)
ylim = switch(TYPE, "class" = NULL, "regr"  = c(0,2.5e5), "multiclass" = c(0,2.5e5))
plotloc = paste0(plotloc,TYPE,"/")




#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

if (TYPE == "class") df.orig = read_csv(paste0(dataloc,"titanic.csv"), col_names = TRUE)
if (TYPE %in% c("regr","multiclass")) {
  df.orig = read_delim(paste0(dataloc,"AmesHousing.txt"), delim = "\t", col_names = TRUE) 
  colnames(df.orig) = str_replace_all(colnames(df.orig), " ", "_")
}

skip = function() {
  # Check some stuff
  df.tmp = df.orig %>% mutate_if(is.character, as.factor) 
  summary(df.tmp)
  if (TYPE == "class") table(df.tmp$survived) / nrow(df.tmp)
  if (TYPE == "regr") { hist(df.tmp$SalePrice, 30); hist(log(df.tmp$SalePrice), 30) }
}

# "Save" orignial data
df = df.orig 




# Feature engineering -------------------------------------------------------------------------------------------------------------

if (TYPE == "class") { 
  df$deck = as.factor(str_sub(df$cabin, 1, 1))
  summary(df$deck) 
  # also possible: df$familysize = df$sibsp + df$parch as well as something with title from name
}
if (TYPE %in% c("regr","multiclass")) {
  # number of rooms, sqm_per_room, ...
}




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
if (TYPE == "class") { 
  df = mutate(df, target = factor(ifelse(survived == 0, "N", "Y"), levels = c("N","Y")),
                  target_num = ifelse(target == "N", 0 ,1))
  summary(df$target_num)
}
if (TYPE == "regr") df$target = df$SalePrice
if (TYPE == "multiclass") {
  df$target = as.factor(paste0("Cat_", 
                  as.numeric(cut(df.orig$SalePrice, c(-Inf,quantile(df.orig$SalePrice, c(.25,.50,.75)),Inf)))))
}
summary(df$target)

# Train/Test fold: usually split by time
df$fold = factor("train", levels = c("train", "test"))
set.seed(123)
df[sample(1:nrow(df), floor(0.3*nrow(df))),"fold"] = "test" #70/30 split
summary(df$fold)




#######################################################################################################################-
#|||| Metric variables: Explore and adapt ||||----
#######################################################################################################################-

# Define metric covariates -------------------------------------------------------------------------------------

if (TYPE == "class") metr = c("age","fare")
if (TYPE %in% c("regr","multiclass")) {
  metr = c("Lot_Frontage","Lot_Area","Year_Built","Year_RemodAdd","Mas_Vnr_Area","BsmtFin_SF_1","BsmtFin_SF_2",
         "Bsmt_Unf_SF","Total_Bsmt_SF","first_Flr_SF","second_Flr_SF","Low_Qual_Fin_SF","Gr_Liv_Area",
         "Garage_Yr_Blt","Garage_Area","Wood_Deck_SF","Open_Porch_SF","Enclosed_Porch","threeSsn_Porch","Screen_Porch",
         "Pool_Area","Misc_Val") 
  df[metr] = map(df[metr], ~ na_if(., 0)) #zeros are always missing here
}
summary(df[metr]) 




# Create nominal variables for all metric variables (for linear models)  -------------------------------

metr_binned = paste0(metr,"_BINNED_")
df[metr_binned] = map(df[metr], ~ {
  # Hint: Adapt sequence increment in case you have lots of data 
  cut(., unique(quantile(., seq(0, 1, 0.1), na.rm = TRUE)), include.lowest = TRUE)  
})

# Convert missings to own level ("(Missing)")
df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))
summary(df[metr_binned],11)

# Remove binned variables with just 1 bin
#(onebin = metr_binned[map_lgl(metr_binned, ~ length(levels(df[[.]])) == 1)])
#metr_binned = setdiff(metr_binned, onebin)




# Missings + Outliers + Skewness -------------------------------------------------------------------------------------

# Remove covariates with too many missings from metr 
misspct = map_dbl(df[metr], ~ round(sum(is.na(.)/nrow(df)), 3)) #misssing percentage
misspct[order(misspct, decreasing = TRUE)] #view in descending order
(remove = names(misspct[misspct > 0.99])) #vars to remove
metr = setdiff(metr, remove) #adapt metadata
metr_binned = setdiff(metr_binned, paste0(remove,"_BINNED_")) #keep "binned" version in sync

# Check for outliers and skewness
summary(df[metr]) 
options(warn = -1)
plots = suppressMessages(get_plot_distr_metr(df, metr, color = color, missinfo = misspct, ylim = ylim))
ggsave(paste0(plotloc, TYPE, "_distr_metr.pdf"), 
       marrangeGrob(suppressMessages(plots), ncol = 4, nrow = 2), width = 18, height = 12)
options(warn = 0)

# Winsorize
df[metr] = map(df[metr], ~ winsorize(., 0.01, 0.99)) #hint: one might want to plot again before deciding for log-trafo

# Log-Transform
if (TYPE == "class") tolog = c("fare")
if (TYPE %in% c("regr","multiclass")) tolog = c("Lot_Area","Open_Porch_SF","Misc_Val")
df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(.+1) else log(.)})
metr = map_chr(metr, ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metadata (keep order)
names(misspct) = map_chr(names(misspct), ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #keep misspct in sync




# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance: ONLY for non-missing observations -> need to consider also NA-percentage!
(varimp_metr = (filterVarImp(df[metr], df$target, nonpara = TRUE) %>% rowMeans() %>% 
                 .[order(., decreasing = TRUE)] %>% round(2)))
(varimp_metr_imputed = (filterVarImp(map_df(df[metr], ~ impute(.)), df$target, nonpara = TRUE) %>% rowMeans() %>% 
                           .[order(., decreasing = TRUE)] %>% round(2)))

# Plot 
options(warn = -1)
plots1 = suppressMessages(get_plot_distr_metr(df, metr, color = color, 
                                             missinfo = misspct, varimpinfo = varimp_metr, ylim = ylim))
plots2 = suppressMessages(get_plot_distr_nomi(df, metr_binned, color = color, varimpinfo = NULL, inner_barplot = FALSE,
                                              min_width = 0.2, ylim = ylim))
plots = list() ; for (i in 1:length(plots1)) {plots = c(plots, plots1[i], plots2[i])} #zip plots
ggsave(paste0(plotloc, TYPE, "_distr_metr_final.pdf"), 
       marrangeGrob(suppressMessages(plots), ncol = 4, nrow = 2), width = 24, height = 18)
options(warn = 0)




# Removing variables -------------------------------------------------------------------------------------------

# Remove Self features
metr = setdiff(metr, "xxx")

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = cutoff)
ggsave(paste0(plotloc, TYPE, "_corr_metr.pdf"), plot, width = 9, height = 9)
remove = c("xxx") #put at xxx the variables to remove
metr = setdiff(metr, c("xxx")) #remove
metr_binned = setdiff(metr_binned, paste0(c("xxx"),"_BINNED_")) #keep "binned" version in sync




# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
df$fold_test = factor(ifelse(df$fold == "test", "Y", "N"))
(varimp_metr_fold = filterVarImp(df[metr], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
    .[order(., decreasing = TRUE)] %>% round(2))

# Plot: only variables with with highest importance
metr_toprint = names(varimp_metr_fold)[varimp_metr_fold >= 0.52]
options(warn = -1)
plots = get_plot_distr_metr(df, metr_toprint, color = twocol, target_name = "fold_test", 
                            missinfo = misspct, varimpinfo = varimp_metr_fold, ylim = ylim)
ggsave(paste0(plotloc, TYPE, "_distr_metr_final_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2), 
       width = 18, height = 12)
options(warn = 0)




# Missing indicator and imputation (must be done at the end of all processing)-----------------------------------------

# Create mising indicators
(miss = metr[map_lgl(df[metr], ~ any(is.na(.)))])
df[paste0("MISS_",miss)] = map(df[miss], ~ as.factor(ifelse(is.na(.x), "miss", "no_miss")))
summary(df[,paste0("MISS_",miss)])

# Impute missings with randomly sampled value (or median, see below)
df[miss] = map(df[miss], ~ impute(., type = "random"))
summary(df[metr]) 



#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------

if (TYPE == "class") nomi = c("pclass","sex","sibsp","parch","deck","embarked","boat","home.dest")
if (TYPE %in% c("regr","multiclass")) {
  nomi = c("MS_SubClass","MS_Zoning","Street","Alley","Lot_Shape","Land_Contour","Utilities","Lot_Config","Land_Slope",
           "Neighborhood","Condition_1","Condition_2","Bldg_Type","House_Style","Overall_Qual","Overall_Cond",
           "Roof_Style","Roof_Matl","Exterior_1st","Exterior_2nd","Mas_Vnr_Type","Exter_Qual","Exter_Cond","Foundation",
           "Bsmt_Qual","Bsmt_Cond","Bsmt_Exposure","BsmtFin_Type_1","BsmtFin_Type_2","Heating","Heating_QC","Central_Air",
           "Electrical","Bsmt_Full_Bath","Bsmt_Half_Bath","Full_Bath","Half_Bath","Bedroom_AbvGr","Kitchen_AbvGr",
           "Kitchen_Qual","TotRms_AbvGrd","Functional","Fireplaces","Fireplace_Qu","Garage_Type","Garage_Finish",
           "Garage_Cars","Garage_Qual","Garage_Cond","Paved_Drive","Pool_QC","Fence","Misc_Feature","Mo_Sold","Yr_Sold",
           "Sale_Type","Sale_Condition") 
}
nomi = union(nomi, paste0("MISS_",miss)) #add missing indicators
df[nomi] = map(df[nomi], ~ as.factor(as.character(.))) #map to factor
summary(df[nomi])




# Handling factor values ----------------------------------------------------------------------------------------------

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))
summary(df[nomi])

# Reorder "numeric" nominal variables
if (TYPE == "class") ord = c("sibsp", "parch")
if (TYPE %in% c("regr","multiclass")) {
  ord = c("MS_SubClass","Overall_Qual","Overall_Cond","Bedroom_AbvGr","TotRms_AbvGrd","Mo_Sold")
}
df[ord] =  map(df[ord], ~ fct_relevel(., levels(.)[order(as.numeric(levels(.)), na.last = FALSE)]))

# Create compact covariates for "too many members" columns 
topn_toomany = 10
(levinfo = map_int(df[nomi], ~ length(levels(.))) %>% .[order(., decreasing = TRUE)]) #number of levels
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("xxx"))) #set exception for important variables
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(., topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #adapt metadata (keep order)
summary(df[nomi], topn_toomany + 2)

# Univariate variable importance
(varimp_nomi = filterVarImp(df[nomi], df$target, nonpara = TRUE) %>% rowMeans() %>% 
    .[order(., decreasing = TRUE)] %>% round(2))


# Check
plots = suppressMessages(get_plot_distr_nomi(df, nomi, color = color, varimpinfo = varimp_nomi, inner_barplot = TRUE,
                                             min_width = 0.2, ylim = ylim))
ggsave(paste0(plotloc,TYPE,"_distr_nomi.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
       width = 18, height = 12)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-features
if (TYPE == "class") nomi = setdiff(nomi, "boat_OTHER_")
if (TYPE %in% c("regr","multiclass")) nomi = setdiff(nomi, "xxx")

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = cutoff, textcol = "white")
ggsave(paste0(plotloc,TYPE,"_corr_nomi.pdf"), plot, width = 14, height = 14)
if (TYPE %in% c("regr","multiclass")) {
  nomi = setdiff(nomi, c("MISS_BsmtFin_SF_2","MISS_BsmtFin_SF_1","MISS_second_Flr_SF","MISS_Misc_Val_LOG_",
                        "MISS_Mas_Vnr_Area","MISS_Garage_Yr_Blt","MISS_Garage_Area","MISS_Total_Bsmt_SF"))
}




# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance
(varimp_nomi_fold = filterVarImp(df[nomi], df$fold_test, nonpara = TRUE) %>% rowMeans() %>% 
   .[order(., decreasing = TRUE)] %>% round(2))

# Plot (Hint: one might want to filter just on variable importance with highest importance)
nomi_toprint = names(varimp_nomi_fold)[varimp_nomi_fold >= 0.52]
plots = get_plot_distr_nomi(df, nomi_toprint, color = twocol, target_name = "fold_test", inner_barplot = FALSE,
                            varimpinfo = varimp_nomi_fold, ylim = ylim)
ggsave(paste0(plotloc,TYPE,"_distr_nomi_folddependency.pdf"), marrangeGrob(plots, ncol = 4, nrow = 3), 
       width = 18, height = 12)




#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final features ----------------------------------------------------------------------------------------

features = c(metr, nomi)
formula = as.formula(paste("target", "~", paste(features, collapse = " + ")))
formula_rightside = as.formula(paste("~", paste(features, collapse = " + ")))
features_binned = c(metr_binned, setdiff(nomi, paste0("MISS_",miss))) #do not need indicators if binned variables
formula_binned = as.formula(paste("target", "~", paste(features_binned, collapse = " + ")))
formula_binned_rightside = as.formula(paste("~", paste(features_binned, collapse = " + ")))

# Check
summary(df[features])
setdiff(features, colnames(df))
summary(df[features_binned])
setdiff(features_binned, colnames(df))




# Save image ----------------------------------------------------------------------------------------------------------
rm(df.orig)
rm(plots)
save.image(paste0(TYPE,"_1_explore.rdata"))



