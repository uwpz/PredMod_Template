
skip = function() {
  TYPE = "class"
  TYPE = "regr"
  TYPE = "multiclass"
}

color = switch(TYPE, "class" = twocol,
               "regr"  = hexcol,
               "multiclass" = fourcol)

#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load libraries and functions
source("./code/0_init.R")




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
df[sample(1:nrow(df), floor(0.3*nrow(df))),"fold"] = "test"
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
}
summary(df[metr]) 



# Create nominal variables for all metric variables (for linear models) before imputing -------------------------------

metr_binned = paste0(metr,"_BINNED_")
df[metr_binned] = map(df[metr], ~ {
  # Adapt sequence increment in case you have lots of data 
  cut(., unique(quantile(., seq(0, 1, 0.1), na.rm = TRUE)), include.lowest = TRUE)
})

# Convert missings to own level ("(Missing)")
df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))
summary(df[metr_binned],11)

# Remove binned variables with just 1 bin
onebin = metr_binned[map_lgl(metr_binned, ~ length(levels(df[[.]])) == 1)]
metr_binned = setdiff(metr_binned, onebin)




# Missings + Outliers + Skewness -------------------------------------------------------------------------------------

# Remove covariates with too many missings from metr 
misspct = map_dbl(df[metr], ~ round(sum(is.na(.)/nrow(df)), 3)) #misssing percentage
misspct[order(misspct, decreasing = TRUE)] #view in descending order
(remove = names(misspct[misspct > 0.99])) 
metr = setdiff(metr, remove)
misspct = misspct[setdiff(names(misspct), remove)]
summary(df[metr]) 

# Check for outliers and skewness
#options(warn = -1)
plots = suppressMessages(get_plot_distr_metr(df, metr, color = color, missinfo = misspct))
ggsave(paste0(plotloc, TYPE, "_distr_metr.pdf"), 
       marrangeGrob(suppressMessages(plots), ncol = 4, nrow = 2), width = 18, height = 12)
#options(warn = 0)

# Winsorize
df[metr] = map(df[metr], ~ winsorize(., 0.01, 0.99))

# Log-Transform
if (TYPE == "class") tolog = c("fare")
if (TYPE %in% c("regr","multiclass")) tolog = c("Lot_Area")
df[paste0(tolog,"_LOG_")] = map(df[tolog], ~ {if(min(., na.rm=TRUE) == 0) log(.+1) else log(.)})
metr = map_chr(metr, ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt metr and keep order
names(misspct) = map_chr(names(misspct), ~ ifelse(. %in% tolog, paste0(.,"_LOG_"), .)) #adapt misspct and keep order




# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance
varimp = filterVarImp(df[metr], df$target, nonpara = TRUE) %>% rowMeans()
varimp[order(varimp, decreasing = TRUE)]

# Plot 
plots = suppressMessages(get_plot_distr_metr(df, metr, color = color, missinfo = misspct, varimpinfo = varimp))
ggsave(paste0(plotloc, TYPE, "_distr_metr_final.pdf"), 
       marrangeGrob(suppressMessages(plots), ncol = 4, nrow = 2), width = 18, height = 12)

