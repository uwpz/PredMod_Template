
#######################################################################################################################-
#|||| Initialize ||||----
#######################################################################################################################-

# Load libraries and functions
source("./code/0_init.R")




#######################################################################################################################-
#|||| ETL ||||----
#######################################################################################################################-

# Read data --------------------------------------------------------------------------------------------------------

df.orig = bind_rows(cbind(read_csv(paste0("./data/","census-income.data"), col_names = FALSE), fold = "train"),
                    cbind(read_csv(paste0("./data/","census-income.test"), col_names = FALSE), fold = "test"))
tmp = read_delim(paste0("./data/","census_colnames.txt"), delim = "\t", col_names = FALSE) %>% .$X1
names = c(setdiff(tmp, c("adjusted_gross_income","federal_income_tax_liability","total_person_earnings",
                         "total_person_income","taxable_income_amount")),
          c("year","income","fold"))
colnames(df.orig) = names

skip = function() {
  # Check some stuff
  df.tmp = df.orig %>% mutate_if(is.character, as.factor) 
  summary(df.tmp)
  table(df.tmp$income) / nrow(df.tmp)
}

# "Save" orignial data
df = df.orig 




# Feature engineering -------------------------------------------------------------------------------------------------------------




# Define target and train/test-fold ----------------------------------------------------------------------------------

# Target
df = mutate(df, target = factor(ifelse(income == "- 50000.", "N", "Y"), levels = c("N","Y")),
                target_num = ifelse(target == "N", 0 ,1))
summary(df$target_num)
summary(df$target)

# Train/Test fold: usually split by time
df$fold = factor(df$fold, levels = c("train", "test"))
summary(df$fold)




#######################################################################################################################-
#|||| Metric variables: Explore and adapt ||||----
#######################################################################################################################-

# Define metric covariates -------------------------------------------------------------------------------------
metr = c("age","wage_per_hour","capital_gains","capital_losses","divdends_from_stocks","instance_weight",
         "weeks_worked_in_year")
summary(df[metr]) 




# Create nominal variables for all metric variables (for linear models)  -------------------------------

metr_binned = paste0(metr,"_BINNED_")
df[metr_binned] = map(df[metr], ~ {
  # Hint: Adapt sequence increment in case you have lots of data 
  cut(., unique(quantile(., seq(0, 1, 0.1), na.rm = TRUE)), include.lowest = TRUE)  
})

# # Remove binned variables with just 1 bin
# (onebin = metr_binned[map_lgl(metr_binned, ~ length(levels(df[[.]])) == 1)])
# metr_binned = setdiff(metr_binned, onebin)

# Need more detailed bins for these
tmp = c("wage_per_hour","capital_gains","capital_losses","divdends_from_stocks")
df[paste0(tmp,"_BINNED_")] = map(df[tmp], ~ {
  cut(., unique(quantile(., seq(0,1,0.01), na.rm = TRUE)), include.lowest = TRUE)
})

# Convert missings to own level ("(Missing)")
df[metr_binned] = map(df[metr_binned], ~ fct_explicit_na(., na_level = "(Missing)"))
summary(df[metr_binned],11)






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
plots = suppressMessages(get_plot_distr_metr(df, metr, color = twocol, missinfo = misspct, ylim = c(0,0.2)))
ggsave(paste0(plotloc,"census_distr_metr.pdf"), 
       marrangeGrob(suppressMessages(plots), ncol = 4, nrow = 2), width = 18, height = 12)
options(warn = 0)

# Winsorize
df[metr] = map(df[metr], ~ winsorize(., 0.01, 0.99)) #hint: one might want to plot again before deciding for log-trafo




# Final variable information --------------------------------------------------------------------------------------------

# Univariate variable importance: ONLY for non-missing observations -> need to consider also NA-percentage!
(varimp_metr = (filterVarImp(df[metr], df$target, nonpara = TRUE) %>% rowMeans() %>% 
                 .[order(., decreasing = TRUE)] %>% round(2)))
(varimp_metr_imputed = (filterVarImp(map_df(df[metr], ~ impute(.)), df$target, nonpara = TRUE) %>% rowMeans() %>% 
                           .[order(., decreasing = TRUE)] %>% round(2)))

# Plot 
options(warn = -1)
plots1 = suppressMessages(get_plot_distr_metr(df, metr, color = twocol, 
                                             missinfo = misspct, varimpinfo = varimp_metr, ylim = c(0,0.2)))
plots2 = suppressMessages(get_plot_distr_nomi(df, metr_binned, color = twocol, varimpinfo = NULL, inner_barplot = FALSE,
                                              min_width = 0.2, ylim = c(0,0.2)))
plots = list() ; for (i in 1:length(plots1)) {plots = c(plots, plots1[i], plots2[i])} #zip plots
ggsave(paste0(plotloc,"census_distr_metr_final.pdf"), 
       marrangeGrob(suppressMessages(plots), ncol = 4, nrow = 2), width = 24, height = 18)
options(warn = 0)




# Removing variables -------------------------------------------------------------------------------------------

# Remove Self features
metr = setdiff(metr, "xxx")

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
summary(df[metr])
plot = get_plot_corr(df, input_type = "metr", vars = metr, missinfo = misspct, cutoff = 0.1)
ggsave(paste0(plotloc,"census_corr_metr.pdf"), plot, width = 9, height = 9)
remove = c("xxx") #put at xxx the variables to remove
metr = setdiff(metr, c("xxx")) #remove
metr_binned = setdiff(metr_binned, paste0(c("xxx"),"_BINNED_")) #keep "binned" version in sync




# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
df$fold_test = factor(ifelse(df$fold == "test", "Y", "N"))
(varimp_metr_fold = filterVarImp(df[metr], df$fold_test, nonpara = TRUE) %>% rowMeans() %>%
    .[order(., decreasing = TRUE)] %>% round(2))




#######################################################################################################################-
#|||| Nominal variables: Explore and adapt ||||----
#######################################################################################################################-

# Define nominal covariates -------------------------------------------------------------------------------------

nomi = c("class_of_worker","industry_code","occupation_code","education","enrolled_in_edu_inst_last_wk",
         "marital_status","major_industry_code","major_occupation_code","race","hispanic_Origin","sex",
         "member_of_a_labor_union","reason_for_unemployment","full_or_part_time_employment_stat",
         "tax_filer_status","region_of_previous_residence","state_of_previous_residence",
         "detailed_household_and_family_stat","detailed_household_summary_in_household","migration_code_change_in_msa",
         "migration_code_change_in_reg","migration_code_move_within_reg","live_in_this_house_1_year_ago",
         "migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18",
         "country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship",
         "own_business_or_self_employed","fill_inc_questionnaire_for_veterans_admin","veterans_benefits","year") 
df[nomi] = map(df[nomi], ~ as.factor(as.character(.))) #map to factor
summary(df[nomi])




# Handling factor values ----------------------------------------------------------------------------------------------

# Convert missings to own level ("(Missing)")
df[nomi] = map(df[nomi], ~ fct_explicit_na(.))
summary(df[nomi])

# Create compact covariates for "too many members" columns 
topn_toomany = 20
(levinfo = map_int(df[nomi], ~ length(levels(.))) %>% .[order(., decreasing = TRUE)]) #number of levels
(toomany = names(levinfo)[which(levinfo > topn_toomany)])
(toomany = setdiff(toomany, c("occupation_code"))) #set exception for important variables
df[paste0(toomany,"_OTHER_")] = map(df[toomany], ~ fct_lump(., topn_toomany, other_level = "_OTHER_")) #collapse
nomi = map_chr(nomi, ~ ifelse(. %in% toomany, paste0(.,"_OTHER_"), .)) #adapt metadata (keep order)
summary(df[nomi], topn_toomany + 2)

# Univariate variable importance
(varimp_nomi = filterVarImp(df[nomi], df$target, nonpara = TRUE) %>% rowMeans() %>% 
    .[order(., decreasing = TRUE)] %>% round(2))


# Check
plots = suppressMessages(get_plot_distr_nomi(df, nomi, color = twocol, varimpinfo = varimp_nomi, inner_barplot = TRUE,
                                             min_width = 0.2, ylim = c(0,0.2)))
ggsave(paste0(plotloc,"census_distr_nomi.pdf"), marrangeGrob(plots, ncol = 3, nrow = 2), 
       width = 18, height = 12)




# Removing variables ----------------------------------------------------------------------------------------------

# Remove Self-features

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!) 
plot = get_plot_corr(df, input_type = "nomi", vars = nomi, cutoff = 0.9, textcol = "white")
ggsave(paste0(plotloc,"census_corr_nomi.pdf"), plot, width = 14, height = 14)





# Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance
(varimp_nomi_fold = filterVarImp(df[nomi], df$fold_test, nonpara = TRUE) %>% rowMeans() %>% 
   .[order(., decreasing = TRUE)] %>% round(2))





#######################################################################################################################-
#|||| Prepare final data ||||----
#######################################################################################################################-

# Define final features ----------------------------------------------------------------------------------------

features = c(metr, nomi)
formula = as.formula(paste("target", "~", paste(features, collapse = " + ")))
formula_rightside = as.formula(paste("~", paste(features, collapse = " + ")))
features_binned = c(metr_binned, nomi) #do not need indicators if binned variables
formula_binned = as.formula(paste("target", "~", paste(features_binned, collapse = " + ")))
formula_binned_rightside = as.formula(paste("~", paste(features_binned, collapse = " + ")))

# Check
summary(df[features])
setdiff(features, colnames(df))
summary(df[features_binned])
setdiff(features_binned, colnames(df))




# Save image ----------------------------------------------------------------------------------------------------------
rm(df.orig, plots, plots1, plots2)
save.image(paste0("census_1_explore.rdata"))



