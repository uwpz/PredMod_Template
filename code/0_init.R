
#TODO: 
# interactions univariate


#######################################################################################################################-
# Libraries + Parallel Processing Start ----
#######################################################################################################################-

skip = function() {
  install.packages(c("tidyverse"))
}

library(Matrix)
library(plyr) #always load plyr before dplyr
library(tidyverse) #ggplot2,tibble,tidyr,readr,purrr,dplyr
library(forcats)
library(stringr)
library(lubridate)
library(bindrcpp)
library(magrittr)
library(scales)
library(zeallot)

library(doParallel)
# 
library(corrplot)
library(vcd)
library(grid)
library(gridExtra)
library(waterfalls)
library(boot)

# library(Hmisc)
# library(d3heatmap)
# library(htmlwidgets)
# library(rgl)
# 
library(caret)
library(xgboost)
library(glmnet)
library(ranger)
library(ROCR)

library(MicrosoftML)

#library(devtools); install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboostExplainer)

#library(devtools); options(devtools.install.args = "--no-multiarch"); install_github("Microsoft/LightGBM", subdir = "R-package")
library(lightgbm)

#library(h2o); h2o.init()
library(keras)


#######################################################################################################################-
# Parameters ----
#######################################################################################################################-

# Locations
dataloc = "./data/"
plotloc = "./output/"


# Colors
twocol = c("red","darkgreen")
threeDcol = colorRampPalette(c("cyan", "white", "magenta"))(100)
hexcol = colorRampPalette(c("white", "blue", "yellow", "red"))(100)
manycol = c('#00FF00','#0000FF','#FF0000','#01FFFE','#FFA6FE','#FFDB66','#006401','#010067','#95003A',
            '#007DB5','#FF00F6','#FFEEE8','#774D00','#90FB92','#0076FF','#D5FF00','#FF937E','#6A826C','#FF029D',
            '#FE8900','#7A4782','#7E2DD2','#85A900','#FF0056','#A42400','#00AE7E','#683D3B','#BDC6FF','#263400',
            '#BDD393','#00B917','#9E008E','#001544','#C28C9F','#FF74A3','#01D0FF','#004754','#E56FFE','#788231',
            '#0E4CA1','#91D0CB','#BE9970','#968AE8','#BB8800','#43002C','#DEFF74','#00FFC6','#FFE502','#620E00',
            '#008F9C','#98FF52','#7544B1','#B500FF','#00FF78','#FF6E41','#005F39','#6B6882','#5FAD4E','#A75740',
            '#A5FFD2','#FFB167','#009BFF','#E85EBE')
#barplot(1:length(manycol), col = manycol)
threecol = c("#00BA38","#B79F00","#F8766D")

# Themes
theme_my = theme_bw() +  theme(plot.title = element_text(hjust = 0.5))





#######################################################################################################################-
# My Functions ----
#######################################################################################################################-

## Calculate predictions from probability of sample data and the corresponding base probabilities (classification)
## ... possibly exponentiate for regression 
scale_pred = function(yhat, b_sample = NULL, b_all = NULL) {
  if (is.null(dim(yhat))) {
    # Regression
    yhat
  } else {
    # Classification
    as.data.frame(t(t(as.matrix(yhat)) * (b_all / b_sample))) %>% (function(x) x/rowSums(x))
  }
}



## Workaround for ggsave and marrangeGrob not to create first page blank
grid.draw.arrangelist <- function(x, ...) {
  for (ii in seq_along(x)) {
    if (ii > 1) grid.newpage()  # skips grid.newpage() call the first time around
    grid.draw(x[[ii]])
  }
}



## Winsorize
winsorize = function(variable, lower = NULL, upper = NULL) {
  if (!is.null(lower)) {
    q_lower = quantile(variable, lower, na.rm = TRUE)
    variable[variable < q_lower] = q_lower
  }
  if (!is.null(upper)) {
    q_upper = quantile(variable, upper, na.rm = TRUE)
    variable[variable > q_upper] = q_upper
  }
  variable
}



## Impute
impute = function(variable, type = "random") {
  i.na = which(is.na(variable))
  if (length(i.na)) {
    variable[i.na] = switch(type, 
        "random" = sample(variable[-i.na], length(i.na) , replace = TRUE) ,
        "median" = median(variable[-i.na], na.rm = TRUE),
        "zero" = 0)
  }
  variable 
}




## Undersample
undersample_n = function(df, target_name = "target", n_maxpersample) {
  #browser()
  i.samp = unlist(map(levels(df[[target_name]]), ~ {
    which(df[[target_name]] == .x) %>% sample(min(length(.), n_maxpersample))
  }))
  list(df = df[i.samp,], 
       b_sample = df[[target_name]][i.samp] %>% (function(.) summary(.)/length(.)), 
       b_all = df[[target_name]] %>% (function(.) summary(.)/length(.)))
}



## Custom summary function for classification performance (use by caret)
mysummary = function(data, lev = NULL, model = NULL) 
{
  #browser()
  # Adapt target observations
  if ("y" %in% colnames(data)) data$obs = data$y

  # Switch colnames in case of classification
  colnames(data) = gsub("yhat.","",colnames(data))
  
  ## Classification or Multiclass-Classifiction
  if (is.factor(data$obs)) {
    # Adapt prediction observations
    if (!("pred" %in% colnames(data))) data$pred = factor(levels(data$obs)[apply(data[levels(data$obs)], 1, 
                                                                                 function(x) which.max(x))], 
                                                          levels = levels(data$obs))
    if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) stop("levels of observed and predicted data do not match")
    
    # Logloss stats
    if (is.null(lev)) lev = levels(data$obs)
    lloss <- mnLogLoss(data = data, lev = lev, model = model)
    
    # AUC stats
    prob_stats = map(levels(data[, "pred"]), function(x) {
      AUCs <- try(ModelMetrics::auc(ifelse(data[, "obs"] == x, 1, 0), data[, x]), silent = TRUE)
      AUCs = max(AUCs, 1 - AUCs)
      return(AUCs)
    })
    roc_stats = c("AUC" = mean(unlist(prob_stats)), 
                  "Weighted_AUC" = sum(unlist(prob_stats) * table(data$obs)/nrow(data)))
    
    # Confusion matrix stats
    CM = confusionMatrix(data[, "pred"], data[, "obs"])
    class_stats = CM$byClass
    if (!is.null(dim(class_stats))) class_stats = colMeans(class_stats)
    names(class_stats) <- paste0("Mean_", names(class_stats))
    
    # Collect metrics
    stats = c(roc_stats, CM$overall[c("Accuracy","Kappa")], lloss, class_stats)
    names(stats) = gsub("[[:blank:]]+", "_", names(stats))
  } 

  ## Regression
  if (is.numeric(data$obs)) {
    
    # Derive concordance
    concord = function(obs, pred, n=100000) {
      i.samp1 = sample(1:length(obs), n, replace = TRUE)
      i.samp2 = sample(1:length(obs), n, replace = TRUE)
      obs1 = obs[i.samp1]
      obs2 = obs[i.samp2]
      pred1 = pred[i.samp1]
      pred2 = pred[i.samp2]
      sum((obs1 > obs2) * (pred1 > pred2) + (obs1 < obs2) * (pred1 < pred2) + 0.5*(obs1 == obs2)) / sum(obs1 != obs2)
    }
    
    # Get y and yhat ("else" is default caret behavior)
    y = data$obs 
    if ("yhat" %in% colnames(data)) yhat = data$yhat else yhat = data$pred
    
    # Remove NA in target
    i.notna = which(!is.na(yhat))
    yhat = yhat[i.notna]
    y = y[i.notna]
    res = y - yhat
    absres = abs(res) #absolute residual
    
    # Derive stats
    spearman = cor(yhat, y, method = "spearman")
    pearson = cor(yhat, y, method = "pearson")
    IqrE = IQR(res)
    AUC = concord(yhat, y)
    MAE = mean(absres)
    MdAE = median(absres)
    MAPE = mean(absres / abs(y))
    MdAPE = median(absres / abs(y))
    sMAPE = mean(2 * absres / (abs(yhat) + abs(y)))
    sMdAPE = median(2 * absres / (abs(yhat) + abs(y)))
    MRAE = mean(absres / abs(y - mean(y)))
    MdRAE = median(absres / abs(y - mean(y)))
    
    # Collect metrics
    stats = c(spearman, pearson, IqrE, AUC, MAE, MdAE, MAPE, MdAPE, sMAPE, sMdAPE, MRAE, MdRAE)
    names(stats) = c("spearman","pearson","IqrE","AUC", "MAE", "MdAE", "MAPE", "MdAPE", "sMAPE", "sMdAPE", "MRAE", "MdRAE")
  }
  stats
}



## Get plot list of metric variables vs target 
get_plot_distr_metr = function(df.plot = df, vars = metr, target_name = "target", 
                               missinfo = NULL, varimpinfo = NULL, nbins = 20, offset = 14, ylim = NULL, 
                               legend_only_in_1stplot = TRUE, color) {
  ## Classification
  if (is.factor(df.plot[[target_name]])) {
    # Get levels of target
    levs_target = levels(df.plot[[target_name]])
    
    # Calculate missinfo
    if (is.null(missinfo)) missinfo = map_dbl(df.plot[vars], ~ round(sum(is.na(.)/nrow(df.plot)), 3))
    
    # Loop over vars
    plots = map(vars, ~ {
      #.x = vars[1]
      print(.x)
      
      # Start histogram plot
      p = ggplot(data = df.plot, aes_string(x = .x)) +
        geom_histogram(aes_string(y = "..density..", fill = target_name, color = target_name), 
                       bins = nbins, position = "identity") +
        geom_density(aes_string(color = target_name)) +
        scale_fill_manual(limits = rev(levs_target), values = alpha(rev(color), .2), name = target_name) + 
        scale_color_manual(limits = rev(levs_target), values = rev(color), name = target_name) +
        #guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE))
        labs(title = paste0(.x, if (!is.null(varimpinfo)) paste0(" (VI: ", round(varimpinfo[.x],2),")")),
             x = paste0(.x," (NA: ", missinfo[.x] * 100,"%)"))
      
      # Get underlying data for max of y-value and range of x-value
      tmp = ggplot_build(p)
      
      # Inner boxplot
      p.inner = ggplot(data = df.plot, aes_string(target_name, y = .x)) +
        geom_boxplot(aes_string(color = target_name)) +
        stat_summary(aes_string(color = target_name), fun.y = mean, geom = "point", shape = 4) +
        coord_flip() +
        scale_y_continuous(limits = c(min(tmp$data[[1]]$xmin), max(tmp$data[[1]]$xmax))) +
        scale_color_manual(values = color, name = target_name) +
        theme_void() +
        theme(legend.position = "none")
      
      # Put all together
      p = p + 
        scale_y_continuous(limits = c(-tmp$layout$panel_ranges[[1]]$y.range[2]/offset, NA)) +
        theme_my +
        annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) 
      if (legend_only_in_1stplot == TRUE) {
        if (.x != vars[1]) p = p + theme(legend.position = "none") #legend only for first plot
      }
      p
    })
  }
  
  ## Regression
  if (is.numeric(df.plot[[target_name]])) {
    df.plot$dummy = "dummy"
    
    # Loop over vars
    plots = map(vars, ~ {
      #.x = vars[1]
      print(.x)
      
      # Scatterplot
      p = ggplot(data = df.plot, aes_string(x = .x, y = target_name)) +
        geom_hex() + 
        scale_fill_gradientn(colours = color) +
        geom_smooth(color = "black", level = 0.95, size = 0.5) +
        labs(title = paste0(.x, if (!is.null(varimpinfo)) paste0(" (VI: ", round(varimpinfo[.x],2),")")),
             x = paste0(.x," (NA: ", missinfo[.x] * 100,"%)"))
      if (!is.null(ylim)) p = p + ylim(ylim)
      
      # Get underlying data for max of y-value and range of x-value
      yrange = ggplot_build(p)$layout$panel_ranges[[1]]$y.range
      
      # Inner Histogram
      p.inner = ggplot(data = df.plot, aes_string(x = .x)) +
        geom_histogram(aes(y = ..density..), bins = nbins, position = "identity", color = "lightgrey") +
        geom_density(color = "black") +
        theme_void()
      
      # Inner Boxplot
      p.innerinner = ggplot(data = df.plot, aes_string(x = "dummy", y = .x)) +
        geom_boxplot() +
        coord_flip() +
        theme_void()
      
      # Put inner plots together
      p.inner = p.inner +
        scale_y_continuous(limits = c(-ggplot_build(p.inner)$layout$panel_ranges[[1]]$y.range[2]/2, NA)) +
        annotation_custom(ggplotGrob(p.innerinner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) 
      
      # Put all together
      p = p + 
        scale_y_continuous(limits = c(yrange[1] - 0.2*(yrange[2] - yrange[1]), ifelse(length(ylim), ylim[2], NA))) +
        theme(plot.title = element_text(hjust = 0.5)) + #default style here due to white spots
        annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = yrange[1]) 
      if (legend_only_in_1stplot == TRUE) {
        if (.x != vars[1]) p = p + theme(legend.position = "none") #legend only for first plot
      }
      p 
    })
  } 
  plots
}



## Get plot list of nomial variables vs target 
get_plot_distr_nomi = function(df.plot = df, vars = nomi, target_name = "target", 
                               varimpinfo = NULL, min_width = 0, inner_barplot = FALSE, ylim = NULL, decimals = 1,
                               legend_only_in_1stplot = TRUE, color) {
  #browser()
  ## Classification
  if (is.factor(df.plot[[target_name]])) {
  
    # Get levels of target
    levs_target = levels(df.plot[[target_name]])
          
    # Proportion target (for reference lines)
    refs = df.plot %>% 
      group_by_(target_name) %>% 
      summarise(ref = n()) %>% 
      mutate(ref = cumsum(ref)/sum(ref)) %>% 
      .$ref
    if (length(levs_target) > 2) refs = refs[-length(refs)] else refs = refs[2] - refs[1]
	
    # Loop over vars
    plots = map(vars, ~ {
      #.x = vars[1]
      print(.x)
      
      # Check for multiclass
      if (length(levels(df.plot[[target_name]])) > 2) multiclass = TRUE else multiclass = FALSE
      
      # Proportion nominal variable
      df.hlp = df.plot %>% 
        group_by_(.x) %>% summarise(n = n()) %>% ungroup() %>% 
        mutate(prop = n/sum(n), width = n/max(n))

      # Prepare data for plotting
      df.ggplot = suppressMessages(
        df.plot %>% 
          group_by_(.x, target_name) %>% 
          summarise(y = n()) %>% ungroup() %>% complete_(c(.x, target_name), fill = list(y = 0)) %>% 
          group_by_(.x) %>% 
          mutate(y = y/sum(y)) %>% 
          left_join(df.hlp) %>% 
          mutate_(.dots = setNames(paste0("factor(as.character(",target_name,"), levels = rev(levs_target),)"), 
                                   target_name)) %>%  #reverse levels for multiclass
          mutate(width = ifelse(width < min_width, min_width, width))
      )
      
      # Adpat color (as levels are reversed)
      color = rev(color)
      
      # Just take "Y"-class in non-multiclass case
      if (!multiclass) {
        df.ggplot = df.ggplot %>% filter_(paste0(target_name," == '",levels(df.ggplot[[target_name]])[1],"'")) 
        df.hlp = df.ggplot #fix problem with missing combinations
      }
      
      # Plot
      p = ggplot(df.ggplot, aes_string(x = .x, y = "y", fill = target_name)) + 
        geom_bar(stat = "identity", position = ifelse(multiclass == TRUE, "fill", "stack"), 
                 width = df.ggplot$width, color = "black") +
        scale_fill_manual(values = alpha(color, 0.2)) +
        scale_x_discrete(labels = paste0(df.hlp[[.x]], " (", round(100 * df.hlp[["prop"]], decimals), "%)")) + 
        geom_hline(yintercept = refs, size = 0.5, colour = "black", linetype = 3) +
        labs(title = paste0(.x, if (!is.null(varimpinfo)) paste0(" (VI:", round(varimpinfo[.x], 2), ")")), 
             x = "", 
             y = paste0("Proportion Target")) +
        coord_flip() +
        theme_my 
      if (!is.null(ylim)) p = p + ylim(ylim)
      
      # Get underlying data for max of y-value and range of x-value
      yrange = ggplot_build(p)$layout$panel_ranges[[1]]$x.range
      
      if (inner_barplot) {   
        # Inner Barplot
        p.inner = ggplot(data = df.hlp, aes_string(x = .x, y = "prop")) +
          geom_bar(stat= "identity", fill = "grey", colour = "black", width = 0.9) +
          coord_flip() +
          #scale_x_discrete(limits = rev(df.tmp2[[.]]), labels = rev(df.tmp2$label)) +
          theme_void()
        
        # Put all together
        p = p + 
          scale_y_continuous(limits = c(yrange[1] - 0.2*(yrange[2] - yrange[1]), ifelse(length(ylim), ylim[2], NA))) +
          theme_my +
          annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) +
          geom_hline(yintercept = 0, size = 0.5, colour = "black", linetype = 1)
      }
      if (legend_only_in_1stplot == TRUE) {
        if (.x != vars[1]) p = p + theme(legend.position = "none") #legend only for first plot
      }
      p
    })
  }
  
  ## Regression
  if (is.numeric(df.plot[[target_name]])) {
    # Drop unused levels
    df.plot = droplevels(df.plot)
    
    # Loop over vars
    plots = map(vars, ~ {
      #.x = vars[1]
      print(.x)
      
      # Main Boxplot
      p = ggplot(df.plot, aes_string(x = .x, y = target_name)) +
        geom_boxplot(varwidth = TRUE, outlier.size = 0.5) +
        stat_summary(aes(group = 1), fun.y = median, geom = "line", color = "black", linetype = 1) +
        stat_summary(aes(group = 1), fun.y = mean, geom = "line", color = "red", linetype = 2) +
        stat_summary(aes(group = 1), fun.y = mean, geom = "point", color = "red", shape = 4) +
        coord_flip() +
        scale_x_discrete(labels = paste0(levels(df.plot[[.x]]), " (", 
                                         round(100 * table(df.plot[[.x]])/nrow(df.plot), decimals), "%)")) +
        labs(title = paste0(.x, if (!is.null(varimpinfo)) paste0(" (VI: ", round(varimpinfo[.x],2),")")), x = "") +
        theme_my +
        theme(legend.position = "none") 
      if (!is.null(ylim)) p = p + ylim(ylim)
      
      # Get underlying data for max of y-value and range of x-value
      yrange = ggplot_build(p)$layout$panel_ranges[[1]]$x.range
      
      if (inner_barplot) {  
        # Inner Barplot
        p.inner = ggplot(data = df.plot, aes_string(x = .x)) +
          geom_bar(fill = "grey", colour = "black", width = 0.9) +
          coord_flip() +
          theme_void()
        
        # Put all together
        p = p + 
          scale_y_continuous(limits = c(yrange[1] - 0.2*(yrange[2] - yrange[1]), ifelse(length(ylim), ylim[2], NA))) +
          theme_my +
          annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = yrange[1]) 
      }
      p   
    })
  }
  plots
}
  

# Get plot list of correlations of variables
get_plot_corr <- function(df.plot = df, input_type = "metr" , vars = metr, cutoff = 0, textcol = "white",
                          missinfo = NULL, method = "spearman") {

  # Correlation matrix
  if (input_type == "metr") {
    ## For metric variables
    m.corr = abs(cor(df.plot[vars], method = tolower(method), use = "pairwise.complete.obs"))
    
    # Calculate missinfo
    if (is.null(missinfo)) missinfo = map_dbl(df.plot[vars], ~ round(sum(is.na(.)/nrow(df.plot)), 3))
    
    # Adapt labels
    rownames(m.corr) = colnames(m.corr) = 
      paste0(rownames(m.corr)," (NA: ", round(100 * missinfo[rownames(m.corr)], 1),"%)")
    
  } else {
    ## For nominal variables
    p = length(vars)
    m.corr = matrix(NA, p, p)
    for (i in 1:p) {
      for (j in 1:p) {
        #print(paste0(i,"...", j))
        
        # Corrected contingency coefficient
        tab = table(df.plot[vars[c(i,j)]])
        M = min(dim(tab))
        m.corr[i,j] = assocstats(tab)$cont * sqrt(M / (M - 1))
      }
    }
    # Adapt labels
    rownames(m.corr) = colnames(m.corr) = 
      paste0(vars," (#Levs: ", map_int(df.plot[vars], ~ length(levels(.))), ")")
  }
  
  # Clip matrix
  if (cutoff != 0) {
    tokeep = which(rowSums(ifelse(m.corr > cutoff, 1, 0), na.rm = TRUE) > 1)
    m.corr = m.corr[tokeep, tokeep]
  }

  # Put in clustered order
  m.corr[which(is.na(m.corr))] = 0 #set NAs to 0
  ord = corrMatOrder(m.corr , order = "hclust")
  m.corr = m.corr[ord, ord]
  #d3heatmap(m.corr, colors = "Blues", sym = TRUE, xaxis_font_size = paste0(12, "pt"), xaxis_height = 120, yaxis_width = 160)
  
  # Output as widget (clusters again and might therefore create different order)
  #saveWidget(d3heatmap(m.corr, colors = "Blues", sym = TRUE, xaxis_font_size = paste0(h, "pt"),
  #                       xaxis_height = h*20, yaxis_width = w*40), 
  #           file = normalizePath(paste0(str_split(outpdf,".pdf", simplify = TRUE)[1,1],".html"), mustWork = FALSE))

  # Output as ggplot
  df.ggplot = as.data.frame(m.corr) %>% 
    mutate(rowvar = rownames(m.corr)) %>% 
    gather(key = colvar, value = corr, -rowvar) %>% 
    mutate(textcol = as.factor(ifelse(corr > cutoff, 2, 1)))
  p = ggplot(df.ggplot, aes(x = rowvar, y = colvar)) +
    geom_tile(aes(fill = corr)) + 
    geom_text(aes(label = round(corr, 2), colour = textcol)) +
    scale_fill_gradient(low = "white", high = "blue") +
    scale_x_discrete(limits = rev(rownames(m.corr))) +
    scale_y_discrete(limits = rownames(m.corr)) +
    scale_color_manual(values = c("black", textcol)) +
    guides(color = FALSE) +
    labs(title = paste0(if (input_type == "metr") paste0("abs. ",method," Correlation") else "Contig.Coef",
                        " (cutoff = ", cutoff, ")"), 
         fill = "", x = "", y = "") +
    theme_my + theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  p
}


## Get plot list for ROC, Confusion, Distribution, Calibration, Gain, Lift, Precision-Recall, Precision
get_plot_performance = function(yhat, y, reduce_factor = NULL, quantiles = seq(0.1, 0.9, 0.1),
                                colors = twocol, gradcol = hexcol, color = "blue",
                                ylim = NULL) {
  
  #browser()
  ## Classification
  if (is.factor(y)) {
    # Derive predicted class
    pred = factor(colnames(yhat)[apply(yhat, 1, which.max)], levels = colnames(yhat))
    
    ## Prepare "direct" information (confusion, distribution, calibration)
    conf_obj = confusionMatrix(pred, y)
    df.confu = as.data.frame(conf_obj$table)
    df.distr_all = c()
    df.calib_all = c()
    for (i in 1:length(levels(y))) {
      #i=1
      lev = levels(y)[i]
      
      # Prepare
      df.distr = data.frame(target = factor(lev, levels = levels(y)), 
                            y = as.factor(ifelse(y == lev, "Y", "N")), yhat = yhat[[lev]])
    
      df.calib = calibration(y~yhat, df.distr[c("y","yhat")], class = "Y",
                             cuts = quantile(df.distr$yhat, quantiles))$data
      df.calib$target = factor(lev, levels = levels(y))
      
      # Collect
      df.distr_all = bind_rows(df.distr_all, df.distr)
      df.calib_all = bind_rows(df.calib_all, df.calib)
    }
    
    
    
    ## Prepare "reducable" information (roc, gain, lift, precrec, prec)
    df.roc_all = c()
    df.precrec_all = c()
    df.gainlift_all = c()
    df.auc_all = c()
    for (i in 1:length(levels(y))) {
      #i=1
      lev = levels(y)[i]
      
      # Subset on level
      df.tmp = data.frame(target = factor(lev, levels = levels(y)), 
                          y = as.factor(ifelse(y == lev, "Y", "N")), yhat = yhat[[lev]])
      
      # Create performance objects
      pred_obj = prediction(df.tmp$yhat, df.tmp$y)
      df.auc = data.frame(target = factor(lev, levels = levels(y)),
                          auc = performance(pred_obj, "auc" )@y.values[[1]])
      tprfpr = performance( pred_obj, "tpr", "fpr")
      precrec = performance( pred_obj, "ppv", "rec")
      tmp = ifelse(df.tmp$y == "Y", 1, 0)[order(df.tmp$yhat, decreasing = TRUE)] 
      df.gainlift = data.frame(target = factor(lev, levels = levels(y)),
                               "x" = 100*(1:length(df.tmp$yhat))/length(df.tmp$yhat),
                               "gain" = 100*cumsum(tmp)/sum(tmp))
      df.gainlift$lift = df.gainlift$gain/df.gainlift$x 
      
      # Thin out reducable objects (for big test data as this reduces plot size)
      if (!is.null(reduce_factor)) {
        set.seed(123)
        i.reduce = sample(1:nrow(df.tmp), floor(reduce_factor * nrow(df.tmp)))
        i.reduce = i.reduce[order(i.reduce)]
        for (type in c("x.values","y.values","alpha.values")) {
          slot(tprfpr, type)[[1]] = slot(tprfpr, type)[[1]][i.reduce]
          slot(precrec, type)[[1]] = slot(precrec, type)[[1]][i.reduce]
        }
        df.gainlift = df.gainlift[i.reduce,]
      } 
      
      # Collect information of reduced information into data frames
      df.roc = data.frame(target = factor(lev, levels = levels(y)),
                          "fpr" = tprfpr@x.values[[1]], "tpr" = tprfpr@y.values[[1]],
                          "fpr_alpha" = NA, "tpr_alpha" = NA)
      i.alpha = map_int(seq(0.1,0.9,0.1), function(.) which.min(abs(tprfpr@alpha.values[[1]] - .)))
      df.roc[i.alpha,"fpr_alpha"] = df.roc[i.alpha,"fpr"]
      df.roc[i.alpha,"tpr_alpha"] = df.roc[i.alpha,"tpr"]
      
      df.precrec = data.frame(target = factor(lev, levels = levels(y)),
                              "rec" = precrec@x.values[[1]], "prec" = precrec@y.values[[1]], 
                              x = 100*(1:length(precrec@x.values[[1]]))/length(precrec@x.values[[1]]))
      df.precrec[is.nan(df.precrec$prec),"prec"] = 1
      df.precrec[i.alpha,"rec_alpha"] = df.precrec[i.alpha,"rec"]
      df.precrec[i.alpha,"prec_alpha"] = df.precrec[i.alpha,"prec"]
      df.precrec[i.alpha,"x_alpha"] = df.precrec[i.alpha,"x"]
      
      # Collect
      df.roc_all = bind_rows(df.roc_all, df.roc)
      df.auc_all = bind_rows(df.auc_all, df.auc)
      df.precrec_all = bind_rows(df.precrec_all, df.precrec)
      df.gainlift_all = bind_rows(df.gainlift_all, df.gainlift)
    }
    
    df.tmp = data.frame(100 * table(y)/length(y))
    colnames(df.tmp) = c("target", "x")
    df.tmp$y = 100
    df.gainlift_help = bind_rows(df.tmp, data.frame(target = df.tmp$target, x = 0, y = 0),
                                 data.frame(target = df.tmp$target, x = 100, y = 100)) %>% arrange(x)
    
    
    ## Plots
    # ROC
    p_roc = ggplot(df.roc_all, aes(x = fpr, y = tpr)) +
      geom_line(aes(color = target), size = .5) +
      geom_abline(intercept = 0, slope = 1, color = "grey") + 
      geom_point(aes(x = fpr_alpha, y = tpr_alpha, color = target), 
                 data = df.roc_all[!is.na(df.roc_all$fpr_alpha),], size = 3, shape = "x") +
      scale_color_manual(values = colors, 
                         labels = paste0(levels(df.roc_all$target)," (",round(df.auc_all$auc,3),")"), name = "Target (AUC)") +
      scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.1)) + 
      scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.1)) +
      labs(title = paste0("ROC (mean(AUC)=", round(mean(df.auc_all$auc),3), ")"), x = expression(paste("fpr: P(", hat(y), "=1|y=0)", sep = "")),
           y = expression(paste("tpr: P(", hat(y), "=1|y=1)", sep = ""))) + 
      #geom_label(data = data.frame(x = 0.9, y = 0.1, text = paste0("AUC: ",round(auc,3))), aes(x = x, y = y, label = text)) +
      #geom_label(aes(x, y, label = text), data.frame(x=0.7, y=0.3, text = lev), size = 7) +
      theme_my +
      theme(legend.position = c(0.75,0.5), legend.background = element_rect(color = "black"))
    
    # Condusion Matrix
    p_confu = ggplot(df.confu, aes(Prediction, Reference)) +
      geom_tile(aes(fill = Freq)) + 
      geom_text(aes(label = Freq)) +
      scale_fill_gradient(low = "white", high = "darkgrey") +
      scale_y_discrete(limits = rev(levels(df.confu$Reference))) +
      labs(title = paste0("Confusion Matrix (Accuracy = ", round(conf_obj$overall["Accuracy"], 3), ")")) +
      theme_my 
    
    # Stratified distribution of predictions (plot similar to plot_distr_metr)
    p_distr = ggplot(data = df.distr_all, aes_string("yhat")) +
      geom_histogram(aes(y = ..density.., fill = y), bins = 40, position = "identity") +
      geom_density(aes(color = y)) +
      scale_fill_manual(values = alpha(c("blue","red"), .2), name = "Target (y)") + 
      scale_color_manual(values = c("blue","red"), name = "Target (y)") +
      labs(title = "Predictions", x = expression(paste("Prediction (", hat(y),")", sep = ""))) +
      #guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE)) +
      theme(legend.position = "none") +
      facet_wrap("target", scales = "free")
    
    # Gain + Lift
    p_gain = ggplot(df.gainlift_all) +
      # geom_polygon(aes(x, y, color = target), data.frame(x = c(0,100,100*sum(tmp)/length(tmp)), y = c(0,100,100)), 
      #              fill = "grey90", alpha = 0.5) +
      geom_line(aes(x, gain, color = target), size = .5) +
      geom_line(aes(x, y, color = target), df.gainlift_help, size = .5, linetype = 2) +
      scale_color_manual(values = colors, name = "Target (y)") +
      scale_x_continuous(limits = c(0,100), breaks = seq(0,100,10)) + 
      labs(title = "Gain", x = "% Samples Tested", y = "% Samples Found") +
      theme_my +
      theme(legend.position = "none")
    p_lift = ggplot(df.gainlift_all) +
      geom_line(aes(x, lift, color = target), size = .5) +
      scale_color_manual(values = colors, name = "Target (y)") +
      scale_x_continuous(limits = c(0,100), breaks = seq(0,100,10)) + 
      labs(title = "Lift", x = "% Samples Tested", y = "Lift") +
      theme_my +
      theme(legend.position = "none") +
      facet_wrap("target", scales = "free_y")
    
    # Calibrate
    p_calib = ggplot(df.calib_all, aes(midpoint, Percent)) +
      geom_line(aes(color = target)) +
      geom_point(aes(color = target)) +  
      scale_color_manual(values = colors, name = "Target (y)") +
      geom_abline(intercept = 0, slope = 1, color = "grey") + 
      scale_x_continuous(limits = c(0,100), breaks = seq(10,90,20)) + 
      scale_y_continuous(limits = c(0,100), breaks = seq(0,100,10)) +
      labs(title = "Calibration", x = "Midpoint Predicted Event Probability", y = "Observed Event Percentage") +
      theme_my +
      theme(legend.position = "none")
    
    # Precision Recall
    p_precrec = ggplot(df.precrec_all, aes(rec, prec)) +
      geom_line(aes(color = target), size = .5) +
      #geom_point(aes(color = target), df.precrec[i.alpha,], color = "red", size = 0.8) +
      geom_point(aes(x = rec_alpha, y = prec_alpha, color = target), 
                 data = df.precrec_all[!is.na(df.precrec_all$rec_alpha),], size = 3, shape = "x") +
      scale_color_manual(values = colors, name = "Target (y)") +
      scale_x_continuous(breaks = seq(0,1,0.1)) +
      labs(title = "Precision Recall Curve", x = expression(paste("recall=tpr: P(", hat(y), "=1|y=1)", sep = "")),
           y = expression(paste("precision: P(y=1|", hat(y), "=1)", sep = ""))) +
      theme_my +
      theme(legend.position = "none")
    
    # Precision
    p_prec = ggplot(df.precrec_all, aes(x, prec)) +
      geom_line(aes(color = target), size = .5) +
      #geom_point(aes(x, prec), df.precrec[i.alpha,], color = "red", size = 0.8) +
      geom_point(aes(x = x_alpha, y = prec_alpha, color = target), 
                 data = df.precrec_all[!is.na(df.precrec_all$x_alpha),], size = 3, shape = "x") +
      scale_color_manual(values = colors, name = "Target (y)") +
      scale_x_continuous(breaks = seq(0,100,10)) +
      labs(title = "Precision", x = "% Samples Tested",
           y = expression(paste("precision: P(y=1|", hat(y), "=1)", sep = ""))) +
      theme_my +
      theme(legend.position = "none")
    
    
    # Plot
    plots = list(p_roc = p_roc, p_confu = p_confu, p_distr = p_distr, p_calib = p_calib, 
                 p_gain = p_gain, p_lift = p_lift, p_precrec = p_precrec, p_prec = p_prec)
  }
  
  ## Regression
  if (is.numeric(y)) {
    ## Prepare
    pred_obj = mysummary(data.frame(y = y, yhat = yhat))
    spearman = round(pred_obj["spearman"], 2)
    df.perf = data.frame(y = y, yhat = yhat, res = y - yhat, absres = abs(y - yhat), rel_absres = abs(y - yhat) / abs(y),
                         midpoint = cut(yhat, quantile(yhat, quantiles), include.lowest = TRUE))
    df.distr = data.frame(type = c(rep("y", length(y)), rep("yhat", length(y))),
                          value = c(y, yhat))
    df.calib = df.perf %>% group_by(midpoint) %>% summarise(y = mean(y), yhat = mean(yhat))
    
    
    ## Performance plot
    p_perf = ggplot(data = df.perf, aes_string("yhat", "y")) +
      geom_hex() + 
      scale_fill_gradientn(colors = gradcol, name = "count") +
      geom_smooth(color = "black", level = 0.95, size = 0.5) +
      geom_abline(intercept = 0, slope = 1, color = "grey") + 
      labs(title = bquote(paste("Observed vs. Fitted (", rho[spearman], " = ", .(spearman), ")", sep = "")),
           x = expression(hat(y))) +
      theme(plot.title = element_text(hjust = 0.5))
    if (length(ylim)) p_perf = p_perf + xlim(ylim) + ylim(ylim)
    
    
    ## Distribution of predictions and target (plot similar to plot_distr_metr)
    p_distr = ggplot(data = df.distr, aes_string("value")) +
      geom_histogram(aes(y = ..density.., fill = type), bins = 40, position = "identity") +
      geom_density(aes(color = type)) +
      scale_fill_manual(values = alpha(colors, .2), labels = c("y", expression(paste(hat(y)))), name = " ") + 
      scale_color_manual(values = colors, labels = c("y", expression(paste(hat(y)))), name = " ") +
      labs(title = "Distribution", x = " ") +
      guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE))
    if (length(ylim)) p_distr = p_distr + xlim(ylim)
    tmp = ggplot_build(p_distr)
    p.inner = ggplot(data = df.distr, aes_string("type", "value")) +
      geom_boxplot(aes_string(color = "type")) +
      coord_flip() +
      scale_y_continuous(limits = c(min(tmp$data[[1]]$xmin), max(tmp$data[[1]]$xmax))) +
      scale_color_manual(values = colors, name = " ") +
      theme_void() +
      theme(legend.position = "none")
    if (length(ylim)) p.inner = p.inner + ylim(ylim)
    p_distr = p_distr + 
      scale_y_continuous(limits = c(-tmp$layout$panel_ranges[[1]]$y.range[2]/10, NA)) +
      theme_my +
      annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) 
    
    
    
    ## Calibration
    p_calib = ggplot(df.calib, aes(yhat, y)) +
      geom_line(color = "black") +
      geom_point(color = "black") +  
      xlim(range(c(df.calib$y,df.calib$yhat))) +
      ylim(range(c(df.calib$y,df.calib$yhat))) +
      geom_abline(intercept = 0, slope = 1, color = "grey") + 
      labs(title = "Calibration", x = "Prediction Average (in quantile bin)", y = "Observation Average") +
      theme_my 
    
    
    ## Two sided scatter: used for Residual plots
    res_plot = function(df.plot = df.perf, x = "yhat", y = "res", title = "Residuals vs. Fitted", 
                        xlab = expression(hat(y)), ylab = expression(paste(y - hat(y)))) {
      
      
      p_res = ggplot(data = df.plot, aes_string(x, y)) +
        geom_hex() + 
        scale_fill_gradientn(colors = gradcol, name = "count") +
        geom_smooth(color = "black", level = 0.95, size = 0.5) +
        labs(title = title, x = xlab, y = ylab) +
        theme(plot.title = element_text(hjust = 0.5))
      if (length(ylim)) p_res = p_res + xlim(ylim)
      tmp = ggplot_build(p_res)
      xrange = tmp$layout$panel_ranges[[1]]$x.range
      yrange = tmp$layout$panel_ranges[[1]]$y.range
      p.inner_x = ggplot(data = df.plot, aes_string(x = x)) +
        geom_histogram(aes(y = ..density..), bins = 50, position = "identity", fill = "grey", color = "black") +
        scale_x_continuous(limits = c(xrange[1] - 0.2*(xrange[2] - xrange[1]), ifelse(length(ylim), ylim[2], NA))) +
        geom_density(color = "black") +
        theme_void()
      tmp = ggplot_build(p.inner_x)
      p.inner_x_inner = ggplot(data = df.plot, aes_string(x = 1, y = x)) +
        geom_boxplot(color = "black") +
        coord_flip() +
        scale_y_continuous(limits = c(min(tmp$data[[1]]$xmin, na.rm = TRUE), max(tmp$data[[1]]$xmax, na.rm = TRUE))) +
        theme_void()
      p.inner_x = p.inner_x + 
        scale_y_continuous(limits = c(-tmp$layout$panel_ranges[[1]]$y.range[2]/3, NA)) +
        theme_void() +
        annotation_custom(ggplotGrob(p.inner_x_inner), xmin = -Inf, xmax = Inf, ymin = -Inf, 
                          ymax = -tmp$layout$panel_ranges[[1]]$y.range[2]/(3*5)) 
      
      p.inner_y = ggplot(data = df.plot, aes_string(x = y)) +
        geom_histogram(aes(y = ..density..), bins = 50, position = "identity", fill = "grey", color = "black") +
        scale_x_continuous(limits = c(yrange[1] - 0.2*(yrange[2] - yrange[1]), NA)) +
        geom_density(color = "black") +
        coord_flip() +
        theme_void()
      tmp = ggplot_build(p.inner_y)
      p.inner_y_inner = ggplot(data = df.plot, aes_string(x = 1, y = y)) +
        geom_boxplot(color = "black") +
        #coord_flip() +
        scale_y_continuous(limits = c(min(tmp$data[[1]]$xmin, na.rm = TRUE), max(tmp$data[[1]]$xmax, na.rm = TRUE))) +
        theme_void()
      p.inner_y = p.inner_y + 
        scale_y_continuous(limits = c(-tmp$layout$panel_ranges[[1]]$x.range[2]/3, NA)) +
        theme_void() +
        annotation_custom(ggplotGrob(p.inner_y_inner), xmin = -Inf, xmax = Inf, ymin = -Inf, 
                          ymax = -tmp$layout$panel_ranges[[1]]$x.range[2]/(3*5))
      
      p_res + 
        scale_x_continuous(limits = c(xrange[1] - 0.2*(xrange[2] - xrange[1]), ifelse(length(ylim), ylim[2], NA))) +
        scale_y_continuous(limits = c(yrange[1] - 0.2*(yrange[2] - yrange[1]), NA)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        annotation_custom(ggplotGrob(p.inner_x), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = yrange[1]) +
        annotation_custom(ggplotGrob(p.inner_y), xmin = -Inf, xmax = xrange[1], ymin = -Inf, ymax = Inf)
    }
    
    p_res = res_plot()
    p_absres = res_plot(y = "absres", title = "absolute Residuals vs. Fitted", 
                        ylab = expression(paste("|",y - hat(y),"|")))
    p_rel_absres = res_plot(y = "rel_absres", title = "rel. absolute Residuals vs. Fitted", 
                            ylab = expression(paste("|",y - hat(y),"| / |y|")))
    
    # Plot
    plots = list(p_perf, p_calib, p_distr, p_res, p_absres, p_rel_absres)
  }
  
  plots
}



## Variable importance by permutation argument 
get_varimp_by_permutation = function(df.for_varimp = df.test, fit.for_varimp = fit, sparse = TRUE,
                                     feature_names = features, target_name = "target",
                                     b_sample = NULL, b_all = NULL,
                                     vars = features,  metric = "AUC") {
  
  #browser()
  
  ## Original performance
  if (is.factor(df.for_varimp[[target_name]])) type = "prob" else type = "raw"
  formula_tmp = as.formula(paste("~ -1 + ", paste(feature_names, collapse = " + "))) 
  if (sparse) {
    yhat_unscaled = predict(fit.for_varimp, 
                            xgb.DMatrix(sparse.model.matrix(formula_tmp, data = df.for_varimp[feature_names])), 
                            type = type)
  } else {
    yhat_unscaled = predict(fit.for_varimp, 
                            xgb.DMatrix(model.matrix(formula_tmp, data = df.for_varimp[feature_names])), 
                            type = type)
  }
  perf_orig = mysummary(data.frame(y = df.for_varimp[[target_name]], 
                                   yhat = scale_pred(yhat_unscaled, b_sample, b_all)))[metric]
  
  
  ## Performance for permuted variables
  set.seed(999)
  i.permute = sample(1:nrow(df.for_varimp)) #permutation vector
  start = Sys.time()
  df.varimp = foreach(i = 1:length(vars), .combine = bind_rows, 
                      .packages = c("caret","xgboost","Matrix","dplyr","purrr"), 
                      .export = c("mysummary","scale_pred")) %dopar% 
  { 
    #i=1
    df.tmp = df.for_varimp
    df.tmp[[vars[i]]] = df.tmp[[vars[i]]][i.permute] #permute
    if (sparse) {
      yhat_unscaled = predict(fit.for_varimp, 
                              xgb.DMatrix(sparse.model.matrix(formula_tmp, data = df.tmp[feature_names])), 
                              type = type)
    } else {
      yhat_unscaled = predict(fit.for_varimp, 
                              xgb.DMatrix(model.matrix(formula_tmp, data = df.tmp[feature_names])), 
                              type = type)
    }
    perf = mysummary(data.frame(y = df.tmp[[target_name]], 
                                yhat = scale_pred(yhat_unscaled, b_sample, b_all)))[metric]
    # Performance difference
    data.frame(variable = vars[i], perfdiff = max(0, perf_orig - perf), stringsAsFactors = FALSE) 
  }
  print(Sys.time() - start)
  
  # Calculate importance as scaled performance difference
  df.varimp = df.varimp %>% 
    arrange(desc(perfdiff)) %>% 
    mutate(importance = 100 * perfdiff/max(perfdiff),
           importance_cum = 100 * cumsum(perfdiff)/sum(perfdiff),
           importance_sumnormed = 100 * perfdiff/sum(perfdiff)) 
  df.varimp 
}



## Get plot list for variable importance
get_plot_varimp = function(df.plot = df.varimp, vars = topn_vars, col = c("blue","orange","red"), 
                           length_features = length(features),
                           df.plot_cv = NULL, run_name = "run", cv_lines = TRUE, cv_CI = TRUE) {
  # Subset
  df.ggplot = df.plot %>% filter(variable %in% vars)
  if (!is.null(df.plot_cv)) df.ggplot_cv = df.plot_cv %>% filter(variable %in% vars)
  
  # Importancve plot 
  p_imp = ggplot(df.ggplot) +
    geom_bar(aes(x = reorder(variable, importance), y = importance, fill = color), stat = "identity") +
    scale_fill_manual(values = col) +
    labs(title = paste0("Top ", min(length(vars), length_features)," Important Variables (of ", 
                        length_features, ")"), 
         x = "", y = "Importance (scaled to 100)") +
    coord_flip() +
    guides(fill = guide_legend(reverse = TRUE, title = "")) +
    theme_my +    
    scale_y_continuous(breaks = seq(0,100,20))
 
  # Add cv information
  if (!is.null(df.plot_cv)) {
    # Add cv lines
    if (cv_lines == TRUE) {
      p_imp = p_imp +
        geom_line(aes_string(x = "variable", y = "importance", group = run_name), data = df.ggplot_cv, 
                  color = "grey", size = 0.1) +
        geom_point(aes_string(x = "variable", y = "importance", group = run_name), data = df.ggplot_cv, 
                   color = "black", size = 0.3) 
    }
    
    # Add cv Confidence Intervals
    if (cv_CI == TRUE) {
      # Calculate confidence intervals
      df.help = df.ggplot_cv %>% 
        group_by(variable) %>% 
        summarise(sd = sd(importance)) %>% 
        left_join(select(df.ggplot, variable, importance)) %>% 
        mutate(lci = importance - 1.96*sd, rci = importance + 1.96*sd)
      p_imp = p_imp +
        geom_errorbar(aes(x = variable, ymin = lci, ymax = rci), data = df.help, size = 0.5, width = 0.25)
    }
  }
  
  
  # Cumulative importance plot
  p_impcum = ggplot(df.ggplot) +
    geom_bar(aes(x = reorder(variable, importance), y = importance_cum, fill = color), stat = "identity") +
    scale_fill_manual(values = col) +
    labs(title = paste0("Top ", min(length(vars), length_features)," Important Variables (of ", 
                        length_features, ")"), 
         x = "", y = "Cumulative Importance (in %)") +
    coord_flip(ylim = c(0, 100)) +
    guides(fill = guide_legend(reverse = TRUE, title = "")) +
    theme_my +
    scale_y_continuous(breaks = seq(0,100,20))
  
  plots = list(p_imp, p_impcum)
  plots
}



## Partial dependance on green field
get_partialdep = function(df.for_partialdep = df.test, fit.for_partialdep = fit, sparse = TRUE,
                          feature_names = features, target_name = "target",
                          b_sample = NULL, b_all = NULL,
                          vars = topn_vars, l.levs, l.quantiles) {
  #browser()
  df.partialdep = foreach(i = 1:length(vars), .combine = bind_rows, .packages = c("caret","xgboost","Matrix","dplyr"),
                          .export = c("scale_pred")) %dopar% 
  {
    print(vars[i])
    
    # Some parameters
    if (is.factor(df.for_partialdep[[target_name]])) type = "prob" else type = "raw"
    formula_tmp = as.formula(paste("~ -1 + ", paste(feature_names, collapse = " + "))) 
    
    # Initialize result set
    df.res = c()
    
    # Define grid to loop over
    if (is.factor(df.for_partialdep[[vars[i]]])) values = l.levs[[vars[i]]] else values = l.quantiles[[vars[i]]]
    
    # Loop over levels for nominal covariables or quantiles for metric covariables
    df.tmp = df.for_partialdep #save original data
    start = Sys.time()
    for (value in values ) {
      #value = values[1]
      print(value)
      df.tmp[1:nrow(df.tmp),vars[i]] = value #keep also original factor levels
      if (sparse) {
        yhat_unscaled = predict(fit.for_partialdep, 
                                xgb.DMatrix(sparse.model.matrix(formula_tmp, data = df.tmp[feature_names])), 
                                type = type)
      } else {
        yhat_unscaled = predict(fit.for_partialdep, 
                                xgb.DMatrix(model.matrix(formula_tmp, data = df.tmp[feature_names])), 
                                type = type)
      }
      # IMPORTANT: rescale before averaging
      yhat = colMeans(as.data.frame(scale_pred(yhat_unscaled, b_sample, b_all)))
      #if (type == "prob") yhat = yhat[,2]
      df.res = rbind(df.res, data.frame(variable = vars[i], value = as.character(value), 
                                        target = names(yhat), yhat = yhat,
                                        stringsAsFactors = FALSE))
    }
    print(Sys.time() - start)
    
    # Return
    df.res                        
  }
  df.partialdep
}



## Get plot list for partial dependance
get_plot_partialdep = function(df.plot = df.partialdep, vars = topn_vars,
                               df.for_partialdep = df.test, target_name = "target", 
                               ylim = NULL, colors = twocol, min_width = 0.2,
                               legend_only_in_1stplot = TRUE,
                               df.plot_cv = NULL, run_name = "run", cv_lines = TRUE, cv_CI = TRUE) {
  
  #browser()
  
  # Determine target type
  if (is.factor(df.for_partialdep[[target_name]])) type = "class" else type = "regr" 
  if (type == "class" & length(levels(df.for_partialdep[[target_name]])) > 2) type = "multiclass"
  
  # Get levels and distribution (for refernce linses) of target
  if (type != "regr") {
    levs_target = levels(df.for_partialdep[[target_name]])
    
    # Proportion target (for reference lines)
    refs = df.for_partialdep %>% 
      group_by_(target_name) %>% 
      summarise(ref = n()) %>% 
      mutate(ref = cumsum(ref)/sum(ref)) %>% 
      .$ref
    if (type == "multiclass") refs = refs[-length(refs)] else refs = refs[2] - refs[1]
  } else {
    refs = mean(df.for_partialdep[[target_name]])
  }
  
  # Adapt ylim if not set
  if (is.null(ylim)) {
    ylim = c(0,1)
    if (type == "regr") ylim = range(df.for_partialdep[[target_name]])
  }
  
  # Adapt color
  #if (type != "regr") colors = rev(colors) else colors = colors[length(colors)]
  if (type == "class") colors = colors[length(colors)]
  
  # Plot
  plots = map(vars, ~ {
    #.x = vars[3]

    print(.x)
    
    # Subset data
    df.ggplot = df.plot[df.plot$variable == .x,] 
    if (!is.null(df.plot_cv)) df.ggplot_cv = df.plot_cv[df.plot_cv$variable == .x,]
    
    # Just take "Y"-class in non-multiclass case
    if (type == "class") {
      df.ggplot = df.ggplot %>% filter_(paste0(target_name," == '",levs_target[2],"'"))
      if (!is.null(df.plot_cv)) df.ggplot_cv = df.ggplot_cv %>% 
                                      filter_(paste0(target_name," == '",levs_target[2],"'")) 
    }

    # Distinguish nominal from metric feature
    if (is.factor(df.for_partialdep[[.x]])) {
      
      ## Plot for nominal feature
      
      # Proportion nominal variable
      df.hlp = df.for_partialdep %>% 
        group_by_(.x) %>% summarise(n = n()) %>% ungroup() %>% complete_(c(.x), fill = list(n = 0)) %>% 
        mutate(prop = n/sum(n), width = n/max(n))
      
      
      # Adapt value to .x
      df.ggplot[[.x]] = factor(as.character(df.ggplot$value), levels = levels(df.for_partialdep[[.x]])) 
      if (!is.null(df.plot_cv)) df.ggplot_cv[[.x]] = 
                      factor(as.character(df.ggplot_cv$value), levels = levels(df.for_partialdep[[.x]])) 
      
      # Prepare data for plotting
      df.ggplot = df.ggplot %>% 
        left_join(df.hlp) %>% 
        mutate(width = ifelse(width < min_width, min_width, width)) 
      if (type != "regr") {
        df.ggplot[[target_name]] = factor(as.character(df.ggplot[[target_name]]), levels = rev(levs_target))
      } else {
        df.ggplot[[target_name]] = "target"
        if (!is.null(df.plot_cv)) df.ggplot_cv[[target_name]] = "target"
      }
      
      #if (type == "class") df.hlp = df.ggplot #fix problem with missing combinations
      
      # Plot 
      p = ggplot(df.ggplot, aes_string(x = .x, y = "yhat", fill = target_name)) + 
        geom_bar(stat = "identity", position = ifelse(type == "multiclass", "fill", "stack"), 
                 width = df.ggplot$width, color = "black") +
        scale_fill_manual(values = alpha(colors, 0.2)) +
        scale_x_discrete(labels = paste0(as.character(df.hlp[[.x]]), " (", round(100 * df.hlp[["prop"]],1), "%)")) +
        geom_hline(yintercept = refs, linetype = 2, color = "darkgrey") +
        labs(title = .x, x = "", y = expression(hat(y))) +
        coord_flip(ylim = ylim) +
        theme_my

    } else {
      ## Plot for metric feature
      
      # Adapt x
      df.ggplot[[.x]] = as.numeric(df.ggplot$value)
      if (!is.null(df.plot_cv)) df.ggplot_cv[[.x]] = as.numeric(df.ggplot_cv$value)
      
      if (type == "regr") {
        df.ggplot[[target_name]] = "target"
        if (!is.null(df.plot_cv)) df.ggplot_cv[[target_name]] = "target"
      }

      # For retrieving max y-axis value for rescaling density plot
      tmp = ggplot_build(ggplot(df.ggplot, aes_string(.x)) +
                           geom_density(aes_string(y = paste0("..density..")), data = df.for_partialdep))
      
      # Plot 
      p = ggplot(df.ggplot, aes_string(x = .x, color = target_name)) +
        geom_density(aes_string(y = paste0(ylim[1]," + ","..density.. * ", 
                                           (ylim[2] - ylim[1]) / tmp$layout$panel_ranges[[1]]$y.range[2])), 
                     data = df.for_partialdep, fill = alpha("grey", 0.3), color = alpha("grey", 0.3)) +
        geom_line(aes_string(y = "yhat")) +
        geom_point(aes_string(y = "yhat")) +
        #scale_color_manual(limits = rev(levs_target), values = rev(color), name = target_name) +
        scale_color_manual(values = colors, name = target_name) +
        geom_rug(aes_string(.x), df.ggplot, sides = "b", col = "darkgrey") +
        geom_hline(yintercept = refs, linetype = 2, color = "darkgrey") +
        labs(title = .x, x = "", y = expression(hat(y))) +
        coord_cartesian(ylim = ylim, expand = FALSE) +
        theme_my
    }
    
    if (legend_only_in_1stplot == TRUE) {
      if (.x != vars[1]) p = p + theme(legend.position = "none") #legend only for first plot
    }
    
    # Add boostrap information
    if (!is.null(df.plot_cv)) {
      
      if (type != "regr") df.ggplot_cv[[target_name]] = factor(as.character(df.ggplot_cv[[target_name]]), 
                                                                 levels = levs_target)
      # Needed to group lines
      df.ggplot_cv$group = interaction(df.ggplot_cv[[run_name]], df.ggplot_cv[[target_name]] )
      
      if (type == "multiclass" & is.factor(df.for_partialdep[[.x]])) {
        df.ggplot_cv = df.ggplot_cv %>% group_by_(.x, "run") %>% mutate(yhat = cumsum(yhat))
      }
      
      # Add cv lines
      if (cv_lines == TRUE) {
        p = p +
          geom_line(aes_string(x = .x, y = "yhat", color = target_name, group = "group"), data = df.ggplot_cv, 
                    linetype = 3, 
                    #size = 0.1, alpha = 0.2, 
                    show.legend = FALSE) +
          scale_color_manual(values = colors, name = target_name) #+
          #geom_line(aes_string(y = "yhat", group = run_name)) + #plot red lines again
          #geom_point(aes_string(y = "yhat", group = run_name)) #plot red dots again
      }
      
      # Add cv Confidence Intervals
      if (cv_CI == TRUE) {
        
        # Calculate confidence intervals
        df.help = df.ggplot_cv %>% 
          group_by_(.x, target_name) %>% 
          summarise(sd = sd(yhat)) %>% 
          left_join(select_(df.ggplot, .x, "yhat", target_name)) 
        if (type == "multiclass" & is.factor(df.for_partialdep[[.x]])) {
          df.help = df.help %>% group_by_(.x) %>% mutate(yhat = cumsum(yhat))
        }
        df.help = df.help %>% mutate(lci = yhat - 1.96*sd, rci = yhat + 1.96*sd)
        
        # Add errorbar or ribbon
        if (is.factor(df.for_partialdep[[.x]])) {
          p = p +
            geom_errorbar(aes_string(x = .x, ymin = "lci", ymax = "rci"), data = df.help, size = 0.5, width = 0.05,
                          show.legend = FALSE)
        } else {
          p = p +
            geom_ribbon(aes_string(x = .x, ymin = "lci", ymax = "rci", fill = target_name, color = target_name), 
                        data = df.help, alpha = 0.1, show.legend = FALSE) +
            scale_color_manual(values = colors, name = target_name) +
            scale_fill_manual(values = colors, name = target_name) 
        }
      }
    }
    p
  })
  plots
}



## Get explanations data

# Overwrite slow explainPredictions
explainPredictions = function (xgb.model, explainer, data) 
{
  nodes = predict(xgb.model, data, predleaf = TRUE)
  colnames = names(explainer)[1:(ncol(explainer) - 2)]
  # preds_breakdown = data.table(matrix(0, nrow = nrow(nodes), 
  #                                     ncol = length(colnames)))
  # setnames(preds_breakdown, colnames)
  # num_trees = ncol(nodes)
  # cat("\n\nExtracting the breakdown of each prediction...\n")
  # pb <- txtProgressBar(style = 3)
  # for (x in 1:num_trees) {
  #   nodes_for_tree = nodes[, x]
  #   tree_breakdown = explainer[tree == x - 1]
  #   preds_breakdown_for_tree = tree_breakdown[match(nodes_for_tree, 
  #                                                   tree_breakdown$leaf), ]
  #   preds_breakdown = preds_breakdown + preds_breakdown_for_tree[, 
  #                                                                colnames, with = FALSE]
  #   setTxtProgressBar(pb, x/num_trees)
  # }
  # cat("\n\nDONE!\n")
  preds_breakdown = gather(bind_cols(id = 1:nrow(nodes), as.data.frame(nodes)), 
                           key = tree, value = leaf, -id) %>% #transpose
    mutate(tree = as.numeric(substring(tree,2)) - 1) %>% #derive tree number
    left_join(explainer) %>% #add explainer data
    select(-tree, -leaf) %>% group_by(id) %>% summarise_all(sum) %>% select(-id) #summarise
  return(preds_breakdown)
}

# get_explanations_old = function(fit.for_explain = fit,
#                             b_sample = NULL, b_all = NULL,
#                             df.train_explain = df.train[features], 
#                             df.test_explain = df.test[i.explain, c("id", features)],
#                             preds = yhat_explain[i.explain],
#                             id_name = "id", feature_names = features, formula = formula_rightside,
#                             type = "class") {
#   #browser()
#   
#   # Get model matrix and DMatrix for train and test (sample) data
#   m.train = sparse.model.matrix(formula, data = df.train_explain[feature_names])
#   DM.train = xgb.DMatrix(m.train) 
#   m.test_explain = sparse.model.matrix(formula, data = df.test_explain[feature_names])
#   DM.test_explain = xgb.DMatrix(m.test_explain)
#   
#   
#   ## Create explainer data table from train data
#   if (type == "class") {
#     df.explainer = buildExplainer(fit.for_explain$finalModel, DM.train, type = "binary")
#     
#     # Switch coefficients (as explainer takes "N" as target = 1)
#     cols = setdiff(colnames(df.explainer), c("leaf","tree"))
#     df.explainer[, (cols) := lapply(.SD, function(x) -x), .SDcols = cols]
#   } else {
#     df.explainer = buildExplainer(fit.for_explain$finalModel, DM.train, type = "regression")
#   }
#   
#   browser()
#   ## Get explanations for predictions of test data
#   df.predictions = explainPredictions(fit.for_explain$finalModel, df.explainer, DM.test_explain)
#   if (type == "class") {
#     tmp = inv.logit(df.predictions$intercept)
#     df.predictions$intercept = logit(scale_pred(data.frame(N = 1-tmp, Y = tmp), b_sample, b_all)[,2])
#   }
#   
#   # Aggregate predictions for all nominal variables
#   df.predictions = as.data.frame(df.predictions)
#   df.map = data.frame(varname = features[attr(model.matrix(formula, data = df.train_explain[1,feature_names]), "assign")],
#                       levname = setdiff(colnames(m.train),"(Intercept)"))
#   for (i in 1:length(nomi)) {
#     #i=1
#     varname = nomi[i]
#     levnames = as.character(df.map[df.map$varname == varname,]$levname)
#     df.predictions[varname] = apply(df.predictions[levnames], 1, function(x) sum(x, na.rm = TRUE))
#     df.predictions[levnames] = NULL
#   }
#   
#   # Check
#   if (type == "class") left = inv.logit(rowSums(df.predictions)) else left = rowSums(df.predictions)
#   if (all.equal(left, preds, tolerance = 1e-5) == TRUE) print("Predictions and explanations map") else 
#     print("Predictions and explanations DO NOT map!!!")
#   
#   df.predictions$id = df.test_explain$id 
#   df.predictions
# }

## Create explainer data frame from train data
get_explainer = function(fit.for_explain = fit,
                         m.train_for_explain = m.train,
                         feature_names = features) {
  #browser()
  
  # Create mapping table for Feature to variable
  df.map = data.frame(levname = colnames(m.train_for_explain),
                      variable = feature_names[attr(m.train_for_explain, "assign")],
                      stringsAsFactors = FALSE) %>% 
    bind_rows(data.frame(levname = "intercept", variable = "intercept", stringsAsFactors = FALSE ))
  
  # Get trees and add lp (linear predictor) info
  df.trees = xgb.model.dt.tree(attr(m.train_for_explain, "Dimnames")[[2]], fit.for_explain$finalModel) %>% 
    as.data.frame()%>% 
    mutate(lp_per_cover = ifelse(Feature == "Leaf", Quality, NA),
           lp = lp_per_cover * Cover)
  
  # Convert Yes/No branches to real parent-child
  df.parent_child = bind_rows(df.trees %>% select(-No) %>% rename("ID_child" = "Yes") %>% filter(Feature != "Leaf"),
                              df.trees %>% select(-Yes) %>% rename("ID_child" = "No")) %>% 
    arrange(Tree, Node, ID)
  
  # Sceleton for appending
  df.explainer = c()
  
  # Set df.leaves for first iteration
  df.leaves = df.parent_child %>% filter(Feature == "Leaf") %>% 
    mutate(count_leaf = 1) %>% 
    select(Tree, Node, ID_child = ID, Cover_leaf = Cover, lp_leaf = lp, lp_per_cover_child = lp_per_cover, count_leaf)
  
  ## Loop from here ...
  while(TRUE) {
    # Get split information
    df.split = df.leaves %>% 
      left_join(df.parent_child %>% select(ID, ID_child, Feature)) %>% #add next level to all leaves
      group_by(Tree, ID) %>% mutate(Cover_parent = sum(Cover_leaf), 
                                    lp_parent = sum(lp_leaf), 
                                    count = n()) %>% ungroup() %>% #aggregate leaves cover and lp to parent
      mutate(lp_per_cover_parent = lp_parent/Cover_parent, 
             weight = lp_per_cover_child - lp_per_cover_parent) %>% #calc diff in lp_per_cover
      mutate(ID = ifelse(count == count_leaf, ID_child, ID)) %>% #reset ID in case of having no split
      mutate(count_leaf = count) #%>% #add leaf count information (needed in next iteration for above chech)
    #filter(!is.na(ID)) # infor
    
    # Append split information and intercepts
    df.explainer = bind_rows(df.explainer, 
                             df.split %>% select(Tree, Node, Feature, weight), #add split data
                             df.split %>% filter(str_split(ID, "-", simplify = TRUE)[,2] == 0) %>% #add intercept
                               mutate(Feature = "intercept", weight = lp_per_cover_parent) %>% 
                               select(Tree, Node, Feature, weight))
    # Calc df.leaves for next iteration
    df.leaves = df.split %>% 
      select(Tree, Node, ID_child = ID, Cover_leaf, lp_leaf, lp_per_cover_child = lp_per_cover_parent, count_leaf) %>% 
      filter(str_split(ID_child, "-", simplify = TRUE)[,2] != 0)
    
    # Exit criteria
    if(nrow(df.leaves) == 0) break
  }
  
  # Aggregate over Feature names and return
  df.explainer %>% 
    left_join(df.map, by = c("Feature" = "levname")) %>% #add original variable name
    group_by(Tree, Node, variable) %>% summarise(weight = sum(weight))
}  


## Get explanations for some predictions from explainer data
get_explanations = function(fit.for_explain = fit,
                            feature_names = features,
                            target_name = "target",
                            df.test_explain,
                            id_name = "id",
                            yhat_explain,
                            df.explainer,
                            b_sample, b_all,
                            rounding = 2) {
  
  #browser()
  # Check for target scale
  levs = fit.for_explain$levels #also needed later
  length_levs = length(fit.for_explain$levels)
  target_type = switch(as.character(length_levs), 
                       "1" = "regr", "2" = "class", "multiclass")

  
  
  ## Get explanations for predictions of test data
  m.test_explain = sparse.model.matrix(as.formula(paste("~ -1 + ", paste(feature_names, collapse = " + "))),
                                df.test_explain[feature_names])
  df.weights = predict(fit.for_explain$finalModel, xgb.DMatrix(m.test_explain), predleaf = TRUE) %>%
    as.data.frame() %>% 
    mutate_(.dots = setNames(list("1:nrow(.)", paste0("df.test_explain$",id_name)), c("row_number",id_name))) %>% 
    gather_(key = "tree", value = "leaf", gather_cols = setdiff(colnames(.), c("row_number", id_name))) %>% 
    mutate(tree = as.numeric(substring(tree,2)) - 1) %>%
    left_join(df.explainer, by = c("tree" = "Tree", "leaf" = "Node")) 
  
  if (target_type == "multiclass") {
    df.weights = df.weights %>% mutate(target = levs[1 + tree%%length_levs]) 
  } else df.weights$target = levs[2]
  
  df.weights = df.weights %>% 
    group_by_("target", "row_number", id_name, "variable") %>% summarise(weight = sum(weight)) 
  
  #  group_by(row_number) %>% mutate(weight = exp(weight)/sum(exp(weight)))

  if (target_type %in% c("class","multiclass")) {
    # Switch coefficients (as explainer takes "N" as target = 1)
    if (target_type == "class") df.weights$weight = -df.weights$weight
    
    # Rescale intercept due to undersampling
    if (target_type == "class") {
      i.intercept = which(df.weights$variable == "intercept")
      tmp = inv.logit(df.weights$weight[i.intercept])
      intercept_new =  logit(scale_pred(data.frame(N = 1-tmp, Y = tmp), b_sample, b_all)[,2])
      df.weights[i.intercept, "weight"] = intercept_new
    } 
    if (target_type == "multiclass") {
      denom = sum(exp(unique(df.weights %>% filter(variable == "intercept") %>% .$weight)))
      for (i in 1:length_levs) {
        i.intercept = which(df.weights$target == levs[i] & df.weights$variable == "intercept")
        tmp = exp(df.weights$weight[i.intercept]) / denom
        intercept_new = log(scale_pred(data.frame(N = 1-tmp, Y = tmp), 
                                         c(1-b_sample[i], b_sample[i]), c(1-b_all[i], b_all[i]))[,2])
        df.weights[i.intercept, "weight"] = intercept_new
      }
    }
  }
  
  # Check
  df.check = df.weights %>% group_by(target, row_number) %>% summarise(weight = sum(weight))
  if (target_type == "multiclass") {
    df.check = df.check %>% 
      group_by(row_number) %>% mutate(weight = exp(weight)/sum(exp(weight))) %>% 
      left_join(yhat_explain %>% mutate(row_number = 1:nrow(.)) %>% gather(key = target, value = yhat, -row_number))
  } else if(target_type == "class") {
    df.check$weight = inv.logit(df.check$weight)
    df.check$yhat = yhat_explain
  } else {
    df.check$yhat = yhat_explain
  }
  if (all.equal(df.check$weight, df.check$yhat, tolerance = 1e-5) == TRUE) {
    print("Predictions and explanations map") 
  } else {
    print("Predictions and explanations DO NOT map!!!")
    print(all.equal(df.check$weight, df.check$yhat, tolerance = 1e-5))
  }
  
  # Add values
  df.values = df.test_explain[c(id_name,feature_names)]
  df.values[feature_names] = map(df.values[feature_names], ~ {
    if (is.numeric(.x)) as.character(round(.x, rounding)) else as.character(.x)
  })
  df.values = df.values %>% gather_(key = "variable", value = "value", setdiff(colnames(.), id_name))
  df.weights = df.weights %>% left_join(df.values)
  
  # Return
  df.weights
}


## Get plot list for xgboost explainer
get_plot_explanations = function(df.plot = df.explanations,  
                                 id_name = "id", scale_target = "class", ylim = c(0.01, 0.99), 
                                 fillcol = alpha(c("red","darkgreen"), 0.5),
                                 threshold = NULL, topn = NULL,
                                 add_to_title = NULL) {
  
  #browser()
  
  # Prepare
  df.plot = df.plot %>% mutate(flag_intercept = ifelse(variable == "intercept", 1, 0))
  
  # Threshold summary of small effects
  if (!is.null(threshold)) {
    df.tmp = df.plot %>% 
      mutate(variable = ifelse(variable != "intercept" & abs(weight) < threshold, "..... the rest", variable))
  }
  
  # Topn summary of small effects
  if (!is.null(topn)) {
    df.tmp = df.plot %>% group_by_(id_name) %>% arrange_(id_name, "desc(flag_intercept)", "desc(abs(weight))") %>% 
      mutate(n = row_number()) %>% 
      mutate(variable = ifelse(n > topn + 1, "..... the rest", variable)) %>% 
      ungroup()
  }
  df.ggplot = df.tmp %>%  
    group_by_(id_name, "flag_intercept", "variable") %>% summarise(weight = sum(weight)) %>%  #summarise small effect
    mutate(flag_therest = ifelse(variable == "..... the rest", 1, 0)) %>% 
    arrange_(id_name, "desc(flag_intercept)","flag_therest","desc(abs(weight))") %>%  #sort descending inside id
    left_join(df.plot %>% select_("row_number", id_name, "variable", "value")) %>%  #add values
    mutate(variableandvalue = ifelse(variable %in% c("intercept","..... the rest"), 
                                     variable, paste0(variable," = ",value)))
  
  # Plot
  ids = unique(df.plot[[id_name]])
  plots = map(1:length(ids), ~ {
    #.x = 1
    print(ids[.x])
    
    df.waterfall = df.ggplot[df.ggplot[[id_name]] == ids[.x],] 
    title = paste0(id_name, " = ", ids[.x])
    if (!is.null(add_to_title)) title = paste0(title, add_to_title[.x])
    p = waterfall(values = df.waterfall$weight, rect_text_labels = round(df.waterfall$weight, 2), 
                  labels = df.waterfall$variableandvalue, total_rect_text = round(sum(df.waterfall$weight), 2),
                  fill_by_sign = FALSE,
                  fill_colours = ifelse(df.waterfall$weight < 0, fillcol[1], fillcol[2]),
                  calc_total = TRUE, total_axis_text = "Prediction") + 
      labs(title = title) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(hjust = 0.5)) 
    if (scale_target %in% c("class","multiclass")) {
      p = p + 
        scale_y_continuous(labels = plogis, breaks = qlogis(seq(0, 100, 2)/100)) #, limits = qlogis(ylim)) 
    } else {
      p = p + coord_cartesian(ylim = ylim, expand = FALSE)
    }
    p 
  })
  plots  
}





# 
# ## Plot Interactiontest
# plot_interactiontest = function(outpdf, vars, df = df.interpret, fit = fit.gbm, l.boot = NULL, 
#                                 ncols = 4, w = 18, h = 12) {
#   
#   # Derive interaction matrix for topn important variables
#   pred_inter = setdiff(vars,"INT") #remove INT from testing variables
#   k = length(pred_inter)
#   m.inter = matrix(0, k, k)
#   for (i in 1:(k - 1)) {
#     for (j in (i + 1):k) {
#       # Interaction Test
#       m.inter[i,j] = interact.gbm(fit$finalModel, df[pred_inter], pred_inter[c(i,j)], fit$finalModel$tuneValue$n.trees)
#       m.inter[j,i] = m.inter[i,j]
#     }
#   }
#   colnames(m.inter) = pred_inter
#   rownames(m.inter) = pred_inter
#   m.inter[is.na(m.inter)] = 0
#   
#   
#   ## Plot in correlation matrix style
#   df.inter = as.data.frame(m.inter) %>% 
#     mutate(rowvar = rownames(m.inter)) %>% 
#     gather(key = colvar, value = inter, -rowvar)
#   p = ggplot(df.inter, aes(rowvar, colvar)) +
#     geom_tile(aes(fill = inter)) + 
#     geom_text(aes(label = round(inter, 2))) +
#     scale_fill_gradient(low = "white", high = "blue") +
#     scale_x_discrete(limits = rownames(m.inter)) + 
#     scale_y_discrete(limits = rev(rownames(m.inter))) +
#     labs(title = "Interaction", fill = "", x = "", y = "") +
#     theme_my +
#     theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
#   ggsave(outpdf, p, width = w, height = h)
#   
#   
#   
#   if (!is.null(l.boot)) {
#     # Do the same as above for each bootstrapped model and collect
#     df.inter_boot = map_df(l.boot, ~ {
#       for (i in 1:(k - 1)) {
#         for (j in (i + 1):k) {
#           # Interaction Test
#           m.inter[i,j] = interact.gbm(.$finalModel, df[pred_inter], pred_inter[c(i,j)], .$finalModel$tuneValue$n.trees)
#           m.inter[j,i] = m.inter[i,j]
#         }
#       }
#       m.inter[is.na(m.inter)] = 0
#       
#       df.inter = as.data.frame(m.inter) %>% 
#         mutate(rowvar = rownames(m.inter)) %>% 
#         gather(key = colvar, value = inter, -rowvar)
#       df.inter
#     }, .id = "run")
#     
#     
#     # Same plot but now facetting
#     p_boot = ggplot(df.inter_boot, aes(rowvar, colvar)) +
#       geom_tile(aes(fill = inter)) + 
#       geom_text(aes(label = round(inter, 2))) +
#       scale_fill_gradient(low = "white", high = "blue") +
#       scale_x_discrete(limits = rownames(m.inter)) + 
#       scale_y_discrete(limits = rev(rownames(m.inter))) +
#       labs(title = "Interaction per Bootstrap Run", fill = "", x = "", y = "") +
#       theme_my +
#       theme(axis.text.x = element_text(angle = 90, hjust = 1))  +
#       facet_wrap( ~ run, ncol = ncols)
#   }
#   ggsave(paste0(str_split(outpdf,".pdf", simplify = TRUE)[1,1],"_boot.pdf"), p_boot, width = w, height = h)
# }
# 
# 
# 
# 
# ## Plot interactions of m.gbm
# plot_inter = function(outpdf, vars = inter, df = df.interpret, fit = fit.gbm, 
#                       ylim = c(0,1), w = 12, h = 8) {
#   # outpdf="./output/interaction1.pdf"; vars=inter1; df=df.interpret; fit=fit; w=12; h=8; ylim = c(0,.3)
#   
#   # Final model
#   model = fit$finalModel
#   
#   # Derive offset
#   offset = model$initF - plot(model, i.var = "INT", type = "link", return.grid = TRUE)[1,"y"]
#   
#   # Marginal plots for digonal
#   plots_marginal = map(vars, ~ {
#     #.=vars
#     # Get interaction data
#     df.plot = plot(model, i.var = ., type = "link", return.grid = TRUE) #get plot data on link level
#     p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
#     df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
#     
#     if (is.factor(df[[.]])) {
#       # Width of bars correspond to freqencies
#       tmp = table(df[,.])
#       df.plot$width = as.numeric(tmp[df.plot[[.]]])/max(tmp)
#       
#       # Plot for a nominal variable
#       p = ggplot(df.plot, aes_string(., "y", fill = .)) +
#         geom_bar(stat = "identity",  width = df.plot$width, color = "black") +
#         labs(title = ., x = "", y = expression(paste("P(", hat(y), "=1)"))) +
#         scale_fill_manual(values = manycol) +
#         scale_y_continuous(limits = ylim) +
#         theme_my + 
#         theme(legend.position = "none")
#     } else {
#       # Plot for a metric variable
#       df.rug = data.frame(q = quantile(df[,.], prob = seq(.05, .95, .1)), y = 0)
#       p = ggplot(df.plot, aes_string(., "y")) +
#         geom_line(stat = "identity", color = "black") +
#         geom_rug(aes(q, y), df.rug, sides = "b", col = "red") +
#         labs(title = ., x = "", y = expression(paste("P(", hat(y), "=1)"))) +
#         scale_y_continuous(limits = ylim) +
#         theme_my      
#     }
#     p + geom_hline(yintercept = b_all, linetype = 3)
#   })
#   
#   # Interaction plots 
#   df.plot = plot(model, i.var = vars, type = "link", return.grid = TRUE) #get plot data on link level
#   p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
#   df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
#   # quantile(df$TT4, seq(0,1, length.out=100))
#   # (max(df$TT4) - min(df$TT4))/99 + min(df$TT4)
#   
#   if (sum(map_lgl(df.plot[vars], ~ is.factor(.))) == 2) {
#     # Mosaicplot for nominal-nominal interaction
#     plots_inter = map(1:2, ~ { 
#       if (.==2) vars = rev(vars)
#       #tmp = table(df[vars[2]])
#       #df.plot[[vars[1]]] = factor(df.plot[[vars[1]]], levels = rev(levels(df.plot[[vars[1]]])))
#       ggplot(df.plot, aes_string(vars[2], "y", fill = vars[1])) +
#         geom_bar(stat = "identity", position = "fill") + #, width = rep(tmp/max(tmp), 5)) +
#         scale_fill_manual(values = manycol) +
#         labs(y = "", x = "") +
#         theme_my      
#     })
#   }  
#   if (sum(map_lgl(df.plot[vars], ~ is.factor(.))) == 1) {
#     # Grouped line chart for metric-nominal interaction
#     p = ggplot(df.plot, aes_string(vars[1], "y", color = vars[2])) +
#       geom_line(stat = "identity") +
#       labs(y = expression(paste("P(", hat(y), "=1)"))) +
#       scale_color_manual(values = manycol) +
#       scale_y_continuous(limits = ylim*2) +
#       guides(color = guide_legend(reverse = TRUE)) +
#       theme_my      
#     plots_inter = list(p,p)
#   }   
#   if (sum(map_lgl(df.plot[vars], ~ is.factor(.))) == 0) {
#     # Grouped (by quantiles) line chart for metric-metric interaction
#     plots_inter = map(1:2, ~ { 
#       if (.==2) vars = rev(vars)
#       val_near_quant =   map_dbl(quantile(df[[vars[2]]], seq(.05,.95,.1)), ~ {
#         df.plot[[vars[2]]][which.min(abs(df.plot[[vars[2]]] - .))]})
#       i.tmp = df.plot[[vars[2]]] %in% val_near_quant
#       df.tmp = df.plot[i.tmp,]
#       df.tmp[vars[2]] = factor( round(df.tmp[[vars[2]]],2) )
#       
#       ggplot(df.tmp, aes_string(vars[1], "y", color = vars[2])) +
#         geom_line(stat = "identity") +
#         labs(y = expression(paste("P(", hat(y), "=1)"))) +
#         scale_color_manual(values = manycol) +
#         scale_y_continuous(limits = ylim) +
#         guides(color = guide_legend(reverse = TRUE)) +
#         theme_my   
#     })
#   } 
#   
#   # Arrange plots
#   plots = list(plots_marginal[[1]], plots_inter[[1]], plots_inter[[2]], plots_marginal[[2]])
#   ggsave(outpdf, marrangeGrob(plots, ncol = 2, nrow = 2, top = NULL), width = w, height = h)
# }
# 


# 
# # Animated Interaction of 2 metric variables
# plot_inter_active = function(outfile, vars = inter, df = df.interpret, fit = fit.gbm, duration = 15) {
#   
#   # Final model
#   model = fit$finalModel
#   
#   # Derive offset
#   offset = model$initF - plot(model, i.var = "INT", type = "link", return.grid = TRUE)[1,"y"]
#   
#   # Get interaction data
#   df.plot = plot(model, i.var = vars, type = "link", return.grid = TRUE) #get plot data on link level
#   p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
#   df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
#   
#   # Prepare 3d plot
#   x = unique(df.plot[[vars[1]]])
#   y = unique(df.plot[[vars[2]]])
#   z = matrix(df.plot$y, length(x), length(y), byrow = FALSE)
#   nrz = nrow(z)
#   colcut = cut((z[-1, -1] + z[-1, -nrz] + z[-nrz, -1] + z[-nrz, -ncol(z)])/4, 100)
#   
#   # html Widget
#   persp3d(x, y, z, col = col3d[colcut], phi = 30, theta = 50, axes = T, ticktype = 'detailed',
#           xlab = vars[1], ylab = vars[2], zlab = "")
#   writeWebGL(dir = file.path(paste0(outfile)), width = 1000)
#   rgl.close()
#   
#   # animated gif
#   open3d("windowRect" = 2*c(20,20,400,400))
#   persp3d(x, y, z, col = col3d[colcut], phi = 30, theta = 50, axes = T, ticktype = 'detailed',
#           xlab = vars[1], ylab = vars[2], zlab = "")
#   movie3d(spin3d(axis = c(0,0,1), rpm = 2), duration = duration, convert = NULL, clean = TRUE, movie = "test",
#           dir = paste0(outfile))
#   rgl.close()
#   
# }
# 


#######################################################################################################################-
# Caret definition of non-caret algorithms ----
#######################################################################################################################-

## rxFastTrees (boosted trees)
ms_boosttree = list()
ms_boosttree$label = "MicrosoftML rxFastTrees"
ms_boosttree$library = c("MicrosoftML","RevoScaleR")
ms_boosttree$type = c("Regression","Classification")
ms_boosttree$parameters = 
  read.table(header = TRUE, sep = ",", strip.white = TRUE, 
             text = "parameter,class,label
             numTrees,numeric,numTrees
             numLeaves,numeric,numLeaves
             minSplit,numeric,minSplit
             learningRate,numeric,learningRate
             featureFraction,numeric,featureFraction
             exampleFraction,numeric,exampleFractione"                             
  )

ms_boosttree$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(numTrees = floor((1:len) * 50),
                       numLeaves = 2^seq(1, len),
                       minSplit = 10,
                       learningRate = .1,
                       featureFraction = 0.7,
                       exampleFraction = 0.7)
  } else {
    out <- data.frame(numTrees = floor(runif(len, min = 10, max = 5000)),
                      numLeaves = 2 ^ sample(1:6, replace = TRUE, size = len), 
                      minSplit = 2 ^ sample(0:6, replace = TRUE, size = len),
                      learningRate = runif(len, min = .001, max = .6),
                      featureFraction = runif(len, min = .1, max = 1),
                      exampleFraction = runif(len, min = .1, max = 1)) 
    out <- out[!duplicated(out),]
  }
  out
}

ms_boosttree$fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
  #browser()
  theDots = list(...)
  #if (is.factor(y) && length(lev) == 2) {y = ifelse(y == lev[1], 1, 0)}
  #y = factor(y, levels = c(1,0))
  #x = as.matrix(x)
  if (is.factor(y)) type = "binary" else type = "regression"
  modArgs <- list(formula = paste("y~", paste0(names(x), collapse = "+")),
                  data = cbind(x, y),
                  numTrees = param$numTrees,
                  numLeaves = param$numLeaves,
                  minSplit = param$minSplit,
                  learningRate = param$learningRate,
                  featureFraction = param$featureFraction,
                  exampleFraction = param$exampleFraction,
                  type = type)
  if (length(theDots) > 0) modArgs <- c(modArgs, theDots)
  do.call(MicrosoftML::rxFastTrees, modArgs)
  #out = MicrosoftML::rxFastTrees(formula = modArgs$formula, data = modArgs$data, type = modArgs$type)
  #out
}

ms_boosttree$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (modelFit$problemType == "Classification") {
    out = RevoScaleR::rxPredict(modelFit, newdata)$Probability.Y
  } else {
    newdata$y = NA
    out = RevoScaleR::rxPredict(modelFit, newdata)$Score
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- ifelse(out >= 0.5, "Y", "N")
  }
  out
}

ms_boosttree$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  out = RevoScaleR::rxPredict(modelFit, newdata)[,"Probability.Y"]
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- c("Y","N")
  }
  out
}

ms_boosttree$levels = function(x) {c("N","Y")}

ms_boosttree$sort = function(x) {
  x[order(x$numTrees, x$numLeaves, x$learningRate), ]
}




## rxForest (random Forest)
ms_forest = list()
ms_forest$label = "MicrosoftML rxFastForest"
ms_forest$library = c("MicrosoftML","RevoScaleR")
ms_forest$type = c("Regression","Classification")
ms_forest$parameters = 
  read.table(header = TRUE, sep = ",", strip.white = TRUE,
             text = "parameter,class,label
             numTrees,numeric,numTrees
             splitFraction,numeric,splitFraction"
  )

ms_forest$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(numTrees = floor((1:len) * 50),
                       splitFraction = seq(0.01, 1, length.out = len))
  } else {
    out <- data.frame(numTrees = floor(runif(len, min = 1, max = 5000)),
                      splitFraction = runif(len, min = 0.01, max = 1))
    out <- out[!duplicated(out),]
  }
  out
}

ms_forest$fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
  theDots = list(...)
  if (is.factor(y)) type = "binary" else type = "regression"
  modArgs <- list(formula = paste("y~", paste0(names(x), collapse = "+")),
                  data = cbind(x, y),
                  numTrees = param$numTrees,
                  splitFraction = param$splitFraction,
                  type = type)
  if (length(theDots) > 0) modArgs <- c(modArgs, theDots)
  do.call(MicrosoftML::rxFastForest, modArgs)
}

ms_forest$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (modelFit$problemType == "Classification") {
    out = RevoScaleR::rxPredict(modelFit, newdata)$Probability.Y
  } else {
    newdata$y = NA
    out = RevoScaleR::rxPredict(modelFit, newdata)$Score
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- ifelse(out >= 0.5, "Y", "N")
  }
  out
}

ms_forest$prob = function(modelFit, newdata, submodels = NULL) {
  out = RevoScaleR::rxPredict(modelFit, newdata)[,"Probability.Y"]
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- c("Y","N")
  }
  out
}

ms_forest$levels = function(x) {c("N","Y")}

ms_forest$sort = function(x) {
  x[order(x$numTrees, x$splitFraction), ]
}



## lightgbm (boosted trees)

lgbm = list()
lgbm$label = "lightgbm"
lgbm$library = c("lightgbm","plyr")
lgbm$type = c("Regression","Classification")
lgbm$parameters = 
  read.table(header = TRUE, sep = ",", strip.white = TRUE, 
             text = "parameter,class,label
             nrounds,numeric,nrounds
             num_leaves,numeric,num_leaves
             min_data_in_leaf,numeric,min_data_in_leaf
             learning_rate,numeric,learning_rate
             feature_fraction,numeric,feature_fraction
             bagging_fraction,numeric,bagging_fraction"                             
  )

lgbm$grid = function(x, y, len = NULL, search = "grid") {
  #browser()
  if (search == "grid") {
    out <- expand.grid(nrounds = floor((1:len) * 50),
                       num_leaves = 2^seq(1, len),
                       min_data_in_leaf = 10,
                       learning_rate = .1,
                       feature_fraction = 0.7,
                       bagging_fraction = 0.7)
  } else {
    out <- data.frame(nrounds = floor(runif(len, min = 10, max = 5000)),
                      num_leaves = 2 ^ sample(1:6, replace = TRUE, size = len), 
                      min_data_in_leaf = 2 ^ sample(0:6, replace = TRUE, size = len),
                      learning_rate = runif(len, min = .001, max = .6),
                      feature_fraction = runif(len, min = .1, max = 1),
                      bagging_fraction = runif(len, min = .1, max = 1)) 
    out <- out[!duplicated(out),]
  }
  out
}

lgbm$loop = function(grid) {
  #browser()
  loop <- plyr::ddply(grid, 
                c("learning_rate", "num_leaves", "feature_fraction", "min_data_in_leaf", "bagging_fraction"), 
                function(x) c(nrounds = max(x$nrounds)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for (i in seq(along = loop$nrounds)) {
    index <- which(grid$learning_rate == loop$learning_rate[i] &
                   grid$num_leaves == loop$num_leaves[i] & 
                   grid$feature_fraction == loop$feature_fraction[i] & 
                   grid$min_data_in_leaf == loop$min_data_in_leaf[i] & 
                   grid$bagging_fraction == loop$bagging_fraction[i])
    trees <- grid[index, "nrounds"]
    submodels[[i]] <- data.frame(nrounds = trees[trees != loop$nrounds[i]])
  }
  list(loop = loop, submodels = submodels)
}

lgbm$fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
  #browser()
  theDots = list(...)
  if (is.factor(y)) objective = "binary" else objective = "regression_l2"
  if (is.factor(y)) y = as.numeric(y) - 1
  modArgs <- list(data = lightgbm::lgb.Dataset(x, label = y),
                  nrounds = param$nrounds,
                  num_leaves = param$num_leaves,
                  min_data_in_leaf = param$min_data_in_leaf,
                  learning_rate = param$learning_rate,
                  feature_fraction = param$feature_fraction,
                  bagging_fraction = param$bagging_fraction,
                  objective = objective)
  if (length(theDots) > 0) modArgs <- c(modArgs, theDots)
  list("model" = do.call(lightgbm::lightgbm, modArgs)) #put it into list as it is a S4 object!
}

lgbm$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (modelFit$problemType == "Classification") {
    out = predict(modelFit$model, newdata)
  } else {
    out = predict(modelFit$model, newdata)
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- ifelse(out >= 0.5, "Y", "N")
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) + 
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit$model, newdata, num_iteration = submodels$nrounds[j])
      if (modelFit$problemType == "Classification") {
        if (length(modelFit$obsLevels) == 2) {
          tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1], 
                             modelFit$obsLevels[2])
        } else {
          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
                             byrow = TRUE)
          tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 
                                               1, which.max)]
        }
      }
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}  
  
lgbm$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  out = predict(modelFit$model, newdata)
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- c("Y","N")
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) + 1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit$model, newdata, num_iteration = submodels$nrounds[j])
      if (length(modelFit$obsLevels) == 2) {
        tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
        colnames(tmp_pred) <- c("Y","N")
      } else {
        tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
                           byrow = TRUE)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      tmp_pred <- as.data.frame(tmp_pred)
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

lgbm$levels = function(x) {c("N","Y")}

lgbm$sort = function(x) {
  #browser()
  x[order(x$nrounds, x$num_leaves, x$learning_rate, x$feature_fraction, 
          x$bagging_fraction), ]
}



## xgboost with alpha and lambda (boosted trees)

xgb = list()

xgb$label = "xgboost"

xgb$library = c("xgboost","plyr")   

xgb$type = c("Regression","Classification")

xgb$parameters = 
  read.table(header = TRUE, sep = ",", strip.white = TRUE, 
             text = "parameter,class,label
             nrounds,numeric,nrounds
             max_depth,numeric,Max Tree Depth
             eta,numeric,Shrinkage
             gamma,numeric,Minimum Loss Reduction
             colsample_bytree,numeric,Subsample Ratio of Columns
             min_child_weight,numeric,Minimum Sum of Instance Weight
             subsample,numeric,Subsample Percentage
             alpha,numeric,alpha
             lambda,numeric,lambda")

xgb$grid = function(x, y, len = NULL, search = "grid") {
  if (search == "grid") {
    out <- expand.grid(max_depth = seq(1, len), nrounds = floor((1:len) * 50), 
                       eta = c(0.3, 0.4), gamma = 0, colsample_bytree = c(0.6,0.8), 
                       min_child_weight = c(1), subsample = seq(0.5,1, length = len),
                       alpha = 0, lambda = 1)
  }
  else {
    out <- data.frame(nrounds = sample(1:1000, size = len, replace = TRUE), 
                      max_depth = sample(1:10, replace = TRUE, size = len), 
                      eta = runif(len, min = 0.001, max = 0.6), 
                      gamma = runif(len, min = 0, max = 10), 
                      colsample_bytree = runif(len, min = 0.3, max = 0.7), 
                      min_child_weight = sample(0:20, size = len, replace = TRUE), 
                      subsample = runif(len, min = 0.25, max = 1),
                      alpha = 0, lambda = 1)
    out$nrounds <- floor(out$nrounds)
    out <- out[!duplicated(out), ]
  }
  out
}

xgb$loop = function(grid) {
  loop <- plyr::ddply(grid, c("eta", "max_depth", "gamma", 
                              "colsample_bytree", "min_child_weight", "subsample","alpha","lambda"), 
                      function(x) c(nrounds = max(x$nrounds)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for (i in seq(along = loop$nrounds)) {
    index <- which(grid$max_depth == loop$max_depth[i] & 
                     grid$eta == loop$eta[i] & grid$gamma == loop$gamma[i] & 
                     grid$colsample_bytree == loop$colsample_bytree[i] & 
                     grid$min_child_weight == loop$min_child_weight[i] & 
                     grid$subsample == loop$subsample[i] &
                     grid$alpha == loop$alpha[i] &
                     grid$lambda == loop$lambda[i])
    trees <- grid[index, "nrounds"]
    submodels[[i]] <- data.frame(nrounds = trees[trees != loop$nrounds[i]])
  }
  list(loop = loop, submodels = submodels)
}

xgb$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
  #browser()
  # if (!inherits(x, "xgb.DMatrix")) 
  #   x <- as.matrix(x)
  if (is.factor(y)) {
    if (length(lev) == 2) {
      y <- ifelse(y == lev[1], 1, 0)
      if (!inherits(x, "xgb.DMatrix")) 
        x <- xgboost::xgb.DMatrix(x, label = y)
      else xgboost::setinfo(x, "label", y)
      if (!is.null(wts)) 
        xgboost::setinfo(x, "weight", wts)
      out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth, 
                                     gamma = param$gamma, colsample_bytree = param$colsample_bytree, 
                                     min_child_weight = param$min_child_weight, subsample = param$subsample,
                                     alpha = param$alpha, lambda = param$lambda), 
                                data = x, nrounds = param$nrounds, objective = "binary:logistic", 
                                ...)
    }
    else {
      y <- as.numeric(y) - 1
      if (!inherits(x, "xgb.DMatrix")) 
        x <- xgboost::xgb.DMatrix(x, label = y)
      else xgboost::setinfo(x, "label", y)
      if (!is.null(wts)) 
        xgboost::setinfo(x, "weight", wts)
      out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth, 
                                     gamma = param$gamma, colsample_bytree = param$colsample_bytree, 
                                     min_child_weight = param$min_child_weight, subsample = param$subsample,
                                     alpha = param$alpha, lambda = param$lambda), 
                                data = x, num_class = length(lev), nrounds = param$nrounds, 
                                objective = "multi:softprob", ...)
    }
  }
  else {
    if (!inherits(x, "xgb.DMatrix")) 
      x <- xgboost::xgb.DMatrix(x, label = y)
    else xgboost::setinfo(x, "label", y)
    if (!is.null(wts)) 
      xgboost::setinfo(x, "weight", wts)
    out <- xgboost::xgb.train(list(eta = param$eta, max_depth = param$max_depth, 
                                   gamma = param$gamma, colsample_bytree = param$colsample_bytree, 
                                   min_child_weight = param$min_child_weight, subsample = param$subsample,
                                   alpha = param$alpha, lambda = param$lambda), 
                              data = x, nrounds = param$nrounds, objective = "reg:linear", 
                              ...)
  }
  out
}

xgb$predict = function(modelFit, newdata, submodels = NULL) {
  if (!inherits(newdata, "xgb.DMatrix")) {
    #newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata)
  }
  out <- predict(modelFit, newdata)
  if (modelFit$problemType == "Classification") {
    if (length(modelFit$obsLevels) == 2) {
      out <- ifelse(out >= 0.5, modelFit$obsLevels[1], 
                    modelFit$obsLevels[2])
    }
    else {
      out <- matrix(out, ncol = length(modelFit$obsLevels), 
                    byrow = TRUE)
      out <- modelFit$obsLevels[apply(out, 1, which.max)]
    }
  }
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) + 
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
      if (modelFit$problemType == "Classification") {
        if (length(modelFit$obsLevels) == 2) {
          tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1], 
                             modelFit$obsLevels[2])
        }
        else {
          tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
                             byrow = TRUE)
          tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 
                                               1, which.max)]
        }
      }
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

xgb$prob = function(modelFit, newdata, submodels = NULL) {
  if (!inherits(newdata, "xgb.DMatrix")) {
    #newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata)
  }
  if (!is.null(modelFit$param$objective) && modelFit$param$objective == 
      "binary:logitraw") {
    p <- predict(modelFit, newdata)
    out <- binomial()$linkinv(p)
  }
  else {
    out <- predict(modelFit, newdata)
  }
  if (length(modelFit$obsLevels) == 2) {
    out <- cbind(out, 1 - out)
    colnames(out) <- modelFit$obsLevels
  }
  else {
    out <- matrix(out, ncol = length(modelFit$obsLevels), 
                  byrow = TRUE)
    colnames(out) <- modelFit$obsLevels
  }
  out <- as.data.frame(out)
  if (!is.null(submodels)) {
    tmp <- vector(mode = "list", length = nrow(submodels) + 
                    1)
    tmp[[1]] <- out
    for (j in seq(along = submodels$nrounds)) {
      tmp_pred <- predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
      if (length(modelFit$obsLevels) == 2) {
        tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      else {
        tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
                           byrow = TRUE)
        colnames(tmp_pred) <- modelFit$obsLevels
      }
      tmp_pred <- as.data.frame(tmp_pred)
      tmp[[j + 1]] <- tmp_pred
    }
    out <- tmp
  }
  out
}

xgb$predictors = function (x, ...) {
  imp <- xgboost::xgb.importance(x$xNames, model = x)
  x$xNames[x$xNames %in% imp$Feature]
}

xgb$varImp = function (object, numTrees = NULL, ...) {
  imp <- xgboost::xgb.importance(object$xNames, model = object)
  imp <- as.data.frame(imp)[, 1:2]
  rownames(imp) <- as.character(imp[, 1])
  imp <- imp[, 2, drop = FALSE]
  colnames(imp) <- "Overall"
  missing <- object$xNames[!(object$xNames %in% rownames(imp))]
  missing_imp <- data.frame(Overall = rep(0, times = length(missing)))
  rownames(missing_imp) <- missing
  imp <- rbind(imp, missing_imp)
  imp
}

xgb$levels = function(x) x$obsLevels

xgb$tags = c("Tree-Based Model" )

xgb$sort = function(x) {
  x[order(x$nrounds, x$max_depth, x$eta, x$gamma, x$colsample_bytree, 
          x$min_child_weight), ]
}



## Deep learning (from mlpKerasDecay)

deepLearn = list()

deepLearn$label = "Deep Learning"

deepLearn$library = "keras"

deepLearn$loop = NULL

deepLearn$type = c("Regression","Classification")

deepLearn$parameters = 
  read.table(header = TRUE, sep = ",", strip.white = TRUE, 
             text = "parameter,class,label
             size,character,Layer shape
             lambda,numeric,L2 L2 Regularization
             dropout,numeric,Dropout Rate
             batch_size,numeric,Batch Size
             lr,numeric,Learning Rate
             batch_normalization,boolean,Batch Normalization
             activation,character,Activation Function
             epochs,numeric,Epochs"
  )

deepLearn$grid = function(x, y, len = NULL, search = "grid") {
  afuncs <- c("sigmoid", "relu", "tanh")
  if (search == "grid") {
    out <- expand.grid(size = "10",
                       lambda = c(0, 10^seq(-1, -4, length = len - 1)),
                       batch_size = floor(nrow(x)/3),
                       lr = 2e-06,
                       dropout = 0,
                       batch_normalization = FALSE,
                       activation = "relu",
                       epochs = 10)
  }
  else {
    n <- nrow(x)
    out <- data.frame(size = "10",
                      lambda = 10^runif(len, min = -5, 1),
                      batch_size = floor(n * runif(len, min = 0.1)),
                      lr = runif(len),
                      dropout = 0,
                      batch_normalization = FALSE,
                      activation = sample(afuncs, size = len, replace = TRUE),
                      epochs = 10)
  }
  out
}

deepLearn$fit = function(x, y, wts, param, lev, last, classProbs, ...) {
 # browser()
  print(param)
  
  require(dplyr)
  K <- keras::backend()
  K$clear_session()
  if (!is.matrix(x)) 
    x <- as.matrix(x)
  model <- keras::keras_model_sequential()
  
  size = as.numeric(str_split(param$size,"-",simplify = TRUE)[1,])
  
  for (i in 1:length(size)) {
    model %>% keras::layer_dense(units = size[i], activation = as.character(param$activation), 
                                 input_shape = ncol(x), kernel_initializer = keras::initializer_glorot_uniform(), 
                                 kernel_regularizer = keras::regularizer_l2(param$lambda))
    if(param$batch_normalization) model %>% keras::layer_batch_normalization()
    if(param$dropout > 0) model %>% keras::layer_dropout(param$dropout)
  }
  if (is.factor(y)) {
    y <- class2ind(y)
    model %>% keras::layer_dense(units = length(lev), activation = "softmax", 
                                 kernel_regularizer = keras::regularizer_l2(param$lambda)) %>% 
      keras::compile(loss = "categorical_crossentropy", 
                     optimizer = keras::optimizer_rmsprop(lr = param$lr), 
                     metrics = "accuracy")
  }
  else {
    model %>% keras::layer_dense(units = 1, activation = "linear", 
                                 kernel_regularizer = keras::regularizer_l2(param$lambda)) %>% 
      compile(loss = "mean_squared_error", 
              optimizer = keras::optimizer_rmsprop(lr = param$lr), 
              metrics = "mean_squared_error")
  }
  model %>% keras::fit(x = x, y = y, batch_size = param$batch_size, epochs = param$epochs, 
                       ...)
  if (last) 
    model <- keras::serialize_model(model)
  list(object = model)
}

deepLearn$predict = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (inherits(modelFit$object, "raw")) 
    modelFit$object <- keras::unserialize_model(modelFit$object)
  if (!is.matrix(newdata)) 
    newdata <- as.matrix(newdata)
  out <- predict(modelFit$object, newdata)
  if (ncol(out) == 1) {
    out <- out[, 1]
  }
  else {
    out <- modelFit$obsLevels[apply(out, 1, which.max)]
  }
  out
}

deepLearn$prob = function(modelFit, newdata, submodels = NULL) {
  #browser()
  if (inherits(modelFit$object, "raw")) 
    modelFit$object <- keras::unserialize_model(modelFit$object)
  if (!is.matrix(newdata)) 
    newdata <- as.matrix(newdata)
  out <- predict(modelFit$object, newdata)
  colnames(out) <- modelFit$obsLevels
  as.data.frame(out)
}

deepLearn$sort = function(x) x[order(x$size, -x$lambda), ]

deepLearn$check = function(pkg) {
  testmod <- try(keras::keras_model_sequential(), silent = TRUE)
  if (inherits(testmod, "try-error")) 
    stop("Could not start a sequential model. ", "`tensorflow` might not be installed. ", 
         "See `?install_tensorflow`.", call. = FALSE)
  TRUE
}



