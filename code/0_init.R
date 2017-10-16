#######################################################################################################################-
# Libraries + Parallel Processing Start ----
#######################################################################################################################-

skip = function() {
  install.packages(c("tidyverse"))
}

library(plyr) #always load plyr before dplyr
library(tidyverse) #ggplot2,tibble,tidyr,readr,purrr,dplyr
library(forcats)
library(stringr)
library(lubridate)
library(bindrcpp)
library(magrittr)

library(doParallel)
# 
library(corrplot)
library(vcd)
library(grid)
library(gridExtra)
# library(Hmisc)
# library(d3heatmap)
# library(htmlwidgets)
# library(rgl)
# 
library(caret)
library(xgboost)
# library(glmnet)
library(ROCR)


#######################################################################################################################-
# Parameters ----
#######################################################################################################################-

# Locations
dataloc = "./data/"
plotloc = "./output/"


# Colors
twocol = c("blue","red")
threeDcol = colorRampPalette(c("cyan", "white", "magenta"))(100)
hexcol = colorRampPalette(c("white", "blue", "yellow", "red"))(100)
manycol = c('#00FF00','#0000FF','#FF0000','#01FFFE','#FFA6FE','#FFDB66','#006401','#010067','#95003A',
            '#007DB5','#FF00F6','#FFEEE8','#774D00','#90FB92','#0076FF','#D5FF00','#FF937E','#6A826C','#FF029D',
            '#FE8900','#7A4782','#7E2DD2','#85A900','#FF0056','#A42400','#00AE7E','#683D3B','#BDC6FF','#263400',
            '#BDD393','#00B917','#9E008E','#001544','#C28C9F','#FF74A3','#01D0FF','#004754','#E56FFE','#788231',
            '#0E4CA1','#91D0CB','#BE9970','#968AE8','#BB8800','#43002C','#DEFF74','#00FFC6','#FFE502','#620E00',
            '#008F9C','#98FF52','#7544B1','#B500FF','#00FF78','#FF6E41','#005F39','#6B6882','#5FAD4E','#A75740',
            '#A5FFD2','#FFB167','#009BFF','#E85EBE')

# Themes
theme_my = theme_bw() +  theme(plot.title = element_text(hjust = 0.5))



#######################################################################################################################-
# My Functions ----
#######################################################################################################################-

## Calculate probabilty on all data from probabilt from sample data and the corresponding (prior) base probabilities 
prob_samp2full = function(p_sample, b_sample, b_all) {
  p_all = b_all * ((p_sample - p_sample*b_sample) / 
                   (b_sample - p_sample*b_sample + b_all*p_sample - b_sample*b_all))
  p_all
}



## Workaround for ggsave and marrangeGrob not to create first page blank
grid.draw.arrangelist <- function(x, ...) {
  for (ii in seq_along(x)) {
    if (ii > 1) grid.newpage()  # skips grid.newpage() call the first time around
    grid.draw(x[[ii]])
  }
}

## Summary function for classification performance
my_twoClassSummary = function (data, lev = NULL, model = NULL) 
{
  # Get y and yhat
  y = data$obs
  yhat = data[[levels(y)[[2]]]]
  
  conf_obj = confusionMatrix(ifelse(yhat > 0.5,"Y","N"), y)
  accuracy = as.numeric(conf_obj$overall["Accuracy"])
  missclassification = 1 - accuracy
  
  pred_obj = ROCR::prediction(yhat, y)
  auc = ROCR::performance(pred_obj, "auc" )@y.values[[1]]
  
  out = c("auc" = auc, "accuracy" = accuracy, "missclassification" = missclassification)
  out
}

## Get plot list of metric variables vs classification target 
get_plot_distr_metr_class = function(df.plot = df, vars = metr, target_name = "target", missinfo = NULL, 
                                     nbins = 20, color = twocol, legend_only_in_1stplot = TRUE) {
  # Get levels of target
  levs_target = levels(df.plot[[target_name]])
  
  # Univariate variable importance
  varimp = filterVarImp(df.plot[vars], df.plot[[target_name]], nonpara = TRUE)[levs_target[2]] %>% 
              mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .[[levs_target[2]]]
  names(varimp) = vars
  
  # Calculate missinfo
  if (is.null(missinfo)) missinfo = map_dbl(df.plot[vars], ~ round(sum(is.na(.)/nrow(df)), 3))
  
  # Loop over vars
  plots = map(vars, ~ {
    #.x = vars[1]
    print(.x)
    
    # Start histogram plot
    p = ggplot(data = df.plot, aes_string(x = .x)) +
      geom_histogram(aes_string(y = "..density..", fill = target_name), bins = nbins, position = "identity") +
      geom_density(aes_string(color = target_name)) +
      scale_fill_manual(limits = rev(levs_target), values = alpha(rev(color), .2), name = target_name) + 
      scale_color_manual(limits = rev(levs_target), values = rev(color), name = target_name) +
      #guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE))
      labs(title = paste0(.x," (VI: ", round(varimp[.x],2),")"),
           x = paste0(.x," (NA: ", missinfo[.x] * 100,"%)"))
    
    # Get underlying data for max of y-value and range of x-value
    tmp = ggplot_build(p)
    
    # Inner boxplot
    p.inner = ggplot(data = df.plot, aes_string(target_name, y = .x)) +
      geom_boxplot(aes_string(color = target_name)) +
      coord_flip() +
      scale_y_continuous(limits = c(min(tmp$data[[1]]$xmin), max(tmp$data[[1]]$xmax))) +
      scale_color_manual(values = color, name = target_name) +
      theme_void() +
      theme(legend.position = "none")
    
    # Put all together
    p = p + 
      scale_y_continuous(limits = c(-tmp$layout$panel_ranges[[1]]$y.range[2]/14, NA)) +
      theme_my +
      annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) 
    if (legend_only_in_1stplot == TRUE) {
      if (.x != vars[1]) p = p + theme(legend.position = "none") #legend only for first plot
    }
    p
  })
  plots
}



## Get plot list of metric variables vs regression target 
get_plot_distr_metr_regr = function(df.plot = df, vars = metr, target_name = "target", missinfo = NULL, 
                                    nbins = 50, color = hexcol, ylim = NULL, legend_only_in_1stplot = TRUE) {
  # Univariate variable importance
  varimp = sqrt(filterVarImp(df.plot[vars], df.plot[[target_name]], nonpara = TRUE)) %>% .[[1]]
  names(varimp) = vars
  
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
      labs(title = paste0(.x," (VI: ", round(varimp[.x],2),")"),
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
  plots
}


## Get plot list of nomial variables vs classification target 
get_plot_distr_nomi_class = function(df.plot = df, vars = nomi, target_name = "target",  
                                     color = twocol) {
  # Get levels of target
  levs_target = levels(df.plot[[target_name]])
  
  # Univariate variable importance
  varimp = filterVarImp(df.plot[vars], df.plot[[target_name]], nonpara = TRUE)[levs_target[2]] %>% 
    mutate(Y = round(ifelse(Y < 0.5, 1 - Y, Y),2)) %>% .[[levs_target[2]]]
  names(varimp) = vars
  
  # Loop over vars
  plots = map(vars, ~ {
    #.x = vars[1]
    print(.x)
    
    # Proportion of Target=Y (height of bars) and relative frequency of levels (width of bars)
    df.plot$target_num = ifelse(df.plot[[target_name]] == levs_target[2], 1, 0)
    df.ggplot = df.plot %>% 
      group_by_(.x) %>% 
      summarise(n = n(), prob = mean(target_num)) %>%
      ungroup() %>% 
      mutate(perc = n/sum(n), width = n/max(n))

    # Plot
    p = ggplot(df.ggplot, aes_string(x = .x, y = "prob")) +
      geom_bar(stat = "identity", width = df.ggplot$width, color = twocol[2], fill = alpha(twocol[2], 0.2)) +
      scale_x_discrete(labels = paste0(df.ggplot[[.x]], " (", round(100 * df.ggplot[["perc"]],1), "%)")) + 
      labs(title = paste0(.x," (VI:", varimp[.x], ")"), 
           x = "", 
           y = paste0("Proportion ",target_name," = ",levs_target[2])) +
      geom_hline(yintercept = sum(df.plot[[target_name]] == levs_target[2]) / nrow(df.plot), 
                 linetype = 2, color = "darkgrey") +
      coord_flip() +
      theme_my 
    p
  })
  plots
}


## Get plot list of nomial variables vs regression target 
get_plot_distr_nomi_regr = function(df.plot = df, vars = nomi, target_name = "target",  
                                    ylim = NULL) {
  # Univariate variable importance
  varimp = sqrt(filterVarImp(df.plot[vars], df.plot[[target_name]], nonpara = TRUE)) %>% .[[1]]
  names(varimp) = vars
  
  # Loop over vars
  plots = map(vars, ~ {
    #.x = vars[1]
    print(.x)
    
    # Main Boxplot
    p = ggplot(df.plot, aes_string(x = .x, y = target_name)) +
      geom_boxplot(varwidth = TRUE, outlier.size = 0.5) +
      stat_summary(aes(group = 1), fun.y = median, geom = "line", color = "black", linetype = 1) +
      stat_summary(aes(group = 1), fun.y = mean, geom = "line", color = "darkgrey", linetype = 2) +
      coord_flip() +
      scale_x_discrete(labels = paste0(levels(df.plot[[.x]]), " (", 
                                       round(100 * table(df.plot[[.x]])/nrow(df.plot), 1), "%)")) +
      labs(title = paste0(.x," (VI: ", round(varimp[.x],2),")"), x = "") +
      theme_my +
      theme(legend.position = "none") 
    if (!is.null(ylim)) p = p + ylim(ylim)
    
    # Get underlying data for max of y-value and range of x-value
    yrange = ggplot_build(p)$layout$panel_ranges[[1]]$x.range
    
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
    p   
  })
  plots
}


## Get plot list of correlations of variables
get_plot_corr <- function(outpdf, df.plot = df, input_type = "metr" , vars = metr, cutoff = NULL,
                          missinfo = NULL, method = "spearman") {

  # Correlation matrix
  if (input_type == "metr") {
    ## For metric variables
    m.corr = abs(cor(df[vars], method = tolower(method), use = "pairwise.complete.obs"))
    
    # Calculate missinfo
    if (is.null(missinfo)) missinfo = map_dbl(df.plot[vars], ~ round(sum(is.na(.)/nrow(df)), 3))
    
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
  if (!is.null(cutoff)) {
    tokeep = which(rowSums(ifelse(m.corr > cutoff, 1, 0)) > 1)
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
    gather(key = colvar, value = corr, -rowvar) 
  p = ggplot(df.ggplot, aes(x = rowvar, y = colvar)) +
    geom_tile(aes(fill = corr)) + 
    geom_text(aes(label = round(corr, 2))) +
    scale_fill_gradient(low = "white", high = "blue") +
    scale_x_discrete(limits = rev(rownames(m.corr))) +
    scale_y_discrete(limits = rownames(m.corr)) +
    labs(title = if (input_type == "metr") paste0(method," Correlation") else "Contig.Coef", fill = "", x = "", y = "") +
    theme_my + theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  p
}


## Get plot list  for ROC, Confusion, Distribution, Calibration, Gain, Lift, Precision-Recall, Precision
get_plot_performance = function(yhat, y, reduce = NULL, color = "blue", colors = twocol) {
  
  ## Prepare "direct" information (confusion, distribution, calibration)
  conf_obj = confusionMatrix(ifelse(yhat > 0.5,"Y","N"), y)
  df.confu = as.data.frame(conf_obj$table)
  df.distr = data.frame(y = y, yhat = yhat)
  df.calib = calibration(y~yhat, data.frame(y = y, yhat = 1 - yhat), cuts = 5)$data

  
  
  ## Prepare "reducable" information (roc, gain, lift, precrec, prec)
  # Create performance objects
  pred_obj = prediction(yhat, y)
  auc = performance(pred_obj, "auc" )@y.values[[1]]
  tprfpr = performance( pred_obj, "tpr", "fpr")
  precrec = performance( pred_obj, "ppv", "rec")
  help_gain = ifelse(y == "Y", 1, 0)[order(yhat, decreasing = TRUE)] 
  df.gainlift = data.frame("x" = 100*(1:length(yhat))/length(yhat),
                           "gain" = 100*cumsum(help_gain)/sum(help_gain)) %>% 
    mutate(lift = gain / x)
  
  # Thin out reducable objects (for big test data as this reduces plot size)
  if (!is.null(reduce)) {
    set.seed(123)
    i.reduce = sample(1:length(yhat), floor(reduce * length(yhat)))
    i.reduce = i.reduce[order(i.reduce)]
    for (type in c("x.values","y.values","alpha.values")) {
      slot(tprfpr, type)[[1]] = slot(tprfpr, type)[[1]][i.reduce]
      slot(precrec, type)[[1]] = slot(precrec, type)[[1]][i.reduce]
    }
    df.gainlift = df.gainlift[i.reduce,]
  } 
  
  # Collect information of reduced information into data frames
  df.roc = data.frame("fpr" = tprfpr@x.values[[1]], "tpr" = tprfpr@y.values[[1]])
  i.alpha = map_int(seq(0.1,0.9,0.1), function(.) which.min(abs(tprfpr@alpha.values[[1]] - .))) #cutoff values
  df.precrec = data.frame("rec" = precrec@x.values[[1]], "prec" = precrec@y.values[[1]], 
                          x = 100*(1:length(precrec@x.values[[1]]))/length(precrec@x.values[[1]]))
  df.precrec[is.nan(df.precrec$prec),"prec"] = 1 #adapt first value
  

  
  ## Plots
  # ROC
  p_roc = ggplot(df.roc, aes(x = fpr, y = tpr)) +
    geom_line(color = "blue", size = .5) +
    geom_abline(intercept = 0, slope = 1, color = "grey") + 
    geom_point(aes(fpr, tpr), data = df.roc[i.alpha,], color = "red", size = 0.8) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.1)) + 
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.1)) +
    labs(title = paste0("ROC (auc=", round(auc,3), ")"), 
         x = expression(paste("fpr: P(", hat(y), "=1|y=0)", sep = "")),
         y = expression(paste("tpr: P(", hat(y), "=1|y=1)", sep = ""))) + 
    theme_my 
  
  # Condusion Matrix
  p_confu = ggplot(df.confu, aes(Prediction, Reference)) +
    geom_tile(aes(fill = Freq)) + 
    geom_text(aes(label = Freq)) +
    scale_fill_gradient(low = "white", high = "blue") +
    scale_y_discrete(limits = rev(levels(df.confu$Reference))) +
    labs(title = paste0("Confusion Matrix (Accuracy = ", round(conf_obj$overall["Accuracy"], 3), ")")) +
    theme_my 
  
  # Stratified distribution of predictions (plot similar to plot_distr_metr)
  p_distr = ggplot(data = df.distr, aes(x = yhat)) +
    geom_histogram(aes(y = ..density.., fill = y), bins = 40, position = "identity") +
    geom_density(aes(color = y)) +
    scale_fill_manual(values = alpha(colors, .2), name = "Target (y)") + 
    scale_color_manual(values = colors, name = "Target (y)") +
    labs(title = "Predictions", 
         x = expression(paste("Prediction (", hat(y),")", sep = ""))) +
    guides(fill = guide_legend(reverse = TRUE), color = guide_legend(reverse = TRUE))
  tmp = ggplot_build(p_distr)
  p.inner = ggplot(data = df.distr, aes_string("y", "yhat")) +
    geom_boxplot(aes_string(color = "y")) +
    coord_flip() +
    scale_y_continuous(limits = c(min(tmp$data[[1]]$xmin), max(tmp$data[[1]]$xmax))) +
    scale_color_manual(values = colors, name = "Target") +
    theme_void() +
    theme(legend.position = "none")
  p_distr = p_distr + 
    scale_y_continuous(limits = c(-tmp$layout$panel_ranges[[1]]$y.range[2]/10, NA)) +
    theme_my +
    annotation_custom(ggplotGrob(p.inner), xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0) 
  
  # Gain + Lift
  p_gain = ggplot(df.gainlift) +
    geom_polygon(aes(x, y), data.frame(x = c(0,100,100*sum(help_gain)/length(help_gain)), y = c(0,100,100)), 
                 fill = "grey90", alpha = 0.5) +
    geom_line(aes(x, gain), df.gainlift, color = "blue", size = .5) +
    scale_x_continuous(limits = c(0,100), breaks = seq(0,100,10)) + 
    labs(title = "Gain", x = "% Samples Tested", y = "% Samples Found") +
    theme_my 
  p_lift = ggplot(df.gainlift) +
    geom_line(aes(x, lift), color = "blue", size = .5) +
    scale_x_continuous(limits = c(0,100), breaks = seq(0,100,10)) + 
    labs(title = "Lift", x = "% Samples Tested", y = "Lift") +
    theme_my 
  
  # Calibrate
  p_calib = ggplot(df.calib, aes(midpoint, Percent)) +
    geom_line(color = "blue") +
    geom_point(color = "blue") +  
    geom_abline(intercept = 0, slope = 1, color = "grey") + 
    scale_x_continuous(limits = c(0,100), breaks = seq(10,90,20)) + 
    scale_y_continuous(limits = c(0,100), breaks = seq(0,100,10)) +
    labs(title = "Calibration", x = "Midpoint Predicted Event Probability", y = "Observed Event Percentage") +
    theme_my 
  
  # Precision Recall
  p_precrec = ggplot(df.precrec, aes(rec, prec)) +
    geom_line(color = "blue", size = .5) +
    geom_point(aes(x = ), df.precrec[i.alpha,], color = "red", size = 0.8) +
    scale_x_continuous(breaks = seq(0,1,0.1)) +
    labs(title = "Precision Recall Curve", x = expression(paste("recall=tpr: P(", hat(y), "=1|y=1)", sep = "")),
         y = expression(paste("precision: P(y=1|", hat(y), "=1)", sep = ""))) +
    theme_my 
  
  # Precision
  p_prec = ggplot(df.precrec, aes(x, prec)) +
    geom_line(color = "blue", size = .5) +
    geom_point(aes(x, prec), df.precrec[i.alpha,], color = "red", size = 0.8) +
    scale_x_continuous(breaks = seq(0,100,10)) +
    labs(title = "Precision", x = "% Samples Tested",
         y = expression(paste("precision: P(y=1|", hat(y), "=1)", sep = ""))) +
    theme_my
  
  
  
  ## Collect Plots and return
  plots = list(p_roc = p_roc, p_confu = p_confu, p_distr = p_distr, p_calib = p_calib, 
               p_gain = p_gain, p_lift = p_lift, p_precrec = p_precrec, p_prec = p_prec)
  plots
}



## Variable importance by permutation argument 
get_varimp_by_permutation = function(df.for_varimp = df.test, fit.for_varimp = fit, predictors,
                                     vars = predictors) {
  
  #browser()
  # Original performance
  perf_orig = my_twoClassSummary(data.frame(
    obs = df.for_varimp$target, 
    predict(fit.for_varimp, df.for_varimp[predictors], type = "prob")["Y"]))["auc"]

  # Permute
  set.seed(999)
  i.permute = sample(1:nrow(df.for_varimp)) #permutation vector
  start = Sys.time()
  df.varimp = foreach(i = 1:length(vars), .combine = bind_rows, .packages = "caret", 
                      .export = "my_twoClassSummary") %dopar% {
    #i=1
    df.tmp = df.for_varimp
    df.tmp[[vars[i]]] = df.tmp[[vars[i]]][i.permute] #permute
    yhat = predict(fit.for_varimp, df.tmp[predictors], type="prob")["Y"]  #predict
    perf = my_twoClassSummary(cbind(obs = df.for_varimp$target, yhat))["auc"]  #performance
    data.frame(variable = vars[i], perfdiff = max(0, perf_orig - perf), stringsAsFactors = FALSE) #performance diff
  }
  print(Sys.time() - start)
  # Calculate importance as scaled performance difference
  df.varimp = df.varimp %>%
    mutate(importance = 100 * perfdiff/max(perfdiff)) %>%     
    arrange(desc(importance))
  df.varimp 
}



## Get plot list for variable importance
get_plot_varimp = function(df.plot = df.varimp, vars = topn_vars, col = c("blue","orange","red"), 
                           df.plot_boot = NULL, run_name = "run", bootstrap_lines = TRUE, bootstrap_CI = TRUE) {
  # Subset
  df.ggplot = df.plot %>% filter(variable %in% vars)
  
  # Plot
  plot = ggplot(df.ggplot) +
    geom_bar(aes(x = reorder(variable, importance), y = importance, fill = color), stat = "identity") +
    scale_fill_manual(values = col) +
    labs(title = paste0("Top ", min(length(vars), length(predictors))," Important Variables (of ", 
                        length(predictors), ")"), 
         x = "", y = "Importance (scaled to 100)") +
    coord_flip() +
    guides(fill = guide_legend(reverse = TRUE, title = "")) +
    theme_my   
  
  # Add boostrap information
  if (!is.null(df.plot_boot)) {
    # Add bootstrap lines
    if (bootstrap_lines == TRUE) {
      plot = plot +
        geom_line(aes_string(x = "variable", y = "importance", group = run_name), data = df.plot_boot, 
                  color = "grey", size = 0.1) +
        geom_point(aes_string(x = "variable", y = "importance", group = run_name), data = df.plot_boot, 
                   color = "black", size = 0.3) 
    }
    
    # Add bootstrap Confidence Intervals
    if (bootstrap_CI == TRUE) {
      # Calculate confidence intervals
      df.help = df.plot_boot %>% 
        group_by(variable) %>% 
        summarise(sd = sd(importance)) %>% 
        left_join(select(df.ggplot, variable, importance)) %>% 
        mutate(lci = importance - 1.96*sd, rci = importance + 1.96*sd)
      plot = plot +
        geom_errorbar(aes(x = variable, ymin = lci, ymax = rci), data = df.help, size = 0.5, width = 0.25)
    }
  }
  plot
}



## Variable Importance
plot_variableimportance = function(outpdf, vars, fit.plot = fit, l.boot = NULL, 
                                   ncols = 5, nrows = 2, w = 18, h = 12) {
  # Group importances
  df.tmp = varImp(fit.plot)$importance %>% 
    mutate(variable = rownames(varImp(fit.plot)$importance)) %>% 
    filter(variable %in% vars) %>% 
    arrange(desc(Overall)) %>% 
    mutate(color = cut(Overall, c(-1,10,50,100), labels = c("low","middle","high"))) 
  
  # Plot
  p = ggplot(df.tmp) +
    geom_bar(aes(variable, Overall, fill = color), stat = "identity") +
    scale_x_discrete(limits = rev(df.tmp$variable)) +
    scale_fill_manual(values = c("blue","orange","red")) +
    labs(title = paste0("Top ", min(topn, length(predictors))," Important Variables (of ", length(predictors), ")"), 
         x = "", y = "Importance (scaled to 100)") +
    coord_flip() +
    #geom_hline(yintercept = c(10,50), color = "grey", linetype = 2) +
    theme_my + theme(legend.position = "none")
  
  # Bootstrap lines
  if (!is.null(l.boot)) {
    df.tmpboot = map_df(l.boot, ~ {
      df = varImp(.)$importance
      df$variable = rownames(df)
      df
    } , .id = "run")
    p = p + geom_line(aes(variable, Overall, group = run), df.tmpboot, color = "grey", size = 0.1) +
      geom_point(aes(variable, Overall, group = run), df.tmpboot, color = "black", size = 0.3)
  }
  ggsave(outpdf, p, width = 8, height = 6)
} 



## Partial Depdendence
plot_partialdependence = function(outpdf, vars, df = df.interpret, fit = fit.gbm, l.boot = NULL, CI = FALSE,
                                  ylim = c(0,1), ncols = 5, nrows = 2, w = 18, h = 12) {

  # Final model
  model = fit$finalModel
  
  # Derive offset
  offset = model$initF - plot(model, i.var = "INT", type = "link", return.grid = TRUE)[1,"y"]
  
  # Plot
  plots = map(vars, ~ {
    #.=vars[2]
    # Plot data (must be adapted due to offset of plot(gbm) and possible undersampling)
    
    df.plot = plot(model, i.var = ., type = "link", return.grid = TRUE) #get plot data on link level
    p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
    df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
    
    if (is.factor(df[[.]])) {
      # Width of bars correspond to freqencies
      tmp = table(df[,.])
      df.plot$width = as.numeric(tmp[df.plot[[.]]])/max(tmp)
      
      # Plot for a nominal variable
      p = ggplot(df.plot, aes_string(., "y")) +
        geom_bar(stat = "identity",  width = df.plot$width, fill = "grey", color = "black") +
        labs(title = ., x = "", y = expression(paste("P(", hat(y), "=1)"))) +
        scale_y_continuous(limits = ylim) +
        theme_my         
    } else {
      # Plot for a metric variable
      df.rug = data.frame(q = quantile(df[,.], prob = seq(.05, .95, .1)), y = 0)
      p = ggplot(df.plot, aes_string(., "y")) +
        geom_line(stat = "identity", color = "black") +
        geom_rug(aes(q, y), df.rug, sides = "b", col = "red") +
        labs(title = ., x = "", y = expression(paste("P(", hat(y), "=1)"))) +
        scale_y_continuous(limits = ylim) +
        theme_my      
    }
    
    # Add Bootstrap lines and dots
    if (!is.null(l.boot)) {
      
      # Do the same as above for each bootstrapped model
      varactual = .
      df.tmpboot = map_df(l.boot, ~ {
        model_boot = .$finalModel
        offset_boot = model_boot$initF - plot(model_boot, i.var = "INT", type = "link", return.grid = TRUE)[1,"y"]
        df.plot_boot = plot(model_boot, i.var = varactual, type = "link", return.grid = TRUE)
        p_sample_boot = 1 - (1 - 1/(1 + exp(df.plot_boot$y + offset_boot)))
        df.plot_boot$y = prob_samp2full(p_sample_boot, b_sample, b_all)
        df.plot_boot
      } , .id = "run")
      
      df.plot = left_join(df.plot, df.tmpboot %>% group_by_(varactual) %>% summarise(sd = sd(y))) %>% 
        mutate(ymin = y - 1.96*sd, ymax = y + 1.96*sd)
      
      if (is.factor(df[,.])) {
        if (CI == TRUE) {
          p = p + 
            geom_errorbar(aes_string(., ymin = "ymin", ymax = "ymax"), data = df.plot, size = 0.5, width = 0.25)
          
        } else {
          p = p + 
            geom_line(aes_string(., "y", group = "run"), df.tmpboot, color = "lightgrey", size = 0.1) +
            geom_point(aes_string(., "y", group = "run"), df.tmpboot, color = "black", size = 0.3)
          }
      } else {
        if (CI == TRUE) {
          p = p +             
            geom_ribbon(aes_string(., ymin = "ymin", ymax = "ymax"), data = df.plot, alpha = 0.2)
        } else {
          p = p + 
            geom_line(aes_string(., "y", group = "run"), df.tmpboot, color = "lightgrey") +
            geom_line(aes_string(., "y"), df.plot, stat = "identity", color = "black") #plot black line again
        }
      }
    }
    # Add (prior) base probability
    p + geom_hline(yintercept = b_all, linetype = 3)
  })
  ggsave(outpdf, marrangeGrob(plots, ncol = ncols, nrow = nrows, top = NULL), width = w, height = h)
}



## Plot Interactiontest
plot_interactiontest = function(outpdf, vars, df = df.interpret, fit = fit.gbm, l.boot = NULL, 
                                ncols = 4, w = 18, h = 12) {
  
  # Derive interaction matrix for topn important variables
  pred_inter = setdiff(vars,"INT") #remove INT from testing variables
  k = length(pred_inter)
  m.inter = matrix(0, k, k)
  for (i in 1:(k - 1)) {
    for (j in (i + 1):k) {
      # Interaction Test
      m.inter[i,j] = interact.gbm(fit$finalModel, df[pred_inter], pred_inter[c(i,j)], fit$finalModel$tuneValue$n.trees)
      m.inter[j,i] = m.inter[i,j]
    }
  }
  colnames(m.inter) = pred_inter
  rownames(m.inter) = pred_inter
  m.inter[is.na(m.inter)] = 0
  
  
  ## Plot in correlation matrix style
  df.inter = as.data.frame(m.inter) %>% 
    mutate(rowvar = rownames(m.inter)) %>% 
    gather(key = colvar, value = inter, -rowvar)
  p = ggplot(df.inter, aes(rowvar, colvar)) +
    geom_tile(aes(fill = inter)) + 
    geom_text(aes(label = round(inter, 2))) +
    scale_fill_gradient(low = "white", high = "blue") +
    scale_x_discrete(limits = rownames(m.inter)) + 
    scale_y_discrete(limits = rev(rownames(m.inter))) +
    labs(title = "Interaction", fill = "", x = "", y = "") +
    theme_my +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) 
  ggsave(outpdf, p, width = w, height = h)
  
  
  
  if (!is.null(l.boot)) {
    # Do the same as above for each bootstrapped model and collect
    df.inter_boot = map_df(l.boot, ~ {
      for (i in 1:(k - 1)) {
        for (j in (i + 1):k) {
          # Interaction Test
          m.inter[i,j] = interact.gbm(.$finalModel, df[pred_inter], pred_inter[c(i,j)], .$finalModel$tuneValue$n.trees)
          m.inter[j,i] = m.inter[i,j]
        }
      }
      m.inter[is.na(m.inter)] = 0
      
      df.inter = as.data.frame(m.inter) %>% 
        mutate(rowvar = rownames(m.inter)) %>% 
        gather(key = colvar, value = inter, -rowvar)
      df.inter
    }, .id = "run")
    
    
    # Same plot but now facetting
    p_boot = ggplot(df.inter_boot, aes(rowvar, colvar)) +
      geom_tile(aes(fill = inter)) + 
      geom_text(aes(label = round(inter, 2))) +
      scale_fill_gradient(low = "white", high = "blue") +
      scale_x_discrete(limits = rownames(m.inter)) + 
      scale_y_discrete(limits = rev(rownames(m.inter))) +
      labs(title = "Interaction per Bootstrap Run", fill = "", x = "", y = "") +
      theme_my +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))  +
      facet_wrap( ~ run, ncol = ncols)
  }
  ggsave(paste0(str_split(outpdf,".pdf", simplify = TRUE)[1,1],"_boot.pdf"), p_boot, width = w, height = h)
}




## Plot interactions of m.gbm
plot_inter = function(outpdf, vars = inter, df = df.interpret, fit = fit.gbm, 
                      ylim = c(0,1), w = 12, h = 8) {
  # outpdf="./output/interaction1.pdf"; vars=inter1; df=df.interpret; fit=fit; w=12; h=8; ylim = c(0,.3)
  
  # Final model
  model = fit$finalModel
  
  # Derive offset
  offset = model$initF - plot(model, i.var = "INT", type = "link", return.grid = TRUE)[1,"y"]
  
  # Marginal plots for digonal
  plots_marginal = map(vars, ~ {
    #.=vars
    # Get interaction data
    df.plot = plot(model, i.var = ., type = "link", return.grid = TRUE) #get plot data on link level
    p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
    df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
    
    if (is.factor(df[[.]])) {
      # Width of bars correspond to freqencies
      tmp = table(df[,.])
      df.plot$width = as.numeric(tmp[df.plot[[.]]])/max(tmp)
      
      # Plot for a nominal variable
      p = ggplot(df.plot, aes_string(., "y", fill = .)) +
        geom_bar(stat = "identity",  width = df.plot$width, color = "black") +
        labs(title = ., x = "", y = expression(paste("P(", hat(y), "=1)"))) +
        scale_fill_manual(values = manycol) +
        scale_y_continuous(limits = ylim) +
        theme_my + 
        theme(legend.position = "none")
    } else {
      # Plot for a metric variable
      df.rug = data.frame(q = quantile(df[,.], prob = seq(.05, .95, .1)), y = 0)
      p = ggplot(df.plot, aes_string(., "y")) +
        geom_line(stat = "identity", color = "black") +
        geom_rug(aes(q, y), df.rug, sides = "b", col = "red") +
        labs(title = ., x = "", y = expression(paste("P(", hat(y), "=1)"))) +
        scale_y_continuous(limits = ylim) +
        theme_my      
    }
    p + geom_hline(yintercept = b_all, linetype = 3)
  })
  
  # Interaction plots 
  df.plot = plot(model, i.var = vars, type = "link", return.grid = TRUE) #get plot data on link level
  p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
  df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
  # quantile(df$TT4, seq(0,1, length.out=100))
  # (max(df$TT4) - min(df$TT4))/99 + min(df$TT4)
  
  if (sum(map_lgl(df.plot[vars], ~ is.factor(.))) == 2) {
    # Mosaicplot for nominal-nominal interaction
    plots_inter = map(1:2, ~ { 
      if (.==2) vars = rev(vars)
      #tmp = table(df[vars[2]])
      #df.plot[[vars[1]]] = factor(df.plot[[vars[1]]], levels = rev(levels(df.plot[[vars[1]]])))
      ggplot(df.plot, aes_string(vars[2], "y", fill = vars[1])) +
        geom_bar(stat = "identity", position = "fill") + #, width = rep(tmp/max(tmp), 5)) +
        scale_fill_manual(values = manycol) +
        labs(y = "", x = "") +
        theme_my      
    })
  }  
  if (sum(map_lgl(df.plot[vars], ~ is.factor(.))) == 1) {
    # Grouped line chart for metric-nominal interaction
    p = ggplot(df.plot, aes_string(vars[1], "y", color = vars[2])) +
      geom_line(stat = "identity") +
      labs(y = expression(paste("P(", hat(y), "=1)"))) +
      scale_color_manual(values = manycol) +
      scale_y_continuous(limits = ylim*2) +
      guides(color = guide_legend(reverse = TRUE)) +
      theme_my      
    plots_inter = list(p,p)
  }   
  if (sum(map_lgl(df.plot[vars], ~ is.factor(.))) == 0) {
    # Grouped (by quantiles) line chart for metric-metric interaction
    plots_inter = map(1:2, ~ { 
      if (.==2) vars = rev(vars)
      val_near_quant =   map_dbl(quantile(df[[vars[2]]], seq(.05,.95,.1)), ~ {
        df.plot[[vars[2]]][which.min(abs(df.plot[[vars[2]]] - .))]})
      i.tmp = df.plot[[vars[2]]] %in% val_near_quant
      df.tmp = df.plot[i.tmp,]
      df.tmp[vars[2]] = factor( round(df.tmp[[vars[2]]],2) )
      
      ggplot(df.tmp, aes_string(vars[1], "y", color = vars[2])) +
        geom_line(stat = "identity") +
        labs(y = expression(paste("P(", hat(y), "=1)"))) +
        scale_color_manual(values = manycol) +
        scale_y_continuous(limits = ylim) +
        guides(color = guide_legend(reverse = TRUE)) +
        theme_my   
    })
  } 
  
  # Arrange plots
  plots = list(plots_marginal[[1]], plots_inter[[1]], plots_inter[[2]], plots_marginal[[2]])
  ggsave(outpdf, marrangeGrob(plots, ncol = 2, nrow = 2, top = NULL), width = w, height = h)
}




# Animated Interaction of 2 metric variables
plot_inter_active = function(outfile, vars = inter, df = df.interpret, fit = fit.gbm, duration = 15) {
  
  # Final model
  model = fit$finalModel
  
  # Derive offset
  offset = model$initF - plot(model, i.var = "INT", type = "link", return.grid = TRUE)[1,"y"]
  
  # Get interaction data
  df.plot = plot(model, i.var = vars, type = "link", return.grid = TRUE) #get plot data on link level
  p_sample = 1 - (1 - 1/(1 + exp(df.plot$y + offset))) #add offset and calcualte dependence on response level
  df.plot$y = prob_samp2full(p_sample, b_sample, b_all) #switch to correct probability of full data
  
  # Prepare 3d plot
  x = unique(df.plot[[vars[1]]])
  y = unique(df.plot[[vars[2]]])
  z = matrix(df.plot$y, length(x), length(y), byrow = FALSE)
  nrz = nrow(z)
  colcut = cut((z[-1, -1] + z[-1, -nrz] + z[-nrz, -1] + z[-nrz, -ncol(z)])/4, 100)
  
  # html Widget
  persp3d(x, y, z, col = col3d[colcut], phi = 30, theta = 50, axes = T, ticktype = 'detailed',
          xlab = vars[1], ylab = vars[2], zlab = "")
  writeWebGL(dir = file.path(paste0(outfile)), width = 1000)
  rgl.close()
  
  # animated gif
  open3d("windowRect" = 2*c(20,20,400,400))
  persp3d(x, y, z, col = col3d[colcut], phi = 30, theta = 50, axes = T, ticktype = 'detailed',
          xlab = vars[1], ylab = vars[2], zlab = "")
  movie3d(spin3d(axis = c(0,0,1), rpm = 2), duration = duration, convert = NULL, clean = TRUE, movie = "test",
          dir = paste0(outfile))
  rgl.close()
  
}


