
library(xgboost)
data(agaricus.train, package='xgboost')
train <- agaricus.train
dtrain <- xgb.DMatrix(train$data, label=train$label)
xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
dtrain <- xgb.DMatrix('xgb.DMatrix.data')

test = xgb.DMatrix(as.data.frame(mtcars))


a = agaricus.train$data
b = as.data.frame(a)


library(Matrix)
m.train = sparse.model.matrix(formula, data = df.train[c("target",predictors)])
m.train@factors

df.tmp = data.frame(factor_a = paste0("L_", 1:10000))
m.sparse = sparse.model.matrix(~factor_a-1, data = df.tmp)
m.xgbd = xgb.DMatrix(m.sparse)
inherits(m.xgbd, "xgb.DMatrix")
format(object.size(m.sparse), "Mb")
format(object.size(m.xgbd), "Mb")
save(m.sparse, file = "m.sparse")

m = model.matrix(~factor_a-1, data = df.tmp)
format(object.size(m), "Mb")
save(m, file = "m")




col_names = attr(m.train, ".Dimnames")[[2]]
trees = xgb.model.dt.tree(col_names, model = fit$finalModel, n_first_tree = NULL)

nodes.train = predict(fit$finalModel, m.train, predleaf = TRUE)
getStatsForTrees(trees, nodes.train, type = type, 
                 base_score = base_score)
tree_list = xgboostExplainer:::getStatsForTrees(trees, nodes.train, type = "binary", 
                             base_score = 0.5)


