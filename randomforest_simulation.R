rm(list=ls())
install.packages("randomForest")
library("randomForest")
install.packages("Metrics")
library("tree")

N = 200 # number of simulation
n = 500 # sample size
# generate 6 predictors 
set.seed(1)
x1=rnorm(n,10,3)
x2=rnorm(n,2,1)
x3=rnorm(n,7,2)
x4=rnorm(n,5,3)
x5=rnorm(n,4,1)
x6=rnorm(n,9,1)
d = data.frame(x1, x2, x3, x4, x5, x6)
E.y = 1 + 2*x1 + 3*x2 + 4*x3 + 5*x4 + 6*x5 + 7*x6 #expected value of y

# partition data into training and test sets
set.seed(1)
id.train = sample(1:n, 400, replace = FALSE)

##########################
##########################
#### using a fixed number of bootstrap samples (default) and number of predictors in RF (2)
#### compare RF, bagging, and LS results 
##########################
##########################
#bias_table=matrix(NA,N,3)

yhat_table = array(NA, c(N, 100, 3)) 

for (i in 1:N) {
  error=rnorm(n, mean = 0, sd = 40) #change sd for error term
  d$y=1 + 2*x1 + 3*x2 + 4*x3 + 5*x4 + 6*x5 + 7*x6 + error
  data.train=d[id.train, ]
  data.test=d[-id.train, -7]

  # single tree
  y.tree = tree(y~x1+x2+x3+x4+x5+x6, data=data.train)
  yhat.tree = predict(y.tree, newdata = data.test)
  yhat_table[i, ,1] = yhat.tree  

  #bagging
  set.seed(i)
  y.bg=randomForest(y~x1+x2+x3+x4+x5+x6, data=data.train,ntree=500, mtry=6, importance=TRUE)
  yhat.bg=predict(y.bg,newdata=data.test)
  yhat_table[i, ,2] = yhat.bg

  #random forest
  set.seed(i)
  y.rf=randomForest(y~x1+x2+x3+x4+x5+x6,data=data.train, ntree=500,mtry=2, importance=TRUE)
  yhat.rf=predict(y.rf,newdata=data.test)
  yhat_table[i, ,3] = yhat.rf
}

#variance
var.tree = apply(yhat_table[,,1], 2, var)
var.bg = apply(yhat_table[,,2], 2, var)
var.rf = apply(yhat_table[,,3], 2, var)
mean(var.tree)
mean(var.bg)
mean(var.rf)
#bias=mean(all predicted y values)-E(y)
#single tree
mean.treehat = apply(yhat_table[,,1], 2, mean)
bias.tree = mean.treehat - E.y[-id.train]
mean(abs(bias.tree))

#bagging
mean.bghat = apply(yhat_table[,,2], 2, mean)
bias.bagging = mean.bghat - E.y[-id.train]
mean(abs(bias.bagging))

#random forest
mean.rfhat = apply(yhat_table[,,3], 2, mean)
bias.rf = mean.rfhat - E.y[-id.train]
mean(abs(bias.rf))


#MSE=bias^2+variance
mse.tree=bias.tree^2+var.tree
mse.bagging=bias.bagging^2+var.bg
mse.rf=bias.rf^2+var.rf
mean(mse.tree)
mean(mse.bagging)
mean(mse.rf)


##########################
##########################
#### change number of bootstrap samples and predictors (for bagging and RF)
##########################
##########################

B = c(5, 10, 20, 50, 100, 200, 500) # number of bootstrap samples 
M = c(2:5) # number of predictors to use in RF 

yhat_bag_table = array(NA, c(N, 100, length(B)))
yhat_rf_m1_table = array(NA, c(N, 100, length(B)))
yhat_rf_m2_table = array(NA, c(N, 100, length(B)))
yhat_rf_m3_table = array(NA, c(N, 100, length(B)))
yhat_rf_m4_table = array(NA, c(N, 100, length(B)))

for (i in 1:N) # for each sample 
{ 
  set.seed(i)
  error = rnorm(n, mean = 0, sd = 40)
  d$y = 1 + 2*x1 + 3*x2 + 4*x3 + 5*x4 + 6*x5 + 7*x6 + error
  data.train=d[id.train, ]
  data.test=d[-id.train, -7] # remove true y from test set
  
  for (b in 1:length(B)) # for each bootstrap sample size 
  { 
    y.bag = randomForest(y ~ x1 + x2 + x3 + x4 + x5 + x6, data = data.train, 
                       ntree = B[b], mtry = 6, importance=TRUE)
    yhat.bag = predict(y.bag, newdata = data.test)
    yhat_bag_table[i, , b] = yhat.bag
    
    for(m in 1:length(M)) # for each number of predictors to use in RF
    {
      set.seed(m)
      y.rf = randomForest(y~x1+x2+x3+x4+x5+x6, data=data.train, ntree = B[b], mtry = M[m],importance=TRUE)
      yhat.rf = predict(y.rf, newdata = data.test)
      
      if (m == 1){
        yhat_rf_m1_table[i, , b] = yhat.rf
      } else if (m == 2){
        yhat_rf_m2_table[i, , b] = yhat.rf
      } else if (m == 3){
        yhat_rf_m3_table[i, , b] = yhat.rf
      } else if (m == 4){
        yhat_rf_m4_table[i, , b] = yhat.rf
      }
      
    }
  }
}

######## bagging summary (bias, var)
var_bag_summary = matrix(NA, length(B), 100) # 100 var for each bootstrap sample size 
for (i in 1:length(B)){
  var_bag_summary[i, ] = apply(yhat_bag_table[,,i], 2, var)
}

mean.yhat.bag = matrix(NA,length(B) , 100)
bias_bag_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mean.yhat.bag[i, ] = apply(yhat_bag_table[ , , i], 2, mean)
  bias_bag_summary[i, ] = mean.yhat.bag[i, ] - E.y[-id.train]
}

mse_bag_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mse_bag_summary[i, ] = (bias_bag_summary[i, ])^2 + var_bag_summary[i, ]
}

######## RF m1 summary (bias, var)
var_rf_summary_m1 = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  var_rf_summary_m1[i, ] = apply(yhat_rf_m1_table[,,i], 2, var)
}

mean.yhat.rf.m1 = matrix(NA, length(B), 100)
bias_rf_m1_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mean.yhat.rf.m1[i, ] = apply(yhat_rf_m1_table[ , , i], 2, mean)
  bias_rf_m1_summary[i, ] = mean.yhat.rf.m1[i, ] - E.y[-id.train]
}

mse_rf_m1_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mse_rf_m1_summary[i, ] = (bias_rf_m1_summary[i, ])^2 + var_rf_summary_m1[i, ]
}

######## RF m2 summary (bias, var)
var_rf_summary_m2 = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  var_rf_summary_m2[i, ] = apply(yhat_rf_m2_table[,,i], 2, var)
}

mean.yhat.rf.m2 = matrix(NA, length(B), 100)
bias_rf_m2_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mean.yhat.rf.m2[i, ] = apply(yhat_rf_m2_table[ , , i], 2, mean)
  bias_rf_m2_summary[i, ] = mean.yhat.rf.m2[i, ] - E.y[-id.train]
}

mse_rf_m2_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mse_rf_m2_summary[i, ] = (bias_rf_m2_summary[i, ])^2 + var_rf_summary_m2[i, ]
}

######## RF m3 summary (bias, var)
var_rf_summary_m3 = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  var_rf_summary_m3[i, ] = apply(yhat_rf_m3_table[,,i], 2, var)
}

mean.yhat.rf.m3 = matrix(NA, length(B), 100)
bias_rf_m3_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mean.yhat.rf.m3[i, ] = apply(yhat_rf_m3_table[ , , i], 2, mean)
  bias_rf_m3_summary[i, ] = mean.yhat.rf.m3[i, ] - E.y[-id.train]
}

mse_rf_m3_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mse_rf_m3_summary[i, ] = (bias_rf_m3_summary[i, ])^2 + var_rf_summary_m3[i, ]
}

######## RF m4 summary (bias, var)
var_rf_summary_m4 = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  var_rf_summary_m4[i, ] = apply(yhat_rf_m4_table[,,i], 2, var)
}

mean.yhat.rf.m4 = matrix(NA, length(B), 100)
bias_rf_m4_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mean.yhat.rf.m4[i, ] = apply(yhat_rf_m4_table[ , , i], 2, mean)
  bias_rf_m4_summary[i, ] = mean.yhat.rf.m4[i, ] - E.y[-id.train]
}

mse_rf_m4_summary = matrix(NA, length(B), 100)
for (i in 1:length(B)){
  mse_rf_m4_summary[i, ] = (bias_rf_m4_summary[i, ])^2 + var_rf_summary_m4[i, ]
}


##############################################################
var_rf_bag_summary = matrix(NA, length(M) + 1, length(B))
var_rf_bag_summary[1, ] = apply(var_rf_summary_m1, 1, mean)
var_rf_bag_summary[2, ] = apply(var_rf_summary_m2, 1, mean)
var_rf_bag_summary[3, ] = apply(var_rf_summary_m3, 1, mean)
var_rf_bag_summary[4, ] = apply(var_rf_summary_m4, 1, mean)
var_rf_bag_summary[5, ] = apply(var_bag_summary, 1, mean)

var.transposedSummary = t(var_rf_bag_summary)
matplot(var.transposedSummary, main = "Variance Across Different Number of Bootstrap Samples", 
        type = "l", col = 1:5, lty = 1:5, lwd = 3, xaxt="n",xlab="Number of Bootstrap Samples",
        ylab="Variance")
axis(1, at = 1:7, labels = c("5", "10", "20", "50","100","200","500"))
legend("topright", legend = c("# predictors = 2", "# predictors = 3", "# predictors = 4", "# predictors = 5", "# predictors = 6 (bagging)"), 
       col = 1:5, lty = 1:5, lwd = 3)

##############################################################
mse_rf_bag_summary = matrix(NA, length(M) + 1, length(B))
mse_rf_bag_summary[1, ] = apply(mse_rf_m1_summary, 1, mean)
mse_rf_bag_summary[2, ] = apply(mse_rf_m2_summary, 1, mean)
mse_rf_bag_summary[3, ] = apply(mse_rf_m3_summary, 1, mean)
mse_rf_bag_summary[4, ] = apply(mse_rf_m4_summary, 1, mean)
mse_rf_bag_summary[5, ] = apply(mse_bag_summary, 1, mean)

mse.transposedSummary = t(mse_rf_bag_summary)
matplot(mse.transposedSummary, main = "MSE Across Different Number of Bootstrap Samples", 
        type = "l", col = 1:5, lty = 1:5, lwd = 3, xaxt="n",xlab="Number of Bootstrap Samples",
        ylab="MSE")
axis(1, at = 1:7, labels = c("5", "10", "20", "50","100","200","500"))
legend("topright", legend = c("# predictors = 2", "# predictors = 3", "# predictors = 4", "# predictors = 5", "# predictors = 6 (bagging)"), 
       col = 1:5, lty = 1:5, lwd = 3)


##############################################################
bias_rf_bag_summary = matrix(NA, length(M) + 1, length(B))
bias_rf_bag_summary[1, ] = apply(abs(bias_rf_m1_summary), 1, mean)
bias_rf_bag_summary[2, ] = apply(abs(bias_rf_m2_summary), 1, mean)
bias_rf_bag_summary[3, ] = apply(abs(bias_rf_m3_summary), 1, mean)
bias_rf_bag_summary[4, ] = apply(abs(bias_rf_m4_summary), 1, mean)
bias_rf_bag_summary[5, ] = apply(abs(bias_bag_summary), 1, mean)

bias.transposedSummary = t(bias_rf_bag_summary)
matplot(bias.transposedSummary, main = "Bias Across Different Number of Bootstrap Samples", 
        type = "l", col = 1:5, lty = 1:5, lwd = 3, xaxt="n",xlab="Number of Bootstrap Samples",
        ylab="MSE",ylim=c(3.4,5.8))
axis(1, at = 1:7, labels = c("5", "10", "20", "50","100","200","500"))
legend("topright",legend = c("# predictors = 2", "# predictors = 3", "# predictors = 4", "# predictors = 5", "# predictors = 6 (bagging)"), 
       col = 1:5, lty = 1:5, lwd = 3)




