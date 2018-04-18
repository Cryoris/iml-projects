#setwd("~/uni/semester_8/iml/task2")

# IDEAL
# for svm poly -> deg=2, cost=exp(-0.5)
# for svm radial -> gamma=0.1 cost=exp(0.5)

library(mlr)
library(e1071)
library(caret)

# SETTING
# What to do?
# - "final" computation
# - cross-validation for "params"
# - cross-validation for "features" (which features to remove)
action <- "features"#"final"

# PART 0 / Read in
data.train <- read.csv("train.csv", header=T)[,-1] # Drop Id column
data.test <- read.csv("test.csv", header=T)[,-1]   # Drop Id column

summary(data.train)
sum(data.train[,1] == 0)                            # How often does 0 appear?
sum(data.train[,1] == 1)
sum(data.train[,1] == 2)

# PART I / Final computation with given parameters
if (action == "final") {
  rem.feat <- c(2,3,7,11,13,15)  #c(2,3,7,11,12,13,15)                   # Bad features
  data.train <- data.train[,-(1+rem.feat)]           # Remove from train set
  data.test <- data.test[,-rem.feat]                 # Remove from test set
  summary(data.train)

  train.x <- scale(data.train[,-1])                  # Normalise train data
  train.y <- data.train[,1]
  data.train <- cbind(train.y, train.x)
  data.test <- scale(data.test)                      # Normalise test data
  summary(data.train)

  # Adjust parameters!
  fit <- svm(train.y~., data.train, kernel='radial', cost=exp(0.5), gamma=0.1, type="C-classification")
  pred <- predict(fit, data.test)
  y <- as.numeric(pred) - 1                          # Convert to useful format

  our.rownames = as.character(1999+c(1:length(y)))   # Set correct output format
  our.columnnames = c("Id","y")
  #write.table(cbind(our.rownames, y), "SVMPred.csv", sep=",", row.names=F, col.names=our.columnnames)
  write.table(cbind(our.rownames, y), "JonasSVMPred2.csv", sep=",", row.names=F, col.names=our.columnnames)
}

# FUNCTION / Perform one cross-validation for (a single set of) given parameters
cvsvm.single <- function(data, p.1, p.2, n.lo=100, normalise=T) {
  idx <- sample(1:nrow(data), size=n.lo)             # Random sample of 1:n

  train.data <- data[-idx,]
  train.y <- data[-idx,1]

  test.data <- data[idx,]
  test.y <- data[idx,1]

  if (normalise) {
    train.x <- scale(train.data[,-1])
    train.data <- cbind(train.y, train.x)
    test.x <- scale(test.data[,-1])
    test.data <- cbind(test.y, test.x)
  }

  summary(train.data)
  summary(test.data)

  fit <- svm(train.y~., train.data, kernel='radial', epsilon=p.1, gamma=p.2, type="C-classification")
  pred <- predict(fit, test.data[,-1])

  confusion <- table(pred, test.y)                    # How many correctly classified?
  return(sum(diag(confusion))/sum(confusion))         # Return accuracy
}

# FUNCTION / Perform n.folds CVs for a given (single set of) parameters
cvsvm <- function(data, n.folds, p.1, p.2, n.lo=100, normalise=T) {
  accs <- replicate(n.folds, cvsvm.single(data, p.1, p.2, n.lo, normalise))
  return(mean(accs))
}

# PART 2 / Cross-validation for parameter-finding
if (action == "params") {
  p.cost <- exp(seq(-1,4,by=0.5))
  p.deg <- c(1:5)
  p.gamma <- 10^seq(-2, 2, by=0.5)
  p.gamma.dummy <- 0.1
  p.eps <- 10^seq(-2,0,by=0.5)
  
  p.1 <- p.eps
  p.2 <- p.gamma.dummy
  n.folds <- 10

  accs <- matrix(NA ,nrow=length(p.1), ncol=length(p.2))

  for (i in 1:length(p.1)) {
    for (j in 1:length(p.2)) {
      accs[i,j] <- cvsvm(data.train, n.folds, p.1[i], p.2[j])
    }
  }
  plot(p.1, accs[1,])
  for (i in 2:nrow(accs)) {
    points(p.1, accs[i,], pch=i)
  }
  print(p.1)
  print(p.2)
  print(accs)
}

# PART 3 / Cross-validation to compute goodness of features
if (action == "features") {
  n <- 100
  feat.nr <- c(1:16)
  cv <- function(x, y, n, feat.nr) {

    idx <- sample(1:length(y), size=n)        # Random sample of 1:n
    #print(idx)

    train.x <- x[-idx,feat.nr]
    train.y <- y[-idx]

    test.x <- x[idx,feat.nr]
    test.y <- y[idx]

    #?svm
    model <- svm(x=train.x, y=train.y, type="C-classification")
    pred.y <- predict(model, test.x)
    #table(pred.y, test.y)
    pred.y <- as.numeric(pred.y)-1
    mse <- sum((test.y - pred.y)^2)
    return(mse)
  }


  feat.sel <- function(x, y) {
    n.cv <- 100
    n.folds <- 10

    n.feats <- ncol(x)
    max.mse <- 0
    all.mse <- c()
    worst.feat <- -1
    for (cur.feat in c(1:n.feats)) {
      mses <- replicate(n.folds, cv(x, y, n.cv, cur.feat))
      all.mse <- c(all.mse, mean(mses))
      if (max.mse < mean(mses)) {
        max.mse <- mean(mses)
        worst.feat <- cur.feat
      }
    }
    plot(all.mse)
    abline(v=worst.feat, col="red")
    return(worst.feat)
  }


  # feat.sel(scale(x),y) # removed since x,y were never defined (Jonas)
        # change in the following: x --> train.x , y --> train.y
  train.x <- data.train[,-1]
  train.y <- data.train[,1]
  feat.sel(scale(train.x),train.y)

  err = c()
  for (i in c(1:16)) {
    control <- trainControl(method="repeatedcv", number=10, repeats=20)
    model <- train(data.frame(train.x[,i], "x"), train.y, method="cforest", preProcess="scale", trControl=control)
    summary(model)
    pred <- predict(model, train.y)
    err <- c(err, sum(abs(train.y - pred)))
  }
  plot(err)
  abline(v=10)
  abline(v=8)
  abline(v=6)
}

