# Task 1a

# Set wd
setwd("~/uni/semester_8/iml/task1a") # for Jules

# Set seed for reproducibility
set.seed(7)

# Load glmnet (for ridge regression)
library(glmnet)

# Load data
data_full <- read.csv("train.csv", header=TRUE)
data <- data_full[,-1]          # Drop column `Id`
n <- length(data[,1])           # Number of observations
data <- data[,-length(data)]    # Drop last column as it carries no information?
summary(data)

# Outer lambda loop
lambdas <- 10^seq(-1, 3, by=1)
rmses <- c()
for (lambda in lambdas) {
  nfolds <- 10                    # Number of folds
  folds <- cut(1:n, breaks=nfolds, labels=FALSE)
  s <- sample(1:n, size=n)        # Random sample of 1:n
  local_rmse <- c()               # Compute avg RMSE
  for (i in 1:nfolds) {
    idx <- s[which(folds == i)]   # Indices for fold
    test <- data[idx,]            # Testset (small)
    train <- data[-idx,]          # Trainset (large)
  
    x <- as.matrix(train[,-1])       # Split in x and y set
    y <- as.matrix(train[,1])
    x_test <- as.matrix(test[,-1])
    y_test <- as.matrix(test[,1])
    
    ridge.fit <- glmnet(x, y, alpha=0, lambda=lambda) 
    ridge.pred <- predict.glmnet(ridge.fit, newx=x_test)
    local_rmse[i] <- sqrt( mean((ridge.pred - y_test)^2) )
  }
  rmses <- cbind(rmses, mean(local_rmse))
}

print(rmses)
write.table(rmses, "rmses.csv", col.names=F, row.names=F, sep="\n")

