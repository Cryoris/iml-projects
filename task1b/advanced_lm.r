# Task 1b

setwd("~/uni/semester_8/iml/task1b")
data <- read.csv("train.csv", header=T)[,-1] # Drop Id column
x <- as.matrix(data[,-1])
x_trans <- cbind(x, x^2, exp(x), cos(x))
y <- as.matrix(data[,1])


library(glmnet)

lambdas <- 10^seq(1,-2, by=-0.01)
model <- cv.glmnet(x_trans, y, alpha=0.8, lambda=lambdas)

coefs <- coef.cv.glmnet(model, s="lambda.1se")
coefs_ordered <- c(coefs[-1], coefs[1])
model

write.table(as.matrix(coefs_ordered), "coefs.csv", sep="\n", row.names=F, col.names=F)
