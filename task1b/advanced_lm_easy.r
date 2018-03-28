# Task 1b

setwd("~/uni/semester_8/iml/task1b")
data <- read.csv("train.csv", header=T)[,-1] # Drop Id column
x <- as.matrix(data[,-1])
y <- as.matrix(data[,1])

model <- lm(y ~ x + I(x^2) + exp(x) + cos(x))
coefs <- coef(model)
length(coefs)
coefs_ordered <- c(coefs[-1], coefs[1])

write.table(as.matrix(coefs_ordered), "coefs.csv", sep="\n", row.names=F, col.names=F)
