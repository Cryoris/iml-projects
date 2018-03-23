n_train = 1000
n_test = 100
m = 4
std = 10
d = 3

# Data point creation
x <- function(n, d, mean, std) {
  return(matrix(rnorm(n*d, mean, std), n, d))
}

# Model
y <- function(x, noise) {
  return(3*x[,1] - x[,2] + 0.5*x[,3] + noise)
}

# Create train and test sets
x_train <- x(n_train, d, m, std)
noise <- rnorm(n_train, -7, 10)
y_train <- y(x_train, noise)

x_test <- x(n_test, d, m, std)
y_test <- y(x_test, 0)


# Get fit through glmnet
library("glmnet")
lambdas = c(1000, 100, 10, 1, 0.1)
fit = glmnet(x_train, y_train, alpha=0, lambda = lambdas)

# Predict on test set
pred = predict.glmnet(fit, newx=x_test, type="response")

# Get coefficients
cf = as.matrix(coef.glmnet(fit))

# Plot dependence on one feature
feature_no = 1
nlambdas = length(lambdas)
cols = c("red", "green", "blue", "yellow", "magenta")
plot(x_test[,feature_no], y_test)
for (i in c(1:nlambdas)) {
  points(x_test[,feature_no], pred[,i], col=cols[i])
}
legend(2,17, lambdas, col=cols[1:nlambdas], pch=21)

# Compute MSE
mse = c()
for (i in c(1:nlambdas)) {
  mse[i] = sum((y_test - pred[,i])^2)/length(y_test)
}

# Estimate MSE through cross validation
cvfit = cv.glmnet(x_train, y_train, lambda=lambdas, type.measure="mse", nfolds=10)
plot(cvfit)
lines(lambdas, mse, col="red")
cvfit$cvm
mse
