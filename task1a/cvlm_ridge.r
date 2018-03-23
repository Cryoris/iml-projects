# Load datasets and drop column `Id`
# Don't forget to set working directory (Session -> Set Working Directory)
train_full <- read.csv("train.csv", header=TRUE)
train <- train_full[,-1]
# Drop last column as it carries no information
train <- train[,-length(train)]
summary(train)

# Load library and have a look at functions
library("glmnet")
?glmnet
?cv.glmnet

# Prepare data
x <- as.matrix(train[,-1])
y <- as.double(train[,1])

# Do CV 10-fold for given lambda values
lambdas = c(1000, 100, 10, 1, 0.1, 0.01)
#lambdas = c(0.1, 0.1, 0.1, 0.2)
cvfit = cv.glmnet(x, y, alpha=0, nfolds=10, lambda=lambdas, type.measure="mse", standardize=FALSE)

# Check results
cvfit$lambda
cvfit$cvm
cvfit$lambda.min
cvfit$lambda.1se
plot(cvfit)

# Manually computing MSE (not sure what cvm exactly is!)
lambdas_final = lambdas[-length(lambdas)]
model = glmnet(x, y, alpha=0, lambda=lambdas_final)

mse = c()
for (l in lambdas_final){
  pred = predict(model, s=l, newx=x)
  current_mse = mean((pred - y)^2)
  mse = c(mse, current_mse)
}

mse
# Write results
# For some reason cv.glmnet only accepts l = 0.1 if we also add 0.01.
# So here, bc we dont need it, we remove the rmse associated with 0.01.
# Also we need to invert the order!
n = length(cvfit$cvm)

rmse <- rev( sqrt(cvfit$cvm[-n]) )

write.table(rmse, file="rmse.csv", sep=",", col.names=FALSE, row.names=FALSE)
?write.csv

