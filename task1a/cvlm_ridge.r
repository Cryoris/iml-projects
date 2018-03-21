# Load datasets and drop column `Id`
train_full <- read.csv("train.csv", header=TRUE)
train <- train_full[,-1]
# Drop last column as it carries no information
train <- train[,-length(train)]
summary(train)

#library("DAAG")
library("boot")
glmfit <- glm(y ~ ., "gaussian", train)
cv.glm(train, glmfit, K=10)$delta

install.packages("glmnet")
library("glmnet")
?glmnet
#x <- model.matrix(y ~ ., train) # Remove y 
#y <- train[,1]
# Prepare data
x <- as.matrix(train[,-1])
y <- as.double(train[,1])
lambdas = 10^seq(-1,3,by=1) # by=1
#fit = glmnet(x, y, alpha=0, lambda=lambdas)
# TODO: is RMSE used? 
# TODO: check that "gaussian" thing
cvfit = cv.glmnet(x, y, alpha=0, nfolds=10, lambda=lambdas)
cvfit$lambda.min
plot(cvfit)
