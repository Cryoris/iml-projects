# Load datasets and drop column `Id`
train_full <- read.csv("train.csv", header=TRUE)
train <- train_full[,-1]
# Drop last column as it carries no information
train <- train[,-length(train)]
summary(train)
glmfit <- glm(y ~ ., "gaussian", train)
cv.glm(train, glmfit, K=10)$delta
