# Load datasets and drop column `Id`
train_full <- read.csv("train.csv", header=TRUE)
train <- train_full[,-1]
# Drop last column as it carries no information
train <- train[,-length(train)]
summary(train)


cross <- function(set, idx){
  train <- set[-idx,] # Drop given rows
  
  # Create test set
  test <- set[idx,] 
  test <- test[,-1] # Drop y column
  y_ref <- set[idx,1]

  model <- lm(y ~ ., train)
  pred <- predict(model, test)

  mse = mean((pred - y_ref)^2)  
  return(mse)
}


cross(train, c(1,4,5))



