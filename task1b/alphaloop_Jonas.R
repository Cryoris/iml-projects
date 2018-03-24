setwd("C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task1b")
data <- read.csv("train.csv", header=T)[,-1] # To drop Id column


# data read-in

x <- as.matrix(data[,-1])
y <- as.matrix(data[,1])


# libraries
library(glmnet)

################
#
# trivial trial
#
################

model <- lm(y ~ x  + I(x**2) + exp(x) + cos(x))
coefs <- coef(model)

coefs.right_order <- c(coefs[-1],coefs[1])


################
#
# more advanced
#
################

# set up parameters for the loops

x_transf <- cbind(x,x^2, exp(x), cos(x))
#lambdas <- 10^seq(10,-4)
lambdas <- seq(10^-8,100,length.out=30)
alphas = seq(0,1, length.out=21)
RMSE_best = 10^10
alpha_best = 0
lambda_best = 1
RMSE.storer = c()

for (alpha in alphas){
    model <- cv.glmnet(x_transf, y, alpha=alpha, lambda = lambdas)
    coefs_adv_unordered <- coef(model)
    coefs_adv <- c(coefs_adv_unordered[-1],coefs_adv_unordered[1])
    y_pred <- cbind(x_transf,1) %*% coefs_adv # 1 for the constant offset
    RMSE <- sqrt(mean((y-y_pred)^2))
    RMSE.storer = cbind(RMSE.storer, RMSE)
    if (RMSE < RMSE_best){
      alpha_best <- alpha  
      RMSE.storer = cbind(RMSE.storer, alpha)
      lambda_best <- model$lambda.1se
      }
}

model_final <- cv.glmnet(x_transf, y, alpha=alpha, lambda = lambdas)
coefs_adv_unordered <- coef(model_final)
coefs_adv <- c(coefs_adv_unordered[-1],coefs_adv_unordered[1])
coefs_adv

RMSE.storer
hist(RMSE.storer)
alpha_best  
lambda_best
