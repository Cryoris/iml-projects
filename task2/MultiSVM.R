
library("e1071")


data.train <- read.csv("train.csv", header=T)[,-1] # Drop Id column
data.test <- read.csv("test.csv", header=T)[,-1] # Drop Id column


## Create the task

# put data into right format
y = data.train[,1]
x = data.train[,-1]

# y has to be cast from 0/1/2 to 3 TRUE/FALSE-statements
data.input <- transform(y0 = y==0, y1 = y==1, y2 = y==2, data.train[,-1])
labels = c("y0", "y1", "y2")

model <- svm(x, y, type="eps-regression", epsilon=.5)

res <- predict(model, newdata=data.test)
res <- as.numeric(res) - 1

our.rownames = as.character(1999+c(1:length(res)))
our.columnnames = c("Id","y")
write.table(cbind(our.rownames, res), "MultiSVMe05.csv", sep=",", row.names=F, col.names=our.columnnames)

