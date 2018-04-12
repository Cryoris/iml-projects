# Task 2
# approach following https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html

## Load libraries and data
setwd("C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task2")

library(mlr) # package: Machine Learning R 
library(randomForestSRC)
library(randomForest)
library(party)

data.train <- read.csv("train.csv", header=T)[,-1] # Drop Id column
data.test <- read.csv("test.csv", header=T)[,-1] # Drop Id column


## Create the task

# put data into right format
train.y = data.train[,1]
x = data.train[,-1]

# y has to be cast from 0/1/2 to 3 TRUE/FALSE-statements
data.input <- transform(y0 = y==0, y1 = y==1, y2 = y==2, data.train[,-1])
labels = c("y0", "y1", "y2")

MultiClass.task = makeMultilabelTask(id = "multi", data = data.input, target = labels)


## Constructing a learner: Algorithm adaptation methods

lrn.rfsrc = makeLearner("multilabel.randomForestSRC", bootstrap="by.root")
#lrn.rfsrc = makeLearner("regr.randomForest")
#lrn.rfsrc = makeLearner("multilabel.cforest")

## Train model

model = train(lrn.rfsrc, MultiClass.task) #, subset = 1:100)


## Predict

#pred = predict(model, newdata = data.test) 
pred = predict(model, newdata = x) 
            # , importance=TRUE,random)
#performance(pred)

# cast responce back to y=0/1/2 
#   incredibly ugly way to get TRUE/FALSE into 0/1:
#       as.numeric(as.matrix((as.data.frame(pred)["response.y0"])))

y = 1*as.numeric(as.matrix((as.data.frame(pred)["response.y1"]))) + 2*as.numeric(as.matrix((as.data.frame(pred)["response.y2"])))


our.rownames = as.character(1999+c(1:length(y)))
our.columnnames = c("Id","y")
write.table(cbind(our.rownames, y), "MultiPredrfsrc_rootBootstrap.csv", sep=",", row.names=F, col.names=our.columnnames)

