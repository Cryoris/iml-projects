# Task 2
# approach following https://mlr-org.github.io/mlr-tutorial/devel/html/multilabel/index.html

## Load libraries and data
setwd("C:/Users/jonas/OneDrive/ETH/Machine Learning/iml-projects/task2")

library(mlr) # package: Machine Learning R 
library(randomForestSRC)
library(randomForest)
library(party)

library(car)
#needed for "vif", but should not override "makeLearner" from mlr, so lower priority


data.train.full <- read.csv("train.csv", header=T)[,-1] # Drop Id column
data.test.full <- read.csv("test.csv", header=T)[,-1] # Drop Id column

data.train = data.train.full
data.test = data.test.full


##utilize Part 2 (remove parameters aproach)

rem.params = c(-(13+1),-(4+1),-(5+1),-(6+1),-(8+1),-(9+1),-(10+1),-(12+1),-(15+1),-(16+1))
rem.params = c(-(13+1),-(9+1),-(10+1),-(12+1),-(15+1),-(16+1))
rem.params = c(-(1+1),-(12+1),-(13+1))#,-(15+1))

# "+1" for y column

data.train <- data.train.full[, rem.params]
data.test <- data.test.full[, rem.params+1] # no y-column -> "+1" not needed 

# from here on, the new approach follows the old one..

## Create the task

# put data into right format
y = data.train[,1]
x = data.train[,-1]
#x = data.train[,c(-1,-13)] # feature13=feature12

# y has to be cast from 0/1/2 to 3 TRUE/FALSE-statements
data.input <- transform(y0 = y==0, y1 = y==1, y2 = y==2, x)
labels = c("y0", "y1", "y2")

MultiClass.task = makeMultilabelTask(id = "multi", data = data.input, target = labels)


## Constructing a learner: Algorithm adaptation methods

lrn.rfsrc = makeLearner("multilabel.randomForestSRC")#, bootstrap="by.root")
#lrn.rfsrc = makeLearner("regr.randomForest")
#lrn.rfsrc = makeLearner("multilabel.cforest")

## Train model
model = mlr::train(lrn.rfsrc, MultiClass.task) #, subset = 1:100)
  # mlr::train instead of train because car-package overrode train from mlr...

## Predict

pred = predict(model, newdata = data.test) 
#pred = predict(model, newdata = x) 
            # , importance=TRUE,random)
#performance(pred)

# cast responce back to y=0/1/2 
#   incredibly ugly way to get TRUE/FALSE into 0/1:
#       as.numeric(as.matrix((as.data.frame(pred)["response.y0"])))

pred.y = 1*as.numeric(as.matrix((as.data.frame(pred)["response.y1"]))) + 2*as.numeric(as.matrix((as.data.frame(pred)["response.y2"])))

sum(abs(myY-pred.y))

our.rownames = as.character(1999+c(1:length(pred.y)))
our.columnnames = c("Id","y")
write.table(cbind(our.rownames, pred.y), "MultiPredrfsrc_VIFremoveNew2.csv", sep=",", row.names=F, col.names=our.columnnames)

###
#Part 2 : remove correlations
###


######## first approach: cor
#full.data <- read.csv("train.csv", header=T)[,-1] # Drop Id column

#cor(full.data) 
# realise 100% correlation between x13&x12; 99% correlation between x12&x6;
# 99% between x8&x1
#data <- full.data[,c(-2,-(13+1))]#,-(6+1),-(8+1))] # "+1" for y column


####################
#new approach: vif

full.data <- read.csv("train.csv", header=T)[,-1] # Drop Id column
cor(full.data) 
# realise 100% correlation between x13&x12 -> aliased variable
data <- full.data[,c(-(13+1))]  # "+1" for y column

# 1,12,13,15
vif(glm(x5~x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x14+x16,data=data))
vif(glm(x12~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x14+x15+x16,data=data))
#remove GVIF >10: x4,5,6,8,9,10,12,15,16
rem.columns = c(-(13+1),-(4+1),-(5+1),-(6+1),-(8+1),-(9+1),-(10+1),-(12+1),-(15+1),-(16+1))
# "+1" for y column

data <- full.data[,rem.columns]
vif(glm(x14~x1+x2+x3+x7+x11+x14,data=data))

