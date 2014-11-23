dataDir <- "./data"
if(!file.exists(dataDir)) {dir.create(dataDir)}
trainFile.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainFile.name <- "pml-training.csv"
trainFile.dpath <- paste(dataDir,"/",trainFile.name,sep="")
download.file(trainFile.url,destfile=trainFile.dpath,method="curl")
testFile.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testFile.name <- "pml-testing.csv"
testFile.dpath <- paste(dataDir,"/",testFile.name,sep="")
download.file(testFile.url,destfile=testFile.dpath,method="curl")

library(caret)
trainData <- read.csv(trainFile.dpath)
zeroCols <- nearZeroVar(trainData)
clipData <- trainData[, -zeroCols]
summary(clipData)

trimData <- clipData[ , -grep("max_",names(clipData))]
trimData <- trimData[ , -grep("min_",names(trimData))]
trimData <- trimData[ , -grep("var_",names(trimData))]
trimData <- trimData[ , -grep("amplitude_",names(trimData))]
trimData <- trimData[ , -grep("stddev_",names(trimData))]
trimData <- trimData[ , -grep("avg_",names(trimData))]
summary(trimData)

exSet1 <- trimData[sample(1:nrow(trimData), 2000,replace=FALSE),]
inTrainEx1 <- createDataPartition(y=exSet1$classe, p=0.7, list=FALSE) 
trainEx1 <- exSet1[inTrainEx1,-c(1:6)]
testEx1 <- exSet1[-inTrainEx1,-c(1:6)]

exSet1Modelrpart <- train(classe ~., method="rpart",data=trainEx1)
print(exSet1Modelrpart$finalModel)
library(rattle)
fancyRpartPlot(exSet1Modelrpart$finalModel)

library(randomForest)
exSet1Modelrf <- randomForest(classe~.,prox=TRUE,importance=TRUE,data=trainEx1)
ex1virf <- varImp(exSet1Modelrf)
ex1virf$sort <- apply(ex1virf,1,sum)
ex1virf[order(ex1virf$sort,decreasing=TRUE),]

inTrain <- createDataPartition(y=trimData$classe, p=0.7, list=FALSE) 
trainSet <- trimData[inTrain,-c(1:6)]
testSet <- trimData[-inTrain,-c(1:6)]


modelRf <- randomForest(classe~.,prox=TRUE,importance=TRUE,data=trainSet)
virf <- varImp(modelRf)
virf$sort <- apply(virf,1,sum)
virf[order(virf$sort,decreasing=TRUE),]

ctrl <- trainControl(method="repeatedcv",number=5,repeats=5)
exSet1Modelrfcc <- train(classe~.,method="rf",trControl = ctrl,prox=TRUE,importance=TRUE,data=trainEx1)
modelRfc <- train(classe~.,method="rf",trControl = ctrl,prox=TRUE,importance=TRUE,data=trainSet)

testData <- read.csv(testFile.dpath)
submitPredictionsModelRf <- predict(modelRf,testData)
submitPredictionsModelRf


answersDir <- "./answers"
if(!file.exists(answersDir)) {dir.create(answersDir)}

pml_write_files = function(x,y){
  n = length(x)
  for(i in 1:n){
    filename = paste(y,"/problem_id_",i,".txt",sep="")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(submitPredictionsModelRf,answersDir)




