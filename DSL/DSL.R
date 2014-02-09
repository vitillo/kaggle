library(caret)
library(doMC)
registerDoMC(8)

# Load data
test <- read.csv("test.csv", header=FALSE)
train <- read.csv("train.csv", header=FALSE)
Labels <- as.factor(read.csv("trainLabels.csv", header=FALSE)$V1)
levels(Labels) <- c("zero", "one")

# PCA (don't scale, noise is over-weighted and the signal (peak) is down-weighted)
pca <- prcomp(rbind(train, test))
train <- as.data.frame(pca$x[1:1000, 1:37]) # 95% Variance
test <- as.data.frame(pca$x[1001:10000, 1:37])
train$Labels <- Labels

# Backward elimination
rfeFuncs <- rfFuncs
rfeFuncs$summary <- twoClassSummary
rfe.control <- rfeControl(rfeFuncs, method = "repeatedcv", number=10 ,repeats=5, verbose=FALSE, returnResamp="final")
rfe.rf <- rfe(train[,-length(train)], train[, length(train)], sizes=10:15, rfeControl=rfe.control, metric="ROC")
train <- train[, c("Labels", predictors(rfe.rf))]
test <- test[,predictors(rfe.rf)]

# Build Model
ctrl <- trainControl(method="repeatedcv", number=10, summaryFunction= twoClassSummary, classProbs=TRUE)
svmFit <- train(Labels ~., data=train, method="svmRadial", metric="ROC", trControl=ctrl)

# Semi-supervised learning
predicted <- predict(svmFit, test, type="prob")[,"one"]
newtraini <- which(test >= 0.975 | test <= 0.025)
newtrain <- test[newtraini,]
newtrain$Labels <- as.factor(ifelse(predicted[newtraini] < 0.5, "zero","one"))
train <- rbind(train,newtrain[,names(train)])

svmSemiFit <- train(Labels ~., data=train, method="svmRadial", metric="ROC", trControl=ctrl)
semiPredicted <- predict(svmSemiFit, test, type="prob")[, "one"]

# Save predictions
predicted <- (predicted + semiPredicted)/2
result <- data.frame(id=1:9000, Solution=ifelse(predicted < 0.5, 0, 1))
write.csv(result, file="prediction.csv", row.names=FALSE, quote=FALSE)