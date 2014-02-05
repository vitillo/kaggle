"VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
(1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
(C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations."

library(caret)

options(na.action='na.pass')

trainFrame <- read.csv("train.csv")
testFrame <- read.csv("test.csv")

clean <- function(data) {
  #data <- data[, !(names(data) %in% c("PassengerId", "Name", "Ticket", "Cabin", "Age"))]
  numRelatives <- data$SibSp + data$Parch
  data <- data[, names(data) %in% c("Survived", "Sex", "Pclass")]
  
  if("Survived" %in% colnames(data)) {
    data$Survived <- as.factor(data$Survived)
  }
  
  data$numRelatives <- numRelatives
  data$Pclass <- as.factor(data$Pclass)
    
  return(data)
}

trainFrame <- clean(trainFrame)

PassengerId <- testFrame$PassengerId
testFrame <- clean(testFrame)
testFrame$PassengerId <- PassengerId

ctrl <- trainControl(method="repeatedcv", repeats=5, summaryFunction= twoClassSummary, classProbs=TRUE)
lrFit <- train(Survived ~., data=trainFrame, method="glm", metric="ROC", trControl=ctrl)
predicted <- predict(lrFit, testFrame)

result <- data.frame(Survived=predicted, PassengerId=testFrame$PassengerId)
write.csv(result, file="prediction.csv", row.names=FALSE, col.names=FALSE, quote=FALSE)