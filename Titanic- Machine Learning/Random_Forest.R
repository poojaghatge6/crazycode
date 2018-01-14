# load library
library(randomForest)
library(plyr)

# Read the data (These are preprocessed data)
train <- read.csv("C:/Users/gokul/Documents/DA Projct/train.csv")
train

test <- read.csv("C:/Users/gokul/Documents/DA Projct/test.csv")
test


# Forming a Random Forest based on SEX, PClass, AGE, Fare and Embarked. 
RF <- randomForest(Survived ~ Sex + Pclass + Age + Fare +Embarked, data = train, ntree = 10000, importance = TRUE)


# Extract the importance, Summary,and Variable Importance Plot.
importance(RF)
summary(RF)
varImpPlot(RF)


# Prediction for Train dataset 
train$survived_pred <- predict(RF, train)


# Prediction for Test dataset
test$survived <- predict(RF, test)

# CSV File with all data and the result
write.csv(test,"Test_RF.csv")

# Extract the CSV File - Result alone 
write.csv(test$survived, "Submission_RF.csv")


