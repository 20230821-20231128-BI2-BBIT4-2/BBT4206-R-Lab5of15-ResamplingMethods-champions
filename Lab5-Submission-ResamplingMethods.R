# STEP 1. Install and Load the Required Packages ----

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


# STEP 2. Loading Pima Indians Diabetes dataset ----
if (!is.element("mlbench", installed.packages()[, 1])) {
  
  install.packages("mlbench", dependencies = TRUE)
}
require("mlbench")

data("PimaIndiansDiabetes")
View (PimaIndiansDiabetes)
summary(PimaIndiansDiabetes)

# The str() function is used to compactly display the structure (variables
# and data types) of the dataset
str(PimaIndiansDiabetes)


## 1. Split the dataset ====
#75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.75,
                                   list = FALSE)
PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]

## 2. Train a Naive Bayes classifier using the training dataset ----

### 2.a. OPTION 1: naiveBayes() function in the e1071 package ----
PimaIndiansDiabetes_model_nb_e1071 <-
  e1071::naiveBayes(diabetes ~ .,
                    data = PimaIndiansDiabetes_train)

### 2.b. OPTION 2: naiveBayes() function in the caret package ====
PimaIndiansDiabetes_model_nb_caret <- # nolint
  caret::train(diabetes ~ ., data =
                 PimaIndiansDiabetes_train[, c("pregnant", "glucose", "pressure",
                                             "triceps", "insulin", "mass",
                                             "pedigree",
                                             "age", 
                                             "diabetes")],
               method = "naive_bayes")

## 3. Test the trained model using the testing dataset ----
### 3.a. Test the trained e1071 Naive Bayes model using the testing dataset ----
predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb_e1071,
          PimaIndiansDiabetes_test[, c("pregnant", "glucose", "pressure",
                                       "triceps", "insulin", "mass",
                                       "pedigree",
                                       "age", 
                                       "diabetes")])
          
### 3.b. Test the trained caret Naive Bayes model using the testing dataset ----
predictions_nb_caret <-
  predict(PimaIndiansDiabetes_model_nb_caret,
          PimaIndiansDiabetes_test[, c("pregnant", "glucose", "pressure",
                                     "triceps", "insulin", "mass",
                                     "pedigree",
                                     "age", 
                                     "diabetes")])

## 4. View the Results ----
### 4.a. e1071 Naive Bayes model and test results using a confusion matrix ----
print(predictions_nb_e1071)
caret::confusionMatrix(predictions_nb_e1071,
                       PimaIndiansDiabetes_test[, c("pregnant", "glucose", "pressure",
                                                    "triceps", "insulin", "mass",
                                                    "pedigree",
                                                    "age", 
                                                    "diabetes")]$diabetes)
plot(table(predictions_nb_e1071,
           PimaIndiansDiabetes_test[, c("pregnant", "glucose", "pressure",
                                      "triceps", "insulin", "mass",
                                      "pedigree",
                                      "age", 
                                      "diabetes")]$diabetes))

### 4.b. caret Naive Bayes model and test results using a confusion matrix ----
print(PimaIndiansDiabetes_model_nb_caret)
caret::confusionMatrix(predictions_nb_caret,
                       PimaIndiansDiabetes_test[, c("pregnant", "glucose", "pressure",
                                                  "triceps", "insulin", "mass",
                                                  "pedigree",
                                                  "age", 
                                                  "diabetes")]$diabetes)
plot(table(predictions_nb_caret,
           PimaIndiansDiabetes_test[, c("pregnant", "glucose", "pressure",
                                      "triceps", "insulin", "mass",
                                      "pedigree",
                                      "age", 
                                      "diabetes")]$diabetes))


# STEP 3. Bootstrapping ----
## 1. Train a linear regression model (for regression) ----

### 1.a. Bootstrapping train control ----
# The "train control" allows you to specify that bootstrapping (sampling with
# replacement) can be used and also the number of times (repetitions or reps)
# the sampling with replacement should be done.

# This increases the size of the training dataset from 768 observations to
# approximately 768 x 50 = 38,400 observations for training the model.
train_control <- trainControl(method = "boot", number = 50)

PimaIndiansDiabetes_model_lm <- # nolint
  caret::train(`insulin` ~
                 `pregnant` + `glucose` +
                 `pressure` + `triceps` +
                 `mass` + `pedigree` +
                 `age`,
               data = PimaIndiansDiabetes_train,
               trControl = train_control,
               na.action = na.omit, method = "lm", metric = "RMSE")

## 2. Test the trained linear regression model using the testing dataset ----
predictions_lm <- predict(PimaIndiansDiabetes_model_lm,
                          PimaIndiansDiabetes_test[, 1:9])

## 3. View the RMSE and the predicted values for the 12 observations ----
# RSME is 101.03
print(PimaIndiansDiabetes_model_lm)
print(predictions_lm)

## 4. Use the model to make a prediction on unseen new data ----
# New data for each of the 12 variables (independent variables) that determine
# the dependent variable can also be specified as follows in a data frame:
new_data <-
  data.frame(`pregnant` = c(13), 
             `glucose` = c(110),
             `pressure` = c(147),
             `triceps` = c(47), `mass` = c(98.4),
             `pedigree` = c(0.54), `diabetes` = c(1),
             `age` = c(21), check.names = FALSE)

# The variables that are factors (categorical) in the training dataset must
# also be defined as factors in the new data

new_data$`diabetes` <-
  as.factor(new_data$`diabetes`)

# We now use the model to predict the output based on the unseen new data:
predictions_lm_new_data <-
  predict(PimaIndiansDiabetes_model_lm, new_data)

# The output below refers to the total orders:
print(predictions_lm_new_data)


# STEP 4. CV, Repeated CV, and LOOCV ----
## 1. Regression: Linear Model ----
### 1.a. 10-fold cross validation ----
train_control <- trainControl(method = "cv", number = 10)

PimaIndiansDiabetes_model_lm <-
  caret::train(`insulin` ~ .,
               data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "lm", metric = "RMSE")

### 1.b. Test the trained linear model using the testing dataset ----
predictions_lm <- predict(PimaIndiansDiabetes_model_lm, PimaIndiansDiabetes_test[, -5])

### 1.c. View the RMSE and the predicted values ====
# The RMSE has a high percentage of 98.05
print(PimaIndiansDiabetes_model_lm)
print(predictions_lm)

## 2. Classification: LDA with k-fold Cross Validation ----

### 2.a. LDA classifier based on a 5-fold cross validation ----
# Train a Linear Discriminant Analysis (LDA) classifier based on a 5-fold
# cross validation train control but this time, using the diabetes variable for
# classification, not the insulin variable for regression.

train_control <- trainControl(method = "cv", number = 5)

PimaIndiansDiabetes_model_lda <-
  caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit, method = "lda2",
               metric = "Accuracy")

### 2.b. Test the trained LDA model using the testing dataset ----
predictions_lda <- predict(PimaIndiansDiabetes_model_lda,
                           PimaIndiansDiabetes_test[, 1:8])

### 2.c. View the summary of the model and view the confusion matrix ----
print(PimaIndiansDiabetes_model_lda)
caret::confusionMatrix(predictions_lda, PimaIndiansDiabetes_test$diabetes)

## 3. Classification: Naive Bayes with Repeated k-fold Cross Validation ----
### 3.a. Train an e1071::naive Bayes classifier based on the churn variable ----
PimaIndiansDiabetes_model_nb <-
  e1071::naiveBayes(`diabetes` ~ ., data = PimaIndiansDiabetes_train)

### 3.b. Test the trained naive Bayes classifier using the testing dataset ----
predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb, PimaIndiansDiabetes_test[, 1:9])

### 3.c. View a summary of the naive Bayes model and the confusion matrix ----
print(PimaIndiansDiabetes_model_nb)
caret::confusionMatrix(predictions_nb_e1071, PimaIndiansDiabetes_test$diabetes)


## 4. Classification: SVM with Repeated k-fold Cross Validation ----
### 4.a. SVM Classifier using 5-fold cross validation with 3 reps ----
# Train a Support Vector Machine (for classification) using "diabetes" variable
# in the training dataset based on a repeated 5-fold cross validation train
# control with 3 reps.

train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

PimaIndiansDiabetes_model_svm <-
  caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")

### 4.b. Test the trained SVM model using the testing dataset ----
predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[, 1:8])

### 4.c. View a summary of the model and view the confusion matrix ----
print(PimaIndiansDiabetes_model_svm)
caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)

## 5. Classification: Naive Bayes with Leave One Out Cross Validation ----
# Leave One Out Cross-Validation (LOOCV), a data instance is left out and a
# model constructed on all other data instances in the training set. This is
# repeated for all data instances.

### 5.a. Train a Naive Bayes classifier based on an LOOCV ----
train_control <- trainControl(method = "LOOCV")

PimaIndiansDiabetes_model_nb_loocv <-
  caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "naive_bayes", metric = "Accuracy")

### 5.b. Test the trained model using the testing dataset ====
predictions_nb_loocv <-
  predict(PimaIndiansDiabetes_model_nb_loocv, PimaIndiansDiabetes_test[, 1:8])

### 5.c. View the confusion matrix ====
print(PimaIndiansDiabetes_model_nb_loocv)
caret::confusionMatrix(predictions_nb_loocv, PimaIndiansDiabetes_test$diabetes)
