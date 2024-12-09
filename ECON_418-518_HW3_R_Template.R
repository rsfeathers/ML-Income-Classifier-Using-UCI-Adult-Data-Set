##################################################
# ECON 418-518 Homework 3
# Roderick Featherstone
# The University of Arizona
# rsfeathers@arizona.edu 
# 08 December 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(ISLR2, glmnet, boot, data.table, ggplot2)

# Set sead
set.seed(418518)

#####################
# Problem 1
#####################

#set working directory
setwd("/Users/roddy/Desktop/CODING/DATA_FILES")
getwd()

# Load the data set into a data table
data <- data.table(read.csv("ECON_418-518_HW3_Data.csv"))
data

#Drop the variables from the data table
data[, c("fnlwgt", "occupation", "relationship", "capital.gain", "capital.loss", "educational.num" ) := NULL]
data

#Change variables if a condition is met to 1, otherwise 0; and add age^2
data[, `:=`(income= ifelse(income == ">50K", 1, 0),
            race= ifelse(race =="White", 1, 0),
            gender= ifelse(gender =="Male", 1, 0),
            workclass= ifelse(workclass =="Private", 1, 0),
            native.country= ifelse(native.country =="United-States", 1, 0),
            marital.status= ifelse(marital.status =="Married-civ-spouse", 1, 0),
            education= ifelse(education == "Masters" | education =="Bachelors" | education =="Doctorate", 1, 0),
            age_sq = age^2)]
data

#Standardize age, age^2, hours per week.
data[, `:=` (age = (age-mean(age))/sd(age),
     age_sq = (age_sq - mean(age_sq))/sd(age_sq),
     hours.per.week = (hours.per.week-mean(hours.per.week))/sd(hours.per.week))]

#Get proportion for income over 50k
prop_income <- mean(data[, income])

#Proportion for private work class
prop_private <- mean(data[, workclass])

#Proportion of married
prop_married <- mean(data[, marital.status])

#Proportion of females
prop_females <- 1-mean(data[, gender])

#Amount of NA's
total_NA <- sum(is.na(data))

#Show all data
prop_income
prop_private
prop_married
prop_females
total_NA

# Convert the "income" variable to a factor
data$income <- as.factor(data$income)

# Check the structure
str(data$income)  

# Calculate training and testing sizes
train_size <- floor(0.7 * nrow(data))
test_size <- floor(0.3 * nrow(dt))

# Shuffle the data indices
shuffled_indices <- sample(nrow(data))

# Split the indices
train_indices <- shuffled_indices[1:train_size]
test_indices <- shuffled_indices[(train_size + 1):nrow(data)]

# Create the training and testing datasets
dt_train <- data[train_indices, ]
dt_test <- data[test_indices, ]

# Show first few rows of training and testing sets
head(dt_train)
head(dt_test)

#################
# Lasso Regression
#################

# Define a sequence of 50 lambda values
lambda_seq <- exp(seq(log(10^5), log(10^-2), length.out = 50))

# Set up training control for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the Lasso regression model
lasso_model <- train(
  income ~ ., data = dt_train,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_seq) # alpha = 1 for Lasso
)

# Output the results
print(lasso_model)

# Evaluate the model on the test set
predictions <- predict(lasso_model, newdata = dt_test)
conf_matrix <- confusionMatrix(predictions, dt_test$income)
conf_matrix

# Extract coefficients for the best lambda from the Lasso model
lasso_coefs <- coef(lasso_model$finalModel, s = lasso_model$bestTune$lambda)
lasso_coefs

##############
# Lasso and Ridge Regression with Non-Zero estimates
##############

# Filter the dataset to include only the specified variables
selected_vars <- c("income", "age", "education", "marital.status", "hours.per.week")
dt_train_filtered <- dt_train[, ..selected_vars]
dt_test_filtered <- dt_test[, ..selected_vars]

# Train the Lasso regression model
lasso_model_filtered <- train(
  income ~ ., data = dt_train_filtered,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda_seq) # alpha = 1 for Lasso
)
lasso_model_filtered

# Train the Ridge regression model
ridge_model_filtered <- train(
  income ~ ., data = dt_train_filtered,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 0, lambda = lambda_seq) # alpha = 0 for Ridge
)
ridge_model_filtered

# Evaluate Lasso on the test set
lasso_predictions <- predict(lasso_model_filtered, newdata = dt_test_filtered)
lasso_conf_matrix <- confusionMatrix(lasso_predictions, dt_test_filtered$income)

# Evaluate Ridge on the test set
ridge_predictions <- predict(ridge_model_filtered, newdata = dt_test_filtered)
ridge_conf_matrix <- confusionMatrix(ridge_predictions, dt_test_filtered$income)

lasso_predictions
lasso_conf_matrix

ridge_predictions
ridge_conf_matrix

#################
# Random Forest Model
#################

# Load the required package
library(randomForest)
library(caret)

# Define the grid for the number of trees and possible features (mtry)
grid <- expand.grid(
  mtry = c(2, 5, 9)  # Number of random features for splitting
)
grid

# Set up training control with 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)

# Random Forest Model 1: 100 Trees
rf_model_100 <- train(
  income ~ ., data = dt_train,
  method = "rf",
  trControl = train_control,
  tuneGrid = grid,
  ntree = 100  # Number of trees
)
rf_model_100

# Random Forest Model 2: 200 Trees
rf_model_200 <- train(
  income ~ ., data = dt_train,
  method = "rf",
  trControl = train_control,
  tuneGrid = grid,
  ntree = 200  # Number of trees
)
rf_model_200

# Random Forest Model 3: 300 Trees
rf_model_300 <- train(
  income ~ ., data = dt_train,
  method = "rf",
  trControl = train_control,
  tuneGrid = grid,
  ntree = 300  # Number of trees
)
rf_model_300

# Evaluate the models
# Model 1: 100 Trees
rf_100_predictions <- predict(rf_model_100, newdata = dt_test)
rf_100_conf_matrix <- confusionMatrix(rf_100_predictions, dt_test$income)

# Model 2: 200 Trees
rf_200_predictions <- predict(rf_model_200, newdata = dt_test)
rf_200_conf_matrix <- confusionMatrix(rf_200_predictions, dt_test$income)

# Model 3: 300 Trees
rf_300_predictions <- predict(rf_model_300, newdata = dt_test)
rf_300_conf_matrix <- confusionMatrix(rf_300_predictions, dt_test$income)

#Show data
rf_100_predictions
rf_100_conf_matrix

rf_200_predictions
rf_200_conf_matrix

rf_300_predictions
rf_300_conf_matrix
















