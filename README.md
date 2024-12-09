# ML Income Classifier Using UCI Adult Data Set

Implementing machine learning techniques like Lasso Regression, Ridge Regression, and Random Forest models to analyze and predict income levels based on a dataset.

## Description

This project is a submission for ECON 518 at The University of Arizona. The purpose is to preprocess a dataset, evaluate machine learning models, and interpret their classification accuracy for income prediction. The analysis involves data cleaning, feature engineering, model training, and cross-validation to identify the best-performing model for the dataset.

## Getting Started

### Dependencies

* R version 4.0 or higher
* Operating System: Windows 10, macOS, or Linux
* Required R packages:
```
ISLR2
```
```
glmnet
```
```
boot
```
```
data.table
```
```
ggplot2
```
```
randomForest
```
```
caret
```
```
pacman
```

### Installing

* Download and install R from CRAN.
* Install RStudio for an enhanced coding environment (optional).
* Install the required libraries using ```"pacman"```:
```
install.packages("pacman")
pacman::p_load(ISLR2, glmnet, boot, data.table, ggplot2, randomForest, caret)
```

### Executing program

* Clone the repository or download the project files to your local machine.
* Place the dataset (```ECON_418-518_HW3_Data.csv```) in the appropriate working directory.
* Open the R script in RStudio.
* Set the working directory:
  ```
  setwd("path_to_your_directory")
  ```
*Execute the script step-by-step or run the entire script

## Authors

Roderick Featherstone
roderickfeatherstone@gmail.com

## Version History

* 0.2
    * Improved feature engineering and added Random Forest models.
    * Optimized hyperparameter tuning and cross-validation.
* 0.1
    * Initial dataset preprocessing and Lasso/Ridge regression implementation.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
William Brasic for guidance on machine learning techniques.
The University of Arizona for providing resources and datasets.
