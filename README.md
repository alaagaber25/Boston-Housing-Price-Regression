# Boston Housing Price Regression Analysis

This repository contains a Jupyter Notebook that performs a comprehensive analysis of the Boston Housing dataset. The goal is to predict the median value of owner-occupied homes (`MEDV`) using various regression models and advanced preprocessing techniques.

## Overview

The notebook demonstrates how to:
- Preprocess data using scikit-learn Pipelines and ColumnTransformers.
- Handle missing values, skewed distributions, and outliers.
- Encode categorical features (`CHAS` and `RAD`) appropriately.
- Train multiple regression models including:
  - **Linear Regression**
  - **Random Forest Regressor**
  - **XGBoost Regressor**
  - **SVM Regressor**
  - **KNeighbors Regressor**
  - **Deep Learning Neural Network (Keras)**
- Evaluate model performance using metrics such as MSE, RMSE, MAE, R², and Cross-Validation R².
- Visualize results by plotting true vs. predicted values and residual distributions.
- Suppress unwanted runtime warnings from Pandas.

## Dataset Description

The Boston Housing dataset includes features such as:
- **CRIM:** Per capita crime rate by town.
- **ZN:** Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS:** Proportion of non-retail business acres per town.
- **CHAS:** Charles River dummy variable (1 if tract bounds river; 0 otherwise) – treated as a categorical variable.
- **NOX:** Nitric oxides concentration (parts per 10 million).
- **RM:** Average number of rooms per dwelling.
- **AGE:** Proportion of owner-occupied units built prior to 1940.
- **DIS:** Weighted distances to five Boston employment centers.
- **RAD:** Index of accessibility to radial highways – treated as an ordinal categorical variable.
- **TAX:** Full-value property tax rate per $10,000.
- **PTRATIO:** Pupil-teacher ratio by town.
- **B:** 1000(Bk - 0.63)² where Bk is the proportion of people of African American descent by town.
- **LSTAT:** Percentage lower status of the population.
- **MEDV:** Median value of owner-occupied homes in $1000s (target variable).

## Preprocessing Pipeline

The notebook uses a robust preprocessing pipeline that includes:
- **Numerical Pipeline:** Imputation using the median and scaling using `RobustScaler`.
- **Categorical Pipeline:** 
  - One-hot encoding for `CHAS` with `drop="first"` (to avoid dummy variable trap).
  - Ordinal encoding for `RAD`.
- The pipelines are combined using `ColumnTransformer` to ensure that the data is preprocessed consistently for both training and testing.

## Model Implementation

### Neural Network
A deep learning model is implemented using TensorFlow Keras with the following architecture:
- Input layer defined using an `Input` layer (with shape determined by the preprocessed training set).
- Several dense layers with `ReLU` activation, interleaved with Batch Normalization and Dropout for regularization.
- The model is compiled with the Adam optimizer and mean squared error (MSE) loss.
- Early stopping (optional) is used to restore the best weights during training.

### Other Models
The notebook also demonstrates how to train traditional regression models such as:
- Random Forest Regressor (with feature importance evaluation)
- Other regressors like SVM, XGBoost, etc. (code for these models can be added similarly)

## Evaluation Metrics

A custom evaluation function calculates:
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**
- **Cross-Validation R² Score** (for scikit-learn models)

The function also plots:
- A scatter plot of true vs. predicted values.
- A residuals vs. predicted values plot.
- A histogram and KDE of the residuals distribution.
