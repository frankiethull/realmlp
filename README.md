
<!-- README.md is generated from README.Rmd. Please edit that file -->

# realmlp

<!-- badges: start -->

<!-- badges: end -->

This repo is transposed from the standalone, default tuned, simplified,
RealMLP implementation with Python found here:
<https://github.com/dholzmueller/realmlp-td-s_standalone/tree/main>.
This is a pure R torch implementation, without reticulate or Python
dependencies, only R & torch are needed.

## Installation

You can install the development version of realmlp like so:

``` r
pak::pak("frankiethull/realmlp")
```

``` r
library(realmlp)
library(torch)
```

## Regression Example

This is a basic example which shows you how to solve a common problem:

``` r
set.seed(42)
reg <- Standalone_RealMLP_TD_S_Regressor$new(device = "cpu")

# Split data for validation
train_idx <- sample(1:150, 120)
X_train <- iris[train_idx, 1:3]
y_train <- iris$Petal.Width[train_idx]
X_val <- iris[-train_idx, 1:3]
y_val <- iris$Petal.Width[-train_idx]

# Fit with validation set
reg$fit(X_train, y_train, X_val = X_val, y_val = y_val)

# Predict
yhat <- reg$predict(iris[, 1:3])
```

## Classification Example

``` r
set.seed(123)
clf <- Standalone_RealMLP_TD_S_Classifier$new(device = "cpu")

# Split data
train_idx <- sample(1:150, 120)
X_train <- iris[train_idx, 1:4]
y_train <- iris$Species[train_idx]
X_val <- iris[-train_idx, 1:4]
y_val <- iris$Species[-train_idx]

# Fit with validation set
clf$fit(X_train, y_train, X_val = X_val, y_val = y_val)

# Predict classes
y_pred <- clf$predict(iris[, 1:4])

# Predict probabilities
probs <- clf$predict_proba(iris[, 1:4])
```

## To-Do’s

- add RealMLP (current version is RealMLP-TD-S)
- register preprocessors as `recipes`
- register RealMLP & RealMLP-TD-S to `parsnip` as an engines
- add `dials` for hyperparameter optimization
