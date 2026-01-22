
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

## Example

``` r
library(realmlp)
library(torch)
library(rsample)

corn_data <- maize::corn_data

corn_splits <- initial_validation_split(corn_data)
train <- training(corn_splits)
validate <- validation(corn_splits)
test <- testing(corn_splits)
```

### Regression

This is a basic example which shows you how to solve a common problem:

``` r
set.seed(42)
reg <- Standalone_RealMLP_TD_S_Regressor$new(device = "cpu")

reg$fit(
  X = train |> dplyr::select(-height), 
  y = train |> dplyr::pull(height), 
  X_val = validate |> dplyr::select(-height), 
  y_val = validate |> dplyr::pull(height)
)

# predictions
yhat <- reg$predict(test |> dplyr::select(-height))
```

### Classification

``` r
set.seed(123)
clf <- Standalone_RealMLP_TD_S_Classifier$new(device = "cpu")

clf$fit(
  X = train |> dplyr::select(-type), 
  y = train |> dplyr::pull(type), 
  X_val = validate |> dplyr::select(-type), 
  y_val = validate |> dplyr::pull(type)
)

# predict classes
y_pred <- clf$predict(test |> dplyr::select(-type))

# predict probabilities
probs <- clf$predict_proba(test |> dplyr::select(-type))
```

## To-Do’s

- [x] create package & implement RealMLP-TD-S based on
  `dholzmueller/realmlp-td-s_standalone`
- [x] export `Mish`
- [ ] add TD dictionaries from `pytabkit`
- [ ] implement RealMLP-TD from `pytabkit`
- [ ] register preprocessors as `recipes`
- [ ] register RealMLP-TD-S & RealMLP-TD to `parsnip` as engines
- [ ] add bagging option for RealMLP engine via cv
- [ ] add `dials` for hyperparameter optimization (RealMLP-HPO)
- [ ] create Caruana ensemble extension for `stacks`
- [ ] add `tests` for Python vs R results
- [ ] note subtle differences (such as sample vs population maths in
  numpy.std vs base::sd())
