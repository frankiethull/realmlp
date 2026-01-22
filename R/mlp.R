# -----------------------------
# Torch modules
# -----------------------------

#' @export
ScalingLayer <- nn_module(
  "ScalingLayer",
  initialize = function(n_features) {
    self$scale <- nn_parameter(torch_ones(n_features))
  },
  forward = function(x) {
    x * self$scale$unsqueeze(1) # [N,F] * [1,F]
  }
)

#' @export
NTPLinear <- nn_module(
  "NTPLinear",
  initialize = function(in_features, out_features, zero_init = FALSE) {
    self$in_features <- as.integer(in_features)
    self$out_features <- as.integer(out_features)
    factor <- if (isTRUE(zero_init)) 0.0 else 1.0
    self$weight <- nn_parameter(factor * torch_randn(c(self$in_features, self$out_features)))
    self$bias <- nn_parameter(factor * torch_randn(c(1, self$out_features)))
  },
  forward = function(x) {
    (1 / sqrt(self$in_features)) * x$matmul(self$weight) + self$bias
  }
)

#' @export
Mish <- nn_module(
  "Mish",
  forward = function(x) {
    x * torch_tanh(nnf_softplus(x))
  }
)

# Cross-entropy with label smoothing if available; otherwise a safe fallback
make_classification_loss <- function(label_smoothing = 0.1) {
  force(label_smoothing)
  function(logits, target) {
    # target expected as long, class indices
    tryCatch(
      nnf_cross_entropy(logits, target, label_smoothing = label_smoothing),
      error = function(e) {
        # fallback: (1-eps)*CE + eps*CE(uniform)  (no target gather required)
        eps <- label_smoothing
        ce <- nnf_cross_entropy(logits, target)
        logp <- nnf_log_softmax(logits, dim = 2)
        uniform <- -logp$mean(dim = 2)$mean()
        (1 - eps) * ce + eps * uniform
      }
    )
  }
}

# -----------------------------
# SimpleMLP (R6)
# -----------------------------
SimpleMLP <- R6Class(
  "SimpleMLP",
  public = list(
    is_classification = NULL,
    device = NULL,

    model_ = NULL,
    classes_ = NULL,   # for classification
    y_mean_ = NULL,    # for regression
    y_std_ = NULL,     # for regression

    initialize = function(is_classification, device = "cpu") {
      self$is_classification <- isTRUE(is_classification)
      self$device <- torch_device(device)
    },

    fit = function(X, y, X_val = NULL, y_val = NULL) {
      stopifnot(is.matrix(X))
      X <- as.matrix(X)
      storage.mode(X) <- "double"

      input_dim <- ncol(X)
      is_cls <- self$is_classification

      # Prepare targets
      if (is_cls) {
        y_fac <- as.factor(y)
        self$classes_ <- levels(y_fac)
        y_idx <- as.integer(y_fac) # 1..K
        output_dim <- length(self$classes_)
      } else {
        y_mat <- if (is.matrix(y) || is.data.frame(y)) as.matrix(y) else matrix(as.numeric(y), ncol = 1)
        storage.mode(y_mat) <- "double"
        self$y_mean_ <- colMeans(y_mat)
        self$y_std_ <- apply(y_mat, 2, stats::sd)
        y_std_safe <- self$y_std_ + 1e-30
        y_mat <- sweep(y_mat, 2, self$y_mean_, "-")
        y_mat <- sweep(y_mat, 2, y_std_safe, "/")
        output_dim <- ncol(y_mat)

        if (!is.null(y_val)) {
          yv <- if (is.matrix(y_val) || is.data.frame(y_val)) as.matrix(y_val) else matrix(as.numeric(y_val), ncol = 1)
          storage.mode(yv) <- "double"
          yv <- sweep(yv, 2, self$y_mean_, "-")
          y_val <- sweep(yv, 2, y_std_safe, "/")
        }
      }

      act <- if (is_cls) nn_selu else Mish

      model <- nn_sequential(
        ScalingLayer(input_dim),
        NTPLinear(input_dim, 256), act(),
        NTPLinear(256, 256), act(),
        NTPLinear(256, 256), act(),
        NTPLinear(256, output_dim, zero_init = TRUE)
      )$to(device = self$device)

      # Loss
      if (is_cls) {
        criterion <- make_classification_loss(label_smoothing = 0.1)
      } else {
        criterion <- function(pred, target) nnf_mse_loss(pred, target, reduction = "mean")
      }

# Optimizer with 3 param groups (scale / weights / biases)
params <- model$parameters

scale_params <- list(params[[1]])
weights      <- params[seq(2, length(params), by = 2)]
biases       <- params[seq(3, length(params), by = 2)]

# Create optimizer with 3 separate parameter groups
opt <- optim_adam(
 params = list(
    list(params = scale_params),
    list(params = weights),
    list(params = biases)
  ),
  betas = c(0.9, 0.95)
)

      # Torch tensors
      x_train <- torch_tensor(X, dtype = torch_float())

      if (is_cls) {
        y_train <- torch_tensor(y_idx, dtype = torch_long())
      } else {
        y_train <- torch_tensor(y_mat, dtype = torch_float())
      }

      if (!is.null(X_val) && !is.null(y_val)) {
        X_val <- as.matrix(X_val); storage.mode(X_val) <- "double"
        x_valid <- torch_tensor(X_val, dtype = torch_float())
        if (is_cls) {
          yv_fac <- factor(y_val, levels = self$classes_)
          yv_idx <- as.integer(yv_fac)
          # unseen labels in validation become NA -> error; force them to 1 (won't be meaningful)
          yv_idx[is.na(yv_idx)] <- 1L
          y_valid <- torch_tensor(yv_idx, dtype = torch_long())
        } else {
          y_valid <- torch_tensor(as.matrix(y_val), dtype = torch_float())
        }
      } else {
        x_valid <- x_train[1:0, ]
        y_valid <- if (is_cls) y_train[1:0] else y_train[1:0, ]
      }

      n_train <- x_train$size()[1]
      n_valid <- x_valid$size()[1]


      n_epochs <- 256L
      train_batch_size <- as.integer(min(256L, n_train))
      n_train_batches <- as.integer(floor(n_train / train_batch_size))
      if (n_train_batches < 1L) n_train_batches <- 1L

      valid_batch_size <- as.integer(max(1L, min(1024L, n_valid)))

      base_lr <- if (is_cls) 0.04 else 0.07
      best_valid_loss <- Inf
      best_valid_params <- NULL

      valid_metric <- function(y_pred, y_true) {
        if (is_cls) {
          # unnormalized classification error count
          pred_idx <- y_pred$argmax(dim = 2)
          (pred_idx != y_true)$sum()
        } else {
          nnf_mse_loss(y_pred, y_true, reduction = "mean")
        }
      }

      # Training loop
      for (epoch in 0:(n_epochs - 1L)) {
        model$train()

        perm <- sample.int(n_train, size = n_train, replace = FALSE)

        for (batch_idx in 0:(n_train_batches - 1L)) {
          start <- batch_idx * train_batch_size + 1L
          end <- start + train_batch_size - 1L
          if (end > n_train) break # because drop_last = TRUE in the python code

          idx <- perm[start:end]

          x_batch <- x_train[idx, ]$to(device = self$device)
          y_batch <- if (is_cls) y_train[idx] else y_train[idx, ]$to(device = self$device)

          if (is_cls) y_batch <- y_batch$to(device = self$device)

          # LR schedule
          t <- (epoch * n_train_batches + batch_idx) / (n_epochs * n_train_batches)
          lr_sched_value <- 0.5 - 0.5 * cos(2 * pi * log2(1 + 15 * t))
          lr <- base_lr * lr_sched_value

          opt$param_groups[[1]]$lr <- 6 * lr     # scale
          opt$param_groups[[2]]$lr <- lr         # weights
          opt$param_groups[[3]]$lr <- 0.1 * lr   # biases

          # Step
          opt$zero_grad()
          y_pred <- model(x_batch)
          loss <- criterion(y_pred, y_batch)
          loss$backward()
          opt$step()
        }

        # Validation / best checkpoint
        model$eval()
        with_no_grad({
          if (n_valid > 0) {
            preds <- list()
            # batch validation to limit memory
            for (start in seq(1L, n_valid, by = valid_batch_size)) {
              end <- min(n_valid, start + valid_batch_size - 1L)
              xb <- x_valid[start:end, ]$to(device = self$device)
              preds[[length(preds) + 1L]] <- model(xb)$detach()$cpu()
            }
            y_pred_valid <- torch_cat(preds, dim = 1)
            v <- valid_metric(y_pred_valid, y_valid$to(device = torch_device("cpu")))
            valid_loss <- as.numeric(v$item())
          } else {
            valid_loss <- 0.0
          }

          if (valid_loss <= best_valid_loss) {
            best_valid_loss <- valid_loss
            best_valid_params <- lapply(model$parameters, function(p) p$detach()$clone())
          }
        })
      }

      # Restore best parameters
      with_no_grad({
        for (i in seq_along(model$parameters)) {
          model$parameters[[i]]$set_(best_valid_params[[i]])
        }
      })

      self$model_ <- model
      invisible(self)
    },

    predict = function(X) {
      stopifnot(is.matrix(X))
      X <- as.matrix(X); storage.mode(X) <- "double"
      x <- torch_tensor(X, dtype = torch_float())$to(device = self$device)

      self$model_$eval()
      out <- with_no_grad({
        self$model_(x)$detach()$cpu()
      })

      if (self$is_classification) {
        idx <- as.integer(as.array(out$argmax(dim = 2)))
        self$classes_[idx]
      } else {
        y_hat <- as.matrix(out)
        # y_hat is N x D; invert standardization
        y_std_safe <- self$y_std_ + 1e-30
        y_hat <- sweep(y_hat, 2, y_std_safe, "*")
        y_hat <- sweep(y_hat, 2, self$y_mean_, "+")
        if (ncol(y_hat) == 1) drop(y_hat[, 1]) else y_hat
      }
    },

    predict_proba = function(X) {
      stopifnot(self$is_classification)
      stopifnot(is.matrix(X))
      X <- as.matrix(X); storage.mode(X) <- "double"
      x <- torch_tensor(X, dtype = torch_float())$to(device = self$device)

      self$model_$eval()
      probs <- with_no_grad({
  as.matrix(nnf_softmax(self$model_(x), dim = 2)$detach()$cpu())
      })
      probs
    }
  )
)

# -----------------------------
# Standalone estimators (R6)
# -----------------------------
#' @export
Standalone_RealMLP_TD_S_Classifier <- R6Class(
  "Standalone_RealMLP_TD_S_Classifier",
  public = list(
    device = NULL,
    prep_ = NULL,
    model_ = NULL,
    classes_ = NULL,

    initialize = function(device = "cpu") {
      self$device <- device
    },

    fit = function(X, y, X_val = NULL, y_val = NULL) {
      self$prep_ <- get_realmlp_td_s_pipeline()
      self$model_ <- SimpleMLP$new(is_classification = TRUE, device = self$device)

      Xp <- prep_fit_transform(self$prep_, X)
      Xvp <- if (!is.null(X_val)) prep_transform(self$prep_, X_val) else NULL

      self$model_$fit(Xp, y, X_val = Xvp, y_val = y_val)
      self$classes_ <- self$model_$classes_
      invisible(self)
    },

    predict = function(X) {
      Xp <- prep_transform(self$prep_, X)
      self$model_$predict(Xp)
    },

    predict_proba = function(X) {
      Xp <- prep_transform(self$prep_, X)
      self$model_$predict_proba(Xp)
    }
  )
)

#' @export
Standalone_RealMLP_TD_S_Regressor <- R6Class(
  "Standalone_RealMLP_TD_S_Regressor",
  public = list(
    device = NULL,
    prep_ = NULL,
    model_ = NULL,

    initialize = function(device = "cpu") {
      self$device <- device
    },

    fit = function(X, y, X_val = NULL, y_val = NULL) {
      self$prep_ <- get_realmlp_td_s_pipeline()
      self$model_ <- SimpleMLP$new(is_classification = FALSE, device = self$device)

      Xp <- prep_fit_transform(self$prep_, X)
      Xvp <- if (!is.null(X_val)) prep_transform(self$prep_, X_val) else NULL

      self$model_$fit(Xp, y, X_val = Xvp, y_val = y_val)
      invisible(self)
    },

    predict = function(X) {
      Xp <- prep_transform(self$prep_, X)
      self$model_$predict(Xp)
    }
  )
)