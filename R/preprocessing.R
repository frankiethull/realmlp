`%||%` <- function(a, b) if (!is.null(a)) a else b

as_data_frame <- function(X) {
  if (is.data.frame(X)) return(X)
  if (is.matrix(X)) return(as.data.frame(X, stringsAsFactors = FALSE))
  as.data.frame(X, stringsAsFactors = FALSE)
}

is_categorical_col <- function(x) {
  is.character(x) || is.factor(x)
}

as_numeric_matrix <- function(df) {
  if (ncol(df) == 0) {
    return(matrix(0, nrow = nrow(df), ncol = 0))
  }
  mats <- lapply(df, function(v) {
    # logical -> 0/1, Date/POSIXct -> numeric, integer -> numeric, etc.
    as.numeric(v)
  })
  out <- do.call(cbind, mats)
  out <- as.matrix(out)
  storage.mode(out) <- "double"
  colnames(out) <- names(df)
  out
}

# -----------------------------
# CustomOneHotEncoder (R6)
# -----------------------------
CustomOneHotEncoder <- R6Class(
  "CustomOneHotEncoder",
  public = list(
    levels_ = NULL,     # list of character vectors (levels per categorical col), excluding NA
    cat_sizes_ = NULL,  # integer vector

    fit = function(X) {
      X <- as_data_frame(X)
      self$levels_ <- vector("list", ncol(X))
      self$cat_sizes_ <- integer(ncol(X))

      for (j in seq_len(ncol(X))) {
        col <- X[[j]]
        # treat factors/characters as categorical; keep as character tokens
        vals <- as.character(col)
        vals[is.na(col)] <- NA_character_
        levs <- unique(vals[!is.na(vals)])
        # preserve stable ordering: first appearance in training data
        self$levels_[[j]] <- levs
        self$cat_sizes_[j] <- length(levs)
      }
      invisible(self)
    },

    transform = function(X) {
      X <- as_data_frame(X)
      n <- nrow(X)
      if (ncol(X) == 0) return(matrix(0, nrow = n, ncol = 0))

      out_list <- vector("list", ncol(X))

      for (j in seq_len(ncol(X))) {
        levs <- self$levels_[[j]]
        k <- self$cat_sizes_[j]

        if (k == 0L) {
          out_list[[j]] <- matrix(0, nrow = n, ncol = 0)
          next
        }

        col <- X[[j]]
        vals <- as.character(col)
        vals[is.na(col)] <- NA_character_

        # match() returns 1..k for known, NA for unknown/missing
        idx <- match(vals, levs)

        out <- matrix(0, nrow = n, ncol = k)
        rows <- which(!is.na(idx))
        if (length(rows) > 0) {
          out[cbind(rows, idx[rows])] <- 1.0
        }

        if (k == 2L) {
          # binary -> single feature: +1 for level1, -1 for level2, 0 for missing/unknown
          out <- out[, 1, drop = FALSE] - out[, 2, drop = FALSE]
        }

        out_list[[j]] <- out
      }

      if (length(out_list) == 0) return(matrix(0, nrow = n, ncol = 0))
      do.call(cbind, out_list)
    },

    fit_transform = function(X) {
      self$fit(X)
      self$transform(X)
    }
  )
)

# -----------------------------
# CustomOneHotPipeline (R6)
# -----------------------------
CustomOneHotPipeline <- R6Class(
  "CustomOneHotPipeline",
  public = list(
    cat_cols_ = NULL,
    num_cols_ = NULL,
    onehot_ = NULL,

    fit = function(X) {
      X <- as_data_frame(X)
      is_cat <- vapply(X, is_categorical_col, logical(1))
      self$cat_cols_ <- names(X)[is_cat]
      self$num_cols_ <- names(X)[!is_cat]

      self$onehot_ <- CustomOneHotEncoder$new()
      if (length(self$cat_cols_) > 0) {
        self$onehot_$fit(X[, self$cat_cols_, drop = FALSE])
      } else {
        self$onehot_$fit(data.frame()) # empty
      }
      invisible(self)
    },

    transform = function(X) {
      X <- as_data_frame(X)

      X_cat <- if (length(self$cat_cols_) > 0) X[, self$cat_cols_, drop = FALSE] else data.frame()
      X_num <- if (length(self$num_cols_) > 0) X[, self$num_cols_, drop = FALSE] else X[0]

      cat_mat <- self$onehot_$transform(X_cat)
      num_mat <- as_numeric_matrix(X_num)

      # ColumnTransformer in the python code outputs categorical first, then remaining.
      if (ncol(cat_mat) == 0 && ncol(num_mat) == 0) {
        matrix(0, nrow = nrow(X), ncol = 0)
      } else if (ncol(cat_mat) == 0) {
        num_mat
      } else if (ncol(num_mat) == 0) {
        cat_mat
      } else {
        cbind(cat_mat, num_mat)
      }
    },

    fit_transform = function(X) {
      self$fit(X)
      self$transform(X)
    }
  )
)

# -----------------------------
# RobustScaleSmoothClipTransform (R6)
# -----------------------------
RobustScaleSmoothClipTransform <- R6Class(
  "RobustScaleSmoothClipTransform",
  public = list(
    median_ = NULL,
    factors_ = NULL,

    fit = function(X) {
      stopifnot(is.matrix(X))
      X <- as.matrix(X)
      storage.mode(X) <- "double"

      self$median_ <- apply(X, 2, stats::median)

      q75 <- apply(X, 2, stats::quantile, probs = 0.75, names = FALSE, type = 7)
      q25 <- apply(X, 2, stats::quantile, probs = 0.25, names = FALSE, type = 7)
      quant_diff <- q75 - q25

      maxv <- apply(X, 2, max)
      minv <- apply(X, 2, min)

      idx0 <- quant_diff == 0
      quant_diff[idx0] <- 0.5 * (maxv[idx0] - minv[idx0])

      factors <- 1.0 / (quant_diff + 1e-30)
      factors[quant_diff == 0] <- 0.0

      self$factors_ <- factors
      invisible(self)
    },

    transform = function(X) {
      stopifnot(is.matrix(X))
      X <- as.matrix(X)
      storage.mode(X) <- "double"

      x_scaled <- sweep(X, 2, self$median_, FUN = "-")
      x_scaled <- sweep(x_scaled, 2, self$factors_, FUN = "*")

      x_scaled / sqrt(1 + (x_scaled / 3)^2)
    },

    fit_transform = function(X) {
      self$fit(X)
      self$transform(X)
    }
  )
)

get_realmlp_td_s_pipeline <- function() {
  list(
    one_hot = CustomOneHotPipeline$new(),
    rssc = RobustScaleSmoothClipTransform$new()
  )
}

# Small helper: fit/transform pipeline
prep_fit_transform <- function(prep, X) {
  X1 <- prep$one_hot$fit_transform(X)
  prep$rssc$fit_transform(X1)
}
prep_transform <- function(prep, X) {
  X1 <- prep$one_hot$transform(X)
  prep$rssc$transform(X1)
}
