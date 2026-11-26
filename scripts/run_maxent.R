#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

suppressPackageStartupMessages({
  library(maxnet)
  library(dplyr)
  library(readr)
})

# ============================================================
# tools
# ============================================================
log_msg <- function(..., .level = "INFO") {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s][%s] ", ts, .level), ..., "\n", sep = "")
  flush.console()
}

fmt_elapsed <- function(t0) {
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (secs < 60) return(sprintf("%.1f sec", secs))
  if (secs < 3600) return(sprintf("%.2f min", secs / 60))
  sprintf("%.2f hr", secs / 3600)
}

progress_loop_msg <- function(i, n, prefix = "Progress", every = 10) {
  if (n <= 0) return(invisible(NULL))
  if (i == 1 || i == n || i %% every == 0) {
    pct <- 100 * i / n
    log_msg(sprintf("%s：%d/%d (%.1f%%)", prefix, i, n, pct))
  }
}

DATA_YEAR_START <- 1999L
DATA_YEAR_END <- 2024L
TIME_WINDOW <- 11L
PROJECT_ROOT <- ".."  
MODEL_SEED <- 1234L
THRESHOLD_CRITERION <- "F1"
# <<< [MAXENT-NODE-MASK-FIX-01]

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit == length(args)) return(default)
  args[hit + 1]
}
parse_years <- function(x) {
  x <- gsub("\\s+", "", x)
  if (grepl(":", x, fixed=TRUE)) {
    yy <- as.integer(strsplit(x, ":", fixed=TRUE)[[1]])
    if (length(yy) != 2L) stop("Invalid year range format: ", x)
    return(yy[1]:yy[2])
  }
  as.integer(strsplit(x, ",", fixed=TRUE)[[1]])
}
as_bool <- function(x) tolower(as.character(x)) %in% c("true", "t", "1", "yes", "y")

root <- normalizePath(get_arg("--root", PROJECT_ROOT), mustWork = FALSE)
out_dir <- file.path(root, get_arg("--out-dir", "results/sdm/MaxEnt"))

data_years <- parse_years(get_arg(
  "--years", paste0(DATA_YEAR_START, ":", DATA_YEAR_END)
))
time_window <- as.integer(get_arg("--time-window", as.character(TIME_WINDOW)))
if (time_window < 1L || time_window > length(data_years)) {
  stop("time_window must be between 1 and the total number of available years.")
}
if (length(data_years) > 1L && any(diff(data_years) != 1L)) {
  stop("--years must specify a continuous range of years.")
}
target_years <- data_years[seq.int(time_window, length(data_years))]
mask_file_default <- file.path(
  root, "results/sdm/shared",
  paste0("tgnn_", min(target_years), "_", max(target_years), ".csv")
)
mask_file <- normalizePath(
  get_arg("--mask-file", mask_file_default), mustWork = FALSE
)

# ---------------------------
# MaxEnt parameters
# ---------------------------
max_bg_per_year <- as.integer(get_arg("--max-bg-per-year", "3000"))

bg_ratio <- as.numeric(get_arg("--bg-ratio", "5"))

min_bg_per_year <- as.integer(get_arg("--min-bg-per-year", "500"))

feature_class <- get_arg("--feature-class", "lq")

max_features <- as.integer(get_arg("--max-features", "25"))

cor_threshold <- as.numeric(get_arg("--cor-threshold", "0.90"))

include_depth <- as_bool(get_arg("--include-depth", "FALSE"))

threshold_criterion <- toupper(get_arg(
  "--threshold-criterion", THRESHOLD_CRITERION
))  # F1 或 TSS
if (!threshold_criterion %in% c("F1", "TSS")) {
  stop("--threshold-criterion must be either F1 or TSS.")
}
seed <- as.integer(get_arg("--seed", as.character(MODEL_SEED)))
set.seed(seed)

pred_dir <- file.path(out_dir, "predictions")
dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)

log_msg("[Configuration] root=", root)
log_msg("[Design] Evaluation using annual node masks; this is not strict temporal extrapolation.")
log_msg("[Configuration] data_years=", paste(data_years, collapse=","))
log_msg("[Configuration] T=", time_window,
        ", target_years=", paste(target_years, collapse=","))
log_msg("[Configuration] canonical_mask=", mask_file)
log_msg("[MaxEnt parameters] max_bg_per_year=", max_bg_per_year,
        ", bg_ratio=", bg_ratio,
        ", min_bg_per_year=", min_bg_per_year,
        ", feature_class=", feature_class,
        ", max_features=", max_features,
        ", cor_threshold=", cor_threshold,
        ", include_depth=", include_depth,
        ", threshold_criterion=", threshold_criterion)

# ============================================================
# 1. read data
# ============================================================
label_file_for_year <- function(year) {
  candidates <- c(

    file.path(root, "data/raw/AnnualLabelsforTGNN", paste0("labels_", year, ".csv")),
    file.path(root, "data/raw/AnnualLabelsforTGNN", paste0("label_", year, ".csv"))

  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) stop("Label file not found: ", paste(candidates, collapse="; "))
  hit[1]
}

env_file_for_year <- function(year) {
  f <- file.path(root, "data/raw/AnnualFeaturesforTGNN", paste0("EnvGNN_", year, ".csv"))
  if (!file.exists(f)) stop("Environmental feature file not found: ", f)
  f
}

normalize_density_columns <- function(df) {
  level_values <- c(0.0, 0.05, 0.25, 0.55, 0.85)
  dens <- suppressWarnings(as.numeric(df$label_density))
  finite <- dens[is.finite(dens)]
  if (length(finite) == 0) {
    df$label_density <- dens
    df$label_density_class <- NA_integer_
    return(df)
  }
  unique_vals <- sort(unique(finite))
  is_integer_class <- all(abs(unique_vals - round(unique_vals)) < 1e-8) &&
    max(unique_vals) > 1 && min(unique_vals) >= 0 && max(unique_vals) <= 4

  if (is_integer_class) {
    cls <- pmin(pmax(round(dens), 0), 4)
    df$label_density_class <- as.integer(cls)
    df$label_density <- level_values[cls + 1]
  } else {
    df$label_density <- dens
    cls <- rep(NA_integer_, length(dens))
    ok <- is.finite(dens)
    if (any(ok)) {
      cls[ok] <- apply(abs(outer(dens[ok], level_values, "-")), 1, which.min) - 1L
    }
    df$label_density_class <- cls
  }
  df
}

standardize_label_columns <- function(df) {
  names(df) <- trimws(names(df))
  df <- df[, !grepl("^Unnamed", names(df)), drop=FALSE]
  if (!"label_presence" %in% names(df)) {
    for (alt in c("label", "presence", "PreorAb")) {
      if (alt %in% names(df)) { df$label_presence <- df[[alt]]; break }
    }
  }
  if (!"label_presence" %in% names(df)) stop("Label file is missing label_presence / label / presence / PreorAb.")
  if (!"label_density" %in% names(df)) {
    for (alt in c("density", "Density", "DENSITY", "cover", "Cover")) {
      if (alt %in% names(df)) { df$label_density <- df[[alt]]; break }
    }
  }
  if (!"label_density" %in% names(df)) df$label_density <- NA_real_

  if (!"label_mask" %in% names(df)) {
    stop("Label file is missing label_mask; execution stopped to prevent training on all nodes.")
  }

  df <- normalize_density_columns(df)
  df$label_presence <- as.integer(df$label_presence)
  df$label_mask <- as.integer(df$label_mask)
  df
}

clean_input_columns <- function(df) {
  names(df) <- trimws(names(df))
  df[, !grepl("^Unnamed", names(df)), drop=FALSE]
}

coalesce_first <- function(df, new_name, candidates) {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) == 0L) return(df)
  v <- df[[hit[1L]]]
  if (length(hit) > 1L) {
    for (h in hit[-1L]) v <- dplyr::coalesce(v, df[[h]])
  }
  df[[new_name]] <- v
  df
}

load_year <- function(year) {
  env <- clean_input_columns(readr::read_csv(
    env_file_for_year(year), show_col_types=FALSE
  ))
  lab <- clean_input_columns(readr::read_csv(
    label_file_for_year(year), show_col_types=FALSE
  ))
  lab <- standardize_label_columns(lab)
  if (!"NodeID" %in% names(env) || !"NodeID" %in% names(lab)) {
    stop(year, " Env/label data are missing NodeID.")
  }
  leak_exact <- c(
    "label_mask", "train_mask", "val_mask", "test_mask", "split",
    "label_presence", "label_density", "label_density_class",
    "label", "presence", "PreorAb", "density", "Density", "DENSITY"
  )
  env <- env[, setdiff(names(env), leak_exact), drop=FALSE]
  keep_lab <- intersect(
    names(lab),
    c("NodeID", "x", "y", "lon", "lat", "Longitude", "Latitude",
      "label_mask", "label_presence", "label_density", "label_density_class")
  )
  lab <- lab[, keep_lab, drop=FALSE]
  df <- dplyr::inner_join(env, lab, by="NodeID", suffix=c("_env", "_lab"))
  df <- coalesce_first(df, "x", c("x", "x_env", "x_lab"))
  df <- coalesce_first(df, "y", c("y", "y_env", "y_lab"))
  df <- coalesce_first(df, "lon", c("lon", "lon_env", "lon_lab", "Longitude", "Longitude_env", "Longitude_lab"))
  df <- coalesce_first(df, "lat", c("lat", "lat_env", "lat_lab", "Latitude", "Latitude_env", "Latitude_lab"))
  df$year <- year
  df
}

infer_features <- function(df) {
  drop_cols <- c("NodeID", "node_id", "lon", "lat", "x", "y", "Longitude", "Latitude",
                 "year", "Year", "label_mask", "label_mask_source",
                 "train_mask", "val_mask", "test_mask", "spatial_split",
                 "label_presence", "label_density","label_density_class",
                 "label", "presence", "density", "PreorAb", "dist_to_coast",
                 "dist_to_SAV", "dist_to_sav", "Dist_to_SAV")
  if (!include_depth) drop_cols <- c(drop_cols, "Depth","depth")
  cand <- setdiff(names(df), drop_cols)
  leak_pattern <- grepl(
    "^(x|y|lon|lat|Longitude|Latitude|year|Year|label|presence|density|PreorAb|mask|Depth|depth|dist_to_coast|dist_to_SAV|dist_to_sav|Dist_to_SAV)(_|\\.)",
    cand, ignore.case=TRUE
  )
  cand <- cand[!leak_pattern]
  cand <- cand[sapply(df[, cand, drop=FALSE], is.numeric)]
  if (length(cand) == 0) stop("No numeric environmental feature columns were identified.")
  cand
}

all_years <- target_years
log_msg("[Data] Loading years: ", paste(all_years, collapse=","))
t_read <- Sys.time()
read_list <- vector("list", length(all_years))
for (ii in seq_along(all_years)) {
  yr <- all_years[ii]
  progress_loop_msg(ii, length(all_years), prefix = "Loading annual EnvGNN/labels", every = 1)
  read_list[[ii]] <- load_year(yr)
}
all_df <- bind_rows(read_list)
log_msg("[Data] Loading completed in ", fmt_elapsed(t_read), ", total rows=", nrow(all_df))

if (!file.exists(mask_file)) {
  stop(
    "Shared TGNN node mask not found: ", mask_file,
    "\nPlease run the updated run_sdm_python_baselines.py first."
  )
}
mask_df <- readr::read_csv(mask_file, show_col_types=FALSE)
required_mask_cols <- c(
  "year", "NodeID", "label_mask", "train_mask", "val_mask", "test_mask"
)
missing_mask_cols <- setdiff(required_mask_cols, names(mask_df))
if (length(missing_mask_cols) > 0L) {
  stop("Shared mask is missing required columns: ", paste(missing_mask_cols, collapse=", "))
}
mask_df <- mask_df %>%
  filter(year %in% target_years) %>%
  select(any_of(c(required_mask_cols, "spatial_split")))
if (anyDuplicated(mask_df[c("year", "NodeID")])) {
  stop("Shared mask contains duplicate year + NodeID entries.")
}
if (nrow(mask_df) != nrow(all_df)) {
  stop("The shared mask and MaxEnt data have different row counts; execution aborted.")
}
all_df <- all_df %>% rename(label_mask_source=label_mask)
all_df <- all_df %>% left_join(mask_df, by=c("year", "NodeID"))
if (any(is.na(all_df$label_mask))) stop("Some MaxEnt nodes did not match the shared mask.")
for (cc in c("label_mask", "train_mask", "val_mask", "test_mask")) {
  if (any(!all_df[[cc]] %in% c(0L, 1L))) {
    stop("Shared mask field contains values other than 0/1: ", cc)
  }
}
if (any(is.na(all_df$label_mask_source))) {
  stop("The label_mask column in the label CSV contains NA values.")
}
mask_sum <- all_df$train_mask + all_df$val_mask + all_df$test_mask
if (any(mask_sum > 1L) ||
    any(all_df$label_mask == 1L & mask_sum != 1L) ||
    any(all_df$label_mask == 0L & mask_sum != 0L)) {
  stop("The shared mask contains overlapping splits or an invalid label_mask range.")
}
if (any(as.integer(all_df$label_mask_source) != as.integer(all_df$label_mask))) {
  stop("The label_mask in the label CSV does not match the shared TGNN mask.")
}

all_features <- infer_features(all_df)
log_msg("[Features] Initial number of numeric features=", length(all_features))

train_all <- all_df %>% filter(label_mask == 1L, train_mask == 1L)
val_all   <- all_df %>% filter(label_mask == 1L, val_mask == 1L)
test_all  <- all_df %>% filter(label_mask == 1L, test_mask == 1L)
if (nrow(train_all) == 0L || nrow(val_all) == 0L || nrow(test_all) == 0L) {
  stop("At least one of the train/validation/test node masks is empty.")
}
log_msg("[Training scope] Only nodes with label_mask == 1 and train_mask == 1 are used; all nodes will not be used.")

# ============================================================
# 2. data fill in and filter 
# ============================================================
impute_by_train_median <- function(train_df, all_df, features) {
  meds <- sapply(features, function(v) median(train_df[[v]], na.rm = TRUE))
  meds[!is.finite(meds)] <- 0
  for (v in features) {
    bad <- !is.finite(all_df[[v]]) | is.na(all_df[[v]])
    if (any(bad)) all_df[[v]][bad] <- meds[[v]]
  }
  list(data = all_df, medians = meds)
}

imp <- impute_by_train_median(train_all, all_df, all_features)
all_df <- imp$data
write_csv(data.frame(feature = names(imp$medians), train_median = as.numeric(imp$medians)),
          file.path(out_dir, "maxent_train_medians.csv"))

train_all <- all_df %>% filter(label_mask == 1L, train_mask == 1L)


cor_filter <- function(df, features, threshold = 0.90, max_rows = 50000) {
  if (threshold >= 1 || length(features) <= 2) return(features)
  dd <- df
  if (nrow(dd) > max_rows) dd <- dd %>% slice_sample(n = max_rows)
  X <- as.data.frame(dd[, features, drop=FALSE])
  cm <- suppressWarnings(abs(cor(X, use = "pairwise.complete.obs")))
  cm[!is.finite(cm)] <- 0
  diag(cm) <- 0
  keep <- features
  repeat {
    sub <- cm[keep, keep, drop=FALSE]
    if (length(keep) <= 2 || max(sub) < threshold) break
    pair <- which(sub == max(sub), arr.ind = TRUE)[1,]
    v1 <- rownames(sub)[pair[1]]; v2 <- colnames(sub)[pair[2]]

    mean1 <- mean(sub[v1, setdiff(keep, v1)], na.rm=TRUE)
    mean2 <- mean(sub[v2, setdiff(keep, v2)], na.rm=TRUE)
    drop <- ifelse(mean1 >= mean2, v1, v2)
    keep <- setdiff(keep, drop)
  }
  keep
}

log_msg("[Features] Starting correlation filtering; threshold=", cor_threshold)
t_cor <- Sys.time()
feature_cols <- cor_filter(train_all, all_features, threshold = cor_threshold)
log_msg("[Features] Number of features after correlation filtering=", length(feature_cols), ", elapsed time ", fmt_elapsed(t_cor))


if (max_features > 0 && length(feature_cols) > max_features) {
  score <- sapply(feature_cols, function(v) {
    suppressWarnings(abs(cor(train_all[[v]], train_all$label_presence, method="spearman", use="complete.obs")))
  })
  score[!is.finite(score)] <- 0
  feature_cols <- names(sort(score, decreasing = TRUE))[1:max_features]
  log_msg("[Features] Number of features after univariate Spearman screening=", length(feature_cols))
}
write_csv(data.frame(feature = feature_cols), file.path(out_dir, "feature_columns_MaxEnt_fast.csv"))
log_msg("[Features] Final features used by MaxEnt:")
print(feature_cols)
flush.console()

# ============================================================
# 3. construct MaxEnt presence + background train samples
# ============================================================
max_train_list <- list()
log_msg("[Samples] Starting annual sampling of presence + background points")
t_sample <- Sys.time()
for (ii in seq_along(target_years)) {
  yr <- target_years[ii]
  yy <- all_df %>% filter(
    year == yr, label_mask == 1L, train_mask == 1L
  )
  pres <- yy %>% filter(label_presence == 1)
  bg <- yy %>% filter(label_presence == 0)
  n_pres <- nrow(pres)
  n_bg <- nrow(bg)
  if (n_bg == 0) next
  desired_bg <- if (n_pres > 0) max(min_bg_per_year, ceiling(bg_ratio * n_pres)) else min_bg_per_year
  desired_bg <- min(max_bg_per_year, desired_bg, n_bg)
  bg_s <- bg %>% slice_sample(n = desired_bg)
  max_train_list[[as.character(yr)]] <- bind_rows(pres, bg_s)
  log_msg("[Samples] year=", yr,
          ", presence=", n_pres,
          ", background_used=", desired_bg,
          ", background_all=", n_bg,
          ", year progress=", ii, "/", length(target_years))
}
max_train <- bind_rows(max_train_list)
if (sum(max_train$label_presence == 1) == 0) stop("No presence samples in the training set; MaxEnt cannot be trained.")
log_msg("[Samples] Total MaxEnt training samples=", nrow(max_train),
        ", presence=", sum(max_train$label_presence == 1),
        ", background=", sum(max_train$label_presence == 0),
        ", elapsed time ", fmt_elapsed(t_sample))
write_csv(max_train %>% count(year, label_presence), file.path(out_dir, "maxent_training_sample_counts.csv"))

# ============================================================
# 4. train MaxEnt / maxnet
# ============================================================
log_msg("\n===== Training MaxEnt (maxnet) - Fast Version =====")
log_msg("[Tip] If this step is still slow, try: --feature-class l --max-bg-per-year 1000 --max-features 15")
log_msg("[Training size] rows=", nrow(max_train),
        ", presence=", sum(max_train$label_presence == 1),
        ", background=", sum(max_train$label_presence == 0),
        ", raw_features=", length(feature_cols),
        ", feature_class=", feature_class)
log_msg("[Training note] maxnet internal optimization does not provide percentage progress; the next log message will appear when model fitting is complete.")

t_form <- Sys.time()
mx_form <- maxnet.formula(
  p = max_train$label_presence,
  data = as.data.frame(max_train[, feature_cols, drop=FALSE]),
  classes = feature_class
)
log_msg("[Training] maxnet.formula construction completed in ", fmt_elapsed(t_form))

start_time <- Sys.time()
log_msg("[Training] Calling maxnet::maxnet() ...")
mx <- maxnet::maxnet(
  p = max_train$label_presence,
  data = as.data.frame(max_train[, feature_cols, drop=FALSE]),
  f = mx_form
)
log_msg("[Completed] MaxEnt training time: ", fmt_elapsed(start_time))
saveRDS(mx, file.path(out_dir, "maxent_fast_model.rds"))
log_msg("[Saved] Model saved to: ", file.path(out_dir, "maxent_fast_model.rds"))

# ============================================================
# 5. metric function
# ============================================================
ap_score <- function(y, p) {
  ok <- is.finite(y) & is.finite(p)
  y <- as.integer(y[ok]); p <- as.numeric(p[ok])
  n_pos <- sum(y == 1L)
  if (length(y) == 0L || n_pos == 0L) return(NA_real_)
  ord <- order(p, decreasing = TRUE)
  y <- y[ord]; p <- p[ord]
  end_idx <- c(which(p[-length(p)] != p[-1L]), length(p))
  tp <- cumsum(y == 1L)[end_idx]
  fp <- cumsum(y == 0L)[end_idx]
  recall <- tp / n_pos
  precision <- tp / pmax(tp + fp, 1)
  sum(diff(c(0, recall)) * precision)
}


pr_auc_trapezoid <- function(y, p) {
  ok <- is.finite(y) & is.finite(p)
  y <- as.integer(y[ok]); p <- as.numeric(p[ok])
  n_pos <- sum(y == 1L)
  if (length(y) == 0L || n_pos == 0L) return(NA_real_)
  ord <- order(p, decreasing=TRUE)
  y <- y[ord]; p <- p[ord]
  end_idx <- c(which(p[-length(p)] != p[-1L]), length(p))
  tp <- cumsum(y == 1L)[end_idx]
  fp <- cumsum(y == 0L)[end_idx]
  recall <- c(0, tp / n_pos)
  precision <- c(1, tp / pmax(tp + fp, 1))
  sum(diff(recall) * (precision[-length(precision)] + precision[-1L]) / 2)
}

roc_auc_rank <- function(y, p) {
  y <- as.integer(y)
  n_pos <- sum(y == 1); n_neg <- sum(y == 0)
  if (n_pos == 0 || n_neg == 0) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}
calc_cls_metrics <- function(y, p, threshold) {
  pred <- ifelse(p >= threshold, 1L, 0L)
  tp <- sum(pred == 1 & y == 1); tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0); fn <- sum(pred == 0 & y == 1)
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  specificity <- ifelse(tn + fp == 0, 0, tn / (tn + fp))
  tss <- recall + specificity - 1
  data.frame(threshold = threshold, F1 = f1, Recall = recall, Precision = precision,
             AP = ap_score(y, p), PR_AUC = pr_auc_trapezoid(y, p),
             ROC_AUC = roc_auc_rank(y, p), TSS = tss,
             TP = tp, TN = tn, FP = fp, FN = fn)
}
best_threshold <- function(y, p, criterion = "F1") {
  criterion <- toupper(criterion)
  ths <- if (criterion == "F1") sort(unique(as.numeric(p[is.finite(p)]))) else seq(0.10, 0.90, by=0.02)
  vals <- bind_rows(lapply(ths, function(th) calc_cls_metrics(y, p, th)))
  idx <- which.max(vals[[criterion]])
  list(threshold = vals$threshold[idx], score = vals[[criterion]][idx], table = vals)
}
add_confusion_type <- function(df) {
  df$confusion_type <- "OUTSIDE_EVAL_MASK"
  m <- df$label_mask == 1L & df$test_mask == 1L
  df$confusion_type[m & df$label_presence == 1 & df$pred_presence == 1] <- "TP"
  df$confusion_type[m & df$label_presence == 0 & df$pred_presence == 0] <- "TN"
  df$confusion_type[m & df$label_presence == 0 & df$pred_presence == 1] <- "FP"
  df$confusion_type[m & df$label_presence == 1 & df$pred_presence == 0] <- "FN"
  df
}

predict_maxnet_chunked <- function(model, df, features, chunk_size = 5000, label = "prediction") {
  out <- numeric(nrow(df))
  starts <- seq(1, nrow(df), by = chunk_size)
  n_chunks <- length(starts)
  t0 <- Sys.time()
  log_msg("[Prediction] ", label, ": ", nrow(df), " rows, ", n_chunks, " chunks, chunk_size=", chunk_size)
  for (k in seq_along(starts)) {
    i <- starts[k]
    idx <- i:min(i + chunk_size - 1, nrow(df))
    nd <- as.data.frame(df[idx, features, drop=FALSE])
    out[idx] <- as.numeric(predict(model, nd, type = "cloglog"))
    progress_loop_msg(k, n_chunks, prefix = paste0("Predicting ", label), every = max(1, floor(n_chunks / 10)))
  }
  log_msg("[Prediction] ", label, " completed in ", fmt_elapsed(t0))
  out
}

# ============================================================
# 6. pooled validation threshold + pooled test assessment
# ============================================================

val_df <- all_df %>% filter(label_mask == 1L, val_mask == 1L)
val_prob <- predict_maxnet_chunked(
  mx, val_df, feature_cols, label="pooled-validation-threshold"
)
bt <- best_threshold(val_df$label_presence, val_prob, threshold_criterion)
threshold <- bt$threshold
write_csv(bt$table, file.path(out_dir, "MaxEnt_pooled_val_threshold_scan.csv"))
log_msg("[Threshold] Best pooled validation ", threshold_criterion,
        " threshold=", threshold, ", score=", round(bt$score, 4))

pred_out <- all_df
pred_out$prob_presence <- predict_maxnet_chunked(
  mx, pred_out, feature_cols, label="all-target-years"
)
pred_out$threshold_used <- threshold
pred_out$pred_presence <- ifelse(
  pred_out$prob_presence >= threshold, 1L, 0L
)
pred_out$pred_density <- NA_real_  

pred_out$model <- "MaxEnt"
pred_out <- add_confusion_type(pred_out)

pooled_cls_row <- function(split_name) {
  mask_col <- paste0(split_name, "_mask")
  eval_df <- pred_out %>% filter(
    label_mask == 1L, .data[[mask_col]] == 1L
  )
  cls <- calc_cls_metrics(
    eval_df$label_presence, eval_df$prob_presence, threshold
  )
  data.frame(
    model="MaxEnt", split=split_name, aggregation="pooled",
    N=nrow(eval_df), N_positive=sum(eval_df$label_presence == 1L),
    prevalence=mean(eval_df$label_presence == 1L),
    threshold=cls$threshold, AP=cls$AP, PR_AUC=cls$PR_AUC,
    F1=cls$F1, Recall=cls$Recall, Precision=cls$Precision,
    RMSE=NA_real_, RMSE_presence=NA_real_,
    ROC_AUC=cls$ROC_AUC, TSS=cls$TSS,
    MAE=NA_real_, R2=NA_real_, Spearman_r=NA_real_,
    TP=cls$TP, TN=cls$TN, FP=cls$FP, FN=cls$FN,
    stringsAsFactors=FALSE
  )
}
metrics <- bind_rows(pooled_cls_row("val"), pooled_cls_row("test"))
write_csv(metrics, file.path(out_dir, "metrics_MaxEnt_node_mask_pooled.csv"))

all_pred_file <- file.path(
  pred_dir,
  paste0("MaxEnt_predictions_all_", min(target_years), "_", max(target_years), ".csv")
)
write_csv(pred_out, all_pred_file)
for (yr in target_years) {
  write_csv(
    pred_out %>% filter(year == yr),
    file.path(pred_dir, paste0("MaxEnt_", yr, ".csv"))
  )
}

for (ii in seq_len(nrow(metrics))) {
  rr <- metrics[ii,]
  log_msg(sprintf(
    "[MaxEnt][POOLED %s] AP=%.4f F1=%.4f R=%.4f P=%.4f RMSE=NA tau=%.4f",
    toupper(rr$split), rr$AP, rr$F1, rr$Recall, rr$Precision,
    rr$threshold
  ))
}

log_msg("\n✅ MaxEnt fast version completed. Output directory: ", out_dir)
log_msg("Node-level predictions: ", all_pred_file)
log_msg("Metrics file: ", file.path(out_dir, "metrics_MaxEnt_node_mask_pooled.csv"))
