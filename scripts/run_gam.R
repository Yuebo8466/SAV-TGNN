#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

suppressPackageStartupMessages({
  library(mgcv)
  library(dplyr)
  library(readr)
})


DATA_YEAR_START <- 1999L
DATA_YEAR_END <- 2024L
TIME_WINDOW <- 11L
PROJECT_ROOT <- ".."
MODEL_SEED <- 1234L 
THRESHOLD_CRITERION <- "F1"

# ---------------------------

# ---------------------------
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit == length(args)) return(default)
  args[hit + 1]
}

parse_years <- function(x) {

  x <- gsub("\\s+", "", x)
  if (grepl(":", x, fixed = TRUE)) {
    tmp <- as.integer(strsplit(x, ":", fixed = TRUE)[[1]])
    if (length(tmp) != 2) stop("Invalid year range format: ", x)
    return(tmp[1]:tmp[2])
  }
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}

root <- normalizePath(get_arg("--root", PROJECT_ROOT), mustWork = FALSE)
out_dir <- file.path(root, get_arg("--out-dir", "results/sdm/GAM"))


data_years <- parse_years(get_arg(
  "--years", paste0(DATA_YEAR_START, ":", DATA_YEAR_END)
))
time_window <- as.integer(get_arg("--time-window", as.character(TIME_WINDOW)))
if (time_window < 1L || time_window > length(data_years)) {
  stop("time_window must be between 1 and the number of available years")
}
if (length(data_years) > 1L && any(diff(data_years) != 1L)) {
  stop("--years must contain consecutive years")
}
target_years <- data_years[seq.int(time_window, length(data_years))]
mask_file_default <- file.path(
  root, "results/sdm/shared",
  paste0("tgnn_", min(target_years), "_", max(target_years), ".csv")
)
mask_file <- normalizePath(
  get_arg("--mask-file", mask_file_default), mustWork = FALSE
)

threshold_criterion <- get_arg(
  "--threshold-criterion", THRESHOLD_CRITERION
)
threshold_criterion <- toupper(threshold_criterion)
if (!threshold_criterion %in% c("F1", "TSS")) {
  stop("--threshold-criterion must be either F1 or TSS")
}


smooth_k_max <- as.integer(get_arg("--smooth-k-max", "5"))


clamp_density <- tolower(get_arg("--clamp-density", "TRUE")) %in% c("true", "t", "1", "yes", "y")

set.seed(as.integer(get_arg("--seed", as.character(MODEL_SEED))))

pred_dir <- file.path(out_dir, "predictions")
dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)

cat("[Configuration] root=", root, "\n")
cat("[Configuration] out_dir=", out_dir, "\n")
cat("[Configuration] data_years=", paste(data_years, collapse=","), "\n")
cat("[Configuration] T=", time_window, ", target_years=",
    paste(target_years, collapse=","), "\n", sep="")
cat("[Configuration] canonical_mask=", mask_file, "\n")
cat("[Configuration] threshold_criterion=", threshold_criterion, "\n")
cat("[Configuration] smooth_k_max=", smooth_k_max, "\n")
cat("[Configuration] clamp_density=", clamp_density, "\n")

# ---------------------------

# ---------------------------
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
  if (!"label_presence" %in% names(df)) stop("The label file is missing label_presence / label / presence / PreorAb")
  if (!"label_density" %in% names(df)) {
    for (alt in c("density", "Density", "DENSITY", "cover", "Cover")) {
      if (alt %in% names(df)) { df$label_density <- df[[alt]]; break }
    }
  }
  if (!"label_density" %in% names(df)) df$label_density <- NA_real_

  if (!"label_mask" %in% names(df)) {
    stop("The label file is missing label_mask; stopped to prevent training on all nodes.")
  }
  df <- normalize_density_columns(df)
  df$label_presence <- as.integer(df$label_presence)
  df$label_mask <- as.integer(df$label_mask)
  df
}

clean_input_columns <- function(df) {
  names(df) <- trimws(names(df))
  df <- df[, !grepl("^Unnamed", names(df)), drop=FALSE]
  df
}

coalesce_first <- function(df, new_name, candidates) {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) == 0) return(df)
  v <- df[[hit[1]]]
  if (length(hit) > 1) {
    for (h in hit[-1]) {
      v <- dplyr::coalesce(v, df[[h]])
    }
  }
  df[[new_name]] <- v
  df
}

load_year <- function(year) {
  env <- readr::read_csv(env_file_for_year(year), show_col_types = FALSE)
  lab <- readr::read_csv(label_file_for_year(year), show_col_types = FALSE)

  env <- clean_input_columns(env)
  lab <- clean_input_columns(lab)
  lab <- standardize_label_columns(lab)

  if (!"NodeID" %in% names(env) || !"NodeID" %in% names(lab)) {
    stop(year, " Env/Label is missing NodeID")
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
      "label_mask", "label_presence", "label_density", "label_density_class",
      "year", "Year", "dist_to_SAV", "dist_to_sav", "Dist_to_SAV")
  )
  lab <- lab[, keep_lab, drop = FALSE]

  df <- dplyr::inner_join(env, lab, by = "NodeID", suffix = c("_env", "_lab"))


  df <- coalesce_first(df, "x", c("x", "x_env", "x_lab"))
  df <- coalesce_first(df, "y", c("y", "y_env", "y_lab"))
  df <- coalesce_first(df, "lon", c("lon", "lon_env", "lon_lab", "Longitude", "Longitude_env", "Longitude_lab"))
  df <- coalesce_first(df, "lat", c("lat", "lat_env", "lat_lab", "Latitude", "Latitude_env", "Latitude_lab"))


  df$year <- year
  df
}


infer_features <- function(df) {
  drop_exact <- c(
    "NodeID", "node_id",
    "lon", "lat", "x", "y", "Longitude", "Latitude",
    "year", "Year",
    "label_mask", "label_mask_source", "train_mask", "val_mask", "test_mask",
    "spatial_split", "label_presence", "label_density", "label_density_class",
    "label", "presence", "density", "PreorAb",
    "dist_to_coast", "dist_to_SAV", "dist_to_sav", "Dist_to_SAV",
    "Depth", "depth"
  )


  drop_pattern <- grepl(
    "^(x|y|lon|lat|Longitude|Latitude|year|Year|label|presence|density|PreorAb|mask|Depth|depth|dist_to_coast|dist_to_SAV|dist_to_sav|Dist_to_SAV)(_|\\.)",
    names(df),
    ignore.case = TRUE
  )

  cand <- setdiff(names(df), drop_exact)
  cand <- cand[!drop_pattern[match(cand, names(df))]]
  cand <- cand[sapply(df[, cand, drop=FALSE], is.numeric)]


  keep <- c()
  for (cc in cand) {
    x <- df[[cc]]
    finite <- x[is.finite(x)]
    if (length(finite) == 0) next
    if (length(unique(finite)) <= 1) next
    keep <- c(keep, cc)
  }
  if (length(keep) == 0) stop("No numeric environmental feature columns were identified for GAM")
  keep
}

fill_feature_na_by_train_median <- function(all_df, train_df, feature_cols) {
  med <- sapply(feature_cols, function(cc) {
    v <- suppressWarnings(as.numeric(train_df[[cc]]))
    m <- median(v[is.finite(v)], na.rm = TRUE)
    if (!is.finite(m)) m <- 0
    m
  })
  for (cc in feature_cols) {
    v <- suppressWarnings(as.numeric(all_df[[cc]]))
    v[!is.finite(v)] <- med[[cc]]
    all_df[[cc]] <- v
  }
  attr(all_df, "feature_median") <- med
  all_df
}


all_years <- target_years
cat("\n===== Reading annual data =====\n")
all_df <- bind_rows(lapply(all_years, function(yr) {
  cat("[Read] year=", yr, "\n")
  load_year(yr)
}))

if (!file.exists(mask_file)) {
  stop(
    "Shared TGNN node mask not found: ", mask_file,
    "\nPlease run the updated run_sdm_python_baselines.py first."
  )
}
mask_df <- readr::read_csv(mask_file, show_col_types = FALSE)
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
  stop("Shared mask contains duplicate year+NodeID entries")
}
if (nrow(mask_df) != nrow(all_df)) {
  stop("Shared mask and GAM data row counts do not match; stopping.")
}
all_df <- all_df %>% rename(label_mask_source = label_mask)
all_df <- all_df %>% left_join(mask_df, by=c("year", "NodeID"))
if (any(is.na(all_df$label_mask))) {
  stop("Some GAM nodes could not be matched to the shared mask")
}
for (cc in c("label_mask", "train_mask", "val_mask", "test_mask")) {
  if (any(!all_df[[cc]] %in% c(0L, 1L))) {
    stop("Shared mask column contains values other than 0/1: ", cc)
  }
}
if (any(is.na(all_df$label_mask_source))) {
  stop("label_mask in the label CSV contains NA values")
}
mask_sum <- all_df$train_mask + all_df$val_mask + all_df$test_mask
if (any(mask_sum > 1L) ||
    any(all_df$label_mask == 1L & mask_sum != 1L) ||
    any(all_df$label_mask == 0L & mask_sum != 0L)) {
  stop("Shared mask contains overlapping splits or an invalid label_mask range")
}
if (any(as.integer(all_df$label_mask_source) != as.integer(all_df$label_mask))) {
  stop("label_mask in the label CSV does not match the shared TGNN mask")
}

feature_cols <- infer_features(all_df)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(data.frame(feature = feature_cols), file.path(out_dir, "feature_columns_GAM.csv"))
cat("[Features] Total: ", length(feature_cols), " features\n")
cat("[Features] ", paste(feature_cols, collapse=", "), "\n")


train_raw <- all_df %>% filter(label_mask == 1L, train_mask == 1L)
all_df <- fill_feature_na_by_train_median(all_df, train_raw, feature_cols)

train <- all_df %>% filter(label_mask == 1L, train_mask == 1L)
val   <- all_df %>% filter(label_mask == 1L, val_mask == 1L)
test  <- all_df %>% filter(label_mask == 1L, test_mask == 1L)
if (nrow(train) == 0L || nrow(val) == 0L || nrow(test) == 0L) {
  stop("At least one of the train/val/test node masks is empty")
}

cat("[Samples] train=", nrow(train), ", val=", nrow(val), ", test=", nrow(test), "\n")
cat("[Samples] train presence=", sum(train$label_presence == 1), ", absence=", sum(train$label_presence == 0), "\n")
cat("[Training scope] label_mask==1 and train_mask==1; all nodes are not used\n")

# ---------------------------

# ---------------------------
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
  vals <- lapply(ths, function(th) calc_cls_metrics(y, p, th))
  vals <- bind_rows(vals)
  if (!criterion %in% names(vals)) stop("Threshold metric not found: ", criterion)
  idx <- which.max(vals[[criterion]])
  list(threshold = vals$threshold[idx], score = vals[[criterion]][idx], table=vals)
}

calc_reg_metrics <- function(y, pred) {
  ok <- is.finite(y) & is.finite(pred)
  y <- y[ok]; pred <- pred[ok]
  if (length(y) == 0) {
    return(data.frame(RMSE=NA_real_, MAE=NA_real_, R2=NA_real_, Spearman_r=NA_real_))
  }
  rmse <- sqrt(mean((y - pred)^2))
  mae <- mean(abs(y - pred))
  sst <- sum((y - mean(y))^2)
  r2 <- ifelse(length(y) > 1 && sst > 0, 1 - sum((y - pred)^2) / sst, NA_real_)
  sp <- suppressWarnings(cor(y, pred, method = "spearman", use = "complete.obs"))
  data.frame(RMSE = rmse, MAE = mae, R2 = r2, Spearman_r = sp)
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

# ---------------------------

# ---------------------------

build_gam_rhs <- function(train_df, feature_cols, k_max = 5) {
  terms <- c()
  for (cc in feature_cols) {
    v <- train_df[[cc]]
    u <- length(unique(v[is.finite(v)]))
    if (u <= 1) next

    if (u <= 3) {
      terms <- c(terms, paste0("`", cc, "`"))
    } else {
      k_use <- max(3, min(k_max, u - 1))
      terms <- c(terms, paste0("s(`", cc, "`, k=", k_use, ")"))
    }
  }
  if (length(terms) == 0) stop("No feature terms are available to build the GAM formula")
  paste(terms, collapse = " + ")
}

smooth_terms <- build_gam_rhs(train, feature_cols, smooth_k_max)
form_cls <- as.formula(paste("label_presence ~", smooth_terms))
form_reg <- as.formula(paste("label_density ~", smooth_terms))

cat("\n[GAM classification formula]\n")
print(form_cls)
cat("\n[GAM regression formula]\n")
print(form_reg)

cat("\n===== Training GAM classification model =====\n")
t0 <- Sys.time()
gam_cls <- mgcv::bam(form_cls, data = train, family = binomial(link = "logit"),
                     method = "fREML", discrete = TRUE, na.action = na.exclude)
cat("[Completed] GAM classification model runtime: ", round(difftime(Sys.time(), t0, units = "mins"), 2), " minutes\n")

cat("\n===== Training GAM density regression model =====\n")
train_reg <- train %>% filter(is.finite(label_density))
t0 <- Sys.time()
gam_reg <- mgcv::bam(form_reg, data = train_reg, family = gaussian(),
                     method = "fREML", discrete = TRUE, na.action = na.exclude)
cat("[Completed] GAM density regression model runtime: ", round(difftime(Sys.time(), t0, units = "mins"), 2), " minutes\n")

# ---------------------------

# ---------------------------

pooled_metric_row <- function(model_name, pred_df, split_name, threshold) {
  mask_col <- paste0(split_name, "_mask")
  eval_df <- pred_df %>% filter(label_mask == 1L, .data[[mask_col]] == 1L)
  cls <- calc_cls_metrics(
    eval_df$label_presence, eval_df$prob_presence, threshold
  )
  reg_all <- calc_reg_metrics(eval_df$label_density, eval_df$pred_density)
  pos_df <- eval_df %>% filter(label_presence == 1L)
  reg_pos <- calc_reg_metrics(pos_df$label_density, pos_df$pred_density)
  data.frame(
    model=model_name, split=split_name, aggregation="pooled",
    N=nrow(eval_df), N_positive=sum(eval_df$label_presence == 1L),
    prevalence=mean(eval_df$label_presence == 1L),
    threshold=cls$threshold, AP=cls$AP, PR_AUC=cls$PR_AUC,
    F1=cls$F1, Recall=cls$Recall, Precision=cls$Precision,
    RMSE=reg_all$RMSE, RMSE_presence=reg_pos$RMSE,
    ROC_AUC=cls$ROC_AUC, TSS=cls$TSS,
    MAE=reg_all$MAE, R2=reg_all$R2, Spearman_r=reg_all$Spearman_r,
    TP=cls$TP, TN=cls$TN, FP=cls$FP, FN=cls$FN,
    stringsAsFactors=FALSE
  )
}

predict_one_model <- function(model_name, df_all, prob_fun, reg_fun=NULL) {
  val_eval <- df_all %>% filter(label_mask == 1L, val_mask == 1L)
  val_prob <- prob_fun(val_eval)
  bt <- best_threshold(
    val_eval$label_presence, val_prob, threshold_criterion
  )
  threshold <- bt$threshold
  write_csv(bt$table, file.path(out_dir, "GAM_pooled_val_threshold_scan.csv"))
  cat(model_name, " pooled validation threshold=", round(threshold, 4),
      ", score=", round(bt$score, 4), "\n", sep="")

  pred_out <- df_all
  pred_out$prob_presence <- as.numeric(prob_fun(pred_out))
  pred_out$threshold_used <- threshold
  pred_out$pred_presence <- ifelse(
    pred_out$prob_presence >= threshold, 1L, 0L
  )
  pred_out$pred_density <- if (is.null(reg_fun)) {
    NA_real_
  } else {
    as.numeric(reg_fun(pred_out))
  }
  if (!is.null(reg_fun) && clamp_density) {
    pred_out$pred_density <- pmin(pmax(pred_out$pred_density, 0.0), 0.85)
  }

  pred_out$model <- model_name
  pred_out <- add_confusion_type(pred_out)

  metrics_out <- bind_rows(
    pooled_metric_row(model_name, pred_out, "val", threshold),
    pooled_metric_row(model_name, pred_out, "test", threshold)
  )
  list(metrics=metrics_out, predictions=pred_out, threshold=threshold)
}

gam_prob_fun <- function(newdata) as.numeric(predict(gam_cls, newdata = newdata, type = "response"))
gam_reg_fun  <- function(newdata) as.numeric(predict(gam_reg, newdata = newdata, type = "response"))

res_gam <- predict_one_model("GAM", all_df, gam_prob_fun, gam_reg_fun)
metrics_gam <- res_gam$metrics

write_csv(metrics_gam, file.path(out_dir, "metrics_GAM_node_mask_pooled.csv"))
all_pred_file <- file.path(
  pred_dir,
  paste0("GAM_predictions_all_", min(target_years), "_", max(target_years), ".csv")
)
write_csv(res_gam$predictions, all_pred_file)
for (yr in target_years) {
  write_csv(
    res_gam$predictions %>% filter(year == yr),
    file.path(pred_dir, paste0("GAM_", yr, ".csv"))
  )
}
saveRDS(gam_cls, file.path(out_dir, "GAM_classifier.rds"))
saveRDS(gam_reg, file.path(out_dir, "GAM_density_regressor.rds"))

for (ii in seq_len(nrow(metrics_gam))) {
  rr <- metrics_gam[ii,]
  cat(sprintf(
    "[GAM][POOLED %s] AP=%.4f F1=%.4f R=%.4f P=%.4f RMSE=%.4f tau=%.4f\n",
    toupper(rr$split), rr$AP, rr$F1, rr$Recall, rr$Precision,
    rr$RMSE, rr$threshold
  ))
}

cat("\n✅ GAM completed. Output directory: ", out_dir, "\n")
cat("Node-level predictions: ", all_pred_file, "\n")
cat("Metrics file: ", file.path(out_dir, "metrics_GAM_node_mask_pooled.csv"), "\n")
