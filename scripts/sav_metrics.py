#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)


def safe_auc(y_true, y_score, kind: str = "roc") -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    ok = np.isfinite(y_score)
    y_true, y_score = y_true[ok], y_score[ok]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if kind == "roc":
            return float(roc_auc_score(y_true, y_score))
        if kind == "ap":
            return float(average_precision_score(y_true, y_score))
        if kind == "pr":
            pr_precision, pr_recall, _ = precision_recall_curve(
                y_true, y_score
            )
            return float(auc(pr_recall, pr_precision))
        raise ValueError(f"Unsupported AUC kind: {kind}")
    except Exception:
        return float("nan")



def tss_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    return float(sensitivity + specificity - 1.0)


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[ok].astype(int), y_prob[ok]

    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "F1": np.nan, "Recall": np.nan, "Precision": np.nan,
            "AP": np.nan, "PR_AUC": np.nan,
            "ROC_AUC": np.nan, "TSS": np.nan,
            "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        }

    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "AP": safe_auc(y_true, y_prob, kind="ap"),
        "PR_AUC": safe_auc(y_true, y_prob, kind="pr"),
        "ROC_AUC": safe_auc(y_true, y_prob, kind="roc"),
        "TSS": tss_score(y_true, y_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})
    return out

def threshold_scan_table(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.9001, 0.02)
    rows = []
    for th in thresholds:
        rows.append(classification_metrics(y_true, y_prob, float(th)))
    return pd.DataFrame(rows)

def find_best_threshold(y_true, y_prob, criterion: str = "F1", thresholds=None) -> tuple[float, float]:
    criterion = criterion.upper()
    if criterion not in {"F1", "TSS"}:
        raise ValueError("criterion must be 'F1' or 'TSS'")

    if criterion == "F1" and thresholds is None:
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        ok = np.isfinite(y_true) & np.isfinite(y_prob)
        y_true, y_prob = y_true[ok].astype(int), y_prob[ok]
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return 0.5, float("nan")

        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            y_true, y_prob
        )
        if len(pr_thresholds) == 0:
            return 0.5, float("nan")
        f1_values = (
            2.0 * pr_precision[:-1] * pr_recall[:-1]
            / (pr_precision[:-1] + pr_recall[:-1] + 1e-12)
        )
        best_idx = int(np.nanargmax(f1_values))
        return (
            float(pr_thresholds[best_idx]),
            float(f1_values[best_idx]),
        )

    scan = threshold_scan_table(y_true, y_prob, thresholds)
    good = scan[np.isfinite(scan[criterion])].copy()
    if good.empty:
        return 0.5, float("nan")

    best_idx = good[criterion].idxmax()
    return float(scan.loc[best_idx, "threshold"]), float(scan.loc[best_idx, criterion])



def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Spearman_r": np.nan}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    if len(y_true) >= 2 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        sp = float(pd.Series(y_true).rank().corr(pd.Series(y_pred).rank()))
    else:
        sp = float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Spearman_r": sp}


def build_evaluation_mask(
    df: pd.DataFrame,
    split: str | None = None,
    label_mask_col: str = "label_mask",
) -> np.ndarray:
    if label_mask_col not in df.columns:
        raise ValueError(f"lack of column: {label_mask_col}")

    mask = df[label_mask_col].astype(int).to_numpy() == 1
    if split is not None:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, test or None")
        split_col = f"{split}_mask"
        if split_col not in df.columns:
            raise ValueError(f"lack of column: {split_col}")
        mask &= df[split_col].astype(int).to_numpy() == 1
    return mask

def add_confusion_type(
    df: pd.DataFrame,
    y_col: str = "label_presence",
    pred_col: str = "pred_presence",
    mask_col: str = "label_mask",
    split: str | None = None,
) -> pd.DataFrame:

    out = df.copy()
    if mask_col not in out.columns:
        out[mask_col] = 1

    out["confusion_type"] = "OUTSIDE_EVAL_MASK"
    m = build_evaluation_mask(out, split=split, label_mask_col=mask_col)
    y = out[y_col].astype(int).to_numpy()
    p = out[pred_col].astype(int).to_numpy()

    out.loc[m & (y == 1) & (p == 1), "confusion_type"] = "TP"
    out.loc[m & (y == 0) & (p == 0), "confusion_type"] = "TN"
    out.loc[m & (y == 0) & (p == 1), "confusion_type"] = "FP"
    out.loc[m & (y == 1) & (p == 0), "confusion_type"] = "FN"
    return out
