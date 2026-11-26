#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_YEAR_START = 1999
DATA_YEAR_END = 2024
TIME_WINDOW = 11
FEATURE_MODE = "current"       # current\lagged
MASK_SEED = 1234               #
MODEL_SEED = 1234              #
THRESHOLD_CRITERION = "F1"

DEFAULT_DATA_YEARS = f"{DATA_YEAR_START}:{DATA_YEAR_END}"
DEFAULT_TIME_WINDOW = TIME_WINDOW
DEFAULT_FEATURE_MODE = FEATURE_MODE
DEFAULT_MASK_SEED = MASK_SEED
DEFAULT_MODEL_SEED = MODEL_SEED
DEFAULT_THRESHOLD_CRITERION = THRESHOLD_CRITERION

from sav_data import (
    DENSITY_LEVEL_VALUES,
    parse_years,
    load_pathA_table,
    split_xy,
)

from sav_metrics import (
    classification_metrics,
    regression_metrics,
    find_best_threshold,
    threshold_scan_table,
    add_confusion_type,
)


def make_models(seed: int, include_xgb: bool = True):
    models = {
        "RF": {
            "cls": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                )),
            ]),
            "reg": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=seed,
                    n_jobs=-1,
                )),
            ]),
        },
        "BRT": {
            "cls": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(
                    n_estimators=800,
                    learning_rate=0.03,
                    max_depth=3,
                    subsample=0.7,
                    random_state=seed,
                )),
            ]),
            "reg": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(
                    n_estimators=800,
                    learning_rate=0.03,
                    max_depth=3,
                    subsample=0.7,
                    random_state=seed,
                )),
            ]),
        },
    }

    if include_xgb:
        try:
            from xgboost import XGBClassifier, XGBRegressor
            models["XGBoost"] = {
                "cls": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBClassifier(
                        n_estimators=700,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.7,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    )),
                ]),
                "reg": Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(
                        n_estimators=700,
                        max_depth=6,
                        learning_rate=0.03,
                        subsample=0.7,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=-1,
                    )),
                ]),
            }
        except Exception as exc:
            warnings.warn(f"xgboost can't use，jump XGBoost：{exc}")
    return models


def predict_positive_proba(model, X):
    clf = model.named_steps["model"] if hasattr(model, "named_steps") else model
    probs = model.predict_proba(X)
    classes = getattr(clf, "classes_", np.array([0, 1]))
    pos = np.where(classes == 1)[0]
    if len(pos) != 1:
        raise RuntimeError(f"Positive class 1 is not uniquely found: {classes}")
    return probs[:, int(pos[0])]


def predict_density(model, X):
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(
        pred,
        float(DENSITY_LEVEL_VALUES.min()),
        float(DENSITY_LEVEL_VALUES.max()),
    )

def pooled_metrics_for_split(
    model_name, df_all, feature_cols, clf, reg, threshold, split
):
    dat, X, y_cls, y_reg = split_xy(df_all, feature_cols, split=split)
    prob = predict_positive_proba(clf, X)
    pred_reg = predict_density(reg, X)
    cls = classification_metrics(y_cls, prob, threshold)
    reg_all = regression_metrics(y_reg, pred_reg)

    pos = y_cls == 1
    reg_pos = regression_metrics(y_reg[pos], pred_reg[pos])
    return {
        "model": model_name,
        "split": split,
        "aggregation": "pooled",
        "N": int(len(dat)),
        "N_positive": int(pos.sum()),
        "prevalence": float(pos.mean()),
        "threshold": float(threshold),
        "AP": cls["AP"],
        "PR_AUC": cls["PR_AUC"],
        "F1": cls["F1"],
        "Recall": cls["Recall"],
        "Precision": cls["Precision"],
        "RMSE": reg_all["RMSE"],
        "RMSE_presence": reg_pos["RMSE"],
        "ROC_AUC": cls["ROC_AUC"],
        "TSS": cls["TSS"],
        "MAE": reg_all["MAE"],
        "R2": reg_all["R2"],
        "Spearman_r": reg_all["Spearman_r"],
        "TP": cls["TP"], "TN": cls["TN"],
        "FP": cls["FP"], "FN": cls["FN"],
    }

def save_full_predictions(model_name, df_all, feature_cols, clf, reg, threshold, out_dir):
    X_all = df_all[feature_cols].replace([np.inf, -np.inf], np.nan)
    out = df_all.copy()
    out["model"] = model_name
    out["prob_presence"] = predict_positive_proba(clf, X_all)
    out["pred_density"] = predict_density(reg, X_all)
    out["threshold_used"] = threshold
    out["pred_presence"] = (
        out["prob_presence"].astype(float) >= threshold
    ).astype(int)

    out = add_confusion_type(out, split="test")

    pred_dir = out_dir / "predictions" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    year_min = int(out["year"].min())
    year_max = int(out["year"].max())
    all_file = pred_dir / (
        f"{model_name}_predictions_all_{year_min}_{year_max}.csv"
    )
    out.to_csv(all_file, index=False)

    files = [str(all_file)]
    for year, g in out.groupby("year"):
        f = pred_dir / f"{model_name}_{int(year)}.csv"
        g.to_csv(f, index=False)
        files.append(str(f))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=None,
        help="Project root.",
    )
    parser.add_argument(
        "--out-dir", default="results/sdm/python_baselines"
    )

    parser.add_argument("--years", default=DEFAULT_DATA_YEARS)
    parser.add_argument(
        "--time-window", "--window", dest="time_window",
        type=int, default=DEFAULT_TIME_WINDOW,
        help="Consisted with the T_Window of TGNN;",
    )
    parser.add_argument(
        "--mask-seed", type=int, default=DEFAULT_MASK_SEED,
        help=" must consisted with maskseed in dataset_AllMap_multitask.py",
    )
    parser.add_argument(
        "--mask-file", default=None,
        help=" alternative ",
    )

    parser.add_argument(
        "--feature-mode", choices=["current", "lagged"],
        default=DEFAULT_FEATURE_MODE,
    )
    parser.add_argument(
        "--threshold-criterion",
        choices=["F1", "TSS"],
        default=DEFAULT_THRESHOLD_CRITERION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--no-xgb", action="store_true")
    args = parser.parse_args()

    root = (
        Path(args.root).resolve()
        if args.root
        else Path(__file__).resolve().parent.parent
    )

    out_dir = (
        Path(args.out_dir)
        if Path(args.out_dir).is_absolute()
        else root / args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)


    data_years = sorted(set(parse_years(args.years)))
    if args.time_window < 1:
        raise ValueError("--time-window must >=1")
    if args.time_window > len(data_years):
        raise ValueError(
            f"time_window={args.time_window}exceed data years{len(data_years)}"
        )
    gaps = [
        (a, b) for a, b in zip(data_years[:-1], data_years[1:])
        if b != a + 1
    ]
    if gaps:
        raise ValueError(f"years must consecutive，found break: {gaps}")
    target_years = data_years[args.time_window - 1:]

    if args.mask_file:
        candidate = Path(args.mask_file)
        mask_file = candidate if candidate.is_absolute() else root / candidate
    else:
        mask_file = None

    print(
        "[DESIGN] annual node-mask evaluation; not strict temporal holdout."
    )
    print(f"[DATA] source_years={data_years}")
    print(
        f"[DATA] T={args.time_window}, target_years={target_years}, "
        f"n_windows={len(target_years)}"
    )
    print(
        f"[MASK] source={'TGNN reproduction' if mask_file is None else mask_file}, "
        f"mask_seed={args.mask_seed}"
    )
    print(f"[FEATURE] mode={args.feature_mode}")

    df_all, feature_cols = load_pathA_table(
        root=root,
        target_years=target_years,
        mask_file=mask_file,
        mask_seed=args.mask_seed,
        feature_mode=args.feature_mode,
        window=args.time_window,
    )

    shared_dir = root / "results" / "sdm" / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    mask_file_out = shared_dir / (
        f"tgnn_{target_years[0]}_{target_years[-1]}.csv"
    )
    mask_cols = [
        "year", "NodeID", "label_mask_source", "label_mask",
        "train_mask", "val_mask", "test_mask", "spatial_split",
    ]
    missing_mask_cols = [c for c in mask_cols if c not in df_all.columns]
    if missing_mask_cols:
        raise ValueError(f"lack of column: {missing_mask_cols}")
    (
        df_all[mask_cols]
        .sort_values(["year", "NodeID"])
        .to_csv(mask_file_out, index=False)
    )
    print(f"[MASK] canonical mask saved: {mask_file_out}")


    train_dat, X_train, y_train_cls, y_train_reg = split_xy(
        df_all, feature_cols, split="train"
    )
    val_dat, X_val, y_val_cls, _ = split_xy(
        df_all, feature_cols, split="val"
    )

    if not (
        (train_dat["label_mask"].astype(int) == 1).all()
        and (train_dat["train_mask"].astype(int) == 1).all()
    ):
        raise AssertionError("Train data exceed range")
    print(
        "[TRAIN SCOPE] only label_mask==1 AND train_mask==1; "
        f"N={len(train_dat)}"
    )

    print(
        "[POOLED N] train/val/test =",
        int(df_all["train_mask"].sum()),
        int(df_all["val_mask"].sum()),
        int(df_all["test_mask"].sum()),
    )
    pd.Series(feature_cols, name="feature").to_csv(
        out_dir / "feature_columns.csv", index=False
    )

    models = make_models(args.seed, include_xgb=not args.no_xgb)
    all_metrics = []
    manifest = {
        "design": (
            "annual node-mask evaluation matching TGNN; "
            "not strict temporal holdout"
        ),
        "data_years": data_years,
        "target_years": target_years,
        "mask_source": (
            "reproduced_from_dataset_AllMap_multitask"
            if mask_file is None else str(mask_file)
        ),
        "mask_seed": args.mask_seed,
        "model_seed": args.seed,
        "feature_mode": args.feature_mode,
        "time_window": args.time_window,
        "feature_cols": feature_cols,
        "canonical_mask_file": str(mask_file_out),
        "models": {},
    }

    for name, pair in models.items():
        print(f"\n===== Train {name} =====")
        clf, reg = pair["cls"], pair["reg"]
        clf.fit(X_train, y_train_cls)
        reg.fit(X_train, y_train_reg)

        val_prob = predict_positive_proba(clf, X_val)
        threshold, val_score = find_best_threshold(
            y_val_cls,
            val_prob,
            criterion=args.threshold_criterion,
        )
        scan_thresholds = np.unique(np.append(
            np.arange(0.10, 0.9001, 0.02), threshold
        ))
        scan = threshold_scan_table(
            y_val_cls, val_prob, thresholds=scan_thresholds
        )
        scan.to_csv(out_dir / f"{name}_pooled_val_threshold_scan.csv", index=False)


        print(
            f"{name}: pooled val {args.threshold_criterion} "
            f"threshold={threshold:.3f}, score={val_score:.4f}"
        )


        val_row = pooled_metrics_for_split(
            name, df_all, feature_cols, clf, reg, threshold, "val"
        )
        test_row = pooled_metrics_for_split(
            name, df_all, feature_cols, clf, reg, threshold, "test"
        )
        all_metrics.extend([val_row, test_row])

        for row in [val_row, test_row]:
            print(
                f"[{name}][POOLED {row['split'].upper()}] "
                f"AP={row['AP']:.4f} F1={row['F1']:.4f} "
                f"R={row['Recall']:.4f} P={row['Precision']:.4f} "
                f"RMSE={row['RMSE']:.4f} tau={row['threshold']:.4f}"
            )

        pred_files = save_full_predictions(
            name, df_all, feature_cols, clf, reg, threshold, out_dir
        )
        joblib.dump(clf, out_dir / "models" / f"{name}_classifier.joblib")
        joblib.dump(reg, out_dir / "models" / f"{name}_regressor.joblib")

        manifest["models"][name] = {
            "threshold": threshold,
            "val_score_for_threshold": val_score,
            "pooled_val_metrics": val_row,
            "pooled_test_metrics": test_row,
            "prediction_files": pred_files,
        }

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(
        out_dir / "metrics_python_baselines_node_mask_pooled.csv", index=False
    )

    with open(
        out_dir / "run_manifest_python_baselines_node_mask.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n✅ Node-mask pooled Python baselines complete：", out_dir)


if __name__ == "__main__":
    main()
