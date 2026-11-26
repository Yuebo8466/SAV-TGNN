#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from __future__ import annotations

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

DENSITY_LEVEL_VALUES = np.array([0.0, 0.05, 0.25, 0.55, 0.85], dtype=float)
DENSITY_CLASS_TO_VALUE = {0: 0.0, 1: 0.05, 2: 0.25, 3: 0.55, 4: 0.85}

def parse_years(s) -> list[int]:
    if isinstance(s, (list, tuple, np.ndarray)):
        return [int(x) for x in s]
    s = str(s).strip()
    if s.startswith("range"):
        nums = s.replace("range(", "").replace(")", "").split(",")
        return list(range(int(nums[0]), int(nums[1])))
    if ":" in s:
        a, b = s.split(":", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def normalize_density_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label_density" not in df.columns:
        df["label_density"] = np.nan
        df["label_density_class"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        return df

    dens = pd.to_numeric(df["label_density"], errors="coerce")
    finite = dens[np.isfinite(dens)]
    if len(finite) == 0:
        df["label_density"] = dens
        df["label_density_class"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        return df

    unique_vals = np.sort(pd.unique(finite))
    is_integer_class = (
        np.all(np.isclose(unique_vals, np.round(unique_vals)))
        and np.nanmax(unique_vals) > 1.0
        and np.nanmin(unique_vals) >= 0
        and np.nanmax(unique_vals) <= 4
    )
    if is_integer_class:
        cls = dens.round().clip(0, 4)
        df["label_density_class"] = cls.astype("Int64")
        df["label_density"] = cls.map(DENSITY_CLASS_TO_VALUE).astype(float)
        return df

    arr = dens.to_numpy(dtype=float)
    cls = np.full(len(arr), np.nan)
    ok = np.isfinite(arr)
    if ok.any():
        cls[ok] = np.argmin(
            np.abs(arr[ok, None] - DENSITY_LEVEL_VALUES[None, :]), axis=1
        )
    df["label_density"] = dens.astype(float)
    df["label_density_class"] = pd.Series(cls).astype("Int64")
    return df


DEFAULT_DROP_COLS = {
    "NodeID", "node_id", "ID", "index",
    "lon", "lat", "x", "y", "Longitude", "Latitude",
    "lon_lab", "lat_lab", "x_lab", "y_lab", "Longitude_lab", "Latitude_lab",
    "year", "Year", "year_lab", "Year_lab", "split", "spatial_split",
    "label_mask", "label_mask_source",
    "train_mask", "val_mask", "test_mask",
    "label_presence", "label_density", "label_density_class",
    "label", "presence", "PreorAb", "density", "Density", "DENSITY", "cover", "Cover",
    "dist_to_coast", "dist_to_SAV", "dist_to_sav", "Dist_to_SAV",
    "Depth", "depth", "h", "H",
    "Unnamed: 0",
}

LEAKAGE_PATTERN = re.compile(
    r"label|presence|density|mask|dist_to_SAV|dist_to_sav|Dist_to_SAV|"
    r"NodeID|node_id|^x$|^y$|lon|lat|Longitude|Latitude|year|Year|Depth|depth|dist_to_coast",
    flags=re.IGNORECASE,
)


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def label_file_for_year(root: str | Path, year: int) -> Path:
    root = Path(root)

    candidates = [
        root / "data" / "raw" / "AnnualLabelsforTGNN" / f"labels_{year}.csv",
        root / "data" / "raw" / "AnnualLabelsforTGNN" / f"label_{year}.csv",
        root / "data" / "raw" / "SAV_TGNNAlllabels_MultiTask" / f"labels_{year}.csv",
        root / "data" / "raw" / "SAV_TGNNAlllabels_MultiTask" / f"label_{year}.csv",
    ]
    p = _first_existing(candidates)
    if p is None:
        raise FileNotFoundError("can't find label files，try the path：\n" + "\n".join(map(str, candidates)))
    return p


def env_file_for_year(root: str | Path, year: int) -> Path:
    root = Path(root)
    p = root / "data" / "raw" / "AnnualFeaturesforTGNN" / f"EnvGNN_{year}.csv"
    if not p.exists():
        raise FileNotFoundError(f"can't find feature files: {p}")
    return p


def standardize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    if "label_presence" not in df.columns:
        for alt in ["label", "presence", "PreorAb"]:
            if alt in df.columns:
                df["label_presence"] = df[alt]
                break
    if "label_presence" not in df.columns:
        raise ValueError("label files lack of label_presence / label / presence / PreorAb")

    if "label_density" not in df.columns:
        for alt in ["density", "Density", "DENSITY", "cover", "Cover"]:
            if alt in df.columns:
                df["label_density"] = df[alt]
                break
    if "label_density" not in df.columns:
        df["label_density"] = np.nan

    if "label_mask" not in df.columns:
        raise ValueError(
            "lable files lack of label_mask；stop。"
        )

    df = normalize_density_column(df)
    df["label_presence"] = pd.to_numeric(
        df["label_presence"], errors="coerce"
    ).fillna(0).astype(int)
    df["label_mask"] = pd.to_numeric(
        df["label_mask"], errors="coerce"
    ).fillna(0).astype(int)
    return df

def load_env_year(root: str | Path, year: int) -> pd.DataFrame:
    env = pd.read_csv(env_file_for_year(root, year))
    env.columns = [c.strip() for c in env.columns]
    env = env.loc[:, ~env.columns.str.contains(r"^Unnamed")]
    if "NodeID" not in env.columns:
        raise ValueError(f"{year} environmental files lack of NodeID")

    exact_leak_cols = {
        "label_presence", "label_density", "label_density_class", "label_mask",
        "train_mask", "val_mask", "test_mask", "split",
        "label", "presence", "PreorAb", "density", "Density", "DENSITY",
    }
    env = env.drop(columns=[c for c in env.columns if c in exact_leak_cols], errors="ignore")
    return env

def load_year(root: str | Path, year: int) -> pd.DataFrame:
    env = load_env_year(root, year)
    lab = standardize_label_columns(pd.read_csv(label_file_for_year(root, year)))
    if "NodeID" not in lab.columns:
        raise ValueError(f"{year} lable files lack of NodeID")

    label_cols = [
        "NodeID", "label_presence", "label_density",
        "label_density_class", "label_mask",
    ]
    lab = lab[label_cols].copy().rename(columns={"label_mask": "label_mask_source"})
    df = env.merge(lab, on="NodeID", how="inner", validate="one_to_one", sort=False)
    df["year"] = int(year)
    return df

def infer_feature_columns(df: pd.DataFrame, extra_drop=None, strict: bool = True) -> list[str]:
    drop = set(DEFAULT_DROP_COLS)
    if extra_drop:
        drop.update(extra_drop)

    cols = []
    for c in df.columns:
        if c in drop or c.endswith("_lab"):
            continue
        if strict and LEAKAGE_PATTERN.search(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    bad = [c for c in cols if LEAKAGE_PATTERN.search(c)]
    if bad:
        raise ValueError("features contained leaked field：" + ", ".join(bad))
    if not cols:
        raise ValueError("unrecognized feature column")
    return cols


def load_years(
    root: str | Path,
    years: list[int],
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:

    frames = []
    for year in years:
        frame = load_year(root, year)
        if feature_cols is None:
            feature_cols = infer_feature_columns(frame)
        missing = [c for c in feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{year}year lacks of feature column: {missing}")
        frames.append(frame)
    df = pd.concat(frames, axis=0, ignore_index=True)

    return df, feature_cols

def build_lagged_table(
    root: str | Path,
    target_years: list[int],
    window: int = 7,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if window < 1:
        raise ValueError("window must >= 1")

    env_by_year = {}
    all_needed = sorted({y - lag for y in target_years for lag in range(window)})
    for yy in all_needed:
        env_by_year[yy] = load_env_year(root, yy)

    if feature_cols is None:
        feature_cols = infer_feature_columns(env_by_year[target_years[0]])

    out_frames = []
    for y in target_years:
        base = load_year(root, y)
        keep_meta = [
            c for c in [
                "NodeID", "year", "label_mask_source",
                "label_presence", "label_density", "label_density_class",
                "lon", "lat", "x", "y", "Longitude", "Latitude",
            ]
            if c in base.columns
        ]
        base = base[keep_meta].copy()

        for lag in range(window):
            yy = y - lag
            feat = env_by_year[yy][["NodeID"] + feature_cols].copy()
            feat = feat.rename(columns={c: f"{c}_lag{lag}" for c in feature_cols})
            base = base.merge(feat, on="NodeID", how="inner", validate="one_to_one")
        out_frames.append(base)

    out = pd.concat(out_frames, axis=0, ignore_index=True)
    lag_cols = [f"{c}_lag{lag}" for lag in range(window) for c in feature_cols]
    return out, lag_cols

def load_spatial_mask_table(
    mask_file: str | Path,
    target_years: list[int] | None = None,
) -> pd.DataFrame:
    p = Path(mask_file)
    if not p.exists():
        raise FileNotFoundError(f"can't find TGNN spatial mask file: {p}")

    mm = pd.read_csv(p)
    mm.columns = [c.strip() for c in mm.columns]

    if "NodeID" not in mm.columns:
        raise ValueError("mask file lack of NodeID")
    if "year" not in mm.columns:
        raise ValueError("path A need mask files contain year column")

    if "split" in mm.columns and not {
        "train_mask", "val_mask", "test_mask"
    }.issubset(mm.columns):
        ss = mm["split"].astype(str).str.lower()
        mm["train_mask"] = (ss == "train").astype(int)
        mm["val_mask"] = ss.isin(["val", "validation"]).astype(int)
        mm["test_mask"] = (ss == "test").astype(int)

    req = ["year", "NodeID", "train_mask", "val_mask", "test_mask"]
    missing = [c for c in req if c not in mm.columns]
    if missing:
        raise ValueError(f"mask file lack of field: {missing}")

    for c in ["year", "train_mask", "val_mask", "test_mask"]:
        mm[c] = pd.to_numeric(mm[c], errors="raise").astype(int)

    split_sum = mm["train_mask"] + mm["val_mask"] + mm["test_mask"]
    if "label_mask" not in mm.columns:
        mm["label_mask"] = (split_sum == 1).astype(int)
    else:
        mm["label_mask"] = pd.to_numeric(mm["label_mask"], errors="raise").astype(int)

    if mm.duplicated(["year", "NodeID"]).any():
        raise ValueError("mask files exist duplicated year+NodeID")
    if (split_sum > 1).any():
        raise ValueError("mask files exist train/val/test duplicate ")
    if ((mm["label_mask"] == 1) & (split_sum != 1)).any():
        raise ValueError("mask files exist label_mask=1 but don't entered a split")
    if ((mm["label_mask"] == 0) & (split_sum != 0)).any():
        raise ValueError("mask files exist label_mask=0 but entered a split")

    if target_years is not None:
        target_years = [int(y) for y in target_years]
        available = sorted(mm["year"].unique().tolist())
        missing_years = sorted(set(target_years) - set(available))
        if missing_years:
            raise ValueError(
                f"mask files lack of target year: {missing_years}; actual year={available}"
            )
        mm = mm[mm["year"].isin(target_years)].copy()

    return mm[
        ["year", "NodeID", "label_mask", "train_mask", "val_mask", "test_mask"]
    ].copy()

def build_tgnn_node_masks(
    df: pd.DataFrame,
    mask_seed: int = 1234,
) -> pd.DataFrame:

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "to reproduce TGNN's torch.randperm mask need PyTorch。TGNN"
            "conda environment to run。"
        ) from exc

    out = df.copy().reset_index(drop=True)
    if "label_mask_source" not in out.columns:
        raise ValueError("lack of label_mask_source，can't reproduce TGNN node mask ")
    if out.duplicated(["year", "NodeID"]).any():
        raise ValueError("exists duplicated year+NodeID，can't reproduce node mask")

    n = len(out)
    label_mask = np.zeros(n, dtype=np.int8)
    train_mask = np.zeros(n, dtype=np.int8)
    val_mask = np.zeros(n, dtype=np.int8)
    test_mask = np.zeros(n, dtype=np.int8)

    for year, group in out.groupby("year", sort=False):
        global_rows = group.index.to_numpy(dtype=int)
        mask_local = pd.to_numeric(
            group["label_mask_source"], errors="raise"
        ).to_numpy(dtype=int)
        if not np.isin(mask_local, [0, 1]).all():
            raise ValueError(f"{int(year)}year label_mask_source isn't 0/1")

        label_mask[global_rows] = mask_local.astype(np.int8)
        idx_all = torch.where(torch.tensor(mask_local) == 1)[0]
        if len(idx_all) == 0:
            continue

        generator = torch.Generator().manual_seed(int(mask_seed))
        perm = idx_all[torch.randperm(len(idx_all), generator=generator)]
        n_train = int(0.70 * len(perm))
        n_val = int(0.15 * len(perm))

        train_local = perm[:n_train].cpu().numpy()
        val_local = perm[n_train:n_train + n_val].cpu().numpy()
        test_local = perm[n_train + n_val:].cpu().numpy()
        train_mask[global_rows[train_local]] = 1
        val_mask[global_rows[val_local]] = 1
        test_mask[global_rows[test_local]] = 1

    out["label_mask"] = label_mask
    out["train_mask"] = train_mask
    out["val_mask"] = val_mask
    out["test_mask"] = test_mask

    split_sum = train_mask + val_mask + test_mask
    if np.any(split_sum > 1):
        raise AssertionError("internal error：train/val/test mask are duplicated")
    if np.any((label_mask == 1) & (split_sum != 1)):
        raise AssertionError("internal error：no label nodes entered split")
    if np.any((label_mask == 0) & (split_sum != 0)):
        raise AssertionError("internal error：no label nodes entered split")

    out["spatial_split"] = np.select(
        [train_mask == 1, val_mask == 1, test_mask == 1],
        ["train", "val", "test"],
        default="outside_split",
    )

    assigned = out[out["label_mask"] == 1]
    changed = assigned.groupby("NodeID")["spatial_split"].nunique()
    n_changed = int((changed > 1).sum())
    if n_changed > 0:
        warnings.warn(
            f"detected {n_changed} NodeID entered different split。"
            "dataset_AllMap_multitask.py.",
            RuntimeWarning,
        )
    return out

def attach_spatial_masks(df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "label_mask" in out.columns:
        if "label_mask_source" not in out.columns:
            out = out.rename(columns={"label_mask": "label_mask_source"})
        else:
            out = out.drop(columns=["label_mask"])

    out = out.drop(
        columns=[
            c for c in ["train_mask", "val_mask", "test_mask", "split", "spatial_split"]
            if c in out.columns
        ],
        errors="ignore",
    )

    out = out.merge(
        mask_df,
        on=["year", "NodeID"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    mask_cols = ["label_mask", "train_mask", "val_mask", "test_mask"]
    if out[mask_cols].isna().any().any():
        bad = (
            out.loc[out[mask_cols].isna().any(axis=1), ["year", "NodeID"]]
            .groupby("year")
            .size()
        )
        raise ValueError(
            "part of year+NodeID can't matched the spatial mask: "
            + "; ".join(f"{int(y)}:{int(n)}行" for y, n in bad.items())
        )

    for c in mask_cols:
        out[c] = out[c].astype(int)

    out["spatial_split"] = np.select(
        [
            out["train_mask"] == 1,
            out["val_mask"] == 1,
            out["test_mask"] == 1,
        ],
        ["train", "val", "test"],
        default="outside_split",
    )
    return out

def load_pathA_table(
    root: str | Path,
    target_years: list[int],
    mask_file: str | Path | None = None,
    mask_seed: int = 1234,
    feature_mode: str = "current",
    window: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    if feature_mode == "current":
        df, feature_cols = load_years(root, target_years)
    elif feature_mode == "lagged":
        df, feature_cols = build_lagged_table(root, target_years, window=window)
    else:
        raise ValueError("feature_mode must be current or lagged")

    generated = build_tgnn_node_masks(df, mask_seed=mask_seed)
    if mask_file is None:
        return generated, feature_cols

    mask_df = load_spatial_mask_table(mask_file, target_years=target_years)
    attached = attach_spatial_masks(df, mask_df)
    mask_cols = ["label_mask", "train_mask", "val_mask", "test_mask"]
    mismatch = np.zeros(len(attached), dtype=bool)
    for col in mask_cols:
        mismatch |= (
            attached[col].to_numpy(dtype=int)
            != generated[col].to_numpy(dtype=int)
        )
    if mismatch.any():
        examples = attached.loc[
            mismatch, ["year", "NodeID"] + mask_cols
        ].head(10)
        raise ValueError(
            "external mask can't consists with dataset_AllMap_multitask.py，"
            f"rows={int(mismatch.sum())}.examples:\n{examples.to_string(index=False)}"
        )
    return attached, feature_cols

def spatial_subset(df: pd.DataFrame, split: str) -> pd.DataFrame:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    return df[
        (df["label_mask"].astype(int) == 1)
        & (df[f"{split}_mask"].astype(int) == 1)
    ].copy()


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    use_mask: bool = True,
    split: str | None = None,
):
    dat = df.copy()
    if split is not None:
        dat = spatial_subset(dat, split)
    elif use_mask and "label_mask" in dat.columns:
        dat = dat[dat["label_mask"].astype(int) == 1].copy()

    X = dat[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_cls = dat["label_presence"].astype(int).to_numpy()
    y_reg = dat["label_density"].astype(float).to_numpy()

    if len(dat) == 0:
        raise ValueError(f"split={split!r}have no available nodes")
    if not np.isin(y_cls, [0, 1]).all():
        raise ValueError(f"split={split!r}labels other than 0/1")
    if not np.isfinite(y_reg).all():
        raise ValueError(f"split={split!r}includes NaN/Inf density labels")

    return dat, X, y_cls, y_reg
