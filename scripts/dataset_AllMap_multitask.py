# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import pandas as pd
import torch
from torch_geometric.data import Data

class SAVFullGraphAnnualDatasetMultiTask:

    def __init__(self, root: str, do_norm: bool = True):
        self.root = root
        self.env_dir = os.path.join(root, "data", "raw", "AnnualFeaturesforTGNN")
        self.label_dir = os.path.join(root, "data", "raw", "AnnualLabelsforTGNN")
        self.data_list = []
        self.do_norm = do_norm

        self.feat_mean = None
        self.feat_std = None

        self._build_dataset()

    def _compute_global_stats(self, env_files):

        feat_list = []
        feat_cols_ref = None

        for f_env in env_files:
            df_env = pd.read_csv(f_env)
            df_env.columns = [c.strip() for c in df_env.columns]
            df_env = df_env.loc[:, ~df_env.columns.str.contains("^Unnamed")]

            drop_cols = [
                "NodeID", "lon", "lat", "Longitude", "Latitude" , "dist_to_coast","Depth"
            ]
            feat_cols = [c for c in df_env.columns if c not in drop_cols]

            if feat_cols_ref is None:
                feat_cols_ref = feat_cols
            else:

                if feat_cols != feat_cols_ref:
                    print(f"⚠️ feature column isn't consist: {os.path.basename(f_env)}")

            feat_list.append(df_env[feat_cols])

        all_feats = pd.concat(feat_list, axis=0, ignore_index=True)
        self.feat_mean = all_feats.mean(axis=0)
        self.feat_std = all_feats.std(axis=0).replace(0, 1.0)  # 防止除 0
        for k, v in self.feat_std.items():
            print(k, v)

        self.feat_cols = feat_cols_ref
        print("✅ standardization is completed.")

    def _build_dataset(self):
        # 找所有 EnvGNN_YYYY.csv
        all_env_files = sorted(glob.glob(os.path.join(self.env_dir, "EnvGNN_*.csv")))
        if len(all_env_files) == 0:
            raise FileNotFoundError("AnnualFeaturesforTGNN can't found EnvGNN_*.csv")

        env_files = []
        for f in all_env_files:
            m = re.search(r"EnvGNN_(\d{4})\.csv$", os.path.basename(f))
            if not m:
                continue
            year = int(m.group(1))

            label_file = os.path.join(self.label_dir, f"labels_{year}.csv")
            if not os.path.exists(label_file):
                continue

            env_files.append(f)

        if self.do_norm:
            self._compute_global_stats(env_files)
        else:

            df0 = pd.read_csv(env_files[0])
            drop_cols = ["NodeID", "lon", "lat", "Longitude", "Latitude"]
            self.feat_cols = [c for c in df0.columns if c not in drop_cols]


        for f_env in env_files:
            basename = os.path.basename(f_env)
            yr = int(re.findall(r"(\d{4})", basename)[0])

            f_label = os.path.join(self.label_dir, f"labels_{yr}.csv")
            if not os.path.exists(f_label):
                print(f"⚠️ can't found label files: {f_label}，jump {yr}")
                continue

            print(f"📂 construct {yr}  Data ")

            df_env = pd.read_csv(f_env)
            df_lab = pd.read_csv(f_label)

            if "NodeID" not in df_env.columns or "NodeID" not in df_lab.columns:
                raise ValueError(f"Env/Label lack of NodeID column: {yr}")

            df = pd.merge(df_env, df_lab,
                          on="NodeID",
                          how="inner",
                          suffixes=("", "_lab"))

            feat_cols = self.feat_cols
            x_raw = df[feat_cols].values

            if self.do_norm:
                x = (x_raw - self.feat_mean[feat_cols].values) / self.feat_std[feat_cols].values
            else:
                x = x_raw

            x = torch.tensor(x, dtype=torch.float32)

            # 标签列 (presence + density)
            if "label_presence" not in df.columns or "label_density" not in df.columns:
                raise ValueError(f"labels_{yr}.csv lack of label_presence or label_density column")

            y_cls = torch.tensor(df["label_presence"].values, dtype=torch.long)
            y_reg = torch.tensor(df["label_density"].values, dtype=torch.float32)

            level_values = torch.tensor([0.0, 0.05, 0.25, 0.55, 0.85], dtype=torch.float32)

            v = y_reg.unsqueeze(1)                             # [N,1]
            diff = torch.abs(v - level_values.unsqueeze(0))    # [N,5]
            y_reg_cls = torch.argmin(diff, dim=1).long()       # [N], 取 0~4

            if "label_mask" in df.columns:
                mask_label = df["label_mask"].values.astype(int)
            else:
                mask_label = (y_cls >= 0).astype(int)

            N = x.shape[0]
            train_mask = torch.zeros(N, dtype=torch.bool)
            val_mask = torch.zeros(N, dtype=torch.bool)
            test_mask = torch.zeros(N, dtype=torch.bool)

            idx_all = torch.where(torch.tensor(mask_label) == 1)[0]
            if len(idx_all) > 0:
                g = torch.Generator().manual_seed(1234)
                perm = idx_all[torch.randperm(len(idx_all), generator=g)]

                n_train = int(0.7 * len(perm))
                n_val = int(0.15 * len(perm))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:n_train + n_val]
                test_idx = perm[n_train + n_val:]

                train_mask[train_idx] = True
                val_mask[val_idx] = True
                test_mask[test_idx] = True

            data = Data(
                x=x,
                y_cls=y_cls,
                y_reg=y_reg,
                y_reg_cls=y_reg_cls,

                node_id=torch.tensor(
                    df["NodeID"].values,
                    dtype=torch.long
                ),
                label_mask=torch.tensor(
                    mask_label == 1,
                    dtype=torch.bool
                ),
                
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            data.year = yr

            self.data_list.append(data)
