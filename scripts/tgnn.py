#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import copy
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    auc,
)

from dataset_AllMap_multitask import SAVFullGraphAnnualDatasetMultiTask

from modelAllMap.GCN_GRU_Temporal_MultiTask import MultiTask_GCN_GRU_Temporal
from modelAllMap.GAT_LSTM_Temporal_MultiTask import MultiTask_GAT_LSTM_Temporal
from modelAllMap.TGCN_Temporal_MultiTask import MultiTask_TGCN_Temporal
from modelAllMap.TGAT_Temporal_MultiTask import MultiTask_TGAT_Temporal
from modelAllMap.A3T_GCN_Temporal_MultiTask import MultiTask_A3TGCN_Temporal

from modelAllMap.GRU_Temporal_MultiTask import MultiTask_GRU_Temporal
from modelAllMap.GCN_Static_MultiTask import MultiTask_GCN_Static
from modelAllMap.MLP_Static_MultiTask import MultiTask_MLP_Static

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_model = 'GCN_GRU'
#exp_model = 'GAT_LSTM'
#exp_model = 'TGCN'
#exp_model = 'TGAT'
#exp_model = 'A3T_GCN'
#对比
#exp_model = 'GRU_ONLY'   #(no graph)
#exp_model = 'GCN_STATIC' #(no temporal)
#exp_model = 'MLP_STATIC' #(optional: no graph no temporal)

task_mode = "multi"   # "multi" | "cls" | "reg"
T_Window = 12

def parse_args():
    parser = argparse.ArgumentParser(
        description="Node-mask TGNN/ablation training (not strict temporal extrapolation)."
    )
    parser.add_argument(
        "--exp-model", default=exp_model,
        choices=["GCN_GRU", "GAT_LSTM", "TGCN", "TGAT", "A3T_GCN",
                 "GRU_ONLY", "GCN_STATIC", "MLP_STATIC"]
    )
    parser.add_argument("--task-mode", default=task_mode, choices=["multi", "cls", "reg"])
    parser.add_argument("--time-window", type=int, default=T_Window)
    parser.add_argument(
        "--root", default=None,
        help="Project root. Defaults to the parent of the script directory."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234) #
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--neg-keep-ratio", type=float, default=0.3)
    parser.add_argument(
        "--optimizer", choices=["adamw", "adam"], default="adamw",
        help="AdamW is recommended; Adam is retained for exact legacy comparison."
    )
    parser.add_argument(
        "--monitor",
        choices=["val_ap", "val_f1", "val_loss", "val_rmse"],
        default="val_ap",
        help="Validation-only checkpoint criterion. AP is recommended for imbalanced presence data."
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--no-shuffle-years", action="store_true",
        help="Disable deterministic shuffling of annual windows within each epoch."
    )


    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--optuna-epochs", type=int, default=60)
    parser.add_argument("--optuna-patience", type=int, default=12)
    parser.add_argument("--optuna-timeout", type=int, default=None)

    parser.add_argument("--optuna-lr-min", type=float, default=1e-4)
    parser.add_argument("--optuna-lr-max", type=float, default=1.5e-3)
    parser.add_argument("--optuna-wd-min", type=float, default=1e-5)
    parser.add_argument("--optuna-wd-max", type=float, default=5e-3)
    parser.add_argument(
        "--run-tag", default="",
        help="Optional checkpoint suffix, e.g. seed1234, for repeated-seed experiments."
    )
    return parser.parse_args()


def set_global_seed(seed):
    """Set the random states used by Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_args(args):

    if args.time_window < 1:
        raise ValueError("--time-window must be >= 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if not (0.0 < args.dropout < 1.0):
        raise ValueError("--dropout must be in (0, 1).")
    if not (0.0 < args.focal_alpha < 1.0):
        raise ValueError("--focal-alpha must be in (0, 1).")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be >= 0.")
    if args.lambda_reg < 0.0:
        raise ValueError("--lambda-reg must be >= 0.")
    if not (0.0 < args.neg_keep_ratio <= 1.0):
        raise ValueError("--neg-keep-ratio must be in (0, 1].")
    if not (0.0 < args.scheduler_factor < 1.0):
        raise ValueError("--scheduler-factor must be in (0, 1).")
    if args.optuna_trials < 0:
        raise ValueError("--optuna-trials must be >= 0.")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be > 0.")
    if not (0.0 < args.optuna_lr_min < args.optuna_lr_max):
        raise ValueError("Require 0 < --optuna-lr-min < --optuna-lr-max.")
    if not (0.0 < args.optuna_wd_min < args.optuna_wd_max):
        raise ValueError("Require 0 < --optuna-wd-min < --optuna-wd-max.")


def assert_disjoint_node_masks(data_list):

    for data in data_list:
        masks = {
            "train": data.train_mask.detach().cpu().bool().view(-1),
            "val": data.val_mask.detach().cpu().bool().view(-1),
            "test": data.test_mask.detach().cpu().bool().view(-1),
        }
        if not (len(masks["train"]) == len(masks["val"]) == len(masks["test"])):
            raise ValueError(f"Mask length mismatch in year {data.year}.")
        if (masks["train"] & masks["val"]).any():
            raise ValueError(f"train_mask overlaps val_mask in year {data.year}.")
        if (masks["train"] & masks["test"]).any():
            raise ValueError(f"train_mask overlaps test_mask in year {data.year}.")
        if (masks["val"] & masks["test"]).any():
            raise ValueError(f"val_mask overlaps test_mask in year {data.year}.")


class NonFiniteTrainingError(FloatingPointError):
    """Raised when NaN/Inf is found in data, outputs, loss, gradients or weights."""
def _year_text(year):
    if year is None:
        return "unknown"
    if torch.is_tensor(year):
        return str(year.detach().cpu().item())
    return str(year)


def require_finite_tensor(name, tensor, stage, year=None):
    """Fail with a compact, attributable diagnostic instead of hiding NaNs."""
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return

    bad = ~finite
    n_bad = int(bad.sum().detach().cpu().item())
    first_bad = torch.nonzero(bad, as_tuple=False)[0].detach().cpu().tolist()
    finite_values = tensor[finite]
    if finite_values.numel() > 0:
        vmin = float(finite_values.min().detach().cpu().item())
        vmax = float(finite_values.max().detach().cpu().item())
        finite_range = f"finite_range=[{vmin:.6g}, {vmax:.6g}]"
    else:
        finite_range = "finite_range=empty"
    raise NonFiniteTrainingError(
        f"stage={stage}, year={_year_text(year)}, tensor={name}, "
        f"nonfinite={n_bad}/{tensor.numel()}, first_bad_index={first_bad}, "
        f"{finite_range}"
    )


def validate_dataset_finite(data_list):
    """Check normalized inputs/continuous targets once before optimization."""
    for data in data_list:
        year = getattr(data, "year", None)
        require_finite_tensor("data.x", data.x, "dataset_check", year)
        supervised = (
            data.train_mask.detach().cpu().bool().view(-1)
            | data.val_mask.detach().cpu().bool().view(-1)
            | data.test_mask.detach().cpu().bool().view(-1)
        )
        require_finite_tensor(
            "data.y_reg[supervised]",
            data.y_reg.detach().cpu().view(-1)[supervised],
            "dataset_check",
            year,
        )

        y_cls = data.y_cls.detach().cpu().view(-1)[supervised]
        y_reg_cls = data.y_reg_cls.detach().cpu().view(-1)[supervised]
        if not bool(((y_cls == 0) | (y_cls == 1)).all()):
            raise ValueError(
                f"Invalid y_cls outside {{0,1}} in year {_year_text(year)}."
            )
        if not bool(((y_reg_cls >= 0) & (y_reg_cls <= 4)).all()):
            raise ValueError(
                f"Invalid y_reg_cls outside [0,4] in year {_year_text(year)}."
            )


def nonfinite_parameter_names(model):
    return [
        name for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
        and not bool(torch.isfinite(parameter).all())
    ]


def nonfinite_gradient_names(model):
    return [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and parameter.grad.is_floating_point()
        and not bool(torch.isfinite(parameter.grad).all())
    ]

class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: [N,2]
        target: [N] (0/1)
        """
        ce = F.cross_entropy(logits, target, reduction='none')  # [N]
        pt = torch.exp(-ce)
        at = torch.where(target == 1,
                         torch.full_like(pt, self.alpha),
                         torch.full_like(pt, 1 - self.alpha))
        loss = at * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def binary_f1_from_label(pred, target):

    pred = pred.view(-1).cpu()
    target = target.view(-1).cpu()
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def binary_f1_from_prob(prob_pos, target, threshold=0.5):

    pred = (prob_pos >= threshold).long()
    return binary_f1_from_label(pred, target)

def build_window(data_list, idx, T):

    assert idx >= T - 1
    window = data_list[idx - T + 1: idx + 1]
    x_seq = torch.stack([d.x for d in window], dim=0)  # [T,N,F]
    last  = window[-1]
    return (x_seq,
            last.y_cls,       # (0/1)
            last.y_reg,       # (0,0.05,0.25,0.55,0.85)
            last.y_reg_cls,   # (0..4)
            last.train_mask,
            last.val_mask,
            last.test_mask,
            last.year)

#============================================================

def forward_model(model, exp_model, x_seq, edge_index):

    if exp_model in {"GCN_GRU", "GAT_LSTM", "TGCN", "TGAT", "A3T_GCN"}:
        return model(x_seq, edge_index)
    elif exp_model == "GRU_ONLY":
        return model(x_seq)
    elif exp_model == "GCN_STATIC":
        x_last = x_seq[-1]  # [N,F]
        return model(x_last, edge_index)
    elif exp_model == "MLP_STATIC":
        x_last = x_seq[-1]
        return model(x_last)
    else:
        raise ValueError(f"Unknown exp_model: {exp_model}")

def train_one_step(model, optimizer,
                   x_seq, edge_index,
                   y_cls, y_reg, y_reg_cls,
                   train_mask,
                   focal_loss_fn,
                   level_values,
                   lambda_reg=0.1,
                   neg_keep_ratio=0.3,
                   grad_clip=1.0,
                   year=None):

    model.train()

    x_seq      = x_seq.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    y_cls      = y_cls.to(DEVICE)
    y_reg      = y_reg.to(DEVICE)
    y_reg_cls  = y_reg_cls.to(DEVICE)
    train_mask = train_mask.to(DEVICE)

    logits_p, logits_d = forward_model(model, exp_model, x_seq, edge_index)

    require_finite_tensor("logits_presence", logits_p, "train_forward", year)
    require_finite_tensor("logits_density", logits_d, "train_forward", year)


    idx_all = torch.where(train_mask)[0]
    if len(idx_all) == 0:
        return None


    with torch.no_grad():

        cls_all = y_cls[idx_all]
        idx_zero = idx_all[(cls_all == 0)]
        idx_pos = idx_all[(cls_all == 1)]

        keep_zero = idx_zero
        if len(idx_zero) > 0 and neg_keep_ratio < 1.0:
            k = max(1, int(len(idx_zero) * neg_keep_ratio))

            perm = torch.randperm(len(idx_zero), device=idx_zero.device)[:k]
            keep_zero = idx_zero[perm]

        idx_use = torch.cat([keep_zero, idx_pos], dim=0)
        if len(idx_use) > 1:
            idx_use = idx_use[
                torch.randperm(len(idx_use), device=idx_use.device)
            ]

    logits_m_cls = logits_p[idx_use]       # [M,2]
    y_cls_m = y_cls[idx_use]               # [M]

    logits_m_reg = logits_d[idx_all]       # [N_train,5]
    y_reg_m = y_reg[idx_all]               # [N_train]
    y_reg_cls_m = y_reg_cls[idx_all]       # [N_train]

    # Focal Loss ----
    loss_cls = focal_loss_fn(logits_m_cls, y_cls_m)

    # Ordinal CE
    loss_reg = F.cross_entropy(logits_m_reg, y_reg_cls_m)

    if task_mode == "cls":
        loss = loss_cls
    elif task_mode == "reg":
        loss = loss_reg
    else:
        loss = loss_cls + lambda_reg * loss_reg

    require_finite_tensor("loss_cls", loss_cls, "train_loss", year)
    require_finite_tensor("loss_reg", loss_reg, "train_loss", year)
    require_finite_tensor("loss_total", loss, "train_loss", year)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    bad_gradients = nonfinite_gradient_names(model)
    if bad_gradients:
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteTrainingError(
            f"stage=train_backward, year={_year_text(year)}, "
            f"nonfinite_gradients={bad_gradients[:12]}"
        )
    if grad_clip is not None and grad_clip > 0:
        try:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
                error_if_nonfinite=True,
            )
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            raise NonFiniteTrainingError(
                f"stage=gradient_clip, year={_year_text(year)}, error={exc}"
            ) from exc
        require_finite_tensor(
            "total_grad_norm", total_grad_norm, "gradient_clip", year
        )
    optimizer.step()

    bad_parameters = nonfinite_parameter_names(model)
    if bad_parameters:
        raise NonFiniteTrainingError(
            f"stage=optimizer_step, year={_year_text(year)}, "
            f"nonfinite_parameters={bad_parameters[:12]}"
        )


    pred_label = logits_m_cls.argmax(dim=-1)
    acc = (pred_label == y_cls_m).float().mean().item()

    prob_pos = torch.softmax(logits_m_cls, dim=-1)[:, 1]
    require_finite_tensor("prob_presence", prob_pos, "train_metric", year)
    best_f1 = 0.0
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        f1_th = binary_f1_from_prob(prob_pos, y_cls_m, threshold=th)
        if f1_th > best_f1:
            best_f1 = f1_th
    f1 = best_f1


    prob_reg = torch.softmax(logits_m_reg, dim=-1)               # [M,5]
    pred_d_cont = (prob_reg * level_values.unsqueeze(0)).sum(dim=-1)  # [M]
    require_finite_tensor("pred_density", pred_d_cont, "train_metric", year)
    rmse = torch.sqrt(F.mse_loss(pred_d_cont, y_reg_m)).item()

    return loss.item(), acc, f1, rmse

@torch.no_grad()
def eval_one_step(model,
                  x_seq, edge_index,
                  y_cls, y_reg, y_reg_cls,
                  mask,
                  focal_loss_fn,
                  level_values,
                  lambda_reg=0.1,
                  year=None):
    model.eval()

    x_seq      = x_seq.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    y_cls      = y_cls.to(DEVICE)
    y_reg      = y_reg.to(DEVICE)
    y_reg_cls  = y_reg_cls.to(DEVICE)
    mask       = mask.to(DEVICE)

    idx_all = torch.where(mask)[0]
    if len(idx_all) == 0:
        return None

    logits_p, logits_d = forward_model(model, exp_model, x_seq, edge_index)
    require_finite_tensor("logits_presence", logits_p, "eval_forward", year)
    require_finite_tensor("logits_density", logits_d, "eval_forward", year)

    logits_m_cls = logits_p[idx_all]
    logits_m_reg = logits_d[idx_all]
    y_cls_m      = y_cls[idx_all]
    y_reg_m      = y_reg[idx_all]
    y_reg_cls_m  = y_reg_cls[idx_all]

    loss_cls = focal_loss_fn(logits_m_cls, y_cls_m)
    loss_reg = F.cross_entropy(logits_m_reg, y_reg_cls_m)

    if task_mode == "cls":
        loss = loss_cls
    elif task_mode == "reg":
        loss = loss_reg
    else:
        loss = loss_cls + lambda_reg * loss_reg
    require_finite_tensor("loss_cls", loss_cls, "eval_loss", year)
    require_finite_tensor("loss_reg", loss_reg, "eval_loss", year)
    require_finite_tensor("loss_total", loss, "eval_loss", year)

    pred_label = logits_m_cls.argmax(dim=-1)
    acc = (pred_label == y_cls_m).float().mean().item()

    prob_pos = torch.softmax(logits_m_cls, dim=-1)[:, 1]
    require_finite_tensor("prob_presence", prob_pos, "eval_metric", year)

    prob_reg = torch.softmax(logits_m_reg, dim=-1)
    pred_d_cont = (prob_reg * level_values.unsqueeze(0)).sum(dim=-1)
    require_finite_tensor("pred_density", pred_d_cont, "eval_metric", year)
    rmse = torch.sqrt(F.mse_loss(pred_d_cont, y_reg_m)).item()

    return {
        "loss": float(loss.item()),
        "n": int(len(idx_all)),
        "acc": float(acc),
        "rmse": float(rmse),
        "prob": prob_pos.detach().cpu().numpy(),
        "node_idx": idx_all.detach().cpu().numpy(),
        "y_cls": y_cls_m.detach().cpu().numpy(),
        "pred_density": pred_d_cont.detach().cpu().numpy(),
        "y_reg": y_reg_m.detach().cpu().numpy(),
    }


def optimal_f1_threshold(prob, y_true):
    """Select one F1-optimal threshold from pooled validation predictions."""
    prob = np.asarray(prob, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    finite = np.isfinite(prob)
    if not np.all(finite):
        bad_positions = np.flatnonzero(~finite)[:20].tolist()
        raise NonFiniteTrainingError(
            f"stage=pooled_pr, tensor=prob_presence, "
            f"nonfinite={int((~finite).sum())}/{len(prob)}, "
            f"first_bad_pooled_positions={bad_positions}"
        )
    if len(prob) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan"), float("nan"), float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5, float("nan"), float("nan"), float("nan")

    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1_t = 2.0 * precision_t * recall_t / (precision_t + recall_t + 1e-12)
    best_idx = int(np.nanargmax(f1_t))
    return (
        float(thresholds[best_idx]),
        float(f1_t[best_idx]),
        float(recall_t[best_idx]),
        float(precision_t[best_idx]),
    )


def evaluate_all_windows(model, data_list, window_indices, edge_index,
                         focal_loss_fn, level_values, lambda_reg, split="val"):

    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    rows = []
    for idx in window_indices:
        (x_seq, y_cls, y_reg, y_reg_cls,
         train_mask, val_mask, test_mask, year) = build_window(data_list, idx, T_Window)
        mask = val_mask if split == "val" else test_mask
        result = eval_one_step(
            model, x_seq, edge_index,
            y_cls, y_reg, y_reg_cls,
            mask, focal_loss_fn, level_values,
            lambda_reg=lambda_reg,
            year=year,
        )
        if result is not None:
            result["year"] = int(year)
            rows.append(result)

    if not rows:
        return {
            "val_loss": float("nan"), "val_ap": float("nan"),"val_pr_auc": float("nan"),
            "val_f1": float("nan"), "val_recall": float("nan"),
            "val_precision": float("nan"), "val_rmse": float("nan"),
            "val_tau": 0.5, "val_n": 0, "val_n_positive": 0,
            "val_prevalence": float("nan"),
        }

    total_n = sum(row["n"] for row in rows)
    weighted_loss = sum(row["loss"] * row["n"] for row in rows) / total_n
    prob = np.concatenate([row["prob"] for row in rows])
    y_cls = np.concatenate([row["y_cls"] for row in rows]).astype(int)
    pred_density = np.concatenate([row["pred_density"] for row in rows])
    y_reg = np.concatenate([row["y_reg"] for row in rows])

    if len(np.unique(y_cls)) < 2:
        ap = float("nan")
        pr_auc = float("nan")
    else:
        ap = float(average_precision_score(y_cls, prob))
        pr_precision, pr_recall, _ = precision_recall_curve(
        y_cls, prob)
        pr_auc = float(auc(pr_recall, pr_precision))

    tau, f1, recall, precision = optimal_f1_threshold(prob, y_cls)
    rmse = float(np.sqrt(np.mean((pred_density - y_reg) ** 2)))
    if not np.isfinite(rmse):
        raise NonFiniteTrainingError(
            "stage=pooled_rmse, tensor=pred_density_or_y_reg, result=nonfinite"
        )

    return {
        "val_loss": float(weighted_loss),
        "val_ap": ap,
        "val_pr_auc": pr_auc,
        "val_f1": f1,
        "val_recall": recall,
        "val_precision": precision,
        "val_rmse": rmse,
        "val_tau": tau,
        "val_n": int(total_n),
        "val_n_positive": int(np.sum(y_cls == 1)),
        "val_prevalence": float(np.mean(y_cls == 1)),
    }

def build_model(exp_model_name, in_dim, dropout):
    if exp_model_name == 'GCN_GRU':
        return MultiTask_GCN_GRU_Temporal(
            in_channels=in_dim,
            gcn_hidden=64,
            gcn_out=64,
            gru_hidden=64,
            gru_layers=1,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'GAT_LSTM':
        return MultiTask_GAT_LSTM_Temporal(
            in_channels=in_dim,
            gat_hidden=64,
            gat_out=64,
            lstm_hidden=64,
            lstm_layers=1,
            heads=4,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'TGCN':
        return MultiTask_TGCN_Temporal(
            in_channels=in_dim,
            gcn_hidden=64,
            gcn_out=64,
            tgcn_hidden=64,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'TGAT':
        return MultiTask_TGAT_Temporal(
            in_channels=in_dim,
            gat_hidden=64,
            gat_out=64,
            gru_hidden=64,
            gru_layers=1,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'A3T_GCN':
        return MultiTask_A3TGCN_Temporal(
            in_channels=in_dim,
            gcn_hidden=64,
            gcn_out=64,
            att_hidden=64,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'GRU_ONLY':
        return MultiTask_GRU_Temporal(
            in_channels=in_dim,
            gru_hidden=64,
            gru_layers=1,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'GCN_STATIC':
        return MultiTask_GCN_Static(
            in_channels=in_dim,
            gcn_hidden=64,
            gcn_out=64,
            dropout=dropout,
        ).to(DEVICE)
    if exp_model_name == 'MLP_STATIC':
        return MultiTask_MLP_Static(
            in_channels=in_dim,
            hidden=128,
            dropout=dropout,
        ).to(DEVICE)
    raise ValueError(f"Unknown exp_model: {exp_model_name}")


def build_optimizer(model, config):
    cls = torch.optim.AdamW if config["optimizer"] == "adamw" else torch.optim.Adam
    return cls(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )


def monitor_mode(monitor):
    return "min" if monitor in {"val_loss", "val_rmse"} else "max"


def is_better(value, best_value, mode, min_delta):
    if not np.isfinite(value):
        return False
    if mode == "min":
        return value < best_value - min_delta
    return value > best_value + min_delta


def fit_once(args, config, data_list, edge_index, in_dim,
             num_epochs, patience, trial=None, verbose=True,
             recovery_path=None):
    """Train one reproducible run using validation data only for selection."""
    set_global_seed(args.seed)
    model = build_model(exp_model, in_dim, config["dropout"])
    optimizer = build_optimizer(model, config)
    focal_loss_fn = FocalLoss(
        gamma=config["focal_gamma"], alpha=config["focal_alpha"]
    )
    level_values = torch.tensor(
        [0.0, 0.05, 0.25, 0.55, 0.85],
        dtype=torch.float32,
        device=DEVICE,
    )

    mode = monitor_mode(args.monitor)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    window_indices = list(range(T_Window - 1, len(data_list)))
    if not window_indices:
        raise ValueError(
            f"No temporal windows available: T={T_Window}, years={len(data_list)}."
        )

    best_value = float("inf") if mode == "min" else -float("inf")
    best_epoch = 0
    best_state = None
    best_val = None
    epochs_without_improvement = 0
    history = []
    numerical_stop_reason = None

    for epoch in range(1, num_epochs + 1):

        epoch_indices = list(window_indices)
        if not args.no_shuffle_years:
            rng = np.random.default_rng(args.seed + epoch)
            rng.shuffle(epoch_indices)

        train_stats = []
        for idx in epoch_indices:
            (x_seq, y_cls, y_reg, y_reg_cls,
             train_mask, val_mask, test_mask, year) = build_window(data_list, idx, T_Window)
            try:
                result = train_one_step(
                    model, optimizer,
                    x_seq, edge_index,
                    y_cls, y_reg, y_reg_cls,
                    train_mask,
                    focal_loss_fn,
                    level_values,
                    lambda_reg=config["lambda_reg"],
                    neg_keep_ratio=config["neg_keep_ratio"],
                    grad_clip=args.grad_clip,
                    year=year,
                )
            except NonFiniteTrainingError as exc:
                numerical_stop_reason = f"epoch={epoch}, {exc}"
                if trial is not None:
                    trial.set_user_attr(
                        "numerical_failure", numerical_stop_reason
                    )
                    import optuna
                    raise optuna.TrialPruned(numerical_stop_reason) from exc
                if verbose:
                    print(f"[NUMERIC STOP] {numerical_stop_reason}")
                break
            if result is not None:
                train_stats.append(result)

        if numerical_stop_reason is not None:
            break

        try:
            val = evaluate_all_windows(
                model, data_list, window_indices, edge_index,
                focal_loss_fn, level_values,
                lambda_reg=config["lambda_reg"], split="val",
            )
        except NonFiniteTrainingError as exc:
            numerical_stop_reason = f"epoch={epoch}, {exc}"
            if trial is not None:
                trial.set_user_attr(
                    "numerical_failure", numerical_stop_reason
                )
                import optuna
                raise optuna.TrialPruned(numerical_stop_reason) from exc
            if verbose:
                print(f"[NUMERIC STOP] {numerical_stop_reason}")
            break
        current_value = float(val[args.monitor])
        if np.isfinite(current_value):
            scheduler.step(current_value)

        if train_stats:
            train_array = np.asarray(train_stats, dtype=float)
            tr_loss, tr_acc, tr_f1, tr_rmse = train_array.mean(axis=0).tolist()
        else:
            tr_loss = tr_acc = tr_f1 = tr_rmse = float("nan")

        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr_loss,
            "train_acc_diag": tr_acc,
            "train_f1_diag": tr_f1,
            "train_rmse_diag": tr_rmse,
            **val,
        }
        history.append(row)

        if verbose:
            print(
                f"[Epoch {epoch:03d}] TrainLoss={tr_loss:.4f} | "
                f"ValLoss={val['val_loss']:.4f} AP={val['val_ap']:.4f} "
                f"PR_AUC={val['val_pr_auc']:.4f} "
                f"F1={val['val_f1']:.4f} R={val['val_recall']:.4f} "
                f"P={val['val_precision']:.4f} RMSE={val['val_rmse']:.4f} "
                f"tau={val['val_tau']:.4f} lr={lr_now:.2e}"
            )

        if is_better(current_value, best_value, mode, args.min_delta):
            best_value = current_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val = dict(val)
            epochs_without_improvement = 0

            if recovery_path is not None:
                torch.save(best_state, recovery_path)
        else:
            epochs_without_improvement += 1

        if trial is not None:
            trial.report(current_value, step=epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if patience > 0 and epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"[EARLY STOP] no {args.monitor} improvement for {patience} "
                    f"epochs; best_epoch={best_epoch}."
                )
            break

    if best_state is None:
        raise RuntimeError(
            f"No finite validation metric was observed for monitor={args.monitor}. "
            f"Numerical diagnostic: {numerical_stop_reason}"
        )

    model.load_state_dict(best_state)
    if numerical_stop_reason is not None and verbose:
        print(
            f"[RECOVERY] Restored best_epoch={best_epoch}; "
            f"{args.monitor}={best_value:.6f}."
        )
    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_monitor_value": float(best_value),
        "best_val": best_val,
        "history": history,
        "numerical_stop_reason": numerical_stop_reason,
    }


def base_training_config(args):
    return {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "lambda_reg": args.lambda_reg,
        "neg_keep_ratio": args.neg_keep_ratio,
        "optimizer": args.optimizer,
    }


def run_optuna_search(args, base_config, data_list, edge_index, in_dim):

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "--optuna-trials > 0 requires Optuna. Install it in the training "
            "environment or run with --optuna-trials 0."
        ) from exc

    direction = "minimize" if monitor_mode(args.monitor) == "min" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, max(1, args.optuna_trials)),
        n_warmup_steps=min(10, max(1, args.optuna_epochs // 4)),
    )
    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner
    )

    def objective(trial):
        config = dict(base_config)
        config.update({

            "lr": trial.suggest_float(
                "lr", args.optuna_lr_min, args.optuna_lr_max, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", args.optuna_wd_min, args.optuna_wd_max,
                log=True
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.40),
            "optimizer": trial.suggest_categorical("optimizer", ["adamw", "adam"]),
            "neg_keep_ratio": trial.suggest_categorical(
                "neg_keep_ratio", [0.3, 0.5, 0.75, 1.0]
            ),
        })
        if task_mode != "reg":
            config["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 2.5)
            config["focal_alpha"] = trial.suggest_float("focal_alpha", 0.50, 0.80)
        if task_mode == "multi":
            config["lambda_reg"] = trial.suggest_float(
                "lambda_reg", 0.03, 0.50, log=True
            )

        result = fit_once(
            args, config, data_list, edge_index, in_dim,
            num_epochs=args.optuna_epochs,
            patience=args.optuna_patience,
            trial=trial,
            verbose=False,
        )
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("best_val", result["best_val"])
        return result["best_monitor_value"]

    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=args.optuna_timeout,
        gc_after_trial=True,
    )

    completed_trials = [
        item for item in study.trials
        if item.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError(
            "Optuna produced no completed finite trial. Inspect the "
            "[NUMERIC STOP] diagnostics before widening the search space."
        )

    tuned_config = dict(base_config)
    tuned_config.update(study.best_trial.params)
    summary = {
        "direction": direction,
        "monitor": args.monitor,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": dict(study.best_trial.params),
        "n_trials_total": int(len(study.trials)),
        "n_trials_completed": int(len(completed_trials)),
        "n_trials_pruned": int(sum(
            item.state == optuna.trial.TrialState.PRUNED
            for item in study.trials
        )),
        "search_bounds": {
            "lr": [args.optuna_lr_min, args.optuna_lr_max],
            "weight_decay": [args.optuna_wd_min, args.optuna_wd_max],
        },
    }
    return tuned_config, summary


def write_history_csv(path, history):
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# ============================================================
#  Train
# ============================================================
def main():

    global exp_model, task_mode, T_Window
    args = parse_args()
    validate_args(args)
    exp_model = args.exp_model
    task_mode = args.task_mode
    T_Window = args.time_window

    if task_mode == "reg" and args.monitor in {"val_ap", "val_f1"}:
        print(
            f"[WARN] monitor={args.monitor} is undefined for a regression-only run; "
            "using val_rmse."
        )
        args.monitor = "val_rmse"

    set_global_seed(args.seed)

    root = (
        os.path.abspath(args.root)
        if args.root
        else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dataset = SAVFullGraphAnnualDatasetMultiTask(root=root, do_norm=True)
    data_list = sorted(dataset.data_list, key=lambda d: d.year)
    if not data_list:
        raise ValueError("Dataset contains no annual graphs.")
    assert_disjoint_node_masks(data_list)
    validate_dataset_finite(data_list)
    print("[NUMERIC CHECK] Dataset features and supervised targets are finite.")

    print(
        "[DESIGN] node-mask evaluation on full annual graphs; "
        "this run is not a strict temporal holdout."
    )
    print(
        f"[CONFIG] model={exp_model}, task={task_mode}, T={T_Window}, "
        f"seed={args.seed}, epochs={args.epochs}, patience={args.patience}, "
        f"monitor={args.monitor}, optuna_trials={args.optuna_trials}"
    )

    edge_file = os.path.join(root, "results", "edge_index_allMap.pth")
    edge_index = torch.load(edge_file).to(DEVICE)
    in_dim = data_list[0].x.shape[1]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
        )
    n_nodes = int(data_list[0].x.shape[0])
    if edge_index.numel() > 0:
        edge_min = int(edge_index.min().detach().cpu().item())
        edge_max = int(edge_index.max().detach().cpu().item())
        if edge_min < 0 or edge_max >= n_nodes:
            raise ValueError(
                f"edge_index outside [0, {n_nodes - 1}]: "
                f"min={edge_min}, max={edge_max}."
            )

    log_dir = os.path.join(root, "results", "training_logs")
    os.makedirs(log_dir, exist_ok=True)
    run_tag = f"_{args.run_tag}" if args.run_tag else ""

    config = base_training_config(args)
    optuna_summary = None
    if args.optuna_trials > 0:
        print(f"[OPTUNA] Starting {args.optuna_trials} validation-only trials...")
        config, optuna_summary = run_optuna_search(
            args, config, data_list, edge_index, in_dim
        )
        print(f"[OPTUNA] Selected training config: {config}")

        optuna_search_path = os.path.join(
            log_dir,
            f"optuna_{exp_model}_{task_mode}_T{T_Window}{run_tag}_summary.json",
        )
        with open(optuna_search_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "selected_training_config": config,
                    "optuna": optuna_summary,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[OPTUNA] Search summary saved to {optuna_search_path}")
    else:
        optuna_search_path = None

    lam_tag = f"{config['lambda_reg']:.4g}"
    neg_tag = f"{config['neg_keep_ratio']:.4g}"
    base_name = (
        f"model_{exp_model}_{task_mode}_T{T_Window}_lam{lam_tag}_neg{neg_tag}"
        f"{run_tag}_OptimizedStable"
    )
    out_path = os.path.join(root, f"{base_name}.pth")

    final_result = fit_once(
        args, config, data_list, edge_index, in_dim,
        num_epochs=args.epochs,
        patience=args.patience,
        trial=None,
        verbose=True,
        recovery_path=out_path,
    )

    model = final_result["model"]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable_parameters={n_params:,}")

    torch.save(final_result["best_state"], out_path)

    history_path = os.path.join(log_dir, f"{base_name}_history.csv")
    config_path = os.path.join(log_dir, f"{base_name}_config.json")
    write_history_csv(history_path, final_result["history"])

    metadata = {
        "design": "annual full graphs with disjoint node masks; not strict temporal holdout",
        "project_root": root,
        "model": exp_model,
        "task_mode": task_mode,
        "time_window": T_Window,
        "seed": args.seed,
        "monitor": args.monitor,
        "trainable_parameters": int(n_params),
        "training_config": config,
        "best_epoch": int(final_result["best_epoch"]),
        "best_monitor_value": float(final_result["best_monitor_value"]),
        "best_validation_metrics": final_result["best_val"],
        "checkpoint": out_path,
        "history_csv": history_path,
        "optuna": optuna_summary,
        "optuna_summary_json": optuna_search_path,
        "numerical_stop_reason": final_result["numerical_stop_reason"],
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    print(
        f"✅ Best-validation checkpoint saved to {out_path}\n"
        f"   best_epoch={final_result['best_epoch']}, "
        f"{args.monitor}={final_result['best_monitor_value']:.6f}\n"
        f"   history={history_path}\n"
        f"   config={config_path}"
    )

if __name__ == "__main__":
    main()
