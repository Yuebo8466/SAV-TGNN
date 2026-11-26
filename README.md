# SAV-TGNN

Code for joint prediction of submerged aquatic vegetation (SAV) occurrence and density using a temporal graph neural network (TGNN), with neural and tree-based baselines.

## 1. Installation

Use Python 3.10 or later in a separate environment. Clone the repository and install the dependencies:

```bash
git clone https://github.com/Yuebo8466/SAV-TGNN.git
cd SAV-TGNN
python -m pip install torch torch-geometric numpy pandas scikit-learn joblib xgboost
```

For GPU use, install a PyTorch build compatible with your CUDA environment. The TGNN script automatically selects CUDA when available; otherwise, it uses the CPU.

## 2. Required files

Keep the following paths relative to the repository root:

| Path | Contents |
| --- | --- |
| `data/raw/AnnualFeaturesforTGNN/EnvGNN_YYYY.csv` | Annual environmental features |
| `data/raw/AnnualLabelsforTGNN/labels_YYYY.csv` | Annual occurrence and density labels |
| `results/edge_index_allMap.pth` | Graph connectivity |
| `modelAllMap/` | TGNN, GRU, GCN and MLP model definitions |
| `scripts/` | Training, data loading and evaluation scripts |

Environmental and label files must share `NodeID` values and consistent node ordering across years. 

## 3. Run TGNN and neural baselines

Run all commands from the repository root. First add the repository root to the Python import path.

Linux/macOS:

```bash
export PYTHONPATH="$PWD"
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Train TGNN:

```bash
python scripts/tgnn.py --exp-model GCN_GRU --task-mode multi --time-window 12 --epochs 200 --seed 1234 --run-tag seed1234
```

Train the neural baselines using the same window:

```bash
python scripts/tgnn.py --exp-model GRU_ONLY --time-window 12 --run-tag seed1234
python scripts/tgnn.py --exp-model GCN_STATIC --time-window 12 --run-tag seed1234
python scripts/tgnn.py --exp-model MLP_STATIC --time-window 12 --run-tag seed1234
```

Change `--time-window` to the desired number of annual inputs. Static models use only the final year of each window; using the same window aligns the target years. For classification-only or density-only experiments, use `--task-mode cls` or `--task-mode reg`. Use distinct `--run-tag` values to avoid overwriting runs with otherwise identical output filenames.

Model checkpoints are saved in the repository root. Training histories and configuration files are saved in `results/training_logs/`. This training entry point saves validation results; it does not automatically export final TGNN test predictions.

## 4. Run tree-based baselines

run Random Forest, boosted regression trees and XGBoost:

```bash
python scripts/run_sdm_python_baselines.py --years 1999:2024 --time-window 12 --feature-mode current --mask-seed 1234 --seed 1234
```

Use `--no-xgb` to run without XGBoost. Match `--time-window` to the TGNN run. The example explicitly uses the currently available labeled years; update `--years` if additional labels are provided.

Metrics, predictions and fitted baseline models are saved in `results/sdm/python_baselines/`. The shared node-mask file is saved in `results/sdm/shared/`.

## 5. Run GAM and MaxEnt

Install the required R packages:

```r
install.packages(c("mgcv", "maxnet", "dplyr", "readr"))
```

Run the following commands from the repository root. Run the Python
baselines first to generate the shared node masks. Use the same time
window for TGNN and all baselines; the example below uses 12 years.

```bash
python scripts/run_sdm_python_baselines.py --root . --years 1999:2024 --time-window 12 --out-dir results/sdm/python_baselines

Rscript scripts/run_gam.R --years 1999:2024 --time-window 12 --out-dir results/sdm/GAM

Rscript scripts/run_maxent.R --years 1999:2024 --time-window 12 --out-dir results/sdm/MaxEnt
```

Results are saved under `results/sdm/`. MaxEnt predicts occurrence only.


## Notes

The commands above describe the currently uploaded scripts. 

For all available options:

```bash
python scripts/tgnn.py --help
python scripts/run_sdm_python_baselines.py --help
```
`







