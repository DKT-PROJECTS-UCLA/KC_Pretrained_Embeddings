#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/examples"
SAVE_DIR="$ROOT_DIR/saved_model/assist2015_dkt_qid_saved_model_42_0_0.2_200_0.001_0_0"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

python "$EXAMPLES_DIR/data_preprocess.py" --dataset_name assist2015
python "$EXAMPLES_DIR/wandb_dkt_train.py" --use_wandb 0 --add_uuid 0
python "$EXAMPLES_DIR/wandb_predict.py" --save_dir "$SAVE_DIR" --use_wandb 0
python "$EXAMPLES_DIR/wandb_eval.py" --save_dir "$SAVE_DIR" --use_wandb 0
