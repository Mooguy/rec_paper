#!/bin/bash
set -euo pipefail

# Resolve directories relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   

# Job variables
MODEL_FILENAME="xgboost_min_cells_20_2250"
TRIALS=200

# Paths (relative to paper_code)
X_TRAIN="$PAPER_CODE_DIR/data/processed/fig5_data/X_train_2250.pkl"
Y_TRAIN="$PAPER_CODE_DIR/data/processed/fig5_data/y_train_2250.npy"
SAVE_PATH="$PAPER_CODE_DIR/models/$MODEL_FILENAME"

cd "$PAPER_CODE_DIR"

python -m scripts.run_model_training \
  --trainer xgboost \
  --X_train-path "$X_TRAIN" \
  --y_train-path "$Y_TRAIN" \
  --save-path "$SAVE_PATH" \
  --model-filename "$MODEL_FILENAME" \
  --trials "$TRIALS" \
  --study-name xgboost_min_cells_20_2250 \
  --group-training "group" \
  --log-level INFO

  #bsub -q interactive-gpu -n 2 -R "rusage[mem=16000]" -R "affinity[thread*2]" -gpu "num=1:j_exclusive=no" -Is /bin/bash 
