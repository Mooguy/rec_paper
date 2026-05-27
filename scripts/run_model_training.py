import numpy as np
import os
import argparse
import sys
import pandas as pd
import logging

from pathlib import Path
from src.models.XGBoostTrainer import XGBoostTrainer

# Resolve the root directory
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data" / "processed" 

def main():
    parser = argparse.ArgumentParser(
        description="Run model training for boosted models with Optuna stepwise tuning."
    )
    parser.add_argument(
        "--trainer",
        choices=["xgboost"],
        default="xgboost",
        help="Which trainer to use (default: xgboost).",
    )
    parser.add_argument(
        "--X_train-path", required=True, help="Path to the pickled X_train data."
    )
    parser.add_argument(
        "--y_train-path", required=True, help="Path to the numpy y_train data."
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Directory path where final model will be saved.",
    )
    parser.add_argument(
        "--model-filename",
        default="final_model",
        help="Filename for the saved model.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of Optuna trials per group (default: 10).",
    )
    parser.add_argument(
        "--study-name", default=None, help="Study name (default: trainer-specific)."
    )
    parser.add_argument(
        "--log-file", default=None, help="Optional path to write logs to a file."
    )
    parser.add_argument(
        "--group-training",
        choices=["group", "all", "all_with_seed"],
        default="group",
        help="If 'group' run groups sequentially; if 'all' tune all params at once.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    # join file name and number of trails to create a folder for saving

    # create save folder (unique name)
    todays_date = pd.Timestamp.now().strftime("%Y%m%d")
    save_path_full = os.path.join(
        args.save_path,
        f"{args.model_filename}_{args.group_training}_{args.trials}_{todays_date}",
    )
    os.makedirs(save_path_full, exist_ok=True)

    # configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]

    # if no explicit log-file provided, place a training.log inside the provided save_path
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    training_log = args.log_file or os.path.join(
        save_path_full, f"{todays_date}_training.log"
    )
    handlers.append(logging.FileHandler(training_log, mode="w"))
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # load data
    if not os.path.exists(args.X_train_path):
        logger.error("X_train file not found: %s", args.X_train_path)
        sys.exit(1)
    if not os.path.exists(args.y_train_path):
        logger.error("y_train file not found: %s", args.y_train_path)
        sys.exit(1)

    X = pd.read_pickle(args.X_train_path)
    y = np.load(args.y_train_path)

    # Run trainer logic
    if args.trainer == "xgboost":
        # Initialize the XGBoost trainer with user-defined trials and study name
        trainer = XGBoostTrainer(trials=args.trials, study_name=args.study_name) 

    best_params = trainer.get_best_params(
        X, y, save_path_full, group_training=args.group_training
    )
    trainer.train_final_model(X, y, best_params)

    trainer.save_model(filename=args.model_filename, path=save_path_full)

    logger.info(
        f"Done. Model saved to {os.path.join(save_path_full, args.model_filename)}"
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        # Fallback logging config if main() didn't configure handlers yet
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                handlers=[logging.StreamHandler(sys.stdout)],
                format="%(asctime)s %(levelname)s: %(message)s",
            )
        logger.exception("Training interrupted by user (KeyboardInterrupt)")
        sys.exit(130)
    except Exception:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logging.basicConfig(
                level=logging.ERROR,
                handlers=[logging.StreamHandler(sys.stderr)],
                format="%(asctime)s %(levelname)s: %(message)s",
            )
        logger.exception("Training failed with exception")
        sys.exit(1)
