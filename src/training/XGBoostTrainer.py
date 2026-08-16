import xgboost as xgb
import numpy as np
import os
import logging
import time
import gc

from optuna.integration.xgboost import XGBoostPruningCallback
from src.training.OptunaStepwiseTrainer import OptunaStepwiseTrainer

logger = logging.getLogger(__name__)


class XGBoostTrainer(OptunaStepwiseTrainer):
    def __init__(self, trials=10, study_name="xgboost"):
        super().__init__(trials, study_name)
        self.gpu_available = self._check_gpu_availability()
        self.base_params = {
            "objective": "multi:softprob",
            "device": "cuda" if self.gpu_available else "cpu",
            "num_class": None,
            "eval_metric": "mlogloss",
            "seed": 42,
            "nthread": -1,
        }

        logger.info(f"Initialized XGBoostTrainer with {trials} trials")
        logger.info(f"Study name: {study_name}")
        logger.info(f"GPU available: {self.gpu_available}")

    def objective(self, trial, X, y, group, params=None, **kwargs):
        try:
            # Ensure params is a dictionary
            if params is None:
                params = {}

            # Safe retrieval of optional kwargs
            weight = kwargs.get("weight", None)

            # Ensure X is an XGBoost DMatrix
            if not isinstance(X, xgb.DMatrix):
                dtrain = xgb.DMatrix(
                    X, label=y, enable_categorical=False, weight=weight
                )
            else:
                dtrain = X

            self.base_params.update(
                {
                    "num_class": len(np.unique(y)),  # Set number of classes dynamically
                }
            )

            # Merge base params with any provided params
            params = {**self.base_params, **params}

            if group == "all":
                # Suggest all parameters
                params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
                params["min_child_weight"] = trial.suggest_float(
                    "min_child_weight", 1, 7
                )
                params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                )
                params["alpha"] = trial.suggest_float("alpha", 1e-8, 1.0, log=True)
                params["lambda"] = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
                params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                )
                params["num_boost_round"] = trial.suggest_int(
                    "num_boost_round", 100, 1000
                )

            elif group == "1":
                params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
                params["min_child_weight"] = trial.suggest_float(
                    "min_child_weight", 1, 7
                )

            elif group == "2":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)
                params["colsample_bytree"] = trial.suggest_float(
                    "colsample_bytree", 0.1, 1.0
                )

            elif group == "3":
                params["alpha"] = trial.suggest_float("alpha", 1e-8, 1.0, log=True)
                params["lambda"] = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
                params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)

            elif group == "4":
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                )
                params["num_boost_round"] = trial.suggest_int(
                    "num_boost_round", 100, 1000
                )

            logger.info(f"Trial {trial.number} params: {params}")
            start_time = time.time()

            # Set up pruning callback
            callbacks = [
                XGBoostPruningCallback(trial, "test-" + self.base_params["eval_metric"])
            ]

            cv_scores = xgb.cv(
                params,
                dtrain,
                num_boost_round=params.pop("num_boost_round", 1000),
                nfold=5,
                stratified=False,
                early_stopping_rounds=20,
                callbacks=callbacks,
                seed=42,
                verbose_eval=False,
            )

            # Extract the final score
            final_score = cv_scores[
                "test-" + self.base_params["eval_metric"] + "-mean"
            ].values[-1]
            logger.info(f"Trial {trial.number} score: {final_score}")
            logger.info(
                f"Trial {trial.number} completed in {time.time() - start_time:.2f} seconds"
            )
            return final_score

        except Exception as e:
            logger.info(f"Trial {trial.number} failed with error: {e}")
            raise  
        finally:
            # free large objects to avoid memory growth across trials
            if "cv_scores" in locals():
                del cv_scores
            gc.collect()

    def train_final_model(self, X_train, y_train, best_params):
        """Train the final model with the best parameters"""
        if not best_params:
            raise ValueError("train_final_model called with empty best_params — refusing to train with defaults.")
            
        logger.info("Training final XGBoost model with best parameters")
        logger.info("Best parameters: %s", best_params)

        self.model = None

        if best_params is None:
            logger.info("Error: No parameters provided for final model training")
            return

        train_params = {**self.base_params, **best_params}
        num_boost_round = int(train_params.pop("num_boost_round", 1000))

        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            self.model = xgb.train(
                train_params,
                dtrain,
                num_boost_round=num_boost_round,
                verbose_eval=False,
            )
            logger.info("XGBoost model training completed successfully")
        except Exception as e:
            logger.info("Final model training failed with error: %s", str(e))
            return

    def save_model(self, filename, path):
        if self.model is None:
            logger.info("No model to save")
            return

        os.makedirs(path, exist_ok=True)
        stem = os.path.splitext(filename)[0]  
        model_path = os.path.join(path, f"{stem}.json")

        logger.info("Saving XGBoost model to %s", model_path)
        self.model.save_model(model_path)
        logger.info("Model saved successfully")


