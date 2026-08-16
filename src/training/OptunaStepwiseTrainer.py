from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import optuna
import numpy as np
import pandas as pd
import json
import subprocess
import os
import logging

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class OptunaStepwiseTrainer(ABC):
    def __init__(self, trials=10, study_name="optuna_study"):
        self.trials = trials
        self.study_name = study_name

    @abstractmethod
    def objective(self, trial, X_train, y_train, group, params, **kwargs):
        """Define the objective function for optimization"""
        pass

    @abstractmethod
    def train_final_model(self, X_train, y_train, params, **kwargs):
        """Train the final model with optimal parameters"""
        pass

    @abstractmethod
    def save_model(self, model, filepath):
        """Save the trained model"""
        pass

    @staticmethod
    def _check_gpu_availability():
        """Check if a GPU is available using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def execute_optimization(
        self, X_train, y_train, study_name, group, trials=10,
        params=None, direction="minimize", **kwargs,
    ):
        if params is None:
            params = {}

        sampler = TPESampler(n_startup_trials=20, seed=42)
        pruner = MedianPruner(n_startup_trials=30, n_warmup_steps=50, interval_steps=1)

        study = optuna.create_study(
            direction=direction,
            study_name=f"{study_name}_group_{group}",
            pruner=pruner,
            sampler=sampler,
        )

        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, group, params, **kwargs),
            n_trials=trials,
            n_jobs=1,
            catch=(),  # don't swallow exceptions silently — let them surface
        )

        if study.best_trial is None:
            raise RuntimeError(
                f"Optimization for group {group} produced no successful trials "
                f"(all {trials} trials failed or were pruned)."
            )

        logger.info("Optimization results for group %s", group)
        logger.info("Best value: %s", study.best_value)
        logger.info("Best parameters: %s", study.best_params)

        return {**params, **study.best_params}

    def stepwise_optimization(
        self,
        X_train,
        y_train,
        trials=10,
        study_name=None,
        **kwargs,
    ):
        """Execute stepwise optimization with proper error propagation."""
        if study_name is None:
            study_name = self.study_name

        final_params = {}

        logger.info("Starting stepwise optimization")
        logger.info("Data shape: %s", X_train.shape)
        logger.info("Number of classes: %s", len(np.unique(y_train)))

        group_training = kwargs.get("group_training", "group")

        if group_training == "group":
            group_list = ["1", "2", "3", "4"]
        else:
            group_list = [kwargs.get("group_training", "group")]

        logger.info("Optimization groups: %s", group_list)

        for g in group_list:
            logger.info("%s Optimizing Group - %s %s", "=" * 20, g, "=" * 20)

            try:
                update_params = self.execute_optimization(
                    X_train=X_train,
                    y_train=y_train,
                    study_name=study_name,
                    group=g,
                    trials=trials,
                    params=final_params,
                    **kwargs,
                )
            except Exception:
                logger.exception("Group %s failed — aborting stepwise optimization", g)
                raise

            final_params.update(update_params)
            logger.info("Parameters after group %s: %s", g, final_params)

        logger.info("Final Optimal Parameters: %s", final_params)
        return final_params

    def get_best_params(self, X_train, y_train, path, study_name=None, **kwargs):
        if study_name is None:
            study_name = self.study_name

        self.final_params = self.stepwise_optimization(
            X_train=X_train,
            y_train=y_train,
            trials=self.trials,
            study_name=study_name,
            **kwargs,
        )
        try:
            os.makedirs(
                f"{path}/best_params/{study_name}_{self.trials}_trials", exist_ok=True
            )
            with open(
                f"{path}/best_params/{study_name}_{self.trials}_trials/best_params_{study_name}_{self.trials}_trials.json",
                "w",
            ) as f:
                json.dump(self.final_params, f)
            print(
                f"Best parameters saved to best_params/{study_name}_{self.trials}_trials/best_params_{study_name}_{self.trials}_trials.json"
            )
        except Exception as e:
            print(f"Error saving best parameters: {str(e)}")

        return self.final_params
