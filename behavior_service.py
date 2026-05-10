import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)


class BehaviorService:
    """
    Central service for developer behavioral intelligence.
    
    Responsibilities:
    - Build behavioral dataset from sprint history
    - Predict consistency
    - Predict learning rate
    """

    def __init__(
        self,
        consistency_model_path="models/consistency_model.pkl",
        learning_model_path="models/learning_rate_model.pkl"
    ):
        self.consistency_model = joblib.load(consistency_model_path)
        self.learning_model = joblib.load(learning_model_path)

        logger.info("✓ Behavior models loaded successfully")

    # =========================================================
    # FEATURE ENGINEERING (FROM RAW SPRINT HISTORY)
    # =========================================================
    def build_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert sprint logs → behavioral features per (dev, sprint)
        """

        grouped = df.groupby(["dev_id", "sprint_id"])
        records = []

        for (dev_id, sprint_id), group in grouped:

            avg_completion_time = group["completion_time"].mean()
            defect_rate = group["defects"].mean()
            velocity = group["velocity_contribution"].sum()

            blocked_tasks = group.get("blocked_tasks", pd.Series([0])).sum()
            reopened_tasks = group.get("reopened_tasks", pd.Series([0])).sum()

            time_variance = group["completion_time"].var()
            time_variance = 0 if np.isnan(time_variance) else time_variance

            # -------------------------
            # CONSISTENCY INPUT FEATURES
            # -------------------------
            records.append({
                "dev_id": dev_id,
                "sprint_id": sprint_id,
                "avg_completion_time": avg_completion_time,
                "defect_rate": defect_rate,
                "velocity": velocity,
                "blocked_tasks": blocked_tasks,
                "reopened_tasks": reopened_tasks,
                "time_variance": time_variance,
            })

        return pd.DataFrame(records)

    # =========================================================
    # CONSISTENCY PREDICTION
    # =========================================================
    def predict_consistency(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict developer consistency score
        """

        X = features_df.drop(columns=["dev_id", "sprint_id"], errors="ignore")

        predictions = self.consistency_model.predict(X)

        features_df["consistency"] = np.clip(predictions, 0, 1)

        logger.info("✓ Consistency predictions completed")

        return features_df

    # =========================================================
    # LEARNING RATE PREDICTION
    # =========================================================
    def predict_learning_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict developer learning rate over time
        """

        X = df.drop(columns=["dev_id", "sprint_id", "consistency"], errors="ignore")

        predictions = self.learning_model.predict(df)

        df["learning_rate"] = np.clip(predictions, 0, 1)

        logger.info("✓ Learning rate predictions completed")

        return df

    # =========================================================
    # FULL PIPELINE
    # =========================================================
    def generate_behavioral_profile(self, sprint_logs: pd.DataFrame) -> pd.DataFrame:
        """
        End-to-end pipeline:
        raw logs → features → consistency → learning rate
        """

        logger.info("🚀 Building behavioral profile...")

        features = self.build_behavior_features(sprint_logs)
        features = self.predict_consistency(features)
        features = self.predict_learning_rate(features)

        logger.info("✅ Behavioral profile generated")

        return features

    # =========================================================
    # SINGLE DEV SNAPSHOT (FOR REAL-TIME API)
    # =========================================================
    def get_dev_behavior(self, dev_id: str, sprint_history: pd.DataFrame) -> dict:
        """
        Get latest behavioral metrics for a developer
        """

        dev_data = sprint_history[sprint_history["dev_id"] == dev_id]

        if dev_data.empty:
            return {
                "dev_id": dev_id,
                "consistency": 0.5,
                "learning_rate": 0.1
            }

        features = self.build_behavior_features(dev_data)
        features = self.predict_consistency(features)
        features = self.predict_learning_rate(features)

        latest = features.sort_values("sprint_id").iloc[-1]

        return {
            "dev_id": dev_id,
            "consistency": float(latest["consistency"]),
            "learning_rate": float(latest["learning_rate"])
        }