import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression


class LearningRateModel:

    def __init__(self):
        # Stores one regression model per developer
        self.models = {}

    # =====================================================
    # TRAIN MODELS
    # =====================================================
    def train(self, df):
        """
        Required columns:
        - dev_id
        - sprint_id
        - performance_score
        """

        for dev_id in df["dev_id"].unique():

            # Get developer sprint history
            dev_data = (
                df[df["dev_id"] == dev_id]
                .sort_values("sprint_id")
            )

            # Need at least 3 sprints
            if len(dev_data) < 3:
                continue

            # X = sprint progression
            X = np.arange(len(dev_data)).reshape(-1, 1)

            # y = performance trend
            y = dev_data["performance_score"].values

            # Train regression
            model = LinearRegression()
            model.fit(X, y)

            # Store model
            self.models[dev_id] = model

        print(f"✓ Trained {len(self.models)} developer learning models")

        return self

    # =====================================================
    # GET SINGLE DEV LEARNING RATE
    # =====================================================
    def get_learning_rate(self, dev_id):

        if dev_id not in self.models:
            return 0.0

        # Slope = improvement rate
        return float(self.models[dev_id].coef_[0])

    # =====================================================
    # PREDICT BATCH
    # =====================================================
    def predict(self, df):

        rates = []

        for dev_id in df["dev_id"]:
            rate = self.get_learning_rate(dev_id)
            rates.append(rate)

        return np.array(rates)

    # =====================================================
    # SAVE MODEL
    # =====================================================
    def save(self, path="models/learning_rate_model.pkl"):

        joblib.dump(self, path)

        print(f"✓ Learning rate model saved to {path}")

    # =====================================================
    # LOAD MODEL
    # =====================================================
    @staticmethod
    def load(path="models/learning_rate_model.pkl"):

        model = joblib.load(path)

        print(f"✓ Learning rate model loaded from {path}")

        return model