import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class ConsistencyModel:

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        self.features = [
            "avg_completion_time",
            "defect_rate",
            "velocity",
            "blocked_tasks",
            "reopened_tasks",
            "time_variance"
        ]

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------
    def train(self, df):
        X = df[self.features]
        y = df["consistency"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        print("Consistency R2:", r2_score(y_test, preds))

        return self

    # -------------------------------
    # PREDICT
    # -------------------------------
    def predict(self, df):
        return self.model.predict(df[self.features])

    # -------------------------------
    # SAVE / LOAD
    # -------------------------------
    def save(self, path="models/consistency_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="models/consistency_model.pkl"):
        self.model = joblib.load(path)
        return self