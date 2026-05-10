import pandas as pd
import joblib
import os

from behavioral.dataset_generator import build_behavioral_dataset
from behavioral.consistency_model import ConsistencyModel
from behavioral.learning_rate_model import LearningRateModel

# =========================================
# LOAD RAW SPRINT LOGS
# =========================================

df = pd.read_csv("datasets/sprint_logs.csv")

print("Loaded sprint logs:", len(df))

# =========================================
# BUILD BEHAVIORAL DATASET
# =========================================

behavior_df = build_behavioral_dataset(df)

print("\nBehavior dataset created")
print(behavior_df.head())

# =========================================
# TRAIN CONSISTENCY MODEL
# =========================================

consistency_model = ConsistencyModel()

consistency_model.train(behavior_df)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

consistency_model.save("models/consistency_model.pkl")

print("\n✓ Consistency model saved")

# =========================================
# PREPARE LEARNING RATE DATA
# =========================================

learning_df = behavior_df.copy()

# Use consistency as performance proxy
learning_df["performance_score"] = (
    learning_df["consistency"] * 100
)

# =========================================
# TRAIN LEARNING RATE MODEL
# =========================================

learning_model = LearningRateModel()

learning_model.train(learning_df)

# Save learning model
joblib.dump(
    learning_model,
    "models/learning_rate_model.pkl"
)

print("✓ Learning rate model saved")

print("\nALL BEHAVIOR MODELS TRAINED SUCCESSFULLY")