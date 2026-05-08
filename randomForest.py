# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("agile_resource_dataset_realistic.csv")

print("Dataset Preview:")
print(df.head())

# =========================================
# 3. PREPROCESSING
# =========================================

# Drop IDs
df = df.drop(columns=["dev_id", "task_id"], errors='ignore')

# ✅ Safe leakage removal (won't crash)
leakage_cols = ["completion_time", "defects", "skill_match_score"]
df = df.drop(columns=leakage_cols, errors='ignore')

# Target
y = df["performance_score"]

# Features
X = df.drop(columns=["performance_score"])

# Check distribution (good debugging step)
print("\nPerformance Score Distribution:")
print(df["performance_score"].describe())

# Encode categorical
X = pd.get_dummies(X, columns=["task_type"], drop_first=True)

# ✅ Save feature names (IMPORTANT for inference)
feature_columns = X.columns

print("\nProcessed Features:")
print(X.head())

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================================
# 5. TRAIN MODEL
# =========================================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# =========================================
# 6. PREDICTIONS
# =========================================
y_pred = model.predict(X_test)

# =========================================
# 7. EVALUATION
# =========================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# =========================================
# 8. FEATURE IMPORTANCE
# =========================================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Plot
plt.figure()
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# =========================================
# 9. TASK ALLOCATION (TOP-N RANKING SYSTEM)
# =========================================

print("\n--- Task Allocation (Top-N Ranking) ---")

sample_task = X_test.iloc[0].copy()

samples = []

N_CANDIDATES = 10   # increase for better ranking realism

for i in range(N_CANDIDATES):
    temp = sample_task.copy()

    # simulate different developers
    for col in ["skill_frontend", "skill_backend", "skill_db"]:
        if col in temp:
            temp[col] = np.random.randint(1, 6)

    if "current_workload" in temp:
        temp["current_workload"] = np.random.randint(5, 30)

    if "availability" in temp:
        temp["availability"] = max(
            0,
            100 - temp["current_workload"] * 3
        )

    samples.append(temp)

samples_df = pd.DataFrame(samples)

# align features
samples_df = samples_df.reindex(columns=feature_columns, fill_value=0)

# predictions
predictions = model.predict(samples_df)

# build ranking table
results = samples_df.copy()
results["predicted_performance"] = predictions

# rank (IMPORTANT PART)
results = results.sort_values(
    by="predicted_performance",
    ascending=False
).reset_index(drop=True)

# add rank column
results["rank"] = np.arange(1, len(results) + 1)

results["score_normalized"] = (
    results["predicted_performance"] /
    results["predicted_performance"].max()
) * 100

# show only useful columns
display_cols = [
    "rank",
    "skill_frontend",
    "skill_backend",
    "skill_db",
    "current_workload",
    "availability",
    "predicted_performance"
]

print("\n🏆 TOP-N DEVELOPER RANKING:")
print(results[display_cols])

# best developer
best = results.iloc[0]

print("\n🥇 BEST DEVELOPER:")
print(best[display_cols])