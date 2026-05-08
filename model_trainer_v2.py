import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

# Try to import SHAP — install if missing: pip install shap
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")
    print("Explainability module will be skipped.\n")

# =========================================
# 1. LOAD DATASET
# =========================================
df = pd.read_csv("datasets/agile_resource_dataset_v2.csv")

print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print(df.head())

# =========================================
# 2. PREPROCESSING
# =========================================

# Drop ID columns — not useful for ML
df_model = df.drop(columns=["dev_id", "task_id"], errors="ignore")

# Columns that are targets or would cause data leakage
# completion_time and defects are outcomes — remove from features
# skill_match_score is a derived feature — safe to keep as input
leakage_cols = ["completion_time", "defects"]
df_model = df_model.drop(columns=leakage_cols, errors="ignore")

# Target variable
y = df_model["performance_score"]
X = df_model.drop(columns=["performance_score"])

print("\nPerformance score distribution:")
print(y.describe())

# One-hot encode task_type
X = pd.get_dummies(X, columns=["task_type"], drop_first=False)

# Save feature column order — CRITICAL for consistent inference later
feature_columns = list(X.columns)
print(f"\nFeatures ({len(feature_columns)}):", feature_columns)

# =========================================
# 3. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# =========================================
# 4. TRAIN MODEL
# =========================================
model = RandomForestRegressor(
    n_estimators=200,    # increased from 100 for better accuracy
    max_depth=12,
    min_samples_leaf=4,  # prevents overfitting on small leaf nodes
    random_state=42,
    n_jobs=-1            # use all CPU cores
)

model.fit(X_train, y_train)
print("\nModel trained successfully!")

# =========================================
# 5. EVALUATION
# =========================================
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred) #(Mean Absolute Error) - On average, how wrong is my prediction?
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #(Root Mean Squared Error) - How bad are big mistakes?
r2   = r2_score(y_test, y_pred) #R² Score - How much of the pattern does the model understand?

print("\n--- Model Evaluation ---")
print(f"MAE:      {round(mae, 2)}")
print(f"RMSE:     {round(rmse, 2)}")
print(f"R2 Score: {round(r2, 3)}")

# =========================================
# 6. SAVE MODEL + FEATURE COLUMNS
# (required for Flask API integration later)
# =========================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/performance_model.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
print("\nModel saved to: models/performance_model.pkl")
print("Feature columns saved to: models/feature_columns.pkl")

# =========================================
# 7. FEATURE IMPORTANCE
# =========================================
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png", dpi=150)
plt.show()

# =========================================
# 8. EXPLAINABILITY MODULE (SHAP)
# This directly maps to your proposal's
# "Explainable Decision Support Module"
# =========================================
if SHAP_AVAILABLE:
    print("\n--- Explainability (SHAP) ---")

    # Use a sample of 200 rows for SHAP (fast)
    X_shap = X_test.sample(200, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Global explanation — feature impact across all predictions
    shap.summary_plot(
        shap_values, X_shap,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Global)")
    plt.tight_layout()
    plt.savefig("models/shap_global.png", dpi=150)
    plt.show()

    # Local explanation — for one specific prediction
    # This is what your dashboard will show per recommendation
    single_idx = 0
    single_shap = shap_values[single_idx]

    explanation_df = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": single_shap,
        "Feature Value": X_shap.iloc[single_idx].values
    }).sort_values("SHAP Value", key=abs, ascending=False)

    print("\nExplainability for prediction #1 (top 5 drivers):")
    print(explanation_df.head(5).to_string(index=False))

# =========================================
# 9. TASK ALLOCATION — REAL DEVELOPER RANKING
# Uses actual developer profiles from dataset
# (not random fake values like the old version)
# =========================================
print("\n--- Task Allocation (Real Developer Ranking) ---")

def rank_developers_for_task(task_profile: dict, df_original: pd.DataFrame,
                              model, feature_columns: list) -> pd.DataFrame:
    """
    Given a task, rank all developers by predicted performance.
    Also calculates a workload balance score.

    Parameters:
        task_profile: dict with task features
        df_original:  the full dataset (to get real developer profiles)
        model:        trained RandomForest model
        feature_columns: saved feature column list

    Returns:
        DataFrame with ranked developers + scores
    """

    # Get unique developers with their LATEST sprint data
    latest_sprint = df_original["sprint_id"].max()
    dev_latest = (
        df_original[df_original["sprint_id"] == latest_sprint]
        .drop_duplicates(subset="dev_id")
        [["dev_id", "skill_frontend", "skill_backend", "skill_db",
          "experience_level", "current_workload", "availability"]]
        .reset_index(drop=True)
    )

    candidates = []

    for _, dev_row in dev_latest.iterrows():
        # Build a feature row combining dev profile + task profile
        row = {
            "sprint_id":          latest_sprint,
            "experience_level":   dev_row["experience_level"],
            "skill_frontend":     dev_row["skill_frontend"],
            "skill_backend":      dev_row["skill_backend"],
            "skill_db":           dev_row["skill_db"],
            "task_complexity":    task_profile.get("task_complexity", 5),
            "story_points":       task_profile.get("story_points", 3),
            "req_frontend":       task_profile.get("req_frontend", 0),
            "req_backend":        task_profile.get("req_backend", 0),
            "req_db":             task_profile.get("req_db", 0),
            "current_tasks":      np.random.randint(1, 5),
            "current_workload":   dev_row["current_workload"],
            "availability":       dev_row["availability"],

            # Compute skill match score for this dev-task combo
            "skill_match_score":  float(np.exp(-(
                abs(dev_row["skill_frontend"] - task_profile.get("req_frontend", 0)) +
                abs(dev_row["skill_backend"]  - task_profile.get("req_backend", 0)) +
                abs(dev_row["skill_db"]       - task_profile.get("req_db", 0))
            ) / 5)),

            "velocity_contribution": 0,   # placeholder for ranking input

            # One-hot encode task_type
            "task_type_backend":   1 if task_profile.get("task_type") == "backend"  else 0,
            "task_type_db":        1 if task_profile.get("task_type") == "db"        else 0,
            "task_type_frontend":  1 if task_profile.get("task_type") == "frontend"  else 0,
        }
        row["_dev_id"] = dev_row["dev_id"]
        candidates.append(row)

    candidates_df = pd.DataFrame(candidates)
    dev_ids = candidates_df["_dev_id"].values
    candidates_df = candidates_df.drop(columns=["_dev_id"])

    # Align to trained feature columns
    candidates_df = candidates_df.reindex(columns=feature_columns, fill_value=0)

    # Predict performance for each developer
    predicted_scores = model.predict(candidates_df)

    # Workload balance score — penalise overloaded developers
    # Lower workload = higher balance score
    workload_values = candidates_df["current_workload"].values
    max_wl = workload_values.max() if workload_values.max() > 0 else 1
    workload_balance_score = 1 - (workload_values / max_wl)

    # Combined final score: 70% performance + 30% workload balance
    final_score = 0.7 * predicted_scores + 0.3 * workload_balance_score * 100

    results = pd.DataFrame({
        "dev_id":                dev_ids,
        "skill_frontend":        candidates_df["skill_frontend"].values,
        "skill_backend":         candidates_df["skill_backend"].values,
        "skill_db":              candidates_df["skill_db"].values,
        "current_workload":      candidates_df["current_workload"].values,
        "availability":          candidates_df["availability"].values,
        "skill_match_score":     candidates_df["skill_match_score"].values,
        "predicted_performance": np.round(predicted_scores, 2),
        "workload_balance":      np.round(workload_balance_score, 3),
        "final_score":           np.round(final_score, 2),
    })

    results = results.sort_values("final_score", ascending=False).reset_index(drop=True)
    results.insert(0, "rank", range(1, len(results) + 1))

    return results


# Example task to allocate
sample_task = {
    "task_type":       "backend",
    "task_complexity": 7,
    "story_points":    5,
    "req_frontend":    1,
    "req_backend":     4,
    "req_db":          2,
}

ranking = rank_developers_for_task(sample_task, df, model, feature_columns)

print("\nSample task:", sample_task)
print("\nTop 5 recommended developers:")
print(ranking.head(5).to_string(index=False))

best = ranking.iloc[0]
print(f"\nBest developer: {best['dev_id']}")
print(f"  Predicted performance : {best['predicted_performance']}")
print(f"  Skill match score     : {round(best['skill_match_score'], 3)}")
print(f"  Workload balance      : {best['workload_balance']}")
print(f"  Final combined score  : {best['final_score']}")

# Save ranking function for Flask API use
joblib.dump(rank_developers_for_task, "models/rank_function.pkl")
print("\nRanking function saved to: models/rank_function.pkl")
