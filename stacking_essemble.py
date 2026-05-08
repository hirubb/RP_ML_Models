import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

print("="*70)
print("TRAINING MODEL WITH CORRECT FEATURES")
print("="*70)

# =========================================
# 1. LOAD DATA
# =========================================
print("\nLoading dataset...")
df = pd.read_csv("datasets/agile_resource_dataset_v2.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# =========================================
# 2. PREPROCESSING (FIXED)
# =========================================
df_model = df.drop(columns=["dev_id", "task_id"], errors="ignore")

# ✅ CRITICAL CHANGE: Only drop OUTCOMES that happen AFTER assignment
# DO NOT drop skill_match_score and velocity_contribution
# These are DERIVED features calculated AT PREDICTION TIME
leakage_cols = ["completion_time", "defects"]
df_model = df_model.drop(columns=leakage_cols, errors="ignore")

print(f"\nFeatures kept for training:")
print(f"  ✓ skill_match_score (derived from dev skills + task requirements)")
print(f"  ✓ velocity_contribution (story points weighted by skill match)")
print(f"\nFeatures dropped (outcomes):")
print(f"  ✗ completion_time (known only after task done)")
print(f"  ✗ defects (known only after task done)")

# Target
y = df_model["performance_score"]
X = df_model.drop(columns=["performance_score"])

# One-hot encode task_type
X = pd.get_dummies(X, columns=["task_type"], drop_first=False)

feature_columns = list(X.columns)

print(f"\nFinal feature set for training ({len(feature_columns)} features):")
for i, col in enumerate(sorted(feature_columns), 1):
    print(f"  {i:2d}. {col}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Train: {len(X_train)} rows")
print(f"  Test:  {len(X_test)} rows")

# =========================================
# 3. DEFINE MODELS
# =========================================
rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=14,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

xgboost_model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

lightgbm_model = lgb.LGBMRegressor(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.04,
    num_leaves=40,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

stacking = StackingRegressor(
    estimators=[
        ("rf", rf),
        ("xgb", xgboost_model),
        ("lgbm", lightgbm_model)
    ],
    final_estimator=Ridge(alpha=0.5),
    cv=5,
    n_jobs=-1
)

models = {
    "Random Forest": rf,
    "XGBoost": xgboost_model,
    "LightGBM": lightgbm_model,
    "Stacking Ensemble": stacking
}

# =========================================
# 4. TRAIN & EVALUATE
# =========================================
results = {}

print("\n" + "="*70)
print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>8}")
print("="*70)

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)

    y_pred = model.predict(X_test)

    mae = round(mean_absolute_error(y_test, y_pred), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)

    results[name] = {
        "model": model,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "time": elapsed,
        "y_pred": y_pred
    }

    print(f"{name:<20} {mae:>8} {rmse:>8} {r2:>8} {elapsed:>7}s")

print("="*70)

# =========================================
# 5. CROSS VALIDATION
# =========================================
print("\n--- 5-Fold Cross Validation (R² Score) ---")

cv_scores_dict = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    mean_score = round(scores.mean(), 3)
    std_score = round(scores.std(), 3)

    cv_scores_dict[name] = mean_score

    print(f"{name:<20} mean={mean_score}  std=±{std_score}")

# =========================================
# 6. SELECT BEST MODEL
# =========================================
best_name = max(cv_scores_dict, key=cv_scores_dict.get)
best_model = models[best_name]

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"  CV R² Score: {cv_scores_dict[best_name]}")
print(f"  Test R²:     {results[best_name]['R2']}")
print(f"  Test MAE:    {results[best_name]['MAE']}")
print(f"  Test RMSE:   {results[best_name]['RMSE']}")
print(f"{'='*70}")

# Retrain on full training data
best_model.fit(X_train, y_train)

# =========================================
# 7. SAVE MODEL
# =========================================
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")
joblib.dump(best_name, "models/best_model_name.pkl")

print(f"\n✓ Model saved:")
print(f"  - models/best_model.pkl ({best_name})")
print(f"  - models/feature_columns.pkl ({len(feature_columns)} features)")

# =========================================
# 8. SUMMARY TABLE
# =========================================
summary = pd.DataFrame([
    {
        "Model": name,
        "R²": res["R2"],
        "MAE": res["MAE"],
        "RMSE": res["RMSE"],
        "CV R²": cv_scores_dict[name],
        "Selected": "✓" if name == best_name else ""
    }
    for name, res in results.items()
]).sort_values("R²", ascending=False)

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(summary.to_string(index=False))

# =========================================
# 9. FEATURE IMPORTANCE
# =========================================
if hasattr(best_model, 'feature_importances_'):
    print(f"\n--- Top 10 Features ({best_name}) ---")
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']:<30} {row['Importance']:>7.4f}")

print(f"\n✓ Training complete! Model ready for deployment.")