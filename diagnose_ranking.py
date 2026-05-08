import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Test task (frontend, high requirement)
task_profile = {
    "taskType": "frontend",
    "taskComplexity": 8,
    "storyPoints": 5,
    "reqFrontend": 5,
    "reqBackend": 1,
    "reqDb": 0
}

# Test developers
developers_data = [
    {
        "dev_id": "D18",
        "skill_frontend": 1,
        "skill_backend": 1,
        "skill_db": 2,
        "experience_level": 1,
        "current_tasks": 1,
        "current_workload": 5,
        "availability": 74.95
    },
    {
        "dev_id": "D_EXPERT",
        "skill_frontend": 5,
        "skill_backend": 4,
        "skill_db": 3,
        "experience_level": 3,
        "current_tasks": 1,
        "current_workload": 8,
        "availability": 70.0
    }
]

dev_df = pd.DataFrame(developers_data)

# =========================================
# SKILL MATCH CALCULATION
# =========================================
print("\n" + "="*60)
print("SKILL MATCH ANALYSIS")
print("="*60)

for _, dev in dev_df.iterrows():
    skill_diff = (
        abs(dev["skill_frontend"] - task_profile["reqFrontend"]) +
        abs(dev["skill_backend"] - task_profile["reqBackend"]) +
        abs(dev["skill_db"] - task_profile["reqDb"])
    )
    
    skill_match = 1 - (skill_diff / 15)
    skill_match = max(0, min(1, skill_match))
    
    print(f"\n{dev['dev_id']}:")
    print(f"  Skill diff: {skill_diff} (frontend: {abs(dev['skill_frontend'] - task_profile['reqFrontend'])}, backend: {abs(dev['skill_backend'] - task_profile['reqBackend'])}, db: {abs(dev['skill_db'] - task_profile['reqDb'])})")
    print(f"  Skill match: {skill_match:.3f}")
    print(f"  Skills: Frontend {dev['skill_frontend']}/5, Backend {dev['skill_backend']}/5, DB {dev['skill_db']}/5")
    print(f"  Task needs: Frontend {task_profile['reqFrontend']}/5, Backend {task_profile['reqBackend']}/5, DB {task_profile['reqDb']}/5")

# =========================================
# ML PREDICTION
# =========================================
print("\n" + "="*60)
print("ML PREDICTION ANALYSIS")
print("="*60)

candidates = []
for _, dev in dev_df.iterrows():
    row = {
        "experience_level": int(dev["experience_level"]),
        "skill_frontend": int(dev["skill_frontend"]),
        "skill_backend": int(dev["skill_backend"]),
        "skill_db": int(dev["skill_db"]),
        "current_tasks": int(dev["current_tasks"]),
        "current_workload": int(dev["current_workload"]),
        "availability": float(dev["availability"]),
        "task_complexity": int(task_profile["taskComplexity"]),
        "story_points": int(task_profile["storyPoints"]),
        "req_frontend": int(task_profile["reqFrontend"]),
        "req_backend": int(task_profile["reqBackend"]),
        "req_db": int(task_profile["reqDb"]),
        "task_type_backend": 0,
        "task_type_frontend": 1,
        "task_type_db": 0,
        "_dev_id": dev["dev_id"]
    }
    candidates.append(row)

candidates_df = pd.DataFrame(candidates)
dev_ids = candidates_df["_dev_id"].values
candidates_df = candidates_df.drop(columns=["_dev_id"])

# Align features
candidates_df = candidates_df.reindex(columns=feature_columns, fill_value=0)

# Predict
predicted = model.predict(candidates_df)

for i, (dev_id, pred) in enumerate(zip(dev_ids, predicted)):
    print(f"\n{dev_id}:")
    print(f"  ML prediction score: {pred:.2f}")
    print(f"  Input features: {candidates_df.iloc[i].to_dict()}")

# =========================================
# FINAL SCORE CALCULATION
# =========================================
print("\n" + "="*60)
print("FINAL SCORE BREAKDOWN")
print("="*60)

max_workload = dev_df["current_workload"].max()
if max_workload == 0:
    max_workload = 1

max_pred = predicted.max()
if max_pred > 0:
    pred_norm = (predicted / max_pred) * 100
else:
    pred_norm = predicted

for i, dev_id in enumerate(dev_ids):
    dev = dev_df[dev_df["dev_id"] == dev_id].iloc[0]
    
    skill_diff = (
        abs(dev["skill_frontend"] - task_profile["reqFrontend"]) +
        abs(dev["skill_backend"] - task_profile["reqBackend"]) +
        abs(dev["skill_db"] - task_profile["reqDb"])
    )
    skill_match = 1 - (skill_diff / 15)
    skill_match = max(0, min(1, skill_match))
    
    workload_balance = 1 - (dev["current_workload"] / max_workload)
    
    final_score = (
        pred_norm[i] * 0.5 +
        skill_match * 30 +
        workload_balance * 20
    )
    
    print(f"\n{dev_id}:")
    print(f"  ML prediction (normalized):  {pred_norm[i]:>6.2f} × 0.50 = {pred_norm[i] * 0.5:>6.2f}")
    print(f"  Skill match score:            {skill_match:>6.3f} × 30.00 = {skill_match * 30:>6.2f}")
    print(f"  Workload balance score:       {workload_balance:>6.3f} × 20.00 = {workload_balance * 20:>6.2f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  FINAL SCORE:                  {final_score:>6.2f}")