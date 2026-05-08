import numpy as np
import pandas as pd

np.random.seed(42)

# =========================================
# CONFIG
# =========================================
NUM_SPRINTS = 10
NUM_DEVS = 30
NUM_TASKS = 100
ROWS_PER_SPRINT = 200

# =========================================
# GENERATE DEVELOPERS
# =========================================
developers = []

for i in range(NUM_DEVS):
    dev = {
        "dev_id": f"D{i+1}",
        "experience_level": np.random.choice([1, 2, 3]),
        "skill_frontend": np.random.randint(1, 6),
        "skill_backend": np.random.randint(1, 6),
        "skill_db": np.random.randint(1, 6),
        "consistency": np.random.uniform(0.6, 1.0),
        "learning_rate": np.random.uniform(0.01, 0.1),
    }
    developers.append(dev)

dev_df = pd.DataFrame(developers)

# =========================================
# GENERATE TASKS
# =========================================
tasks = []

for i in range(NUM_TASKS):
    task_type = np.random.choice(["frontend", "backend", "db"])
    task = {
        "task_id": f"T{i+1}",
        "task_type": task_type,
        "task_complexity": np.random.randint(1, 11),
        "story_points": np.random.randint(1, 9),
        "ambiguity": np.random.uniform(0, 1),
        "dependency_risk": np.random.uniform(0, 1),
        "req_frontend": np.random.randint(1, 6) if task_type == "frontend" else np.random.randint(0, 3),
        "req_backend": np.random.randint(1, 6) if task_type == "backend" else np.random.randint(0, 3),
        "req_db": np.random.randint(1, 6) if task_type == "db" else np.random.randint(0, 3),
    }
    tasks.append(task)

task_df = pd.DataFrame(tasks)

# =========================================
# GENERATE DATASET ACROSS SPRINTS
# =========================================
data = []

# working copy of skills that evolve per sprint
dev_skills = dev_df.copy()

for sprint_num in range(1, NUM_SPRINTS + 1):

    for _ in range(ROWS_PER_SPRINT):

        dev = dev_skills.sample(1).iloc[0]
        task = task_df.sample(1).iloc[0]

        # ---------------------------
        # WORKLOAD
        # ---------------------------
        current_tasks = np.random.randint(1, 7)
        current_workload = current_tasks * np.random.randint(3, 9)
        burnout = np.tanh(current_workload / 40)
        availability = max(0, 100 - current_workload * np.random.uniform(3, 6))

        # ---------------------------
        # SKILL MATCH SCORE
        # This is CRITICAL — it's used both in the dataset AND in prediction
        # Calculate per-skill match and weight by task type
        # ---------------------------
        frontend_match = max(0, 1 - (abs(dev["skill_frontend"] - task["req_frontend"]) / 5))
        backend_match = max(0, 1 - (abs(dev["skill_backend"] - task["req_backend"]) / 5))
        db_match = max(0, 1 - (abs(dev["skill_db"] - task["req_db"]) / 5))

        # Weight by task type (primary skill gets 60%, others split 40%)
        if task["task_type"] == "frontend":
            skill_match_score = frontend_match * 0.6 + backend_match * 0.25 + db_match * 0.15
        elif task["task_type"] == "backend":
            skill_match_score = backend_match * 0.6 + frontend_match * 0.25 + db_match * 0.15
        else:  # db
            skill_match_score = db_match * 0.6 + backend_match * 0.25 + frontend_match * 0.15

        skill_match_score = max(0, min(1, skill_match_score))
        skill_match_score = round(float(skill_match_score), 4)

        # ---------------------------
        # VELOCITY CONTRIBUTION
        # Story points weighted by skill match and workload (burnout reduces capacity)
        # ---------------------------
        velocity_contribution = round(
            float(task["story_points"] * skill_match_score * (1 - burnout * 0.5)),
            2
        )

        # ---------------------------
        # HIDDEN FACTORS
        # ---------------------------
        interruption = np.random.uniform(0, 1)
        random_blocker = np.random.choice([0, 1], p=[0.8, 0.2])

        # ---------------------------
        # COMPLETION TIME
        # Better developers (higher skill match) complete faster
        # ---------------------------
        base_time = task["task_complexity"] * np.random.uniform(1.5, 2.5)
        completion_time = base_time * (
            (1.3 - skill_match_score)  # Skill helps reduce time
            + burnout                  # Burnout increases time
            + task["ambiguity"]        # Unclear reqs increase time
            + task["dependency_risk"]  # Blockers increase time
            + interruption             # Random interruptions
        )
        completion_time *= np.random.normal(dev["consistency"], 0.2)
        completion_time = round(float(max(1, completion_time)), 2)

        # ---------------------------
        # DEFECTS
        # Better skill match = fewer bugs
        # ---------------------------
        defect_rate = (
            (1 - skill_match_score)   # Skill mismatch causes bugs
            + task["ambiguity"]       # Unclear reqs cause bugs
            + burnout                 # Tired devs make mistakes
            + random_blocker * 0.5    # Random issues
        )
        defects = int(np.random.poisson(
            lam=max(0.5, defect_rate * task["task_complexity"] / 3)
        ))

        # ---------------------------
        # PERFORMANCE SCORE (TARGET VARIABLE)
        # High when: skill match is good, workload is manageable, completion is fast, defects are low
        # ---------------------------
        performance_score = (
            (skill_match_score ** 1.5) * 50    # Skill match dominates
            + np.log1p(availability) * 10      # Available capacity helps
            - (completion_time ** 0.7) * 5     # Speed matters
            - (defects ** 1.2) * 4             # Quality matters
        )
        performance_score += np.random.normal(0, 10)  # Real-world noise
        performance_score = round(float(max(0, min(100, performance_score))), 2)

        # ---------------------------
        # STORE ROW
        # ---------------------------
        row = {
            "sprint_id": sprint_num,
            "dev_id": dev["dev_id"],
            "task_id": task["task_id"],
            "experience_level": dev["experience_level"],
            "skill_frontend": dev["skill_frontend"],
            "skill_backend": dev["skill_backend"],
            "skill_db": dev["skill_db"],
            "task_type": task["task_type"],
            "task_complexity": task["task_complexity"],
            "story_points": task["story_points"],
            "req_frontend": task["req_frontend"],
            "req_backend": task["req_backend"],
            "req_db": task["req_db"],
            "current_tasks": current_tasks,
            "current_workload": current_workload,
            "availability": round(float(availability), 2),
            
            # ✅ CRITICAL: These are NOW CALCULATED PROPERLY
            "skill_match_score": skill_match_score,
            "velocity_contribution": velocity_contribution,
            
            # Outcomes (will be dropped during training)
            "completion_time": completion_time,
            "defects": defects,
            "performance_score": performance_score,
        }
        data.append(row)

    # ---------------------------
    # APPLY LEARNING RATE AFTER EACH SPRINT
    # ---------------------------
    for idx in dev_skills.index:
        lr = dev_skills.at[idx, "learning_rate"]
        for skill_col in ["skill_frontend", "skill_backend", "skill_db"]:
            new_val = dev_skills.at[idx, skill_col] + np.random.uniform(0, lr)
            dev_skills.at[idx, skill_col] = round(min(5.0, new_val), 3)

    print(f"Sprint {sprint_num} generated — {ROWS_PER_SPRINT} rows")

# =========================================
# SAVE
# =========================================
df = pd.DataFrame(data)
df.to_csv("agile_resource_dataset_v2.csv", index=False)

print(f"\n✓ Dataset saved: {len(df)} rows, {len(df.columns)} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample row:")
print(df.head(1).to_dict(orient='records')[0])
print(f"\nPerformance score distribution:")
print(df["performance_score"].describe())
print(f"\nSkill match distribution:")
print(df["skill_match_score"].describe())