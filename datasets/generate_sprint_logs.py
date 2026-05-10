import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# =========================================
# CONFIG
# =========================================

NUM_DEVELOPERS = 40
NUM_SPRINTS = 25
TASKS_PER_SPRINT = 30

# =========================================
# DEV TYPES
# =========================================

developer_types = [
    "stable_senior",
    "fast_risky",
    "junior_learning",
    "burned_out",
    "average"
]

developers = []

for i in range(NUM_DEVELOPERS):

    developers.append({
        "dev_id": f"D{i+1}",
        "type": random.choice(developer_types)
    })

# =========================================
# GENERATE DATA
# =========================================

records = []

for sprint in range(1, NUM_SPRINTS + 1):

    for _ in range(TASKS_PER_SPRINT):

        dev = random.choice(developers)

        dev_id = dev["dev_id"]
        dev_type = dev["type"]

        # ---------------------------------
        # STABLE SENIOR
        # ---------------------------------

        if dev_type == "stable_senior":

            completion_time = np.random.normal(3, 0.8)
            defects = np.random.poisson(0.3)
            velocity = np.random.normal(10, 1)

        # ---------------------------------
        # FAST RISKY
        # ---------------------------------

        elif dev_type == "fast_risky":

            completion_time = np.random.normal(2, 0.7)
            defects = np.random.poisson(2)
            velocity = np.random.normal(12, 2)

        # ---------------------------------
        # JUNIOR LEARNING
        # IMPROVES OVER TIME
        # ---------------------------------

        elif dev_type == "junior_learning":

            improvement = sprint * 0.08

            completion_time = np.random.normal(
                max(2.5, 7 - improvement),
                1
            )

            defects = max(
                0,
                int(np.random.poisson(max(0.5, 3 - improvement)))
            )

            velocity = np.random.normal(
                4 + improvement,
                1
            )

        # ---------------------------------
        # BURNED OUT
        # GETS WORSE OVER TIME
        # ---------------------------------

        elif dev_type == "burned_out":

            degradation = sprint * 0.07

            completion_time = np.random.normal(
                4 + degradation,
                1
            )

            defects = np.random.poisson(
                1 + degradation
            )

            velocity = np.random.normal(
                max(3, 8 - degradation),
                1
            )

        # ---------------------------------
        # AVERAGE DEV
        # ---------------------------------

        else:

            completion_time = np.random.normal(5, 1)
            defects = np.random.poisson(1)
            velocity = np.random.normal(7, 1)

        # ---------------------------------
        # OTHER FIELDS
        # ---------------------------------

        blocked_tasks = np.random.binomial(1, 0.12)

        reopened_tasks = np.random.binomial(1, 0.08)

        records.append({
            "dev_id": dev_id,
            "sprint_id": sprint,
            "completion_time": round(abs(completion_time), 2),
            "defects": int(defects),
            "velocity_contribution": round(abs(velocity), 2),
            "blocked_tasks": int(blocked_tasks),
            "reopened_tasks": int(reopened_tasks)
        })

# =========================================
# SAVE CSV
# =========================================

df = pd.DataFrame(records)

os.makedirs("data", exist_ok=True)

df.to_csv("sprint_logs.csv", index=False)

print(df.head())

print("\nDataset Generated Successfully")
print("Total Rows:", len(df))