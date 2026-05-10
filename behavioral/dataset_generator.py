import pandas as pd
import numpy as np

def build_behavioral_dataset(df: pd.DataFrame):
    """
    Convert raw sprint logs into behavioral ML dataset.
    """

    grouped = df.groupby(["dev_id", "sprint_id"])

    records = []

    for (dev_id, sprint_id), group in grouped:

        avg_completion_time = group["completion_time"].mean()
        defect_rate = group["defects"].mean()
        velocity = group["velocity_contribution"].sum()
        blocked = group.get("blocked_tasks", pd.Series([0])).sum()
        reopened = group.get("reopened_tasks", pd.Series([0])).sum()

        # Variability = stability proxy
        variance_time = group["completion_time"].var() if len(group) > 1 else 0

        # -------------------------------
        # TARGET: CONSISTENCY (INITIAL LABEL)
        # -------------------------------
        consistency = 1 / (1 + variance_time + defect_rate + blocked * 0.5)

        records.append({
            "dev_id": dev_id,
            "sprint_id": sprint_id,
            "avg_completion_time": avg_completion_time,
            "defect_rate": defect_rate,
            "velocity": velocity,
            "blocked_tasks": blocked,
            "reopened_tasks": reopened,
            "time_variance": variance_time,
            "consistency": consistency
        })

    return pd.DataFrame(records)