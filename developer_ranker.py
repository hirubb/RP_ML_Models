import pandas as pd
import numpy as np
import math
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def rank_developers_for_task(
        task_profile: dict,
        dev_df: pd.DataFrame,
        model,
        feature_columns: list
) -> pd.DataFrame:
    """
    Rank developers using ML prediction + explicit skill matching + workload balance + growth incentive
    
    CRITICAL: skill_match_score and velocity_contribution are CALCULATED HERE
    (not retrieved from a database or CSV)
    They must match exactly how they were calculated during training
    """

    # Normalize column names (Spring Boot may use camelCase)
    column_mapping = {
        'devid': 'dev_id',
        'experienceinlevel': 'experience_level',
        'skillfrontend': 'skill_frontend',
        'skillbackend': 'skill_backend',
        'skilldb': 'skill_db',
        'currenttasks': 'current_tasks',
        'currentworkload': 'current_workload',
    }

    for old_name, new_name in column_mapping.items():
        if old_name in dev_df.columns:
            dev_df = dev_df.rename(columns={old_name: new_name})

    candidates = []

    # =========================================
    # BUILD CANDIDATE ROWS WITH CALCULATED FEATURES
    # =========================================
    for _, dev in dev_df.iterrows():

        # ✅ STEP 1: Calculate skill_match_score
        # This MUST match the training data calculation
        frontend_match = max(0, 1 - (abs(dev["skill_frontend"] - task_profile.get("reqFrontend", task_profile.get("req_frontend", 0))) / 5))
        backend_match = max(0, 1 - (abs(dev["skill_backend"] - task_profile.get("reqBackend", task_profile.get("req_backend", 0))) / 5))
        db_match = max(0, 1 - (abs(dev["skill_db"] - task_profile.get("reqDb", task_profile.get("req_db", 0))) / 5))

        # Weight by task type (primary skill gets 60%)
        task_type = task_profile.get("taskType", "").lower()
        if task_type == "frontend":
            skill_match_score = frontend_match * 0.6 + backend_match * 0.25 + db_match * 0.15
        elif task_type == "backend":
            skill_match_score = backend_match * 0.6 + frontend_match * 0.25 + db_match * 0.15
        elif task_type == "db":
            skill_match_score = db_match * 0.6 + backend_match * 0.25 + frontend_match * 0.15
        else:
            skill_match_score = (frontend_match + backend_match + db_match) / 3

        skill_match_score = max(0, min(1, skill_match_score))

        # ✅ STEP 2: Calculate velocity_contribution
        # Story points weighted by skill match and workload (burnout reduces capacity)
        current_workload = dev["current_workload"]
        burnout = np.tanh(current_workload / 40)
        story_points = task_profile.get("storyPoints", task_profile.get("story_points", 3))
        velocity_contribution = story_points * skill_match_score * (1 - burnout * 0.5)

        # ✅ STEP 3: Interaction Features (New for Stacking V2)
        skill_exp_inter = float(skill_match_score * dev["experience_level"])
        workload_density = float(dev["current_workload"] / (dev["availability"] + 1))
        task_complexity = int(task_profile.get("taskComplexity", task_profile.get("task_complexity", 5)))
        story_points = int(task_profile.get("storyPoints", task_profile.get("story_points", 3)))
        complexity_sp_ratio = float(task_complexity / (story_points + 1))

        # Build feature row (MUST have all training features)
        row = {
            "sprint_id": 0,  # For prediction, sprint_id is not relevant
            "experience_level": int(dev["experience_level"]),
            "skill_frontend": int(dev["skill_frontend"]),
            "skill_backend": int(dev["skill_backend"]),
            "skill_db": int(dev["skill_db"]),
            "task_complexity": task_complexity,
            "story_points": story_points,
            "req_frontend": int(task_profile.get("reqFrontend", task_profile.get("req_frontend", 0))),
            "req_backend": int(task_profile.get("reqBackend", task_profile.get("req_backend", 0))),
            "req_db": int(task_profile.get("reqDb", task_profile.get("req_db", 0))),
            "current_tasks": int(dev["current_tasks"]),
            "current_workload": int(dev["current_workload"]),
            "availability": float(dev["availability"]),
            
            # ✅ CRITICAL: These are calculated ABOVE
            "skill_match_score": float(skill_match_score),
            "velocity_contribution": float(velocity_contribution),
            
            # ✅ NEW: Interaction Features
            "skill_exp_inter": skill_exp_inter,
            "workload_density": workload_density,
            "complexity_sp_ratio": complexity_sp_ratio,

            # ✅ BEHAVIORAL METRICS (Updated after each sprint)
            "consistency": float(dev.get("consistency", 0.5)),
            "learning_rate": float(dev.get("learning_rate", 0.1)),

            # One-hot task type
            "task_type_backend": 1 if task_type == "backend" else 0,
            "task_type_frontend": 1 if task_type == "frontend" else 0,
            "task_type_db": 1 if task_type == "db" else 0,
            
            # Store dev ID separately
            "_dev_id": dev["dev_id"],
        }

        candidates.append(row)

    # =========================================
    # CREATE DATAFRAME AND ALIGN FEATURES
    # =========================================
    candidates_df = pd.DataFrame(candidates)
    dev_ids = candidates_df["_dev_id"].values
    candidates_df = candidates_df.drop(columns=["_dev_id"])

    # ✅ Align to model's expected features (CRITICAL)
    candidates_df = candidates_df.reindex(columns=feature_columns, fill_value=0)

    logger.info(f"✓ Prepared {len(candidates_df)} candidates for prediction")
    logger.debug(f"  Columns: {list(candidates_df.columns)}")

    # =========================================
    # ML PREDICTION
    # =========================================
    try:
        predicted = model.predict(candidates_df)
        logger.info(f"✓ ML predictions: min={predicted.min():.2f}, max={predicted.max():.2f}, mean={predicted.mean():.2f}")
    except Exception as e:
        logger.error(f"Error in model.predict(): {e}")
        raise

    # =========================================
    # EXTRACT FEATURES FOR FINAL SCORING
    # =========================================
    skill_match_array = candidates_df["skill_match_score"].values
    raw_workloads = candidates_df["current_workload"].values.astype(float)

    # ✅ WORKLOAD BALANCE & FAIRNESS:
    # 1. Exponential decay relative to team mean workload
    mean_workload = raw_workloads.mean() if raw_workloads.mean() > 0 else 1.0
    workload_balance_array = np.exp(-raw_workloads / (mean_workload + 1e-6))
    
    # 2. Task count fairness (penalize devs with active tasks when others are free)
    current_tasks_array = candidates_df["current_tasks"].values.astype(float)
    min_tasks = current_tasks_array.min()
    task_fairness = []
    for t in current_tasks_array:
        if t == 0:
            task_fairness.append(1.0)  # Idle developer boost
        elif t > min_tasks:
            task_fairness.append(1.0 / (1.0 + 0.6 * (t - min_tasks) ** 1.5))
        else:
            task_fairness.append(0.9)
    
    combined_workload_balance = np.clip(workload_balance_array * np.array(task_fairness), 0, 1)

    logger.info(
        f"  Workload balance — mean={mean_workload:.1f}, "
        f"min={combined_workload_balance.min():.3f}, max={combined_workload_balance.max():.3f}"
    )

    # =========================================
    # NORMALIZE ML PREDICTIONS
    # =========================================
    max_pred = predicted.max()
    if max_pred > 0:
        pred_norm = (predicted / max_pred) * 100
    else:
        pred_norm = np.zeros_like(predicted)

    # =========================================
    # BEHAVIORAL & GROWTH INCENTIVES (Option 3 Logic)
    # =========================================
    consistency_array = candidates_df["consistency"].values if "consistency" in candidates_df.columns else np.array([0.5] * len(candidates_df))
    learning_rate_array = candidates_df["learning_rate"].values if "learning_rate" in candidates_df.columns else np.array([0.1] * len(candidates_df))

    # Growth Incentive: Enable junior (exp=1) and mid-level (exp=2) developers for manageable tasks (complexity <= 6)
    # when they meet the baseline requirement threshold (skill_match >= 0.30)
    task_complexity = int(task_profile.get("taskComplexity", task_profile.get("task_complexity", 5)))
    growth_bonuses = []
    for _, dev_row in candidates_df.iterrows():
        exp = dev_row.get("experience_level", 1)
        lr = dev_row.get("learning_rate", 0.1)
        sm = dev_row.get("skill_match_score", 0.0)
        
        if task_complexity <= 6 and sm >= 0.30:
            # Junior (1) gets 1.0x, Mid (2) gets 0.5x, Senior (3) gets 0.0x
            growth_mult = max(0.0, (3 - min(exp, 3)) / 2.0)
            complexity_fit = max(0.0, 1.0 - (task_complexity / 8.0))
            bonus = lr * growth_mult * complexity_fit * 12.0
        else:
            bonus = 0.0
        growth_bonuses.append(bonus)

    growth_bonus_array = np.array(growth_bonuses)

    # =========================================
    # FINAL COMPOSITE SCORE
    # =========================================
    final_score = (
        pred_norm * 0.15 +                        # ML prediction (15%)
        skill_match_array * 35 +                 # Skill match (35%)
        combined_workload_balance * 30 +         # Workload & fairness (30%)
        consistency_array * 10 +                 # Consistency (10%)
        learning_rate_array * 10 +               # Learning rate (10%)
        growth_bonus_array                       # Growth incentive bonus
    )
    final_score = np.clip(final_score, 0, 100)

    # =========================================
    # BUILD RESULTS
    # =========================================
    results = pd.DataFrame({
        "dev_id": dev_ids,
        "skill_frontend": dev_df["skill_frontend"].values,
        "skill_backend": dev_df["skill_backend"].values,
        "skill_db": dev_df["skill_db"].values,
        "experience_level": dev_df["experience_level"].values,
        "current_tasks": dev_df["current_tasks"].values,
        "current_workload": dev_df["current_workload"].values,
        "availability": dev_df["availability"].values,
        "skill_match_score": np.round(skill_match_array, 3),
        "predicted_performance": np.round(predicted, 2),
        "workload_balance": np.round(combined_workload_balance, 3),
        "learning_rate": np.round(learning_rate_array, 2),
        "consistency": np.round(consistency_array, 2),
        "growth_bonus": np.round(growth_bonus_array, 2),
        "final_score": np.round(final_score, 2)
    })

    # =========================================
    # SORT AND RANK
    # =========================================
    results = results.sort_values("final_score", ascending=False).reset_index(drop=True)
    results.insert(0, "rank", range(1, len(results) + 1))

    logger.info(f"✓ Ranked {len(results)} developers")
    if len(results) > 0:
        top = results.iloc[0]
        logger.info(f"  Top: {top['dev_id']} | skill_match={top['skill_match_score']:.3f} | final_score={top['final_score']:.2f}")

    return results


def allocate_sprint(
        tasks: list,
        dev_df: pd.DataFrame,
        model,
        feature_columns: list
) -> list:
    """
    Allocate an entire sprint using GLOBAL LINEAR ASSIGNMENT (Hungarian Algorithm).
    
    Mathematically optimizes total sprint score while strictly balancing workload
    across the entire team and preventing single-developer task hoarding.
    """
    if not tasks or dev_df.empty:
        return []

    # Normalize column names immediately
    current_dev_df = dev_df.copy()
    column_mapping = {
        'devid': 'dev_id',
        'experienceinlevel': 'experience_level',
        'skillfrontend': 'skill_frontend',
        'skillbackend': 'skill_backend',
        'skilldb': 'skill_db',
        'currenttasks': 'current_tasks',
        'currentworkload': 'current_workload',
    }
    for old_name, new_name in column_mapping.items():
        if old_name in current_dev_df.columns:
            current_dev_df = current_dev_df.rename(columns={old_name: new_name})

    n_tasks = len(tasks)
    n_devs = len(current_dev_df)
    dev_ids = current_dev_df["dev_id"].tolist()

    logger.info(f"🚀 Starting GLOBAL LINEAR ASSIGNMENT for {n_tasks} tasks and {n_devs} developers...")

    # Calculate capacity slots per developer
    # If tasks > devs (e.g. 7 tasks, 3 devs), each developer gets ceil(7/3) = 3 slots
    max_slots_per_dev = max(1, math.ceil(n_tasks / n_devs))
    
    # Build candidate slot mapping: (dev_idx, dev_id, slot_num)
    slots = []
    for dev_idx, dev_id in enumerate(dev_ids):
        for slot_num in range(max_slots_per_dev):
            slots.append({
                "dev_idx": dev_idx,
                "dev_id": dev_id,
                "slot_num": slot_num
            })
            
    total_slots = len(slots)

    # Build Cost Matrix (N_tasks x Total_Slots)
    # Higher final_score -> Lower Cost in Hungarian minimization
    cost_matrix = np.zeros((n_tasks, total_slots))
    rankings_cache = {}

    for i, task in enumerate(tasks):
        # Rank developers for this task
        ranking_df = rank_developers_for_task(
            task_profile=task,
            dev_df=current_dev_df,
            model=model,
            feature_columns=feature_columns
        )
        rankings_cache[i] = ranking_df

        for j, slot in enumerate(slots):
            dev_id = slot["dev_id"]
            slot_num = slot["slot_num"]
            
            dev_row = ranking_df[ranking_df["dev_id"] == dev_id]
            if not dev_row.empty:
                base_score = float(dev_row.iloc[0]["final_score"])
            else:
                base_score = 0.0

            # Progressive slot penalty:
            # Slot 0 (1st task for dev) = base cost
            # Slot 1 (2nd task for dev) = +25 cost penalty
            # Slot 2 (3rd task for dev) = +50 cost penalty
            # This mathematically ensures the solver fills 1st slots across ALL devs before giving any dev a 2nd slot!
            slot_penalty = slot_num * 25.0
            cost = (100.0 - base_score) + slot_penalty
            cost_matrix[i, j] = cost

    # Solve Global Minimum Cost Bipartite Matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build final allocations
    allocations = []
    for task_idx, slot_col in zip(row_ind, col_ind):
        assigned_slot = slots[slot_col]
        dev_id = assigned_slot["dev_id"]
        task = tasks[task_idx]
        
        ranking_df = rankings_cache[task_idx]
        dev_row = ranking_df[ranking_df["dev_id"] == dev_id].iloc[0]

        allocations.append({
            "task_id": task.get("id", task.get("taskId", f"task-{task_idx}")),
            "dev_id": dev_id,
            "match_score": float(dev_row["final_score"]),
            "skill_match": float(dev_row["skill_match_score"])
        })
        logger.info(f"✅ Globally Assigned Task {task.get('id')} to {dev_id} (Score: {dev_row['final_score']:.1f})")

    return allocations


def rank_sprint_tasks(
        tasks: list,
        dev_df: pd.DataFrame,
        model,
        feature_columns: list,
        top_n: int = 5
) -> list:
    """
    Rank developers for each task in a sprint without final allocation.
    This provides top-N recommendations for an admin to choose from.

    KEY DESIGN: We maintain a LIVE copy of dev_df and update the virtual
    workload of the #1-ranked developer after each task. This ensures
    the recommendations promote other developers as loads accumulate.
    """
    logger.info(f"📋 Generating recommendations for {len(tasks)} tasks...")

    current_dev_df = dev_df.copy()

    # Normalise column names once
    column_mapping = {
        'devid': 'dev_id',
        'experienceinlevel': 'experience_level',
        'skillfrontend': 'skill_frontend',
        'skillbackend': 'skill_backend',
        'skilldb': 'skill_db',
        'currenttasks': 'current_tasks',
        'currentworkload': 'current_workload',
    }
    for old_name, new_name in column_mapping.items():
        if old_name in current_dev_df.columns:
            current_dev_df = current_dev_df.rename(columns={old_name: new_name})

    sorted_tasks = sorted(
        tasks,
        key=lambda x: x.get("storyPoints", x.get("story_points", 0)),
        reverse=True
    )

    sprint_recommendations = []

    for task in sorted_tasks:
        story_points = task.get("storyPoints", task.get("story_points", 3))

        # Rank all developers against current workload snapshot
        ranking_df = rank_developers_for_task(
            task_profile=task,
            dev_df=current_dev_df,
            model=model,
            feature_columns=feature_columns
        )

        top_recommendations = ranking_df.head(top_n).to_dict(orient="records")

        sprint_recommendations.append({
            "task_id": task.get("id", task.get("taskId", "unknown")),
            "task_title": task.get("title", task.get("name", "Unnamed Task")),
            "recommendations": top_recommendations
        })

        # Virtual workload update for top developer
        if not ranking_df.empty:
            top_dev_id = ranking_df.iloc[0]["dev_id"]
            mask = current_dev_df['dev_id'] == top_dev_id
            current_dev_df.loc[mask, "current_tasks"] += 1
            current_dev_df.loc[mask, "current_workload"] += story_points
            current_avail = current_dev_df.loc[mask, "availability"].values[0]
            new_avail = max(0.0, float(current_avail) - (story_points * 5))
            current_dev_df.loc[mask, "availability"] = new_avail
            logger.info(
                f"  📌 Virtual assign: {top_dev_id} → task {task.get('id')} "
                f"(+{story_points} SP | workload now {current_dev_df.loc[mask, 'current_workload'].values[0]:.0f})"
            )

    return sprint_recommendations