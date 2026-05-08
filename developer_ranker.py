import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def rank_developers_for_task(
        task_profile: dict,
        dev_df: pd.DataFrame,
        model,
        feature_columns: list
) -> pd.DataFrame:
    """
    Rank developers using ML prediction + explicit skill matching + workload balance
    
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

        # Build feature row (MUST have all training features)
        row = {
            "sprint_id": 0,  # For prediction, sprint_id is not relevant
            "experience_level": int(dev["experience_level"]),
            "skill_frontend": int(dev["skill_frontend"]),
            "skill_backend": int(dev["skill_backend"]),
            "skill_db": int(dev["skill_db"]),
            "task_complexity": int(task_profile.get("taskComplexity", task_profile.get("task_complexity", 5))),
            "story_points": int(task_profile.get("storyPoints", task_profile.get("story_points", 3))),
            "req_frontend": int(task_profile.get("reqFrontend", task_profile.get("req_frontend", 0))),
            "req_backend": int(task_profile.get("reqBackend", task_profile.get("req_backend", 0))),
            "req_db": int(task_profile.get("reqDb", task_profile.get("req_db", 0))),
            "current_tasks": int(dev["current_tasks"]),
            "current_workload": int(dev["current_workload"]),
            "availability": float(dev["availability"]),
            
            # ✅ CRITICAL: These are calculated ABOVE
            "skill_match_score": float(skill_match_score),
            "velocity_contribution": float(velocity_contribution),
            
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
    workload_balance_array = 1 - (candidates_df["current_workload"].values / max(candidates_df["current_workload"].max(), 1))

    # =========================================
    # NORMALIZE ML PREDICTIONS
    # =========================================
    max_pred = predicted.max()
    if max_pred > 0:
        pred_norm = (predicted / max_pred) * 100
    else:
        pred_norm = np.zeros_like(predicted)

    # =========================================
    # FINAL COMPOSITE SCORE
    # Weights are tuned to prioritize skill match
    # =========================================
    final_score = (
        pred_norm * 0.3 +           # ML prediction (30%)
        skill_match_array * 50 +    # Skill match (50%) ← DOMINANT
        workload_balance_array * 20 # Workload balance (20%)
    )

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
        "workload_balance": np.round(workload_balance_array, 3),
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
    Allocate an entire list of tasks to developers greedily.
    Updates developer workload after each assignment to ensure balanced distribution.
    """
    allocations = []
    
    # Sort tasks by story points descending (more complex tasks assigned first)
    sorted_tasks = sorted(
        tasks, 
        key=lambda x: x.get("storyPoints", x.get("story_points", 0)), 
        reverse=True
    )

    # Work with a copy to avoid mutating the original df passed in
    current_dev_df = dev_df.copy()

    # Normalize dev_df columns immediately
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

    logger.info(f"🚀 Starting bulk allocation for {len(tasks)} tasks...")

    for task in sorted_tasks:
        # 1. Rank developers for this specific task
        ranking_df = rank_developers_for_task(
            task_profile=task,
            dev_df=current_dev_df,
            model=model,
            feature_columns=feature_columns
        )

        if ranking_df.empty:
            logger.warning(f"No developers available for task {task.get('id', 'unknown')}")
            continue

        # 2. Pick the #1 ranked developer
        top_dev = ranking_df.iloc[0]
        dev_id = top_dev["dev_id"]
        
        # 3. Store the allocation
        allocations.append({
            "task_id": task.get("id", task.get("taskId", "unknown")),
            "dev_id": dev_id,
            "match_score": float(top_dev["final_score"]),
            "skill_match": float(top_dev["skill_match_score"])
        })

        # 4. Update the developer's workload in current_dev_df for the next task
        story_points = task.get("storyPoints", task.get("story_points", 3))
        
        current_dev_df.loc[current_dev_df['dev_id'] == dev_id, "current_tasks"] += 1
        current_dev_df.loc[current_dev_df['dev_id'] == dev_id, "current_workload"] += story_points
        
        logger.info(f"✅ Assigned Task {task.get('id')} to {dev_id}")

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
    This provides recommendations for an admin to choose from.
    """
    logger.info(f"📋 Generating recommendations for {len(tasks)} tasks...")
    
    sprint_recommendations = []

    for task in tasks:
        # Rank developers for this specific task
        ranking_df = rank_developers_for_task(
            task_profile=task,
            dev_df=dev_df,
            model=model,
            feature_columns=feature_columns
        )

        # Get top N recommendations
        top_recommendations = ranking_df.head(top_n).to_dict(orient="records")

        sprint_recommendations.append({
            "task_id": task.get("id", task.get("taskId", "unknown")),
            "task_title": task.get("title", task.get("name", "Unnamed Task")),
            "recommendations": top_recommendations
        })

    return sprint_recommendations