from fastapi import FastAPI
import pandas as pd
import joblib

# Import ranking function
from developer_ranker import rank_developers_for_task, allocate_sprint, rank_sprint_tasks

app = FastAPI()

# =========================================
# LOAD MODEL
# =========================================
model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("ML model loaded successfully!")


# =========================================
# HEALTH CHECK
# =========================================
@app.get("/")
def home():
    return {"message": "ML API running"}


# =========================================
# PREDICT / RANK DEVELOPERS
# =========================================
@app.post("/predict")
def predict(data: dict):

    print("\nFULL REQUEST:")
    print(data)

    # Task from Spring Boot
    task = data["task"]

    # Developers from Spring Boot DB
    developers = data["developers"]

    print("\nDEVELOPERSSSSSSSSSSS:")
    print(developers)

    # Convert JSON -> DataFrame
    dev_df = pd.DataFrame(developers)

    # Call ranking logic
    ranking_df = rank_developers_for_task(
        task_profile=task,
        dev_df=dev_df,
        model=model,
        feature_columns=feature_columns
    )

    # Return top developers
    return ranking_df.head(5).to_dict(orient="records")
@app.post("/allocate-sprint")
def allocate_sprint_endpoint(data: dict):
    """
    Allocates an entire list of tasks to developers.
    Input: { "tasks": [...], "developers": [...] }
    """
    tasks = data.get("tasks", [])
    developers = data.get("developers", [])

    print(f"\n🚀 RECEIVED SPRINT ALLOCATION REQUEST: {len(tasks)} tasks, {len(developers)} developers")

    if not tasks or not developers:
        return {"error": "Tasks and developers list cannot be empty"}

    dev_df = pd.DataFrame(developers)

    # Call allocation logic
    allocations = allocate_sprint(
        tasks=tasks,
        dev_df=dev_df,
        model=model,
        feature_columns=feature_columns
    )

    return {
        "status": "success",
        "allocations": allocations
    }


@app.post("/rank-sprint")
def rank_sprint_endpoint(data: dict):
    """
    Ranks developers for each task in a sprint.
    Input: { "tasks": [...], "developers": [...] }
    """
    tasks = data.get("tasks", [])
    developers = data.get("developers", [])

    print(f"\n📋 RECEIVED SPRINT RANKING REQUEST: {len(tasks)} tasks, {len(developers)} developers")

    if not tasks or not developers:
        return {"error": "Tasks and developers list cannot be empty"}

    dev_df = pd.DataFrame(developers)

    # Call ranking logic
    recommendations = rank_sprint_tasks(
        tasks=tasks,
        dev_df=dev_df,
        model=model,
        feature_columns=feature_columns
    )

    return {
        "status": "success",
        "recommendations": recommendations
    }


