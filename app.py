from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import logging
from typing import Dict, Any

# Import ranking and explainability functions
from developer_ranker import rank_developers_for_task, allocate_sprint, rank_sprint_tasks
from explainability import ExplainabilityEngine, explain_ranking

#import behaviour service
from behavior_service import BehaviorService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agile Resource Allocation with Explainability", version="2.0.0")

# =========================================
# CORS Configuration
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# LOAD MODEL & EXPLAINABILITY ENGINE
# =========================================
try:
    model = joblib.load("models/best_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    logger.info("✓ ML model loaded successfully!")
    
    # Initialize explainability engine
    explainability_engine = ExplainabilityEngine(
        model=model,
        feature_columns=feature_columns
    )
    logger.info("✓ Explainability engine initialized!")
    
except Exception as e:
    logger.error(f"✗ Error loading models: {e}")
    model = None
    feature_columns = None
    explainability_engine = None

# =========================================
# HEALTH CHECK
# =========================================
@app.get("/")
def home():
    return {
        "message": "ML API with Explainability running",
        "status": "healthy" if model else "unhealthy",
        "version": "2.0.0"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "explainability_ready": explainability_engine is not None,
        "feature_count": len(feature_columns) if feature_columns else 0
    }

# =========================================
# PREDICT / RANK DEVELOPERS (WITH EXPLANATIONS)
# =========================================
@app.post("/predict")
def predict(data: dict):
    """
    Rank developers for a task with EXPLANATIONS.
    
    Request:
    {
        "task": {
            "taskType": "frontend",
            "taskComplexity": 7,
            "storyPoints": 5,
            "reqFrontend": 4,
            "reqBackend": 1,
            "reqDb": 0
        },
        "developers": [
            {
                "dev_id": "D1",
                "experience_level": 2,
                "skill_frontend": 4,
                "skill_backend": 2,
                "skill_db": 1,
                "current_tasks": 1,
                "current_workload": 10,
                "availability": 75.5
            },
            ...
        ]
    }
    """
    
    if model is None or explainability_engine is None:
        return {"error": "Model not loaded", "status": "error"}

    logger.info("📊 RECEIVED PREDICTION REQUEST")
    
    try:
        task = data.get("task")
        developers = data.get("developers")
        
        if not task or not developers:
            return {"error": "Task and developers are required", "status": "error"}

        # Convert to DataFrame
        dev_df = pd.DataFrame(developers)
        logger.info(f"Processing {len(dev_df)} developers")
        
        # Get rankings
        ranking_df = rank_developers_for_task(
            task_profile=task,
            dev_df=dev_df,
            model=model,
            feature_columns=feature_columns
        )
        
        # Generate explanations for top 5
        explained_rankings = explain_ranking(
            ranking_df=ranking_df.head(5),
            dev_features=dev_df,
            task_profile=task,
            explainability_engine=explainability_engine
        )
        
        logger.info(f"✓ Generated explanations for {len(explained_rankings)} recommendations")
        
        return {
            "status": "success",
            "task_id": task.get("id", "unknown"),
            "task_type": task.get("taskType", "unknown"),
            "total_developers": len(dev_df),
            "recommendations": explained_rankings
        }
    
    except Exception as e:
        logger.error(f"✗ Error during prediction: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

# =========================================
# ALLOCATE SPRINT (WITH EXPLANATIONS)
# =========================================
@app.post("/allocate-sprint")
def allocate_sprint_endpoint(data: dict):
    """
    Allocate an entire sprint with explanations for each assignment.
    
    Request:
    {
        "tasks": [
            {"id": "T1", "taskType": "frontend", "taskComplexity": 7, ...},
            ...
        ],
        "developers": [...]
    }
    """
    
    if model is None or explainability_engine is None:
        return {"error": "Model not loaded", "status": "error"}

    tasks = data.get("tasks", [])
    developers = data.get("developers", [])

    logger.info(f"🚀 SPRINT ALLOCATION: {len(tasks)} tasks, {len(developers)} devs")

    try:
        if not tasks or not developers:
            return {"error": "Tasks and developers are required", "status": "error"}

        dev_df = pd.DataFrame(developers)
        
        # Allocate sprint
        allocations = allocate_sprint(
            tasks=tasks,
            dev_df=dev_df,
            model=model,
            feature_columns=feature_columns
        )
        
        # Generate explanations for each allocation
        explained_allocations = []
        
        for alloc in allocations:
            # Find the task
            task = next((t for t in tasks if t.get("id") == alloc["task_id"]), None)
            if not task:
                continue
            
            # Rank developers for this task to get detailed scores
            ranking_df = rank_developers_for_task(
                task_profile=task,
                dev_df=dev_df,
                model=model,
                feature_columns=feature_columns
            )
            
            # Get the allocated developer's scores
            alloc_dev_row = ranking_df[ranking_df["dev_id"] == alloc["dev_id"]]
            if alloc_dev_row.empty:
                continue
                
            top_dev_row = alloc_dev_row.iloc[0]
            
            # Generate explanation
            explanation = explainability_engine.explain_recommendation(
                developer_id=alloc["dev_id"],
                task_profile=task,
                dev_features=dev_df,
                ml_score=top_dev_row["predicted_performance"],
                skill_match_score=top_dev_row["skill_match_score"],
                workload_balance=top_dev_row["workload_balance"],
                final_score=top_dev_row["final_score"],
                rank=int(top_dev_row["rank"])
            )
            
            explained_allocations.append({
                "task_id": alloc["task_id"],
                "allocated_to": alloc["dev_id"],
                "match_score": round(float(alloc["match_score"]), 2),
                "skill_match": round(float(alloc["skill_match"]), 3),
                "explanation": explanation
            })
        
        logger.info(f"✓ Allocated {len(explained_allocations)} tasks")
        
        return {
            "status": "success",
            "total_tasks": len(tasks),
            "allocated": len(explained_allocations),
            "allocations": explained_allocations
        }
    
    except Exception as e:
        logger.error(f"✗ Error during allocation: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

# =========================================
# RANK SPRINT TASKS (WITH EXPLANATIONS)
# =========================================
@app.post("/rank-sprint")
def rank_sprint_endpoint(data: dict):
    """
    Rank developers for each task in a sprint (no final allocation).
    Each task gets top 5 recommendations with explanations.
    """
    
    if model is None or explainability_engine is None:
        return {"error": "Model not loaded", "status": "error"}

    tasks = data.get("tasks", [])
    developers = data.get("developers", [])

    logger.info(f"📋 SPRINT RANKING: {len(tasks)} tasks, {len(developers)} devs")

    try:
        if not tasks or not developers:
            return {"error": "Tasks and developers are required", "status": "error"}

        dev_df = pd.DataFrame(developers)
        
        # Get recommendations (this returns ranking for each task)
        recommendations = rank_sprint_tasks(
            tasks=tasks,
            dev_df=dev_df,
            model=model,
            feature_columns=feature_columns,
            top_n=5
        )
        
        # Add explanations to each recommendation
        for task_rec in recommendations:
            task = next((t for t in tasks if t.get("id") == task_rec["task_id"]), None)
            if not task:
                continue
            
            # Generate explanations for each developer in recommendations
            for dev_rec in task_rec["recommendations"]:
                explanation = explainability_engine.explain_recommendation(
                    developer_id=dev_rec["dev_id"],
                    task_profile=task,
                    dev_features=dev_df,
                    ml_score=dev_rec["predicted_performance"],
                    skill_match_score=dev_rec["skill_match_score"],
                    workload_balance=dev_rec["workload_balance"],
                    final_score=dev_rec["final_score"],
                    rank=int(dev_rec["rank"])
                )
                dev_rec["explanation"] = explanation
        
        logger.info(f"✓ Generated ranking explanations for {len(recommendations)} tasks")
        
        return {
            "status": "success",
            "total_tasks": len(tasks),
            "recommendations": recommendations
        }
    
    except Exception as e:
        logger.error(f"✗ Error during ranking: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

# =========================================
# EXPLAIN SINGLE RECOMMENDATION
# =========================================
@app.post("/explain")
def explain_recommendation(data: dict):
    """
    Get detailed explanation for a single developer recommendation.
    
    Request:
    {
        "dev_id": "D1",
        "task": {...},
        "developers": [...],
        "predicted_performance": 72.5,
        "skill_match_score": 0.85,
        "workload_balance": 0.65,
        "final_score": 82.1
    }
    """
    
    if model is None or explainability_engine is None:
        return {"error": "Model not loaded", "status": "error"}

    try:
        dev_id = data.get("dev_id")
        task = data.get("task")
        developers = data.get("developers", [])
        ml_score = data.get("predicted_performance", 0)
        skill_match = data.get("skill_match_score", 0)
        workload_balance = data.get("workload_balance", 0)
        final_score = data.get("final_score", 0)
        
        if not dev_id or not task or not developers:
            return {"error": "dev_id, task, and developers are required", "status": "error"}
        
        dev_df = pd.DataFrame(developers)
        
        explanation = explainability_engine.explain_recommendation(
            developer_id=dev_id,
            task_profile=task,
            dev_features=dev_df,
            ml_score=ml_score,
            skill_match_score=skill_match,
            workload_balance=workload_balance,
            final_score=final_score,
            rank=1
        )
        
        return {
            "status": "success",
            "dev_id": dev_id,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"✗ Error during explanation: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

@app.post("/behavior/predict-all")
def predict_all(data: dict):
    sprint_logs = pd.DataFrame(data["logs"])

    service = BehaviorService()

    result_df = service.generate_behavioral_profile(sprint_logs)

    return {
        "developers": result_df.to_dict(orient="records")
    }


# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("Starting Agile Resource Allocation API with Explainability")
    logger.info("=" * 70)
    logger.info("Endpoints available:")
    logger.info("  GET  /            (health check)")
    logger.info("  GET  /health      (detailed health)")
    logger.info("  POST /predict     (rank developers with explanations)")
    logger.info("  POST /allocate-sprint (allocate sprint with explanations)")
    logger.info("  POST /rank-sprint (rank all sprint tasks)")
    logger.info("  POST /explain     (explain single recommendation)")
    logger.info("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )