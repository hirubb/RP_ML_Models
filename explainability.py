import pandas as pd
import numpy as np
import shap
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Generates human-readable explanations for developer recommendations.
    Uses multiple methods: SHAP values, feature importance, and rule-based reasoning.
    """

    def __init__(self, model, feature_columns: List[str], shap_background_data=None):
        """
        Initialize explainability engine.
        
        Args:
            model: Trained ML model
            feature_columns: List of feature names used in training
            shap_background_data: Background data for SHAP (if None, uses synthetic)
        """
        self.model = model
        self.feature_columns = feature_columns
        
        # Create SHAP explainer
        try:
            # For tree-based models (RandomForest, XGBoost, LightGBM)
            self.shap_explainer = shap.TreeExplainer(model)
            logger.info("✓ SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP TreeExplainer: {e}")
            self.shap_explainer = None

    def explain_recommendation(self,
                               developer_id: str,
                               task_profile: Dict[str, Any],
                               dev_features: pd.DataFrame,
                               ml_score: float,
                               skill_match_score: float,
                               workload_balance: float,
                               consistency: float,
                               learning_rate: float,
                               final_score: float,
                               rank: int) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for why this developer was recommended.
        
        Returns a dict with:
        - summary: One-sentence explanation
        - breakdown: Component scores and their contributions
        - strengths: Top 3 reasons this developer is good for this task
        - concerns: Potential concerns or constraints
        - shap_values: Feature importance from SHAP if available
        """

        explanation = {
            "dev_id": developer_id,
            "rank": rank,
            "final_score": round(final_score, 2),
            "summary": "",
            "breakdown": {},
            "strengths": [],
            "concerns": [],
            "shap_explanation": None,
            "recommendation_confidence": self._calculate_confidence(final_score, rank)
        }

        # =========================================
        # 1. SUMMARY (One-liner)
        # =========================================
        explanation["summary"] = self._generate_summary(
            developer_id, rank, final_score,
            skill_match_score, workload_balance, 
            consistency, learning_rate, task_profile
        )

        # =========================================
        # 2. SCORE BREAKDOWN (Component Analysis)
        # =========================================
        explanation["breakdown"] = self._explain_score_breakdown(
            ml_score, skill_match_score, workload_balance, 
            consistency, learning_rate, final_score
        )

        # =========================================
        # 3. STRENGTHS (Why they're a good fit)
        # =========================================
        explanation["strengths"] = self._identify_strengths(
            developer_id, dev_features, task_profile,
            skill_match_score, workload_balance,
            consistency, learning_rate
        )

        # =========================================
        # 4. CONCERNS (Potential issues)
        # =========================================
        explanation["concerns"] = self._identify_concerns(
            developer_id, dev_features, task_profile,
            skill_match_score, workload_balance, ml_score
        )

        # =========================================
        # 5. SHAP EXPLANATION (Feature importance)
        # =========================================
        if self.shap_explainer is not None:
            try:
                explanation["shap_explanation"] = self._explain_with_shap(
                    developer_id, dev_features
                )
            except Exception as e:
                logger.warning(f"Could not generate SHAP explanation: {e}")

        return explanation

    def _generate_summary(self, dev_id: str, rank: int, final_score: float,
                          skill_match: float, workload_balance: float,
                          consistency: float, learning_rate: float,
                          task_profile: Dict) -> str:
        """
        Generate a one-sentence explanation.
        """
        task_type = task_profile.get("taskType", "this task").lower()
        
        if rank == 1:
            prefix = "Top choice:"
        elif rank <= 3:
            prefix = f"Strong option (#{rank}):"
        else:
            prefix = f"Viable option (#{rank}):"

        if skill_match > 0.8:
            skill_phrase = "excellent skill match"
        elif skill_match > 0.6:
            skill_phrase = "good skill fit"
        else:
            skill_phrase = "adequate skills"

        if workload_balance > 0.7:
            workload_phrase = "light workload"
        elif workload_balance > 0.4:
            workload_phrase = "moderate workload"
        else:
            workload_phrase = "heavy workload"

        if consistency > 0.7:
            behavior_phrase = "high consistency"
        elif learning_rate > 0.6:
            behavior_phrase = "fast learning rate"
        else:
            behavior_phrase = "reliable performance"

        return f"{prefix} {dev_id} has {skill_phrase} for {task_type}, {workload_phrase}, and {behavior_phrase} (score: {final_score:.1f})"

    def _explain_score_breakdown(self, ml_score: float, skill_match: float,
                                  workload_balance: float, consistency: float,
                                  learning_rate: float, final_score: float) -> Dict:
        """
        Break down how the final score was calculated.
        Weights: 15% ML + 35% Skill + 30% Workload + 10% Consistency + 10% Learning
        """
        ml_contribution = (ml_score / max(ml_score, 1)) * 100 * 0.15 if ml_score > 0 else 0
        skill_contribution = skill_match * 35
        workload_contribution = workload_balance * 30
        consistency_contribution = consistency * 10
        learning_contribution = learning_rate * 10

        return {
            "ml_prediction": {
                "value": round(ml_score, 2),
                "weight": "15%",
                "contribution": round(ml_contribution, 2),
                "explanation": "ML model's estimate of task success"
            },
            "skill_match": {
                "value": round(skill_match, 3),
                "weight": "35%",
                "contribution": round(skill_contribution, 2),
                "explanation": "Alignment with task requirements"
            },
            "workload_balance": {
                "value": round(workload_balance, 3),
                "weight": "30%",
                "contribution": round(workload_contribution, 2),
                "explanation": "Availability and capacity"
            },
            "consistency": {
                "value": round(consistency, 2),
                "weight": "10%",
                "contribution": round(consistency_contribution, 2),
                "explanation": "Historical performance stability"
            },
            "learning_rate": {
                "value": round(learning_rate, 2),
                "weight": "10%",
                "contribution": round(learning_contribution, 2),
                "explanation": "Speed of skill improvement"
            },
            "total": round(final_score, 2)
        }

    def _identify_strengths(self, dev_id: str, dev_features: pd.DataFrame,
                            task_profile: Dict, skill_match: float,
                            workload_balance: float, consistency: float,
                            learning_rate: float) -> List[Dict]:
        """
        Identify top 3 reasons this developer is good for this task.
        """
        strengths = []

        # Extract dev data
        dev = dev_features[dev_features['dev_id'] == dev_id]
        if dev.empty:
            return strengths

        dev = dev.iloc[0]
        task_type = task_profile.get("taskType", "").lower()

        # =========================================
        # Strength 1: Skill Match
        # =========================================
        if skill_match > 0.8:
            strength_text = f"Excellent skill match ({skill_match:.0%})"
            detail = "Developer's skills closely align with task requirements"
        elif skill_match > 0.6:
            strength_text = f"Good skill match ({skill_match:.0%})"
            detail = "Developer has relevant experience for this task type"
        elif skill_match > 0.4:
            strength_text = f"Adequate skill match ({skill_match:.0%})"
            detail = "Developer can handle this task with their experience"
        else:
            strength_text = None

        if strength_text:
            strengths.append({
                "strength": strength_text,
                "detail": detail,
                "metric": "Skill alignment"
            })

        # =========================================
        # Strength 2: Experience Level & Specialization
        # =========================================
        exp_level = dev.get("experience_level", 1)
        exp_map = {1: "junior", 2: "mid-level", 3: "senior"}
        exp_text = exp_map.get(int(exp_level), "experienced")

        # Check task-specific skills
        if task_type == "frontend":
            primary_skill = dev.get("skill_frontend", 1)
            skill_name = "Frontend skills"
        elif task_type == "backend":
            primary_skill = dev.get("skill_backend", 1)
            skill_name = "Backend skills"
        elif task_type == "db":
            primary_skill = dev.get("skill_db", 1)
            skill_name = "Database skills"
        else:
            primary_skill = (dev.get("skill_frontend", 1) + dev.get("skill_backend", 1) + dev.get("skill_db", 1)) / 3
            skill_name = "Overall skills"

        if primary_skill >= 4:
            strengths.append({
                "strength": f"Strong {exp_text} developer ({skill_name}: {int(primary_skill)}/5)",
                "detail": f"Has deep expertise in the required area",
                "metric": "Experience"
            })
        elif primary_skill >= 2:
            strengths.append({
                "strength": f"{exp_text.capitalize()} developer with relevant experience",
                "detail": f"Has worked on similar tasks before",
                "metric": "Experience"
            })

        # =========================================
        # Strength 3: Availability
        # =========================================
        if workload_balance > 0.7:
            strengths.append({
                "strength": "High availability and capacity",
                "detail": f"Currently has plenty of spare capacity ({workload_balance:.0%})",
                "metric": "Workload"
            })
        elif workload_balance > 0.4:
            strengths.append({
                "strength": "Reasonable availability",
                "detail": f"Can take on this task without excessive overload",
                "metric": "Workload"
            })

        # =========================================
        # Strength 4: Behavioral Metrics
        # =========================================
        if consistency > 0.8:
            strengths.append({
                "strength": "Highly consistent performance",
                "detail": "Historically maintains very stable delivery times and quality",
                "metric": "Behavior"
            })
        elif learning_rate > 0.7:
            strengths.append({
                "strength": "Exceptional learning rate",
                "detail": "Demonstrates rapid skill acquisition and improvement",
                "metric": "Behavior"
            })

        return strengths[:3]  # Top 3 only

    def _identify_concerns(self, dev_id: str, dev_features: pd.DataFrame,
                          task_profile: Dict, skill_match: float,
                          workload_balance: float, ml_score: float) -> List[Dict]:
        """
        Identify potential concerns or constraints.
        """
        concerns = []

        dev = dev_features[dev_features['dev_id'] == dev_id]
        if dev.empty:
            return concerns

        dev = dev.iloc[0]
        task_type = task_profile.get("taskType", "").lower()

        # =========================================
        # Concern 1: Skill Gap
        # =========================================
        if skill_match < 0.5:
            concerns.append({
                "concern": "Skill gap detected",
                "detail": f"Developer's skills don't strongly match task requirements (match: {skill_match:.0%})",
                "severity": "medium"
            })

        # =========================================
        # Concern 2: Workload
        # =========================================
        if workload_balance < 0.3:
            current_workload = dev.get("current_workload", 0)
            current_tasks = dev.get("current_tasks", 0)
            concerns.append({
                "concern": "High current workload",
                "detail": f"Developer is already busy with {int(current_tasks)} tasks (workload: {current_workload} units)",
                "severity": "medium"
            })

        # =========================================
        # Concern 3: Low ML Confidence
        # =========================================
        if ml_score < 30:  # Raw ML score is low
            concerns.append({
                "concern": "Low model confidence",
                "detail": "ML model is uncertain about this developer's success on this task type",
                "severity": "low"
            })

        # =========================================
        # Concern 4: Skill Imbalance
        # =========================================
        skills = [
            dev.get("skill_frontend", 1),
            dev.get("skill_backend", 1),
            dev.get("skill_db", 1)
        ]
        skill_variance = np.std(skills)
        if task_type == "db" and dev.get("skill_db", 1) < 2 and skill_variance < 1:
            concerns.append({
                "concern": "Generalist without DB focus",
                "detail": "Task requires DB work, but developer is a generalist",
                "severity": "low"
            })

        return concerns[:3]  # Top 3 concerns

    def _explain_with_shap(self, dev_id: str, dev_features: pd.DataFrame) -> Dict:
        """
        Use SHAP to explain which features most influenced the prediction.
        """
        if self.shap_explainer is None:
            return None

        try:
            # Get single developer row
            dev_row = dev_features[dev_features['dev_id'] == dev_id]
            if dev_row.empty:
                return None

            # Align to model features
            dev_row = dev_row[self.feature_columns]

            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(dev_row)

            # Convert to human-readable format
            feature_importance = []
            for i, feature in enumerate(self.feature_columns):
                if isinstance(shap_values, list):  # For some models
                    shap_val = shap_values[0][0, i] if len(shap_values) > 0 else 0
                else:
                    shap_val = shap_values[0, i]

                feature_importance.append({
                    "feature": feature,
                    "shap_value": round(float(shap_val), 4),
                    "direction": "positive" if shap_val > 0 else "negative",
                    "magnitude": abs(round(float(shap_val), 4))
                })

            # Sort by magnitude
            feature_importance.sort(key=lambda x: x["magnitude"], reverse=True)

            return {
                "method": "SHAP (TreeExplainer)",
                "top_features": feature_importance[:5],  # Top 5
                "interpretation": self._interpret_shap_features(feature_importance[:3])
            }

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None

    def _interpret_shap_features(self, top_features: List[Dict]) -> str:
        """
        Convert SHAP features into plain English.
        """
        if not top_features:
            return "No significant features found."

        interpretations = []
        for feat in top_features:
            feature_name = feat["feature"].replace("_", " ").title()
            direction = "increased" if feat["direction"] == "positive" else "decreased"
            interpretations.append(
                f"{feature_name} {direction} the score"
            )

        return "; ".join(interpretations) + "."

    def _calculate_confidence(self, final_score: float, rank: int) -> Dict:
        """
        Calculate how confident we are in this recommendation.
        """
        if rank == 1 and final_score > 75:
            confidence_level = "high"
            confidence_pct = 0.85
        elif rank <= 3 and final_score > 60:
            confidence_level = "medium"
            confidence_pct = 0.70
        elif final_score > 50:
            confidence_level = "medium"
            confidence_pct = 0.60
        else:
            confidence_level = "low"
            confidence_pct = 0.40

        return {
            "level": confidence_level,
            "percentage": round(confidence_pct * 100),
            "explanation": self._confidence_explanation(confidence_level, rank)
        }

    def _confidence_explanation(self, confidence_level: str, rank: int) -> str:
        """
        Generate confidence explanation text.
        """
        if confidence_level == "high":
            return "Strong recommendation - clear fit for this task"
        elif confidence_level == "medium":
            if rank == 1:
                return "Good recommendation - best available option"
            else:
                return "Viable option - consider if top choice unavailable"
        else:
            return "Alternative option - may require additional support or monitoring"


def explain_ranking(ranking_df: pd.DataFrame,
                    dev_features: pd.DataFrame,
                    task_profile: Dict,
                    explainability_engine: ExplainabilityEngine) -> List[Dict]:
    """
    Generate explanations for all developers in a ranking.
    """
    explained_rankings = []

    for idx, (_, row) in enumerate(ranking_df.iterrows()):
        explanation = explainability_engine.explain_recommendation(
            developer_id=row["dev_id"],
            task_profile=task_profile,
            dev_features=dev_features,
            ml_score=row["predicted_performance"],
            skill_match_score=row["skill_match_score"],
            workload_balance=row["workload_balance"],
            consistency=row.get("consistency", 0.5),
            learning_rate=row.get("learning_rate", 0.1),
            final_score=row["final_score"],
            rank=row["rank"]
        )

        explained_rankings.append(explanation)

    return explained_rankings