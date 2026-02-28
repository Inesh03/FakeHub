# analysis/score_engine.py
import pandas as pd

# Tune these weights based on model performance
WEIGHTS = {
    "nlp_bot_prob":      0.40,
    "timing_risk":       0.35,
    "graph_cluster_score": 0.25
}

def compute_per_commenter_scores(
    comments_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    graph_scores: dict
) -> pd.DataFrame:
    """
    Merge all signal sources and compute final bot_score per commenter.
    Returns a DataFrame sorted by bot_score descending.
    """
    merged = comments_df.groupby("author").agg(
        nlp_bot_prob=("nlp_bot_prob", "mean"),
        comment_count=("text", "count")
    ).reset_index()

    merged = merged.merge(timing_df[["author", "timing_risk", "anomaly_explanation"]], on="author", how="left")
    merged["timing_risk"] = merged["timing_risk"].fillna(0.0)
    merged["anomaly_explanation"] = merged["anomaly_explanation"].fillna("Insufficient data for analysis.")

    merged["graph_cluster_score"] = merged["author"].map(graph_scores).fillna(0.0)

    merged["bot_score"] = (
        WEIGHTS["nlp_bot_prob"]       * merged["nlp_bot_prob"] +
        WEIGHTS["timing_risk"]         * merged["timing_risk"] +
        WEIGHTS["graph_cluster_score"] * merged["graph_cluster_score"]
    )

    merged["authenticity_score"] = ((1 - merged["bot_score"]) * 100).round(1)
    merged["risk_label"] = merged["bot_score"].apply(classify_risk)

    # Append graph-level explanations
    def augment_explanation(row):
        base = row["anomaly_explanation"]
        if row["graph_cluster_score"] > 0:
            base += " Identified in a coordinated posting ring."
        if row["nlp_bot_prob"] > 0.7:
            base += " Text shows high linguistic similarity to known bot spam."
        return base

    merged["anomaly_explanation"] = merged.apply(augment_explanation, axis=1)

    return merged.sort_values("bot_score", ascending=False)

def classify_risk(score: float) -> str:
    if score >= 0.75:
        return "🔴 High Risk"
    elif score >= 0.45:
        return "🟡 Suspicious"
    else:
        return "🟢 Likely Human"

def compute_overall_authenticity(per_commenter_df: pd.DataFrame) -> float:
    """
    Overall video-level authenticity score (0–100%).
    Weighted by comment count so prolific bots have more impact.
    """
    total_comments = per_commenter_df["comment_count"].sum()
    weighted_score = (
        per_commenter_df["authenticity_score"] * per_commenter_df["comment_count"]
    ).sum() / max(total_comments, 1)
    return round(weighted_score, 1)
