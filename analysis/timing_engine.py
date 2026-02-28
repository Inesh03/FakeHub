# analysis/timing_engine.py
import pandas as pd
import numpy as np

HUMAN_STD_THRESHOLD = 30.0   # seconds; below this = suspicious
EXACT_INTERVAL_MAX  = 5.0    # seconds; below this = almost certainly automated

def analyze_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze posting timing regularity per author.
    Returns DataFrame with timing risk scores and behavioral explanations.
    """
    results = []

    for author, group in df.groupby("author"):
        group = group.sort_values("published_at")

        if len(group) < 2:
            results.append({
                "author":           author,
                "timing_risk":      0.0,
                "interval_std":     None,
                "interval_mean":    None,
                "is_exact_interval": False,
                "anomaly_explanation": "Insufficient comment history to analyze timing."
            })
            continue

        deltas = group["published_at"].diff().dt.total_seconds().dropna()
        std  = deltas.std()
        mean = deltas.mean()
        explanation = "Natural posting rhythm."

        if std is None or np.isnan(std):
            timing_risk = 0.0
            explanation = "Unable to compute timing variance."
        elif std < EXACT_INTERVAL_MAX:
            timing_risk = 1.0   # Almost certainly a bot
            explanation = f"Posted {len(group)} comments at exact machine-like intervals ({round(std, 2)}s variance)."
        elif std < HUMAN_STD_THRESHOLD:
            # Linearly scale risk between 0.4 and 0.9
            timing_risk = 0.9 - (std / HUMAN_STD_THRESHOLD) * 0.5
            explanation = f"High posting frequency (average {round(mean, 1)}s between comments) is faster than normal human thresholds."
        else:
            timing_risk = max(0.0, 0.4 - (std / 300))
            if timing_risk > 0.2:
                explanation = "Slightly elevated posting speed."

        results.append({
            "author":            author,
            "timing_risk":       round(timing_risk, 3),
            "interval_std":      round(std, 2),
            "interval_mean":     round(mean, 2),
            "is_exact_interval": std < EXACT_INTERVAL_MAX,
            "anomaly_explanation": explanation
        })

    return pd.DataFrame(results)

def detect_engagement_bursts(df: pd.DataFrame, bucket_freq="1T", threshold_std=3.0) -> pd.DataFrame:
    """
    Detect statistical bursts in overall comment volume to flag "Engagement Burst Patterns".
    Default bucket_freq="1T" (1 minute buckets).
    """
    if df.empty or "published_at" not in df.columns:
        return pd.DataFrame()
        
    df = df.copy()
    df.set_index("published_at", inplace=True)
    df = df.sort_index()
    
    # Resample to count comments per bucket
    volume = df.resample(bucket_freq).size().to_frame(name="comment_count")
    
    if len(volume) < 2:
        volume["is_burst"] = False
        return volume.reset_index()
    
    mean = volume["comment_count"].mean()
    std = volume["comment_count"].std()
    
    # Flag buckets that are X standard deviations above the mean (or outright massive for small datasets)
    volume["is_burst"] = (volume["comment_count"] > (mean + (threshold_std * std))) & (volume["comment_count"] > 3)
    
    return volume.reset_index()
