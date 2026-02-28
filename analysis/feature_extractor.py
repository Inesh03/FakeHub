import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have: author, text, published_at, author_channel_id
    Returns df with feature columns added.
    """
    features = []

    for author, group in df.groupby("author"):
        group = group.sort_values("published_at")
        texts = group["text"].tolist()

        # Timing features
        if len(group) > 1:
            deltas = group["published_at"].diff().dt.total_seconds().dropna()
            interval_std  = deltas.std()
            interval_mean = deltas.mean()
            exact_flag    = 1 if interval_std < 5 else 0
        else:
            interval_std  = np.nan
            interval_mean = np.nan
            exact_flag    = 0

        # Text similarity features
        if len(texts) > 1:
            vec = TfidfVectorizer().fit_transform(texts)
            sim_matrix = cosine_similarity(vec)
            np.fill_diagonal(sim_matrix, 0)
            text_sim_avg = sim_matrix.mean()
        else:
            text_sim_avg = 0.0

        # Linguistic features
        all_text = " ".join(texts)
        exclamation_ratio = all_text.count("!") / max(len(all_text), 1)
        caps_ratio        = sum(1 for c in all_text if c.isupper()) / max(len(all_text), 1)
        url_count         = sum(1 for t in texts if "http" in t or "www." in t)

        features.append({
            "author":             author,
            "comment_count":      len(group),
            "interval_std":       interval_std,
            "interval_mean":      interval_mean,
            "exact_interval_flag": exact_flag,
            "text_similarity_avg": text_sim_avg,
            "exclamation_ratio":  exclamation_ratio,
            "caps_ratio":         caps_ratio,
            "url_count":          url_count,
        })

    return pd.DataFrame(features)
