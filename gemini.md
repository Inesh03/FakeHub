# 🛡️ Fake Engagement Detection — Hackathon Project Documentation

> **Problem Statement 3:** Detect fake/bot engagement on social media using real-time data, NLP, graph analysis, and an interactive dashboard.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Environment Setup](#environment-setup)
6. [Step 1 — YouTube API Integration](#step-1--youtube-api-integration)
7. [Step 2 — Training Data (Kaggle Datasets)](#step-2--training-data-kaggle-datasets)
8. [Step 3 — Feature Engineering](#step-3--feature-engineering)
9. [Step 4 — NLP Model (TF-IDF + Random Forest)](#step-4--nlp-model-tf-idf--random-forest)
10. [Step 5 — Neo4j Graph Database Integration](#step-5--neo4j-graph-database-integration)
11. [Step 6 — Timing Regularity Detection](#step-6--timing-regularity-detection)
12. [Step 7 — Authenticity Score Engine](#step-7--authenticity-score-engine)
13. [Step 8 — Streamlit Dashboard](#step-8--streamlit-dashboard)
14. [Step 9 — Live Demo Flow](#step-9--live-demo-flow)
15. [API Quota Management](#api-quota-management)
16. [Model Evaluation Metrics](#model-evaluation-metrics)
17. [Deployment](#deployment)
18. [Judging Criteria Alignment](#judging-criteria-alignment)
19. [Troubleshooting](#troubleshooting)

---

## Project Overview

This system detects fake engagement (bot activity) on YouTube videos in **real time**. A judge pastes a YouTube video URL into the Streamlit dashboard, the backend fetches the latest 500 comments using the **YouTube Data API v3**, runs them through:

- **NLP pipeline** (TF-IDF + Random Forest) to flag linguistic bot patterns
- **Neo4j graph analysis** to detect coordinated bot ring behavior
- **Timing regularity engine** to flag machine-like posting intervals

...and outputs an **Authenticity Score (0–100%)** with a per-commenter Bot Probability breakdown.

### Why This Is a Hackathon Winner
- Live demo on stage — no static slides needed
- Combines 3 distinct AI/data techniques (NLP + Graph DB + Time-Series)
- Visual Neo4j graph showing bot clusters is highly compelling for judges
- Real-time, end-to-end pipeline built in Python

---

## Architecture Diagram

```
YouTube Video URL
       │
       ▼
┌──────────────────┐
│  YouTube Data    │  ← Google Cloud Console API Key
│  API v3 (PRAW    │
│  fallback: Reddit│
│  PRAW API)       │
└────────┬─────────┘
         │ Raw Comments JSON (500 comments)
         ▼
┌──────────────────────────────────────────────┐
│             Feature Extraction Layer          │
│  ┌──────────────┐  ┌───────────┐  ┌────────┐ │
│  │ NLP Pipeline │  │  Graph DB │  │ Timing │ │
│  │ TF-IDF + RF  │  │  (Neo4j)  │  │ Engine │ │
│  └──────┬───────┘  └─────┬─────┘  └───┬────┘ │
└─────────┼────────────────┼────────────┼──────┘
          │                │            │
          ▼                ▼            ▼
┌──────────────────────────────────────────────┐
│         Score Fusion Module                   │
│   Authenticity Score = weighted average of    │
│   NLP score + Graph score + Timing score      │
└───────────────────┬──────────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │  Streamlit Dashboard │
        │  - Authenticity %   │
        │  - Bot Probability  │
        │    per commenter    │
        │  - Neo4j Graph Viz  │
        └─────────────────────┘
```

---

## Tech Stack

| Layer              | Technology                          | Purpose                              |
|--------------------|-------------------------------------|--------------------------------------|
| Data Source        | YouTube Data API v3                 | Fetch live comments                  |
| Training Data      | TwiBot-22, Cresci-2017 (Kaggle)     | Pre-labeled bot/human interactions   |
| NLP                | scikit-learn (TF-IDF + RF)          | Linguistic bot pattern detection     |
| Graph DB           | Neo4j + neo4j Python driver         | User interaction network mapping     |
| Graph Viz          | py2neo / neovis.js                  | Visual bot cluster rendering         |
| Timing Analysis    | pandas, numpy                       | Interval regularity detection        |
| Dashboard          | Streamlit                           | Live demo interface                  |
| Score Engine       | Python (custom weighted average)    | Unified authenticity score           |
| Environment        | Python 3.10+, venv                  | Dependency isolation                 |

---

## Project Structure

```
fake-engagement-detector/
│
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── components/
│   │   ├── score_card.py        # Authenticity Score UI component
│   │   ├── graph_viz.py         # Neo4j graph visualization
│   │   └── commenter_table.py   # Per-commenter bot probability table
│
├── data_fetcher/
│   ├── youtube_fetcher.py       # YouTube Data API v3 integration
│   └── comment_parser.py        # Parse and clean raw API response
│
├── models/
│   ├── train_nlp_model.py       # TF-IDF + Random Forest training script
│   ├── nlp_model.pkl            # Saved trained model (joblib)
│   ├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
│   └── evaluate_model.py        # Evaluation script (precision, recall, F1)
│
├── graph/
│   ├── neo4j_connector.py       # Neo4j driver setup and session management
│   ├── graph_builder.py         # Insert users/replies as nodes and edges
│   ├── bot_cluster_detector.py  # Cypher queries to find bot ring patterns
│   └── queries.cypher           # Raw Cypher query file for reference
│
├── analysis/
│   ├── timing_engine.py         # Posting interval analysis
│   ├── feature_extractor.py     # Composite feature builder per commenter
│   └── score_engine.py          # Weighted score fusion
│
├── data/
│   ├── twibot22_sample.csv      # Sample of TwiBot-22 training data
│   ├── cresci2017_sample.csv    # Sample of Cresci-2017 training data
│   └── raw_comments_cache/      # Cached API responses (JSON)
│
├── config/
│   └── config.yaml              # API keys, Neo4j URI, model weights
│
├── requirements.txt
├── .env                         # Secrets (never commit to git)
├── .gitignore
└── README.md
```

---

## Environment Setup

### 1. Clone and Create Virtual Environment
```bash
git clone https://github.com/your-username/fake-engagement-detector.git
cd fake-engagement-detector

python -m venv venv
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
google-api-python-client==2.120.0
google-auth==2.29.0
google-auth-oauthlib==1.2.0
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1
joblib==1.3.2
neo4j==5.19.0
py2neo==2021.2.4
streamlit==1.33.0
plotly==5.21.0
python-dotenv==1.0.1
praw==7.7.1
matplotlib==3.8.4
seaborn==0.13.2
scipy==1.13.0
nltk==3.8.1
```

### 3. Create `.env` File
```env
YOUTUBE_API_KEY=your_google_console_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=FakeEngagementDetector/1.0
```

---

## Step 1 — YouTube API Integration

### 1a. Google Cloud Console Setup (You Already Have This ✅)

Your existing setup should have:
- A project created at [console.cloud.google.com](https://console.cloud.google.com)
- **YouTube Data API v3** enabled under "APIs & Services > Library"
- An **API Key** generated under "APIs & Services > Credentials"

**Important Quotas to Know:**
- Default daily quota: **10,000 units/day**
- `commentThreads.list` costs **1 unit per call**, returns up to 100 results
- Fetching 500 comments = 5 API calls = 5 units (very cheap)

### 1b. YouTube Comment Fetcher (`data_fetcher/youtube_fetcher.py`)

```python
import os
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")

def fetch_comments(video_url: str, max_comments: int = 500) -> pd.DataFrame:
    """
    Fetch up to max_comments top-level comments from a YouTube video.
    Returns a DataFrame with columns:
      author, text, likes, published_at, updated_at, author_channel_id
    """
    video_id = extract_video_id(video_url)
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="time"  # "time" or "relevance"
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id":        item["id"],
                "author":            snippet.get("authorDisplayName", "Unknown"),
                "author_channel_id": snippet.get("authorChannelId", {}).get("value", ""),
                "text":              snippet.get("textDisplay", ""),
                "likes":             snippet.get("likeCount", 0),
                "published_at":      snippet.get("publishedAt", ""),
                "updated_at":        snippet.get("updatedAt", ""),
                "reply_count":       item["snippet"].get("totalReplyCount", 0)
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(comments)
    # Convert timestamps to datetime objects
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["updated_at"]   = pd.to_datetime(df["updated_at"])
    return df

def fetch_video_metadata(video_url: str) -> dict:
    """Fetch title, view count, like count, comment count for the video."""
    video_id = extract_video_id(video_url)
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()
    if not response["items"]:
        return {}
    item = response["items"][0]
    return {
        "title":         item["snippet"]["title"],
        "channel":       item["snippet"]["channelTitle"],
        "view_count":    int(item["statistics"].get("viewCount", 0)),
        "like_count":    int(item["statistics"].get("likeCount", 0)),
        "comment_count": int(item["statistics"].get("commentCount", 0)),
        "published_at":  item["snippet"]["publishedAt"]
    }
```

---

## Step 2 — Training Data (Kaggle Datasets)

### TwiBot-22 Dataset
- **Source:** [Kaggle / twibot22.github.io](https://twibot22.github.io)
- **Size:** ~1 million Twitter users with bot/human labels
- **Features available:**
  - User profile metadata (account age, follower/following ratio, bio presence)
  - Tweet text content
  - Graph structure (follower/following edges)
  - 4 entity types, 14 relation types
- **Use for:** Training the NLP + metadata-based classifier

### Cresci-2017 Dataset
- **Source:** [Kaggle — cresci-2017](https://www.kaggle.com/datasets/lujing929/cresci-2017)
- **Size:** ~14,000 accounts (genuine + 3 types of bots)
- **Bot categories:** Social spambots, traditional spambots, fake followers
- **Use for:** Multi-class bot type classification (not just binary)

### Loading Training Data (`models/train_nlp_model.py`)

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.corpus import stopwords
import re

nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Clean and normalize comment text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)        # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()        # Normalize whitespace
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def train_model(data_path: str):
    df = pd.read_csv(data_path)
    # Expected columns: "text", "label" (0=human, 1=bot)
    df["clean_text"] = df["text"].fillna("").apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),    # unigrams + bigrams
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",  # handles imbalanced bot/human ratio
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))

    joblib.dump(clf,        "models/nlp_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Model and vectorizer saved.")

if __name__ == "__main__":
    train_model("data/twibot22_sample.csv")
```

---

## Step 3 — Feature Engineering

Each commenter gets a **feature vector** combining text, metadata, and behavioral signals.

### Feature Table

| Feature                | Description                                             | Signal Strength |
|------------------------|---------------------------------------------------------|-----------------|
| `nlp_bot_prob`         | TF-IDF + RF probability output                          | High            |
| `comment_interval_std` | Std dev of time between comments (low = bot)            | High            |
| `comment_interval_mean`| Mean posting interval                                   | Medium          |
| `exact_interval_flag`  | Flag if intervals are suspiciously uniform (< 5s std)   | High            |
| `text_similarity_avg`  | Cosine similarity between user's own comments           | High            |
| `exclamation_ratio`    | Ratio of `!` to total characters                        | Medium          |
| `caps_ratio`           | Ratio of uppercase letters                              | Low             |
| `url_count`            | Number of URLs in comments                              | Medium          |
| `reply_depth`          | How deep in reply chains the account operates           | Medium          |
| `graph_cluster_score`  | Neo4j cluster membership score                          | High            |
| `account_age_days`     | Days since account creation (if available)              | Medium          |

### Feature Extractor (`analysis/feature_extractor.py`)

```python
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
```

---

## Step 4 — NLP Model (TF-IDF + Random Forest)

### Running Inference on Live Comments

```python
import joblib
import pandas as pd

def predict_bot_probability(texts: list) -> list:
    """
    Takes a list of comment texts and returns bot probability for each.
    """
    from models.train_nlp_model import preprocess_text

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model      = joblib.load("models/nlp_model.pkl")

    clean_texts = [preprocess_text(t) for t in texts]
    tfidf_matrix = vectorizer.transform(clean_texts)
    probs = model.predict_proba(tfidf_matrix)[:, 1]  # probability of class=1 (bot)
    return probs.tolist()

def score_comments_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add nlp_bot_prob column to a comments dataframe."""
    df["nlp_bot_prob"] = predict_bot_probability(df["text"].fillna("").tolist())
    return df
```

### Linguistic Bot Signals to Detect
- **Repetitive phrasing:** Same sentence structure repeated across accounts
- **Generic positive sentiment:** "Amazing video!", "Great content!", "I love this!" with no personalization
- **Coordinated hashtag usage:** Multiple accounts using identical hashtags within seconds
- **Non-contextual replies:** Comments unrelated to the video topic
- **Emoji spam:** Excessive or identical emoji sequences

---

## Step 5 — Neo4j Graph Database Integration

### Why Neo4j for This Project
Standard relational databases cannot efficiently traverse multi-hop relationships (e.g., "find all users who replied to the same 3 accounts within 60 seconds"). Neo4j's native graph storage makes these queries 100x faster and produces visually compelling output.

### 5a. Installing and Running Neo4j Locally

```bash
# Option 1: Docker (recommended for hackathon)
docker pull neo4j:5.19
docker run \
  --name neo4j-bot-detector \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:5.19

# Access Neo4j Browser at: http://localhost:7474
```

Or use **Neo4j AuraDB Free Tier** (cloud): [https://neo4j.com/cloud/platform/aura-graph-database/](https://neo4j.com/cloud/platform/aura-graph-database/)

### 5b. Neo4j Connector (`graph/neo4j_connector.py`)

```python
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def clear_graph(self):
        """Clear all nodes and edges — call before each new video analysis."""
        self.run_query("MATCH (n) DETACH DELETE n")
```

### 5c. Graph Builder (`graph/graph_builder.py`)

```python
import pandas as pd
from graph.neo4j_connector import Neo4jConnector

def build_interaction_graph(df: pd.DataFrame):
    """
    Build a graph where:
    - Nodes: User accounts
    - Edges: REPLIED_TO (with timestamp and comment_id properties)
    """
    connector = Neo4jConnector()
    connector.clear_graph()

    # Create User nodes
    create_user_query = """
    MERGE (u:User {channel_id: $channel_id})
    SET u.name = $name,
        u.comment_count = $comment_count
    """
    for _, row in df.drop_duplicates("author_channel_id").iterrows():
        connector.run_query(create_user_query, {
            "channel_id":    row["author_channel_id"] or row["author"],
            "name":          row["author"],
            "comment_count": int(df[df["author"] == row["author"]].shape[0])
        })

    # Create COMMENTED_ON edges (User → Video)
    video_node_query = "MERGE (v:Video {id: $video_id})"
    connector.run_query(video_node_query, {"video_id": "current_video"})

    edge_query = """
    MATCH (u:User {channel_id: $channel_id})
    MATCH (v:Video {id: $video_id})
    CREATE (u)-[:COMMENTED_ON {
        timestamp: $timestamp,
        text:      $text,
        comment_id: $comment_id
    }]->(v)
    """
    for _, row in df.iterrows():
        connector.run_query(edge_query, {
            "channel_id": row["author_channel_id"] or row["author"],
            "video_id":   "current_video",
            "timestamp":  str(row["published_at"]),
            "text":       row["text"][:200],
            "comment_id": row["comment_id"]
        })

    connector.close()
    print(f"Graph built: {len(df)} comment edges created.")
```

### 5d. Bot Cluster Detector (`graph/bot_cluster_detector.py`)

```python
from graph.neo4j_connector import Neo4jConnector

def detect_bot_clusters() -> list:
    """
    Detect groups of users who all commented within a tight time window.
    Returns list of suspected bot cluster groups.
    """
    connector = Neo4jConnector()

    # Find users who commented within 60 seconds of each other
    cluster_query = """
    MATCH (u1:User)-[c1:COMMENTED_ON]->(v:Video)
    MATCH (u2:User)-[c2:COMMENTED_ON]->(v)
    WHERE u1 <> u2
      AND abs(duration.between(
            datetime(c1.timestamp),
            datetime(c2.timestamp)
          ).seconds) < 60
    RETURN u1.name AS user1, u2.name AS user2,
           c1.timestamp AS time1, c2.timestamp AS time2
    ORDER BY c1.timestamp
    LIMIT 200
    """
    results = connector.run_query(cluster_query)

    # Find users with suspiciously high comment counts relative to the pool
    high_volume_query = """
    MATCH (u:User)-[:COMMENTED_ON]->(v:Video)
    WITH u, count(*) AS comment_count
    WHERE comment_count > 3
    RETURN u.name AS author, comment_count
    ORDER BY comment_count DESC
    """
    high_volume = connector.run_query(high_volume_query)

    connector.close()
    return {"coordinated_pairs": results, "high_volume_users": high_volume}

def get_graph_cluster_scores(authors: list) -> dict:
    """
    Assign a graph-based suspicion score (0-1) to each author.
    Authors in bot clusters get higher scores.
    """
    clusters = detect_bot_clusters()
    suspicious_authors = set()

    for pair in clusters["coordinated_pairs"]:
        suspicious_authors.add(pair["user1"])
        suspicious_authors.add(pair["user2"])
    for user in clusters["high_volume_users"]:
        suspicious_authors.add(user["author"])

    return {author: (1.0 if author in suspicious_authors else 0.0) for author in authors}
```

### 5e. Useful Cypher Queries (Reference)

Save these in `graph/queries.cypher`:

```cypher
-- View all users and their comment counts
MATCH (u:User)-[:COMMENTED_ON]->(v:Video)
RETURN u.name, count(*) AS comments
ORDER BY comments DESC;

-- Find coordinated posting rings (within 30 seconds)
MATCH (u1:User)-[c1:COMMENTED_ON]->(v:Video)<-[c2:COMMENTED_ON]-(u2:User)
WHERE u1 <> u2
  AND abs(duration.between(datetime(c1.timestamp), datetime(c2.timestamp)).seconds) < 30
RETURN u1.name, u2.name, c1.timestamp, c2.timestamp;

-- Identify isolated bot-like nodes (high output, no engagement received)
MATCH (u:User)-[c:COMMENTED_ON]->(v:Video)
WITH u, count(c) AS total_comments
WHERE total_comments > 5
RETURN u.name, total_comments
ORDER BY total_comments DESC;
```

---

## Step 6 — Timing Regularity Detection

Bot accounts often post at machine-like intervals. This module flags accounts where the **standard deviation of inter-comment times is below a human threshold**.

```python
# analysis/timing_engine.py
import pandas as pd
import numpy as np

HUMAN_STD_THRESHOLD = 30.0   # seconds; below this = suspicious
EXACT_INTERVAL_MAX  = 5.0    # seconds; below this = almost certainly automated

def analyze_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze posting timing regularity per author.
    Returns DataFrame with timing risk scores per author.
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
                "is_exact_interval": False
            })
            continue

        deltas = group["published_at"].diff().dt.total_seconds().dropna()
        std  = deltas.std()
        mean = deltas.mean()

        if std is None or np.isnan(std):
            timing_risk = 0.0
        elif std < EXACT_INTERVAL_MAX:
            timing_risk = 1.0   # Almost certainly a bot
        elif std < HUMAN_STD_THRESHOLD:
            # Linearly scale risk between 0.4 and 0.9
            timing_risk = 0.9 - (std / HUMAN_STD_THRESHOLD) * 0.5
        else:
            timing_risk = max(0.0, 0.4 - (std / 300))

        results.append({
            "author":            author,
            "timing_risk":       round(timing_risk, 3),
            "interval_std":      round(std, 2),
            "interval_mean":     round(mean, 2),
            "is_exact_interval": std < EXACT_INTERVAL_MAX
        })

    return pd.DataFrame(results)
```

---

## Step 7 — Authenticity Score Engine

The final score fuses all three signals using a **weighted average**.

```python
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

    merged = merged.merge(timing_df[["author", "timing_risk"]], on="author", how="left")
    merged["timing_risk"] = merged["timing_risk"].fillna(0.0)

    merged["graph_cluster_score"] = merged["author"].map(graph_scores).fillna(0.0)

    merged["bot_score"] = (
        WEIGHTS["nlp_bot_prob"]       * merged["nlp_bot_prob"] +
        WEIGHTS["timing_risk"]         * merged["timing_risk"] +
        WEIGHTS["graph_cluster_score"] * merged["graph_cluster_score"]
    )

    merged["authenticity_score"] = ((1 - merged["bot_score"]) * 100).round(1)
    merged["risk_label"] = merged["bot_score"].apply(classify_risk)

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
```

---

## Step 8 — Streamlit Dashboard

### Running the App
```bash
streamlit run app/main.py
```

### Main App (`app/main.py`)

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_fetcher.youtube_fetcher import fetch_comments, fetch_video_metadata
from models.train_nlp_model import preprocess_text
from analysis.feature_extractor import extract_features
from analysis.timing_engine import analyze_timing
from analysis.score_engine import (
    compute_per_commenter_scores, compute_overall_authenticity
)
from graph.graph_builder import build_interaction_graph
from graph.bot_cluster_detector import get_graph_cluster_scores

import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Fake Engagement Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fake Engagement Detector")
st.markdown("Paste a YouTube video URL to analyze its comment section for bot activity.")

# --- Input ---
url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
analyze_btn = st.button("🔍 Analyze Now", type="primary")

if analyze_btn and url:
    with st.spinner("Fetching comments from YouTube API..."):
        meta = fetch_video_metadata(url)
        df   = fetch_comments(url, max_comments=500)

    st.success(f"✅ Fetched {len(df)} comments from: **{meta.get('title', 'Unknown Video')}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👁️ Views",    f"{meta.get('view_count', 0):,}")
    col2.metric("👍 Likes",    f"{meta.get('like_count', 0):,}")
    col3.metric("💬 Comments", f"{meta.get('comment_count', 0):,}")
    col4.metric("📥 Fetched",  f"{len(df):,}")

    with st.spinner("Running NLP bot detection..."):
        model      = joblib.load("models/nlp_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        clean_texts = [preprocess_text(t) for t in df["text"].fillna("").tolist()]
        tfidf_matrix = vectorizer.transform(clean_texts)
        df["nlp_bot_prob"] = model.predict_proba(tfidf_matrix)[:, 1]

    with st.spinner("Analyzing timing patterns..."):
        timing_df = analyze_timing(df)

    with st.spinner("Building Neo4j interaction graph..."):
        build_interaction_graph(df)
        graph_scores = get_graph_cluster_scores(df["author"].unique().tolist())

    with st.spinner("Computing authenticity scores..."):
        scores_df   = compute_per_commenter_scores(df, timing_df, graph_scores)
        overall_score = compute_overall_authenticity(scores_df)

    # --- Score Display ---
    st.markdown("---")
    st.markdown("## 🎯 Overall Authenticity Score")

    color = "green" if overall_score >= 70 else ("orange" if overall_score >= 40 else "red")
    st.markdown(
        f"<h1 style='color:{color}; font-size:72px; text-align:center'>"
        f"{overall_score}%</h1>",
        unsafe_allow_html=True
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        title={"text": "Authenticity Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 40],   "color": "#ffcccc"},
                {"range": [40, 70],  "color": "#fff3cc"},
                {"range": [70, 100], "color": "#ccffcc"},
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # --- Per-Commenter Table ---
    st.markdown("## 👤 Commenter Bot Probability Breakdown")
    display_df = scores_df[[
        "author", "comment_count", "nlp_bot_prob",
        "timing_risk", "graph_cluster_score", "bot_score", "risk_label"
    ]].rename(columns={
        "author":               "Author",
        "comment_count":        "Comments",
        "nlp_bot_prob":         "NLP Score",
        "timing_risk":          "Timing Risk",
        "graph_cluster_score":  "Graph Risk",
        "bot_score":            "Bot Score",
        "risk_label":           "Risk Level"
    })
    display_df["NLP Score"]   = display_df["NLP Score"].round(3)
    display_df["Timing Risk"] = display_df["Timing Risk"].round(3)
    display_df["Bot Score"]   = display_df["Bot Score"].round(3)
    st.dataframe(display_df, use_container_width=True, height=400)

    # --- Bot Score Distribution ---
    st.markdown("## 📊 Bot Score Distribution")
    fig_hist = px.histogram(
        scores_df, x="bot_score", nbins=30,
        color="risk_label",
        color_discrete_map={
            "🔴 High Risk": "red",
            "🟡 Suspicious": "orange",
            "🟢 Likely Human": "green"
        },
        labels={"bot_score": "Bot Score", "count": "Number of Commenters"},
        title="Distribution of Bot Scores Across All Commenters"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("## 🕸️ Network Graph (Neo4j)")
    st.info(
        "Open Neo4j Browser at http://localhost:7474 and run:\n"
        "`MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100`\n"
        "to see the interactive bot coordination graph."
    )
```

---

## Step 9 — Live Demo Flow

This is the **exact script to follow on stage** for maximum judge impact:

### Before You Go On Stage
- [ ] Pre-train and save `nlp_model.pkl` and `tfidf_vectorizer.pkl`
- [ ] Start Neo4j Docker container: `docker start neo4j-bot-detector`
- [ ] Run `streamlit run app/main.py` — keep browser tab open
- [ ] Have Neo4j Browser open at `http://localhost:7474` in a second tab
- [ ] Pre-select 2 YouTube video URLs to demo:
  - One video with **suspected bot engagement** (highly viral in short time)
  - One video with **organic engagement** (slow-growth creator)

### On Stage Demo Script (3 minutes)

**Minute 1 — Setup Hook:**
> *"We've all seen videos go viral overnight with 10,000 comments in 6 hours. Our system tells you if that was real humans — or an army of bots."*

Paste the suspicious video URL → click Analyze → let it fetch live.

**Minute 2 — Show the Score:**
Point to the gauge needle landing in the red zone. Point to the Neo4j graph showing a "ring" of users who all commented within 10 seconds of each other.

**Minute 3 — The Contrast:**
Paste the organic video URL → show the gauge landing green → show the scattered, natural timing distribution.

---

## API Quota Management

| Action                            | Units Used | Daily Limit |
|-----------------------------------|------------|-------------|
| `commentThreads.list` (100 items) | 1 unit     | 10,000      |
| 500 comments (5 requests)         | 5 units    | 10,000      |
| `videos.list` (metadata)          | 1 unit     | 10,000      |
| Full demo (2 videos × 500 comments)| 12 units  | 10,000      |

**You have massive headroom.** Even running the demo 100 times uses only 1,200 units out of 10,000.

To avoid hitting quota during development, **cache API responses**:

```python
import json, os

def fetch_with_cache(video_url: str, cache_dir: str = "data/raw_comments_cache") -> pd.DataFrame:
    video_id = extract_video_id(video_url)
    cache_path = os.path.join(cache_dir, f"{video_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return pd.DataFrame(json.load(f))
    df = fetch_comments(video_url)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, default=str)
    return df
```

---

## Model Evaluation Metrics

After training, evaluate your model to present to judges:

| Metric        | Target    | Description                                          |
|---------------|-----------|------------------------------------------------------|
| Precision     | > 0.85    | Of flagged bots, % that are actually bots            |
| Recall        | > 0.80    | Of actual bots, % that we caught                     |
| F1 Score      | > 0.82    | Harmonic mean of precision and recall                |
| AUC-ROC       | > 0.90    | Overall classifier discrimination ability            |
| False Positive Rate | < 0.10 | % of real humans wrongly flagged as bots          |

**Note:** On TwiBot-22, state-of-the-art graph-based models achieve ~80% accuracy. A TF-IDF + Random Forest baseline typically achieves 72–76%. Your combined system should outperform either alone.

---

## Deployment

### Option 1: Local Demo (Hackathon Day)
```bash
# Terminal 1: Start Neo4j
docker start neo4j-bot-detector

# Terminal 2: Start Streamlit
streamlit run app/main.py --server.port=8501

# Access at: http://localhost:8501
```

### Option 2: Streamlit Cloud (Free, Public Demo Link)
1. Push to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app/main.py` as entry point
4. Add secrets in Streamlit Cloud settings (replaces `.env`):
   ```
   YOUTUBE_API_KEY = "your_key_here"
   NEO4J_URI = "neo4j+s://your-aura-uri"
   NEO4J_USERNAME = "neo4j"
   NEO4J_PASSWORD = "your_password"
   ```

### Option 3: Hugging Face Spaces (Gradio Alternative)
If switching to Gradio, deploy directly to Hugging Face Spaces for free GPU access and a shareable link.

---

## Judging Criteria Alignment

| Judging Criterion       | How This Project Addresses It                              |
|-------------------------|------------------------------------------------------------|
| Innovation              | Combines NLP + Graph DB + Time-Series in one unified score |
| Technical Complexity    | Neo4j Cypher queries, TF-IDF pipeline, weighted fusion     |
| Practical Impact        | Real-world problem: ad fraud, influencer marketing trust   |
| Demo Quality            | Live real-time analysis on stage, not static slides        |
| Completeness            | Full pipeline from raw URL to scored output                |
| Presentation            | Gauge chart, data table, Neo4j visual — 3 compelling views |

---

## Troubleshooting

### YouTube API Errors

| Error                          | Cause                              | Fix                                            |
|--------------------------------|------------------------------------|------------------------------------------------|
| `403 quotaExceeded`            | Daily quota exhausted               | Wait 24 hrs or use cached responses            |
| `403 forbidden`                | API key restrictions               | Check "API restrictions" in Cloud Console      |
| `400 commentsDisabled`         | Video has comments turned off      | Use a different video for demo                 |
| `KeyError: authorChannelId`    | Anonymous commenter                | Add `.get("authorChannelId", {})` with default |

### Neo4j Errors

| Error                              | Fix                                                    |
|------------------------------------|--------------------------------------------------------|
| `ServiceUnavailable`               | Start Docker: `docker start neo4j-bot-detector`        |
| `AuthError`                        | Check `.env` NEO4J_PASSWORD matches Docker `-e` flag   |
| `ClientError: datetime() parsing`  | Ensure timestamps are ISO 8601 format before inserting |

### Streamlit Errors

| Error                         | Fix                                                        |
|-------------------------------|------------------------------------------------------------|
| `ModuleNotFoundError`         | Activate venv and run `pip install -r requirements.txt`    |
| `Model file not found`        | Run `python models/train_nlp_model.py` first               |
| `Port 8501 already in use`    | `streamlit run app/main.py --server.port=8502`             |

---

*Built for Hackathon — Problem Statement 3: Fake Engagement Detection*
*Stack: Python · YouTube Data API v3 · scikit-learn · Neo4j · Streamlit · Plotly*
