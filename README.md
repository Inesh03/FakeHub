# 🛡️ FakeHub: AI-Powered Fake Engagement Detector

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg?logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Streamlit-1.33-FF4B4B.svg?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Neo4j-Graph_DB-008CC1.svg?logo=neo4j" alt="Neo4j">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg?logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/LIME-Explainable_AI-10B981.svg" alt="LIME XAI">
</div>

<br>

**FakeHub** is a real-time behavioral detection model designed to differentiate organic engagement from artificial bot activity on YouTube. It detects coordinated behavioral anomalies, bot rings, and engagement bursts using a fusion of Deep Learning semantic analysis and Graph Database traversal.

---

## ✨ Key Features
- **Deep Semantic NLP**: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) and a Multi-Layer Perceptron neural network to detect bot linguistic patterns and spam.
- **Neo4j Graph Analysis**: Reconstructs user interactions into an animated, physics-based 3D graph to visually identify **coordinated posting rings**.
- **Time-Series Burst Detection**: Mathematical interval tracking to flag exact machine-like posting delays and unusual spikes in traffic.
- **Explainable AI (XAI)**: Integrated `lime` interpretability to generate human-readable explanations of exactly *which* words triggered the AI to flag a comment as a bot.
- **Live Streamlit Dashboard**: A sleek, dark-themed command center to run analysis in real-time.

---

## 📊 Model Performance & Metrics
Trained against the highly imbalanced TwiBot-22 dataset using 384-dimensional dense semantic vector embeddings.

| Metric | Score |
| ------ | ----- |
| **Accuracy** | 91.0% |
| **F1-Score (Bot)** | 0.92 |
| **F1-Score (Human)** | 0.89 |
| **Precision (Bot)** | 0.94 |
| **Recall (Bot)** | 0.90 |

---

## 🏗️ Architecture Stack
1. **Data Ingestion**: YouTube Data API v3 
2. **AI Semantic Engine**: HuggingFace `sentence-transformers` + scikit-learn `MLPClassifier`
3. **Graph Engine**: Local `Neo4j` database + `pyvis` for frontend physics rendering
4. **Scoring Fusion**: Custom python heuristics weighting NLP, Graph, and Timestamps
5. **Frontend UI**: `Streamlit`, HTML components, Custom CSS, Plotly

---

## 🚀 Quick Setup Guide

### 1. Prerequisites
- Python 3.10+
- An active `Neo4j` database (Desktop, Docker, or AuraDB cloud)
- A Google Cloud Console API Key (for YouTube Data API v3)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Inesh03/FakeHub.git
cd FakeHub

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add:
```env
YOUTUBE_API_KEY=your_google_console_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

*(Note: Data sets and `.pkl` model files have been excluded from this repository via `.gitignore` to comply with GitHub's LFS limits. To retrain the model, place the TwiBot-22 CSV in `data/` and run `python models/train_nlp_model.py`)*

### 4. Running the Dashboard
Make sure your Neo4j database is actively running, then start the AI command center:
```bash
streamlit run app/main.py
```

---

## 🔬 Explainable Anomaly Insights
FakeHub doesn't just output a single score; it yields **actionable intelligence** by generating localized explanations for *why* an account is suspicious:
- 🔴 *"Posted at exact machine-like intervals"*
- 🔴 *"Identified in a coordinated posting ring"*
- 🟡 *"Engagement burst anomaly detected"*

---

## 📁 Dataset Provenance

### Training Data — TwiBot-22 (Public Dataset)
| Property | Details |
| --- | --- |
| **Type** | Public, pre-labeled |
| **Source** | [TwiBot-22 Benchmark](https://twibot22.github.io) |
| **Records Used** | ~100,000 labeled text samples |
| **Labels** | Binary: `0 = Human`, `1 = Bot` |
| **Class Distribution** | ~42% Human, ~58% Bot |

**Why it fits behavioural analytics:** TwiBot-22 is among the largest public bot detection benchmarks, encompassing diverse bot typologies (social spambots, fake followers, political bots). The textual patterns transfer directly to YouTube comment bot detection.

**Behavioural features engineered from it:**
- 384-dimensional dense semantic embeddings via `all-MiniLM-L6-v2` transformer
- Inter-comment timing regularity (std deviation, mean interval, exact-interval flags)
- Network cluster membership via Neo4j Cypher traversal
- Engagement burst anomaly detection (1-minute bucketed time-series)

### Live Inference Data — YouTube Data API v3
During live demo, up to 500 comments are fetched in real-time per video with fields: `author`, `text`, `published_at`, `likes`, `author_channel_id`, `reply_count`.

---

## 📄 Submission Documents
- **[Model Explanation Document](model_explanation.md)** — 5-page technical writeup covering problem understanding, data assumptions, feature engineering, model selection, evaluation metrics, and behavioural insights.

---

> Built for Hackathon Problem Statement 3: Fake Engagement Detection in Social Media
