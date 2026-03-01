# Model Explanation Document — Fake Engagement Detection

**Team Project: FakeHub**
**Problem Statement 3: Fake Engagement Detection in Social Media**

---

## 1. Problem Understanding

Social media platforms are increasingly plagued by artificially manufactured engagement—bot farms, coordinated comment rings, and automated posting tools designed to inflate metrics, manipulate algorithms, and deceive audiences. This fake engagement distorts genuine behavioural patterns, erodes platform trust, and misleads brands making data-driven decisions.

**Our objective:** Build a real-time behavioural detection model that:
- Differentiates organic versus artificial engagement on YouTube.
- Detects coordinated behavioural anomalies across multiple accounts.
- Outputs an **Authenticity Score (0–100%)**, **Bot Probability per commenter**, and **human-readable Behavioural Anomaly Explanations**.

The system operates as a live Streamlit dashboard: a user pastes any YouTube URL, and the backend fetches, analyses, and scores up to 500 comments in real time—no pre-processing, no static data required.

---

## 2. Data Assumptions & Dataset Description

### Assumptions
1. **Cross-platform transferability:** Bot linguistic patterns (generic praise, spam URLs, repetitive phrasing) are platform-agnostic; models trained on Twitter-origin text generalize well to YouTube comments.
2. **Timing as a signal:** Human posting behaviour follows irregular, context-dependent timing; bots post at statistically improbable, near-constant intervals.
3. **Graph clustering:** Multiple previously-unrelated accounts commenting on the same video within a tight time window (< 60 seconds) is strongly indicative of coordination.
4. **Weighted fusion:** No single signal (NLP, timing, graph) is individually sufficient; combining all three via calibrated weights produces the most robust assessment.

### Training Data: TwiBot-22 (Public Dataset)

| Property | Details |
| --- | --- |
| **Dataset Type** | Public, pre-labeled |
| **Source** | TwiBot-22 Benchmark — [twibot22.github.io](https://twibot22.github.io) |
| **Records Used** | ~100,000 labeled text samples |
| **Labels** | Binary: `0 = Human`, `1 = Bot` |
| **Class Distribution** | ~42% Human, ~58% Bot |
| **Format** | CSV (`text`, `label`) |

**Why it fits behavioural analytics:** TwiBot-22 is among the largest public bot detection benchmarks, spanning diverse bot typologies—social spambots, fake followers, self-declared bots, and political bots. The textual samples capture the exact linguistic patterns that transfer directly to YouTube comment bot detection.

### Live Inference Data: YouTube Data API v3
During live demo, up to 500 top-level comments are fetched per video. Each record includes: `author`, `text`, `published_at`, `likes`, `author_channel_id`, `reply_count`.

---

## 3. Feature Engineering Logic

We engineer **three independent categories** of behavioural features per commenter:

### A. Semantic Text Features (Deep NLP)
Raw comment text is cleaned and encoded into **384-dimensional dense semantic vectors** using the `all-MiniLM-L6-v2` Sentence Transformer. Unlike TF-IDF word-counting, these embeddings capture deep meaning, sarcasm, multilingual context, and semantically equivalent but lexically different bot messages. Each commenter's average embedding-based bot probability is computed by the neural network classifier.

### B. Timing Regularity & Engagement Burst Features
| Feature | Description | Bot Signal |
| --- | --- | --- |
| `interval_std` | Std dev of inter-comment intervals | < 5s = machine-like |
| `interval_mean` | Average seconds between comments | < 30s = elevated risk |
| `is_exact_interval` | Boolean flag for near-zero variance | Strong bot indicator |
| `engagement_burst` | 1-min bucketed volume anomaly (z-score) | Statistically improbable spikes |

Thresholds: `std < 5s` → risk = 1.0 (automated); `std < 30s` → linearly scaled 0.4–0.9; above 30s → natural decay toward 0. Burst detection flags any 1-minute bucket exceeding `mean + 2σ` in volume.

### C. Network Interaction Features (Graph Analysis)
User interactions are modelled as a directed graph in **Neo4j** (`User -[:COMMENTED_ON]→ Video`). Two Cypher queries detect coordination:
1. **Coordinated Pairs**: Accounts that commented within 60 seconds of each other.
2. **High Volume Accounts**: Users with > 3 comments on a single video.
Accounts appearing in coordinated clusters receive `graph_cluster_score = 1.0`; others receive `0.0`.

**Innovation:** The combination of temporal, semantic, and structural features into a unified scoring engine is what separates this approach from single-signal classifiers. Each feature category compensates for the blind spots of the others.

---

## 4. Model Selection & Reasoning

### NLP Classifier: LLM Embeddings + MLP Neural Network

| Component | Choice | Justification |
| --- | --- | --- |
| **Text Encoder** | `all-MiniLM-L6-v2` (SentenceTransformers) | Compact 22M-parameter transformer; 384-dim dense vectors; captures semantics, sarcasm, context; runs on CPU. |
| **Classifier** | `MLPClassifier(128, 64)` | Multi-Layer Perceptron finds complex non-linear boundaries. ReLU + Adam + early stopping prevents overfitting. |
| **Previous Baseline** | TF-IDF + Random Forest | Replaced because bag-of-words fails on multilingual content and semantically equivalent bot messages. |

### Score Fusion Engine (Weighted Average)
| Signal | Weight | Rationale |
| --- | --- | --- |
| NLP Bot Probability | **0.40** | Strongest individual predictor; captures linguistic intent |
| Timing Risk | **0.35** | Bots exhibit statistically impossible regularity |
| Graph Cluster Score | **0.25** | Coordinated rings are high-confidence but sparse |

**Final Authenticity Score** = `(1 − bot_score) × 100%`, weighted by each commenter's comment count so prolific bots have proportionally greater impact on the video-level score.

### Explainability Layer: LIME
LIME (Local Interpretable Model-agnostic Explanations) perturbs input text and observes prediction shifts, producing a ranked feature-importance list of words influencing the Human vs. Bot decision—satisfying the mandatory "Behavioural Anomaly Explanation" requirement.

---

## 5. Evaluation Metrics

Evaluated on a stratified 80/20 train-test split of ~100K records:

| Metric | Human | Bot |
| --- | --- | --- |
| **Precision** | 0.87 | **0.94** |
| **Recall** | 0.92 | 0.90 |
| **F1-Score** | 0.89 | **0.92** |

| Overall Metric | Value |
| --- | --- |
| **Accuracy** | **91.0%** |
| **Weighted Avg F1** | **0.91** |
| **Macro Avg F1** | **0.91** |

**Interpretation:** High bot precision (0.94) means flagged accounts are overwhelmingly true positives—critical for avoiding false accusations. Balanced recall across both classes ensures neither bots nor humans are systematically missed.

**Validation approach:** Stratified split preserves class distribution; `class_weight` and early stopping mitigate overfitting; the model generalises to unseen YouTube data during live inference without retraining.

---

## 6. Behavioural Insights Derived

The system generates per-commenter natural-language explanations:

| Detected Pattern | Example Explanation |
| --- | --- |
| Machine-like intervals | *"Posted 12 comments at exact machine-like intervals (2.3s variance)."* |
| Coordinated ring | *"Identified in a coordinated posting ring."* |
| Linguistic bot patterns | *"Text shows high linguistic similarity to known bot spam."* |
| Engagement burst | *"Comment volume spike: 47 comments/min (4.2σ above mean)."* |
| Elevated posting speed | *"Avg 18.5s between comments—faster than human thresholds."* |

These insights bridge the gap between raw algorithmic output and actionable intelligence, enabling moderators and analysts to understand *who* is a bot, *why* they were flagged, and *how* they coordinate.

---

## 7. Practical Feasibility

| Aspect | Details |
| --- | --- |
| **Deployment** | Runs entirely on a local machine (Mac/Linux/Windows) with Python 3.10+ and a Neo4j instance. No GPU required. |
| **API Cost** | Fetching 500 YouTube comments = 5 API units (well under the 10,000 daily free quota). |
| **Inference Speed** | Full pipeline (fetch → embed → score → graph → render) completes in under 30 seconds for 500 comments. |
| **Scalability** | SentenceTransformer batch-encodes efficiently; Neo4j scales to millions of nodes; the architecture is production-ready. |
| **Portability** | Single `requirements.txt`, single `streamlit run` command. No cloud infra needed for demo. |

---

## 8. Visualization & Presentation Quality

The Streamlit dashboard provides a unified, dark-themed command center:

- **Authenticity Gauge**: Plotly gauge chart (0–100%) with colour-coded risk banding.
- **Engagement Burst Timeline**: Interactive line chart with anomaly markers (red ✕) for statistically significant spikes.
- **Bot Score Distribution**: Histogram colour-coded by risk label (🔴 High / 🟡 Suspicious / 🟢 Human).
- **Interactive Network Graph**: Physics-based 3D PyVis rendering of the Neo4j interaction graph—bot-ring members highlighted in orange, organic users in green, with a central video node.
- **LIME XAI Panel**: Per-comment feature importance rendered in a high-contrast white container for readability.
- **Detailed Breakdown Table**: Per-commenter NLP Score, Timing Risk, Graph Risk, Bot Score, Risk Label, and Anomaly Explanation.
