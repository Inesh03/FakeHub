# Model Explanation Document — Fake Engagement Detection

**Team Project: FakeHub**
**Problem Statement 3: Fake Engagement Detection in Social Media**

---

## 1. Problem Understanding

Social media platforms are increasingly plagued by artificially manufactured engagement—bot farms, coordinated comment rings, and automated posting tools designed to inflate metrics, manipulate algorithms, and deceive audiences. This fake engagement distorts genuine behavioral patterns, erodes platform trust, and misleads brands making data-driven decisions.

Our objective is to build a **real-time behavioral detection model** that:
- Differentiates organic versus artificial engagement on YouTube videos.
- Detects coordinated behavioral anomalies across multiple accounts.
- Outputs an **Authenticity Score (0–100%)**, **Bot Probability per commenter**, and **human-readable Behavioural Anomaly Explanations**.

The system operates as a live Streamlit dashboard: a judge pastes any YouTube URL, and the backend fetches, analyzes, and scores up to 500 comments in real time.

---

## 2. Data Assumptions & Dataset Description

### Primary Training Data: TwiBot-22 (Public Dataset)

| Property | Details |
| --- | --- |
| **Dataset Type** | Public, pre-labeled |
| **Source** | TwiBot-22 Benchmark — [twibot22.github.io](https://twibot22.github.io) |
| **Records Used** | ~100,000 labeled text samples |
| **Labels** | Binary: `0 = Human`, `1 = Bot` |
| **Class Distribution** | ~42% Human, ~58% Bot |
| **Format** | CSV with columns: `text`, `label` |

**Why TwiBot-22 fits behavioural analytics:**
TwiBot-22 is among the largest and most comprehensive public bot detection benchmarks available. It includes diverse bot typologies—social spambots, fake followers, self-declared bots, and political bots—spanning over 1 million accounts with 4 entity types and 14 relation types. The textual samples capture the exact linguistic patterns (generic praise, repetitive phrasing, spam URLs) that transfer directly to YouTube comment bot detection, making it ideal for cross-platform behavioral modeling.

### Live Inference Data: YouTube Data API v3

During inference (live demo), the system fetches up to 500 top-level comments directly from YouTube using the Google Cloud YouTube Data API v3. Each comment record includes: `author`, `text`, `published_at`, `likes`, `author_channel_id`, and `reply_count`. These are processed in real-time through the detection pipeline.

---

## 3. Feature Engineering Logic

We engineer three distinct categories of behavioral features per commenter:

### A. Semantic Text Features (NLP)
Raw comment text is cleaned (URL removal, whitespace normalization) and encoded into **384-dimensional dense semantic vectors** using the `all-MiniLM-L6-v2` Sentence Transformer. These embeddings capture deep meaning, sarcasm, and contextual relationships—far beyond keyword frequency. Each commenter's average embedding-based bot probability is computed by the neural network.

### B. Timing & Burst Features
| Feature | Description | Bot Signal |
| --- | --- | --- |
| `interval_std` | Standard deviation of inter-comment intervals | Low std (< 5s) = machine-like |
| `interval_mean` | Average seconds between comments | Sub-30s = elevated risk |
| `is_exact_interval` | Boolean flag for near-zero variance | Strong bot indicator |
| `engagement_burst` | 1-minute bucketed comment volume anomaly | Statistically improbable spikes |

Timing thresholds: `std < 5s` → risk = 1.0 (automated); `std < 30s` → linearly scaled 0.4–0.9; above 30s → natural decay toward 0.

### C. Graph / Network Features
User interactions are modeled as a directed graph in **Neo4j** (Nodes = Users, Edges = COMMENTED_ON → Video, with timestamp and text properties). Two Cypher queries detect coordinated rings:
1. **Coordinated Pairs**: Accounts that commented within 60 seconds of each other.
2. **High Volume Accounts**: Users with > 3 comments on a single video.

Accounts appearing in coordinated clusters receive a `graph_cluster_score = 1.0`; others receive `0.0`.

---

## 4. Model Selection & Reasoning

### NLP Classification: LLM Embeddings + MLP Neural Network

| Component | Choice | Justification |
| --- | --- | --- |
| **Text Encoder** | `all-MiniLM-L6-v2` (SentenceTransformers) | Compact 22M-parameter transformer producing 384-dim dense vectors. Captures semantic meaning, sarcasm detection, and contextual patterns. Runs efficiently on CPU without a GPU. |
| **Classifier** | `MLPClassifier` (128, 64) hidden layers | Multi-Layer Perceptron finds complex non-linear boundaries in the dense embedding space. Relu activation + Adam optimizer + early stopping prevent overfitting. |
| **Previous Baseline** | TF-IDF + Random Forest | Replaced because bag-of-words approaches fail on multilingual content, contextual sarcasm, and semantically equivalent but lexically different bot messages. |

### Score Fusion Engine
Three independent signals are fused via weighted averaging into a final `bot_score`:

| Signal | Weight | Rationale |
| --- | --- | --- |
| NLP Bot Probability | **0.40** | Strongest individual predictor; captures linguistic intent |
| Timing Risk | **0.35** | Bots exhibit statistically impossible posting regularity |
| Graph Cluster Score | **0.25** | Coordinated rings are high-confidence but sparse |

Final **Authenticity Score** = `(1 - bot_score) × 100%`, weighted by each commenter's comment count so prolific bots have proportionally greater impact on the video-level score.

### Explainability: LIME (Local Interpretable Model-agnostic Explanations)
For any individual comment, LIME perturbs the input text and observes shifts in the neural network's output probability, producing a ranked list of words that most influenced the Human vs. Bot classification. This satisfies the "Behavioural Anomaly Explanation" requirement.

---

## 5. Evaluation Metrics

Evaluated on a stratified 80/20 train-test split of the TwiBot-22 dataset (~100K records):

| Metric | Human | Bot |
| --- | --- | --- |
| **Precision** | 0.87 | 0.94 |
| **Recall** | 0.92 | 0.90 |
| **F1-Score** | 0.89 | 0.92 |

| Overall Metric | Value |
| --- | --- |
| **Accuracy** | **91.0%** |
| **Weighted Avg F1** | **0.91** |
| **Macro Avg F1** | **0.91** |

The model maintains high precision for bots (0.94), meaning flagged accounts are overwhelmingly true positives—critical for avoiding false accusations. Balanced recall across both classes ensures neither bots nor humans are systematically missed.

---

## 6. Behavioural Insights Derived

The system generates per-commenter natural-language explanations that communicate the *reasoning* behind each flag:

| Detected Pattern | Example Explanation |
| --- | --- |
| Machine-like posting intervals | *"Posted 12 comments at exact machine-like intervals (2.3s variance)."* |
| Coordinated ring membership | *"Identified in a coordinated posting ring."* |
| Linguistic bot patterns | *"Text shows high linguistic similarity to known bot spam."* |
| Engagement burst anomaly | *"Comment volume spike: 47 comments/minute (4.2σ above mean)."* |
| Elevated posting speed | *"High posting frequency (avg 18.5s between comments) is faster than normal human thresholds."* |

These insights bridge the gap between raw algorithmic output and actionable cybersecurity intelligence, enabling platform moderators and brand analysts to understand not just *who* is a bot, but *why* they were flagged and *how* they are coordinating.
