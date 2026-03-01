# Model Explanation Document — Fake Engagement Detection

**Team Project: FakeHub**
**Problem Statement 3: Fake Engagement Detection in Social Media**

---

## 1. Problem Understanding

Social media platforms are increasingly plagued by artificially manufactured engagement — bot farms, coordinated comment rings, and automated posting tools designed to inflate metrics, manipulate algorithms, and deceive audiences. This fake engagement distorts genuine behavioural patterns, erodes platform trust, and misleads brands making data-driven decisions.

**Our objective:** Build a real-time behavioural detection model that:
- Differentiates organic versus artificial engagement on YouTube.
- Detects coordinated behavioural anomalies across multiple accounts.
- Outputs an **Authenticity Score (0–100%)**, **Bot Probability per commenter**, and **human-readable Behavioural Anomaly Explanations**.
- Generates a comprehensive **AI-powered analysis summary** synthesizing all detection signals.

The system operates as a live Streamlit dashboard: a user pastes any YouTube URL (or clicks a pre-loaded demo button), and the backend fetches, analyses, and scores up to 500 comments in real time — no pre-processing, no static data required. Results include 8 distinct visualizations, an animated gauge countdown, LIME explainability, and an auto-generated natural-language verdict.

---

## 2. Data Assumptions & Dataset Description

### Assumptions
1. **Cross-platform transferability:** Bot linguistic patterns (generic praise, spam URLs, repetitive phrasing) are platform-agnostic; models trained on Twitter-origin text generalize well to YouTube comments.
2. **Timing as a signal:** Human posting behaviour follows irregular, context-dependent timing; bots post at statistically improbable, near-constant intervals.
3. **Graph clustering:** Multiple previously-unrelated accounts commenting on the same video within a tight time window (< 60 seconds) is strongly indicative of coordination.
4. **Weighted fusion:** No single signal (NLP, timing, graph) is individually sufficient; combining all three via calibrated weights produces the most robust assessment.
5. **Sentiment as a secondary signal:** Bot accounts disproportionately produce generic positive sentiment ("Amazing video!", "Great content!") compared to the varied sentiment distribution of organic commenters.

### Training Data: TwiBot-22 (Public Dataset)

| Property | Details |
| --- | --- |
| **Dataset Type** | Public, pre-labeled |
| **Source** | TwiBot-22 Benchmark — [twibot22.github.io](https://twibot22.github.io) |
| **Records Used** | ~100,000 labeled text samples |
| **Labels** | Binary: `0 = Human`, `1 = Bot` |
| **Class Distribution** | ~42% Human, ~58% Bot |
| **Format** | CSV (`text`, `label`) |

**Why it fits behavioural analytics:** TwiBot-22 is among the largest public bot detection benchmarks, spanning diverse bot typologies — social spambots, fake followers, self-declared bots, and political bots. The textual samples capture the exact linguistic patterns that transfer directly to YouTube comment bot detection.

### Live Inference Data: YouTube Data API v3
During live demo, up to **500 top-level comments** are fetched per video. Each record includes: `author`, `text`, `published_at`, `likes`, `author_channel_id`, `reply_count`. Cost: 5 API units per analysis (well under the 10,000 daily free quota).

---

## 3. Feature Engineering Logic

We engineer **three primary categories** and **five supplementary categories** of behavioural features per commenter — totalling **8 independent detection indicators**:

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
Accounts appearing in coordinated clusters receive `graph_cluster_score = 1.0`.

### D. Supplementary Feature Innovations
| # | Feature | Method | Purpose |
|---|---------|--------|---------|
| 1 | **Sentiment Polarity** | Embedding-anchor cosine similarity | Detect generic positive spam patterns in bot comments |
| 2 | **User Similarity Clustering** | Pairwise cosine similarity heatmap | Flag copy-paste bot rings using identical text |
| 3 | **Temporal Activity Patterns** | Hour × Day comment density heatmap | Reveal bot farm operating schedules |
| 4 | **Account Age Analysis** | First-appearance timing scatter | Identify coordinated account activation |
| 5 | **AI-Generated Summary** | Rule-based metric synthesis engine | Produce natural-language risk verdict |

**Innovation:** The combination of 3 primary engines + 5 supplementary signals into a unified scoring engine is what separates this approach from single-signal classifiers. Each feature category compensates for the blind spots of the others.

---

## 4. Model Selection & Reasoning

### NLP Classifier: LLM Embeddings + MLP Neural Network

| Component | Choice | Justification |
| --- | --- | --- |
| **Text Encoder** | `all-MiniLM-L6-v2` (SentenceTransformers) | Compact 22M-parameter transformer; 384-dim dense vectors; captures semantics, sarcasm, multilingual context; runs on CPU. |
| **Classifier** | `MLPClassifier(128, 64)` | Multi-Layer Perceptron finds complex non-linear boundaries. ReLU + Adam + early stopping prevents overfitting. |
| **Previous Baseline** | TF-IDF + Random Forest (85%) | Replaced because bag-of-words fails on multilingual content and semantically equivalent bot messages. Upgrade to LLM embeddings yielded a **6% accuracy improvement**. |

### Score Fusion Engine (Weighted Average)
| Signal | Weight | Rationale |
| --- | --- | --- |
| NLP Bot Probability | **0.40** | Strongest individual predictor; captures linguistic intent |
| Timing Risk | **0.35** | Bots exhibit statistically impossible regularity |
| Graph Cluster Score | **0.25** | Coordinated rings are high-confidence but sparse |

**Final Authenticity Score** = `(1 − weighted_bot_score) × 100%`, weighted by each commenter's comment count so prolific bots have proportionally greater impact on the video-level score.

### Explainability Layer: LIME
LIME (Local Interpretable Model-agnostic Explanations) perturbs input text and observes prediction shifts, producing a ranked feature-importance list of words influencing the Human vs. Bot decision — satisfying the mandatory "Behavioural Anomaly Explanation" requirement. Integrated with session-state caching to support interactive comment selection without re-running the full pipeline.

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

**Interpretation:** High bot precision (0.94) means flagged accounts are overwhelmingly true positives — critical for avoiding false accusations. Balanced recall across both classes ensures neither bots nor humans are systematically missed.

**Validation approach:** Stratified split preserves class distribution; `class_weight` and early stopping mitigate overfitting; the model generalises to unseen YouTube data during live inference without retraining.

---

## 6. Behavioural Insights Derived

The system generates both **per-commenter anomaly explanations** and a comprehensive **AI-generated analysis summary**.

### Per-Commenter Explanations
| Detected Pattern | Example Explanation |
| --- | --- |
| Machine-like intervals | *"Posted 12 comments at exact machine-like intervals (2.3s variance)."* |
| Coordinated ring | *"Identified in a coordinated posting ring."* |
| Linguistic bot patterns | *"Text shows high linguistic similarity to known bot spam."* |
| Engagement burst | *"Comment volume spike: 47 comments/min (4.2σ above mean)."* |
| Elevated posting speed | *"Avg 18.5s between comments — faster than human thresholds."* |

### AI-Generated Summary (4 Risk Levels)
The system automatically generates a natural-language verdict based on all gathered metrics:
- **✅ Low Risk (75%+):** Predominantly organic engagement with natural language and varied timing.
- **⚠️ Moderate Risk (50–74%):** Some inorganic signals detected; engagement may be partially inflated.
- **🔶 High Risk (30–49%):** Significant bot activity with repetitive patterns and coordinated posting.
- **🚨 Critical Risk (<30%):** Engagement heavily dominated by automated systems.

Each summary includes detailed per-engine findings (NLP, Timing, Bursts, Graph, Sentiment) and identifies the most suspicious account with its probability and anomaly explanation.

---

## 7. Practical Feasibility

| Aspect | Details |
| --- | --- |
| **Deployment** | Runs entirely on a local machine (Mac/Linux/Windows) with Python 3.10+ and a Neo4j instance. No GPU required. |
| **API Cost** | Fetching 500 YouTube comments = 5 API units (well under the 10,000 daily free quota). |
| **Inference Speed** | Full pipeline (fetch → embed → score → graph → render) completes in **under 30 seconds** for 500 comments. |
| **Scalability** | SentenceTransformer batch-encodes efficiently; Neo4j scales to millions of nodes; architecture is production-ready. |
| **Portability** | Single `requirements.txt`, single `streamlit run` command. No cloud infra needed for demo. |
| **Session Persistence** | Analysis results cached in `st.session_state` — interactive LIME exploration, selectbox changes, and chart interactions don't trigger re-analysis. |
| **Demo-Ready** | Pre-loaded demo URL buttons enable one-click analysis without manual URL searching during live presentations. |

---

## 8. Visualization & Presentation Quality

The Streamlit dashboard provides a unified, premium dark-mode interface with **red/black/white** colour palette, animated backgrounds, and micro-interactions:

| Visualization | Description |
| --- | --- |
| **Animated Authenticity Gauge** | JS-animated countdown from 100% → actual score with CSS progress bar |
| **Engagement Burst Timeline** | Area chart with red anomaly markers (✕) for statistically significant spikes |
| **Bot Score Distribution** | Histogram colour-coded by risk label (🔴 High / 🟡 Suspicious / 🟢 Human) |
| **Sentiment Polarity Pie** | Pie chart showing Positive/Negative/Neutral distribution across comments |
| **Temporal Activity Heatmap** | Hour × Day grid revealing bot farm operating schedules |
| **User Similarity Heatmap** | Cosine similarity matrix exposing copy-paste bot rings |
| **Account Age Scatter** | When suspicious accounts first appeared on the video |
| **Interactive Network Graph** | Physics-based PyVis rendering of Neo4j graph — bots in orange, humans in green |
| **LIME XAI Panel** | Per-comment word-level feature importance for bot classification |
| **AI Summary Card** | Frosted-glass card with verdict, detailed findings, and risk breakdown |
| **Detailed Data Table** | Per-commenter NLP Score, Timing Risk, Graph Risk, Bot Score, Risk Label, and Anomaly Explanation |

**Visual Enhancements:**
- Animated radial gradient background with dot-grid overlay
- Floating particle system with neural-network-style connecting lines
- Cursor glow trail (300px red radial gradient)
- Glassmorphism section titles with backdrop blur
- Metric cards with animated red underline on hover
- Frosted-glass containers around all chart panels
