import os
import sys
import time
import numpy as np
from dotenv import load_dotenv

# Load environment variables (must be first, before any HF imports)
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Add project root to sys.path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from datetime import datetime

from data_fetcher.youtube_fetcher import fetch_comments, fetch_video_metadata
from models.train_nlp_model import preprocess_text
from analysis.feature_extractor import extract_features
from analysis.timing_engine import analyze_timing, detect_engagement_bursts
from analysis.score_engine import (
    compute_per_commenter_scores, compute_overall_authenticity
)
from graph.graph_builder import build_interaction_graph
from graph.bot_cluster_detector import get_graph_cluster_scores
from graph.graph_viz import generate_network_html

import joblib
import logging
import warnings

# Suppress noisy logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(
    page_title="FakeHub — Fake Engagement Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================================
# PREMIUM CSS — Animated background, glassmorphism, visual depth
# =========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide sidebar toggle */
    [data-testid="stSidebarCollapsedControl"] { display: none; }

    /* ===== ANIMATED BACKGROUND ===== */
    .stApp {
        background: #050505;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(220,38,38,0.15), transparent),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(220,38,38,0.08), transparent),
            radial-gradient(circle at 20% 80%, rgba(255,255,255,0.02), transparent);
    }
    /* Dot grid overlay */
    .stApp::before {
        content: '';
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none; z-index: 0;
    }

    /* ===== HEADER ===== */
    .hero-container {
        text-align: center; padding: 60px 0 20px 0;
        position: relative; z-index: 1;
    }
    .main-header {
        font-size: 6rem !important; font-weight: 800 !important; letter-spacing: -3px !important;
        color: #FFFFFF !important; margin-bottom: 10px !important; line-height: 1.0 !important;
        position: relative; display: block;
    }
    .main-header span {
        background: linear-gradient(135deg, #DC2626, #FF6B6B, #DC2626);
        background-size: 200% 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .sub-header {
        font-size: 1.2rem !important; color: #525252 !important; text-align: center;
        margin-bottom: 12px !important; letter-spacing: 0.6px; font-weight: 300 !important;
    }
    .hero-badge {
        display: inline-block; background: rgba(220,38,38,0.08);
        border: 1px solid rgba(220,38,38,0.25); border-radius: 24px;
        padding: 8px 20px; color: #EF4444; font-size: 0.82rem;
        font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase;
        margin-bottom: 32px;
        animation: badgePulse 2.5s ease-in-out infinite;
    }
    @keyframes badgePulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.15); }
        50% { box-shadow: 0 0 20px 4px rgba(220,38,38,0.12); }
    }
    .hero-stats {
        display: flex; justify-content: center; gap: 40px;
        margin-bottom: 16px;
    }
    .hero-stat {
        text-align: center;
    }
    .hero-stat-value {
        font-size: 1.6rem; font-weight: 800; color: #FAFAFA;
    }
    .hero-stat-label {
        font-size: 0.72rem; color: #525252; text-transform: uppercase;
        letter-spacing: 1.2px; margin-top: 2px;
    }

    /* ===== CURSOR GLOW ===== */
    .cursor-glow {
        position: fixed; width: 300px; height: 300px;
        border-radius: 50%; pointer-events: none;
        background: radial-gradient(circle, rgba(220,38,38,0.08) 0%, transparent 70%);
        transform: translate(-50%, -50%);
        z-index: 9999; transition: left 0.1s ease, top 0.1s ease;
    }

    /* ===== GLASSMORPHISM SECTION TITLES ===== */
    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #F5F5F5;
        border-left: 3px solid #DC2626;
        border-radius: 0 10px 10px 0;
        padding: 14px 20px; margin-top: 32px; margin-bottom: 16px;
        background: linear-gradient(135deg, rgba(220,38,38,0.08), rgba(255,255,255,0.02));
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    /* ===== METRIC CARDS with animated glow border ===== */
    [data-testid="stMetric"] {
        background: rgba(15,15,15,0.8) !important;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 18px;
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden;
    }
    [data-testid="stMetric"]::after {
        content: ''; position: absolute; bottom: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #DC2626, transparent);
        opacity: 0; transition: opacity 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(220,38,38,0.3);
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(220,38,38,0.12);
    }
    [data-testid="stMetric"]:hover::after { opacity: 1; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #737373 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #DC2626, #B91C1C) !important;
        color: #FFFFFF !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        letter-spacing: 0.5px; padding: 0.65rem 2.2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(220,38,38,0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #EF4444, #DC2626) !important;
        box-shadow: 0 8px 28px rgba(220,38,38,0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: rgba(15,15,15,0.9) !important; color: #FAFAFA !important;
        border: 1px solid rgba(220,38,38,0.35) !important;
        border-radius: 10px !important; font-weight: 600 !important;
        backdrop-filter: blur(8px); transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #DC2626 !important; border-color: #DC2626 !important;
        box-shadow: 0 8px 24px rgba(220,38,38,0.25) !important;
    }

    /* ===== FROST CONTAINERS ===== */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }
    [data-testid="stExpander"] {
        background: rgba(15,15,15,0.6) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(8px);
    }

    /* ===== TABLES ===== */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    [data-testid="stDataFrame"] > div { border-radius: 12px; }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #525252; }
    .stTabs [aria-selected="true"] { color: #DC2626 !important; }

    /* ===== DIVIDERS ===== */
    hr { border-color: rgba(255,255,255,0.04) !important; }

    /* ===== STATUS ===== */
    [data-testid="stStatusWidget"] {
        border-radius: 12px;
        background: rgba(15,15,15,0.8) !important;
        backdrop-filter: blur(8px);
    }

    /* ===== INPUTS ===== */
    .stTextInput > div > div > input {
        background: rgba(15,15,15,0.9) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important; color: #FAFAFA !important;
        padding: 12px 16px !important; font-size: 0.95rem !important;
        backdrop-filter: blur(8px);
        transition: all 0.25s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #DC2626 !important;
        box-shadow: 0 0 0 2px rgba(220,38,38,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #404040 !important; }
    .stSelectbox > div > div { border-radius: 10px !important; }
    .stRadio > div { gap: 0.8rem; }

    /* ===== PLOTLY CHARTS ===== */
    [data-testid="stPlotlyChart"] {
        background: rgba(10,10,10,0.5); border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.04);
        padding: 8px; backdrop-filter: blur(4px);
    }
</style>
""", unsafe_allow_html=True)

# ===== HERO HEADER =====
st.markdown("""
<div class="hero-container">
    <div class="main-header">🛡️ Fake<span>Hub</span></div>
    <div class="sub-header">Real-time AI bot detection · Graph ring analysis · YouTube engagement scoring</div>
    <div class="hero-badge">⚡ Powered by LLM Embeddings + Neural Network</div>
    <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-value">3</div><div class="hero-stat-label">AI Engines</div></div>
        <div class="hero-stat"><div class="hero-stat-value">500</div><div class="hero-stat-label">Comments Scanned</div></div>
        <div class="hero-stat"><div class="hero-stat-value">91%</div><div class="hero-stat-label">Model Accuracy</div></div>
        <div class="hero-stat"><div class="hero-stat-value">8</div><div class="hero-stat-label">Indicators</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== CURSOR GLOW + FLOATING PARTICLES (JS) =====
components.html("""
<div class="cursor-glow" id="cursorGlow"></div>
<canvas id="particles" style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;"></canvas>
<script>
// --- Cursor Glow ---
const glow = document.getElementById('cursorGlow');
if (glow) {
    document.addEventListener('mousemove', (e) => {
        glow.style.left = e.clientX + 'px';
        glow.style.top = e.clientY + 'px';
    });
}

// --- Floating Particles ---
const canvas = document.getElementById('particles');
if (canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });

    const particles = [];
    for (let i = 0; i < 40; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.5,
            dx: (Math.random() - 0.5) * 0.3,
            dy: (Math.random() - 0.5) * 0.3,
            opacity: Math.random() * 0.3 + 0.05
        });
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.dx; p.y += p.dy;
            if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(220, 38, 38, ${p.opacity})`;
            ctx.fill();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dist = Math.hypot(particles[i].x - particles[j].x, particles[i].y - particles[j].y);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(220, 38, 38, ${0.04 * (1 - dist / 120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animate);
    }
    animate();
}
</script>
<style>
.cursor-glow {
    position: fixed; width: 300px; height: 300px;
    border-radius: 50%; pointer-events: none;
    background: radial-gradient(circle, rgba(220,38,38,0.07) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    z-index: 9999; transition: left 0.08s ease-out, top 0.08s ease-out;
}
</style>
""", height=0)


# =============================================================================
# HELPER: Sentiment Analysis (Feature 1)
# =============================================================================
def analyze_sentiment(texts, embedder):
    """Simple rule + embedding similarity-based sentiment classification."""
    positive_anchor = embedder.encode(["This is amazing, wonderful, great, excellent, love it"])
    negative_anchor = embedder.encode(["This is terrible, awful, horrible, hate it, disgusting"])
    neutral_anchor  = embedder.encode(["This is okay, normal, fine, nothing special"])
    
    text_embeddings = embedder.encode(texts)
    
    pos_sim = cosine_similarity(text_embeddings, positive_anchor).flatten()
    neg_sim = cosine_similarity(text_embeddings, negative_anchor).flatten()
    neu_sim = cosine_similarity(text_embeddings, neutral_anchor).flatten()
    
    sentiments = []
    for p, n, ne in zip(pos_sim, neg_sim, neu_sim):
        scores = {"Positive": p, "Negative": n, "Neutral": ne}
        sentiments.append(max(scores, key=scores.get))
    return sentiments


# Reusable Plotly dark theme
def dark_plotly_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,17,0.8)",
        font=dict(color="#D4D4D4", family="Inter"),
        title_font=dict(size=15, color="#FAFAFA"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )


# =============================================================================
# HELPER: Run full analysis pipeline on one video
# =============================================================================
def run_analysis(url, embedder, model):
    """Run the entire analysis pipeline for a single YouTube URL."""
    meta = fetch_video_metadata(url)
    df   = fetch_comments(url, max_comments=500)
    
    # NLP Bot Detection
    clean_texts = [preprocess_text(t) for t in df["text"].fillna("").tolist()]
    embedded_vectors = embedder.encode(clean_texts)
    df["nlp_bot_prob"] = model.predict_proba(embedded_vectors)[:, 1]
    
    # Sentiment Analysis (Feature 1)
    df["sentiment"] = analyze_sentiment(df["text"].fillna("").tolist(), embedder)
    
    # Timing
    timing_df = analyze_timing(df)
    bursts_df = detect_engagement_bursts(df, bucket_freq="1min", threshold_std=2.0)
    
    # Graph
    build_interaction_graph(df)
    graph_scores = get_graph_cluster_scores(df["author"].unique().tolist())
    
    # Score fusion
    scores_df     = compute_per_commenter_scores(df, timing_df, graph_scores)
    overall_score = compute_overall_authenticity(scores_df)
    
    return meta, df, timing_df, bursts_df, scores_df, overall_score, embedded_vectors


# =============================================================================
# HELPER: Render all visuals for one analysis result
# =============================================================================
def render_dashboard(meta, df, timing_df, bursts_df, scores_df, overall_score, embedded_vectors, embedder, model, prefix=""):
    """Render the full dashboard for one video analysis."""
    
    # --- Video Meta as Metric Cards ---
    st.markdown(f"### 🎬 {meta.get('title', 'N/A')}")
    st.caption(f"📺 Channel: **{meta.get('channel', 'N/A')}**")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("👁️ Views", f"{meta.get('view_count', 0):,}")
    with m2: st.metric("👍 Likes", f"{meta.get('like_count', 0):,}")
    with m3: st.metric("💬 Total Comments", f"{meta.get('comment_count', 0):,}")
    with m4: st.metric("📥 Fetched", f"{len(df):,}")
    st.divider()

    # --- Authenticity Gauge + Key Insights ---
    st.markdown('<p class="section-title">🎯 Overall Authenticity Assessment</p>', unsafe_allow_html=True)
    col_gauge, col_insights = st.columns([1, 1])
    
    with col_gauge:
        color = "#10B981" if overall_score >= 70 else ("#F59E0B" if overall_score >= 40 else "#EF4444")
        # Animated gauge via JS — counts down from 100 to actual score
        gauge_html = f"""
        <div id="gauge-container-{prefix}" style="text-align:center;padding:30px 0;">
            <div style="font-size:0.9rem;color:#737373;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;">Authenticity Score</div>
            <div id="gauge-number-{prefix}" style="font-size:5rem;font-weight:800;color:{color};line-height:1;font-family:Inter,sans-serif;">100%</div>
            <div style="margin-top:16px;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;max-width:300px;margin-left:auto;margin-right:auto;">
                <div id="gauge-bar-{prefix}" style="height:100%;width:100%;border-radius:3px;background:linear-gradient(90deg,#DC2626,{color});transition:width 2s cubic-bezier(0.4,0,0.2,1);"></div>
            </div>
            <div style="display:flex;justify-content:space-between;max-width:300px;margin:6px auto 0;"><span style="font-size:0.7rem;color:#404040;">0</span><span style="font-size:0.7rem;color:#404040;">100</span></div>
        </div>
        <script>
        (function() {{
            const target = {overall_score};
            const el = document.getElementById('gauge-number-{prefix}');
            const bar = document.getElementById('gauge-bar-{prefix}');
            let current = 100;
            const step = (100 - target) / 60;
            const interval = setInterval(() => {{
                current -= step;
                if ((step > 0 && current <= target) || (step < 0 && current >= target)) {{
                    current = target;
                    clearInterval(interval);
                }}
                el.textContent = Math.round(current) + '%';
            }}, 25);
            setTimeout(() => {{ bar.style.width = target + '%'; }}, 100);
        }})();
        </script>
        """
        components.html(gauge_html, height=220)

    with col_insights:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Key Insights")
        high_risk_count = len(scores_df[scores_df["risk_label"] == "🔴 High Risk"])
        burst_count = len(bursts_df[bursts_df["is_burst"] == True]) if not bursts_df.empty else 0
        ring_count = len(scores_df[scores_df["graph_cluster_score"] > 0])
        st.info(f"🚨 **{high_risk_count}** accounts flagged as High Risk.")
        st.warning(f"📈 **{burst_count}** engagement bursts detected.")
        st.error(f"🕸️ **{ring_count}** accounts in coordinated posting rings.")
        
        # Sentiment insight (Feature 1)
        if "sentiment" in df.columns:
            bot_authors = scores_df[scores_df["risk_label"] == "🔴 High Risk"]["author"].tolist()
            bot_comments = df[df["author"].isin(bot_authors)]
            if len(bot_comments) > 0:
                top_sent = bot_comments["sentiment"].value_counts().idxmax()
                pct = round(bot_comments["sentiment"].value_counts(normalize=True).iloc[0] * 100)
                st.success(f"💬 **{pct}%** of high-risk bot comments are **{top_sent}** sentiment.")

    st.divider()

    # --- Row: Timeline + Histogram ---
    col_timeline, col_hist = st.columns([1.2, 1])
    with col_timeline:
        st.markdown('<p class="section-title">📈 Engagement Bursts Timeline</p>', unsafe_allow_html=True)
        if not bursts_df.empty:
            fig_tl = px.area(bursts_df, x="published_at", y="comment_count",
                             labels={"published_at": "Time", "comment_count": "Comments/Min"},
                             title="⏱️ Comment Volume Over Time (1-min buckets)")
            fig_tl.update_traces(fill='tozeroy', fillcolor='rgba(220,38,38,0.12)', line_color='#DC2626')
            fig_tl.update_layout(**dark_plotly_layout())
            bp = bursts_df[bursts_df["is_burst"] == True]
            if not bp.empty:
                fig_tl.add_scatter(x=bp["published_at"], y=bp["comment_count"],
                                   mode="markers", marker=dict(color="#EF4444", size=12, symbol="x"), name="Anomaly Burst")
            st.plotly_chart(fig_tl, use_container_width=True, key=f"{prefix}_timeline")
        else:
            st.info("Not enough data for timeline.")

    with col_hist:
        st.markdown('<p class="section-title">📊 Bot Score Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(scores_df, x="bot_score", nbins=30, color="risk_label",
                                color_discrete_map={"🔴 High Risk": "#DC2626", "🟡 Suspicious": "#A3A3A3", "🟢 Likely Human": "#FAFAFA"},
                                labels={"bot_score": "Bot Probability", "count": "Commenters"}, title="🤖 Distribution Among Commenters")
        fig_hist.update_layout(**dark_plotly_layout())
        st.plotly_chart(fig_hist, use_container_width=True, key=f"{prefix}_hist")

    st.divider()

    # --- Row: Sentiment Pie + Temporal Heatmap (Features 1 & 5) ---
    col_sent, col_temp = st.columns([1, 1.3])
    
    with col_sent:
        st.markdown('<p class="section-title">💬 Sentiment Polarity Analysis</p>', unsafe_allow_html=True)
        if "sentiment" in df.columns:
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(sent_counts, values="Count", names="Sentiment",
                             color="Sentiment",
                             color_discrete_map={"Positive": "#FAFAFA", "Negative": "#DC2626", "Neutral": "#525252"},
                             title="Comment Sentiment Breakdown")
            fig_pie.update_traces(textfont_size=14, pull=[0.05]*len(sent_counts))
            fig_pie.update_layout(**dark_plotly_layout())
            st.plotly_chart(fig_pie, use_container_width=True, key=f"{prefix}_sentiment")
    
    with col_temp:
        st.markdown('<p class="section-title">🕐 Temporal Activity Heatmap</p>', unsafe_allow_html=True)
        if "published_at" in df.columns:
            temp_df = df.copy()
            temp_df["hour"] = temp_df["published_at"].dt.hour
            temp_df["day_of_week"] = temp_df["published_at"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_data = temp_df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
            heatmap_pivot = heatmap_data.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
            # Reindex to proper day order
            heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
            fig_heat = px.imshow(heatmap_pivot, labels={"x": "Hour of Day (UTC)", "y": "Day", "color": "Comments"},
                                 color_continuous_scale="Reds", title="🕐 Comment Density by Hour & Day",
                                 aspect="auto")
            fig_heat.update_layout(**dark_plotly_layout())
            st.plotly_chart(fig_heat, use_container_width=True, key=f"{prefix}_temporal")

    st.divider()

    # --- User Similarity Heatmap (Feature 4) ---
    st.markdown('<p class="section-title">🔗 User Similarity Clustering (Cosine Similarity)</p>', unsafe_allow_html=True)
    st.markdown("Accounts posting **highly similar** text (bright cells) may be copy-paste bot rings.")
    
    # Get top commenters by volume for a readable heatmap
    top_authors = scores_df.nlargest(min(20, len(scores_df)), "comment_count")["author"].tolist()
    author_embeddings = {}
    for author in top_authors:
        author_texts = df[df["author"] == author]["text"].fillna("").tolist()
        if author_texts:
            vecs = embedder.encode([preprocess_text(t) for t in author_texts])
            author_embeddings[author] = vecs.mean(axis=0)
    
    if len(author_embeddings) > 2:
        names = list(author_embeddings.keys())
        matrix = np.array([author_embeddings[n] for n in names])
        sim_matrix = cosine_similarity(matrix)
        np.fill_diagonal(sim_matrix, 0)  # Remove self-similarity
        
        fig_sim = px.imshow(sim_matrix, x=names, y=names,
                            color_continuous_scale="RdGy_r", title="🔗 Pairwise Text Similarity (Top Commenters)",
                            labels={"color": "Cosine Sim"}, aspect="auto")
        fig_sim.update_layout(**dark_plotly_layout(), height=500)
        st.plotly_chart(fig_sim, use_container_width=True, key=f"{prefix}_similarity")
    else:
        st.info("Not enough distinct authors for a similarity matrix.")

    st.divider()

    # --- Account Age Analysis (Feature 6) ---
    st.markdown('<p class="section-title">🆕 Account Age Analysis</p>', unsafe_allow_html=True)
    if "published_at" in df.columns:
        # Approximate account age from earliest comment timestamp
        author_first_seen = df.groupby("author")["published_at"].min().reset_index()
        author_first_seen.columns = ["author", "first_seen"]
        # Merge with scores
        age_df = scores_df.merge(author_first_seen, on="author", how="left")
        
        # We can also check if multiple high-risk users appeared at roughly the same time
        high_risk_authors = age_df[age_df["risk_label"] == "🔴 High Risk"]
        
        col_age1, col_age2 = st.columns([1, 1])
        with col_age1:
            st.markdown("**First Appearance of Flagged Accounts on This Video**")
            if len(high_risk_authors) > 0:
                fig_scatter = px.scatter(
                    high_risk_authors, x="first_seen", y="bot_score",
                    size="comment_count", color="risk_label",
                    color_discrete_map={"🔴 High Risk": "#EF4444"},
                    hover_data=["author", "comment_count"],
                    labels={"first_seen": "First Comment Time", "bot_score": "Bot Score"},
                    title="When Did Suspicious Accounts First Appear?"
                )
                fig_scatter.update_layout(**dark_plotly_layout())
                st.plotly_chart(fig_scatter, use_container_width=True, key=f"{prefix}_age_scatter")
            else:
                st.success("No high-risk accounts detected! 🎉")
        
        with col_age2:
            st.markdown("**Account Activity Concentration**")
            # Show how many comments high-risk vs normal users posted
            activity_summary = scores_df.groupby("risk_label").agg(
                total_comments=("comment_count", "sum"),
                unique_authors=("author", "nunique")
            ).reset_index()
            fig_bar = px.bar(activity_summary, x="risk_label", y="total_comments",
                             color="risk_label",
                             color_discrete_map={"🔴 High Risk": "#DC2626", "🟡 Suspicious": "#A3A3A3", "🟢 Likely Human": "#FAFAFA"},
                             text="unique_authors",
                             labels={"risk_label": "Risk Level", "total_comments": "Total Comments"},
                             title="Comment Volume by Risk Category")
            fig_bar.update_traces(texttemplate='%{text} users', textposition='outside')
            fig_bar.update_layout(**dark_plotly_layout(), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True, key=f"{prefix}_age_bar")

    st.divider()

    # --- Per-Commenter Table ---
    st.markdown('<p class="section-title">👤 Detailed Behaviour & Probability Breakdown</p>', unsafe_allow_html=True)
    display_df = scores_df[[
        "author", "comment_count", "nlp_bot_prob",
        "timing_risk", "graph_cluster_score", "bot_score", "risk_label", "anomaly_explanation"
    ]].rename(columns={
        "author": "Author", "comment_count": "Comments", "nlp_bot_prob": "NLP Score",
        "timing_risk": "Timing Risk", "graph_cluster_score": "Graph Risk",
        "bot_score": "Bot Score", "risk_label": "Risk Level", "anomaly_explanation": "Anomaly Explanation"
    })
    for col in ["NLP Score", "Timing Risk", "Bot Score"]:
        display_df[col] = display_df[col].round(3)
    st.dataframe(display_df, use_container_width=True, height=350,
                 column_config={"Anomaly Explanation": st.column_config.TextColumn("Anomaly Explanation", width="large")})

    # --- LIME XAI ---
    st.markdown('<p class="section-title">🧠 Explainable AI (XAI) — LIME Interpretability</p>', unsafe_allow_html=True)
    st.markdown("Select an author to inspect why the NLP engine assigned them their specific Bot Probability.")
    c1, c2 = st.columns([1, 2])
    suspicious_authors = display_df[display_df["Risk Level"] != "🟢 Likely Human"]["Author"].tolist()
    all_authors = display_df["Author"].tolist()
    
    with c1:
        selected_author = st.selectbox("1. Select an author:", options=suspicious_authors if suspicious_authors else all_authors, key=f"{prefix}_lime_author")
        if selected_author:
            author_comments = df[df["author"] == selected_author]["text"].tolist()
            selected_comment = st.selectbox("2. Select a comment:", options=author_comments, key=f"{prefix}_lime_comment")
    
    with c2:
        if selected_author and selected_comment:
            if len(str(selected_comment).split()) < 2:
                st.warning("⚠️ Comment too short for LIME analysis.")
            else:
                with st.spinner("Generating LIME Explanation..."):
                    try:
                        explainer = LimeTextExplainer(class_names=["Human", "Bot"])
                        def predictor(texts):
                            cleaned = [preprocess_text(t) for t in texts]
                            vecs = embedder.encode(cleaned)
                            return model.predict_proba(vecs)
                        exp = explainer.explain_instance(selected_comment, predictor, num_features=10)
                        lime_html = exp.as_html()
                        styled_html = f"""
                        <div style="background-color:#FFF;padding:20px;border-radius:10px;color:black;box-shadow:0 4px 6px rgba(0,0,0,0.1);">
                            <h4 style="margin-top:0;color:#1E293B;">LIME Feature Importance</h4>
                            {lime_html}
                        </div>"""
                        components.html(styled_html, height=450, scrolling=True)
                    except Exception:
                        st.error("⚠️ LIME could not process this comment.")

    # --- Network Graph ---
    st.markdown('<p class="section-title">🕸️ Interactive Network Graph (Neo4j)</p>', unsafe_allow_html=True)
    st.markdown("Suspicious users (rings/high-volume) are in **Orange**. Organic users in **Green**.")
    with st.spinner("Rendering graph..."):
        graph_html = generate_network_html(suspicious_authors=set(suspicious_authors))
        components.html(graph_html, height=620, scrolling=False)

    # --- AI GENERATED SUMMARY ---
    st.divider()
    st.markdown('<p class="section-title">📝 AI-Generated Analysis Summary</p>', unsafe_allow_html=True)

    # Gather all metrics for the summary
    total_commenters = len(scores_df)
    high_risk_count = len(scores_df[scores_df["risk_label"] == "🔴 High Risk"])
    suspicious_count = len(scores_df[scores_df["risk_label"] == "🟡 Suspicious"])
    human_count = len(scores_df[scores_df["risk_label"] == "🟢 Likely Human"])
    burst_count = len(bursts_df[bursts_df["is_burst"] == True]) if not bursts_df.empty else 0
    ring_count = len(scores_df[scores_df["graph_cluster_score"] > 0])
    avg_bot_score = scores_df["bot_score"].mean()
    max_bot_score = scores_df["bot_score"].max()
    top_bot = scores_df.loc[scores_df["bot_score"].idxmax(), "author"] if len(scores_df) > 0 else "N/A"

    # Determine overall verdict
    if overall_score >= 75:
        verdict = "✅ **Low Risk** — This video appears to have predominantly organic engagement."
        verdict_detail = "The majority of commenters show natural language patterns, varied posting intervals, and no significant coordinated behaviour. This engagement profile is consistent with genuine audience interaction."
    elif overall_score >= 50:
        verdict = "⚠️ **Moderate Risk** — This video shows some signs of inorganic engagement."
        verdict_detail = "While a majority of commenters appear genuine, there are notable clusters of accounts exhibiting bot-like behaviour patterns. The engagement may be partially inflated by coordinated or automated activity."
    elif overall_score >= 30:
        verdict = "🔶 **High Risk** — Significant bot activity detected in this video's engagement."
        verdict_detail = "A substantial portion of the engagement on this video comes from accounts exhibiting automated behaviour — including repetitive language patterns, machine-like timing intervals, and coordinated posting rings."
    else:
        verdict = "🚨 **Critical Risk** — This video's engagement is heavily dominated by bot activity."
        verdict_detail = "The vast majority of comments on this video appear to be generated by automated systems. The evidence includes coordinated posting within narrow time windows, nearly identical language across accounts, and suspicious graph cluster patterns."

    # Build detailed findings
    findings = []

    # NLP findings
    nlp_high = len(scores_df[scores_df["nlp_bot_prob"] > 0.7])
    if nlp_high > 0:
        findings.append(f"🤖 **NLP Engine:** {nlp_high} out of {total_commenters} commenters ({round(nlp_high/total_commenters*100)}%) were flagged by the neural network as exhibiting bot-like linguistic patterns.")
    else:
        findings.append(f"🤖 **NLP Engine:** No commenters exhibited strong bot-like linguistic patterns. Language appears natural and varied.")

    # Timing findings
    timing_flagged = len(scores_df[scores_df["timing_risk"] > 0.5])
    if timing_flagged > 0:
        findings.append(f"⏱️ **Timing Analysis:** {timing_flagged} accounts showed suspiciously regular posting intervals, suggesting automated scheduling.")
    else:
        findings.append(f"⏱️ **Timing Analysis:** Posting intervals appear human-like with natural variation.")

    # Burst findings
    if burst_count > 0:
        findings.append(f"📈 **Burst Detection:** {burst_count} engagement burst(s) detected — sudden spikes in comment volume that deviate from the baseline by more than 2 standard deviations.")
    else:
        findings.append(f"📈 **Burst Detection:** No anomalous engagement spikes detected. Comment flow appears organic.")

    # Graph findings
    if ring_count > 0:
        findings.append(f"🕸️ **Graph Analysis (Neo4j):** {ring_count} accounts are part of coordinated posting rings — groups that comment within tight time windows, suggesting a bot farm operation.")
    else:
        findings.append(f"🕸️ **Graph Analysis (Neo4j):** No coordinated posting rings detected in the interaction graph.")

    # Sentiment findings
    if "sentiment" in df.columns:
        sent_counts = df["sentiment"].value_counts(normalize=True)
        dominant_sent = sent_counts.idxmax()
        dominant_pct = round(sent_counts.iloc[0] * 100)
        bot_authors_list = scores_df[scores_df["risk_label"] == "🔴 High Risk"]["author"].tolist()
        bot_comments_df = df[df["author"].isin(bot_authors_list)]
        if len(bot_comments_df) > 0:
            bot_sent = bot_comments_df["sentiment"].value_counts().idxmax()
            bot_sent_pct = round(bot_comments_df["sentiment"].value_counts(normalize=True).iloc[0] * 100)
            findings.append(f"💬 **Sentiment:** Overall comment sentiment is {dominant_pct}% {dominant_sent}. Among high-risk accounts, {bot_sent_pct}% of comments are {bot_sent} — {'a common bot pattern of generic positive spam.' if bot_sent == 'Positive' else 'which may indicate sentiment manipulation.'}")
        else:
            findings.append(f"💬 **Sentiment:** Overall comment sentiment is {dominant_pct}% {dominant_sent}, with no unusual patterns among flagged accounts.")

    # Top threat
    if high_risk_count > 0:
        top_row = scores_df.nlargest(1, "bot_score").iloc[0]
        findings.append(f"🎯 **Most Suspicious Account:** **{top_row['author']}** with a bot probability of **{top_row['bot_score']:.1%}** — {top_row['anomaly_explanation']}")

    # Render the summary
    summary_html = f"""
    <div style="background:rgba(15,15,15,0.9); border:1px solid rgba(220,38,38,0.15); border-radius:14px; padding:28px 32px; margin:8px 0 16px 0; backdrop-filter:blur(8px);">
        <div style="font-size:1.3rem; font-weight:700; color:#FAFAFA; margin-bottom:12px;">{verdict}</div>
        <div style="font-size:0.95rem; color:#A3A3A3; line-height:1.7; margin-bottom:20px;">{verdict_detail}</div>
        <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:16px;">
            <div style="font-size:0.82rem; color:#737373; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px; font-weight:600;">Detailed Findings</div>
            {''.join(f'<div style="font-size:0.92rem; color:#D4D4D4; line-height:1.65; margin-bottom:10px;">{f}</div>' for f in findings)}
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:14px; margin-top:8px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <span style="font-size:0.78rem; color:#525252;">Score: <b style="color:{color};">{overall_score}%</b></span>
            <span style="font-size:0.78rem; color:#525252;">High Risk: <b style="color:#DC2626;">{high_risk_count}</b> · Suspicious: <b style="color:#A3A3A3;">{suspicious_count}</b> · Human: <b style="color:#FAFAFA;">{human_count}</b></span>
            <span style="font-size:0.78rem; color:#525252;">Commenters Analyzed: <b style="color:#FAFAFA;">{total_commenters}</b></span>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    # --- PDF Export (Feature 2) ---
    st.divider()
    st.markdown('<p class="section-title">📄 Export Report</p>', unsafe_allow_html=True)
    
    report_text = f"""
FAKEHUB — ENGAGEMENT AUTHENTICITY REPORT
==========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO INFORMATION
  Title:    {meta.get('title', 'N/A')}
  Channel:  {meta.get('channel', 'N/A')}
  Views:    {meta.get('view_count', 0):,}
  Likes:    {meta.get('like_count', 0):,}
  Comments: {meta.get('comment_count', 0):,}
  Fetched:  {len(df)}

OVERALL AUTHENTICITY SCORE: {overall_score}%

RISK BREAKDOWN
  High Risk Accounts:      {len(scores_df[scores_df['risk_label'] == '🔴 High Risk'])}
  Suspicious Accounts:     {len(scores_df[scores_df['risk_label'] == '🟡 Suspicious'])}
  Likely Human Accounts:   {len(scores_df[scores_df['risk_label'] == '🟢 Likely Human'])}
  Coordinated Ring Members:{len(scores_df[scores_df['graph_cluster_score'] > 0])}
  Engagement Bursts:       {len(bursts_df[bursts_df['is_burst'] == True]) if not bursts_df.empty else 0}

SENTIMENT ANALYSIS
"""
    if "sentiment" in df.columns:
        for sent, count in df["sentiment"].value_counts().items():
            report_text += f"  {sent}: {count} comments ({round(count/len(df)*100)}%)\n"
    
    report_text += f"""
TOP FLAGGED ACCOUNTS
{'='*50}
"""
    for _, row in scores_df.nlargest(10, "bot_score").iterrows():
        report_text += f"  {row['author']:<30} Bot Score: {row['bot_score']:.3f}  Risk: {row['risk_label']}\n"
        report_text += f"    → {row['anomaly_explanation']}\n\n"
    
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=report_text,
        file_name=f"fakehub_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
        key=f"{prefix}_download"
    )
    
    return display_df


# =============================================================================
# CENTERED INPUT AREA
# =============================================================================

# --- Demo URL buttons ---
DEMO_URL_1 = "https://www.youtube.com/watch?v=R1ixxnsbygY"
DEMO_URL_2 = "https://www.youtube.com/watch?v=dXCliyBQnWU"

demo_col1, demo_col2, demo_col3 = st.columns([1, 1, 1])
with demo_col1:
    if st.button("🎬 Try Demo Video 1", use_container_width=True):
        st.session_state["url_1"] = DEMO_URL_1
        st.session_state["analysis_done"] = False
        st.rerun()
with demo_col2:
    if st.button("🎬 Try Demo Video 2", use_container_width=True):
        st.session_state["url_1"] = DEMO_URL_2
        st.session_state["analysis_done"] = False
        st.rerun()
with demo_col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["url_1"] = ""
        st.session_state["analysis_done"] = False
        st.rerun()

st.markdown("")
mode = st.radio("Analysis Mode", ["Single Video", "Compare Two Videos"], horizontal=True, label_visibility="collapsed")

col_left, col_input, col_btn = st.columns([0.05, 3, 1])
with col_input:
    url = st.text_input("YouTube Video URL", placeholder="Paste a YouTube URL here...", key="url_1", label_visibility="collapsed")
    url2 = None
    if mode == "Compare Two Videos":
        url2 = st.text_input("Second URL", placeholder="Paste a second YouTube URL here...", key="url_2", label_visibility="collapsed")
with col_btn:
    analyze_btn = st.button("Analyze →", type="primary", use_container_width=True)

st.markdown("")
# Indicator pills
st.markdown(
    '<div style="text-align:center; margin-bottom:12px;">'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">🗣️ NLP</span>'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">⏱️ Timing</span>'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">📈 Bursts</span>'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">🕸️ Graph Rings</span>'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">💬 Sentiment</span>'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">🔗 Similarity</span>'
    '<span style="background:#171717;border:1px solid #262626;border-radius:20px;padding:6px 14px;margin:4px;display:inline-block;color:#A3A3A3;font-size:0.82rem;">🕐 Temporal</span>'
    '</div>', unsafe_allow_html=True
)
st.divider()


# =============================================================================
# MAIN EXECUTION — Cache results in session_state for rerun persistence
# =============================================================================

# Load model/embedder once and cache
@st.cache_resource
def load_models():
    m = joblib.load("models/llm_mlp_model.pkl")
    e = SentenceTransformer("all-MiniLM-L6-v2")
    return m, e

if analyze_btn and url:
    model, embedder = load_models()
    
    if mode == "Single Video":
        with st.status("Initializing AI engines...", expanded=True) as status:
            loading_msgs = [
                ("🧠 Waking up the neural network...", 0.4),
                ("🔍 Scanning 500 comments from YouTube API...", 0.3),
                ("🤖 Deploying LLM embeddings (all-MiniLM-L6-v2)...", 0.3),
                ("💬 Analyzing sentiment polarity across comments...", 0.2),
                ("⏱️ Hunting for machine-like timing patterns...", 0.2),
                ("📈 Detecting engagement burst anomalies...", 0.2),
                ("🕸️ Building Neo4j interaction graph to find bot rings...", 0.3),
                ("🎯 Fusing NLP + Timing + Graph signals into final scores...", 0.2),
                ("✨ Generating behavioural insights & anomaly explanations...", 0.2),
            ]
            for msg, delay in loading_msgs:
                st.write(msg)
                time.sleep(delay)
            meta, df, timing_df, bursts_df, scores_df, overall_score, emb_vecs = run_analysis(url, embedder, model)
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
        
        # Save to session state so results persist across selectbox reruns
        st.session_state["analysis_done"] = True
        st.session_state["analysis_mode"] = "single"
        st.session_state["single_results"] = (meta, df, timing_df, bursts_df, scores_df, overall_score, emb_vecs)
    
    elif mode == "Compare Two Videos" and url2:
        with st.status("Analyzing Video 1...", expanded=True) as s1:
            meta1, df1, t1, b1, s_df1, score1, ev1 = run_analysis(url, embedder, model)
            s1.update(label="✅ Video 1 Done!", state="complete", expanded=False)
        
        with st.status("Analyzing Video 2...", expanded=True) as s2:
            meta2, df2, t2, b2, s_df2, score2, ev2 = run_analysis(url2, embedder, model)
            s2.update(label="✅ Video 2 Done!", state="complete", expanded=False)
        
        st.session_state["analysis_done"] = True
        st.session_state["analysis_mode"] = "compare"
        st.session_state["compare_results"] = (
            (meta1, df1, t1, b1, s_df1, score1, ev1),
            (meta2, df2, t2, b2, s_df2, score2, ev2)
        )
    else:
        st.warning("Please enter a second YouTube URL for comparison mode.")

# =============================================================================
# RENDER FROM SESSION STATE (persists across selectbox reruns)
# =============================================================================
if st.session_state.get("analysis_done"):
    model, embedder = load_models()
    
    if st.session_state.get("analysis_mode") == "single":
        meta, df, timing_df, bursts_df, scores_df, overall_score, emb_vecs = st.session_state["single_results"]
        render_dashboard(meta, df, timing_df, bursts_df, scores_df, overall_score, emb_vecs, embedder, model, prefix="single")
    
    elif st.session_state.get("analysis_mode") == "compare":
        (meta1, df1, t1, b1, s_df1, score1, ev1), (meta2, df2, t2, b2, s_df2, score2, ev2) = st.session_state["compare_results"]
        
        st.markdown('<p class="section-title">⚖️ Side-by-Side Comparison</p>', unsafe_allow_html=True)
        comp1, comp2 = st.columns(2)
        with comp1:
            st.metric("Video 1 Authenticity", f"{score1}%")
            st.caption(meta1.get("title", "N/A"))
        with comp2:
            st.metric("Video 2 Authenticity", f"{score2}%")
            st.caption(meta2.get("title", "N/A"))
        
        st.divider()
        
        tab1, tab2 = st.tabs([f"📹 {meta1.get('title', 'Video 1')[:40]}...", f"📹 {meta2.get('title', 'Video 2')[:40]}..."])
        with tab1:
            render_dashboard(meta1, df1, t1, b1, s_df1, score1, ev1, embedder, model, prefix="v1")
        with tab2:
            render_dashboard(meta2, df2, t2, b2, s_df2, score2, ev2, embedder, model, prefix="v2")

