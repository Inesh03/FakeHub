import os
import sys

# Add project root to sys.path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
import os

# Suppress annoying HuggingFace/Torch logs in the Streamlit terminal
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(
    page_title="Fake Engagement Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #E2E8F0;
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛡️ Fake Engagement Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered behavioral analysis to detect bots, coordinated rings, and engagement bursts on YouTube.</p>', unsafe_allow_html=True)

# --- Sidebar Input ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=60) # YouTube icon
    st.markdown("### Analysis Configuration")
    url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    analyze_btn = st.button("🔍 Analyze Video", type="primary", use_container_width=True)
    st.divider()
    st.markdown("""
    **Indicators Monitored:**
    - 🗣️ Linguistic NLP
    - ⏱️ Timing Regularity 
    - 📈 Engagement Bursts
    - 🕸️ Coordinated Graph Postings
    """)

if analyze_btn and url:
    with st.spinner("Fetching comments from YouTube API..."):
        meta = fetch_video_metadata(url)
        df   = fetch_comments(url, max_comments=500)

    st.sidebar.success(f"✅ Fetched {len(df)} comments!")

    # Video Meta Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.info(f"**👁️ Views:** {meta.get('view_count', 0):,}")
    with m2:
        st.success(f"**👍 Likes:** {meta.get('like_count', 0):,}")
    with m3:
        st.warning(f"**💬 Comments:** {meta.get('comment_count', 0):,}")
    with m4:
        st.error(f"**📥 Fetched:** {len(df):,}")

    st.markdown(f"**Video Title:** {meta.get('title', 'Unknown Video')} | **Channel:** {meta.get('channel', 'Unknown')}")
    st.divider()

    with st.status("Running AI Behavioral Analysis Pipeline...", expanded=True) as status:
        st.write("🤖 1/4: Running NLP bot detection (LLM Embeddings + Neural Net)...")
        model = joblib.load("models/llm_mlp_model.pkl")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        clean_texts = [preprocess_text(t) for t in df["text"].fillna("").tolist()]
        embedded_vectors = embedder.encode(clean_texts)
        df["nlp_bot_prob"] = model.predict_proba(embedded_vectors)[:, 1]

        st.write("⏱️ 2/4: Analyzing timing patterns & engagement bursts...")
        timing_df = analyze_timing(df)
        bursts_df = detect_engagement_bursts(df, bucket_freq="1min", threshold_std=2.0)

        st.write("🕸️ 3/4: Building Neo4j interaction graph to find rings...")
        build_interaction_graph(df)
        graph_scores = get_graph_cluster_scores(df["author"].unique().tolist())

        st.write("📊 4/4: Fusing behavioural signals into authenticity scores...")
        scores_df   = compute_per_commenter_scores(df, timing_df, graph_scores)
        overall_score = compute_overall_authenticity(scores_df)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # --- Score Display ---
    st.markdown('<p class="section-title">🎯 Overall Authenticity Assessment</p>', unsafe_allow_html=True)

    col_gauge, col_insights = st.columns([1, 1])
    
    with col_gauge:
        color = "#10B981" if overall_score >= 70 else ("#F59E0B" if overall_score >= 40 else "#EF4444")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall_score,
            title={"text": "Authenticity Score", "font": {"size": 24, "color": "white"}},
            number={"font": {"size": 48, "color": color}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar":  {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [40, 70],  "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.2)"},
                ]
            }
        ))
        gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(gauge, use_container_width=True)

    with col_insights:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Key Insights")
        high_risk_count = len(scores_df[scores_df["risk_label"] == "🔴 High Risk"])
        burst_count = len(bursts_df[bursts_df["is_burst"] == True]) if not bursts_df.empty else 0
        ring_count = len(scores_df[scores_df["graph_cluster_score"] > 0])
        
        st.info(f"🚨 **{high_risk_count}** accounts flagged as High Risk.")
        st.warning(f"📈 **{burst_count}** engagement bursts detected in timeline.")
        st.error(f"🕸️ **{ring_count}** accounts detected in coordinated posting rings.")

    st.divider()

    col_timeline, col_hist = st.columns([1.2, 1])
    
    with col_timeline:
        st.markdown('<p class="section-title">📈 Engagement Bursts Timeline</p>', unsafe_allow_html=True)
        if not bursts_df.empty:
            fig_timeline = px.line(
                bursts_df, x="published_at", y="comment_count",
                labels={"published_at": "Time", "comment_count": "Comments per Minute"},
                title="Comment Volume over Time (1m buckets)"
            )
            fig_timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)")
            burst_points = bursts_df[bursts_df["is_burst"] == True]
            if not burst_points.empty:
                fig_timeline.add_scatter(
                    x=burst_points["published_at"], 
                    y=burst_points["comment_count"],
                    mode="markers",
                    marker=dict(color="#EF4444", size=12, symbol="x"),
                    name="Anomaly Burst"
                )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Not enough data to plot an timeline.")

    with col_hist:
        st.markdown('<p class="section-title">📊 Bot Score Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            scores_df, x="bot_score", nbins=30,
            color="risk_label",
            color_discrete_map={
                "🔴 High Risk": "#EF4444",
                "🟡 Suspicious": "#F59E0B",
                "🟢 Likely Human": "#10B981"
            },
            labels={"bot_score": "Bot Probability Score", "count": "Commenters"},
            title="Distribution among Commenters"
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.2)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # --- Per-Commenter Table ---
    st.markdown('<p class="section-title">👤 Detailed Behaviour & Probability Breakdown</p>', unsafe_allow_html=True)
    display_df = scores_df[[
        "author", "comment_count", "nlp_bot_prob",
        "timing_risk", "graph_cluster_score", "bot_score", "risk_label", "anomaly_explanation"
    ]].rename(columns={
        "author":               "Author",
        "comment_count":        "Comments",
        "nlp_bot_prob":         "NLP Score",
        "timing_risk":          "Timing Risk",
        "graph_cluster_score":  "Graph Risk",
        "bot_score":            "Bot Score",
        "risk_label":           "Risk Level",
        "anomaly_explanation":  "Anomaly Explanation"
    })
    for col in ["NLP Score", "Timing Risk", "Bot Score"]:
        display_df[col] = display_df[col].round(3)
    
    st.dataframe(
        display_df, 
        use_container_width=True, 
        height=350,
        column_config={
            "Anomaly Explanation": st.column_config.TextColumn("Anomaly Explanation", width="large")
        }
    )

    # --- Explainable AI (LIME) ---
    st.markdown('<p class="section-title">🧠 Explainable AI (XAI) - Local Interpretability</p>', unsafe_allow_html=True)
    st.markdown("Select an author to inspect why the NLP engine assigned them their specific Bot Probability.")
    
    c1, c2 = st.columns([1, 2])
    
    suspicious_authors = display_df[display_df["Risk Level"] != "🟢 Likely Human"]["Author"].tolist()
    all_authors = display_df["Author"].tolist()
    
    with c1:
        selected_author = st.selectbox(
            "1. Select an author:", 
            options=suspicious_authors if suspicious_authors else all_authors
        )
        if selected_author:
            author_comments = df[df["author"] == selected_author]["text"].tolist()
            selected_comment = st.selectbox("2. Select a comment:", options=author_comments)
            
    with c2:
        if selected_author and selected_comment:
            if len(str(selected_comment).split()) < 2:
                st.warning("⚠️ This comment is too short for LIME to generate a meaningful AI feature breakdown.")
            else:
                with st.spinner("Generating LIME Explanation for this comment..."):
                    try:
                        explainer = LimeTextExplainer(class_names=["Human", "Bot"])
                        def predictor(texts):
                            cleaned = [preprocess_text(t) for t in texts]
                            # Encode text dynamically with the LLM embedder for LIME
                            vecs = embedder.encode(cleaned)
                            return model.predict_proba(vecs)
                        
                        exp = explainer.explain_instance(
                            selected_comment, 
                            predictor, 
                            num_features=10
                        )
                        
                        # --- FIX LIME DARK MODE VISIBILITY ---
                        lime_html = exp.as_html()
                        styled_html = f"""
                        <div style="background-color: #FFFFFF; padding: 20px; border-radius: 10px; color: black; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h4 style="margin-top:0; color:#1E293B; font-family: sans-serif;">LIME Feature Importance</h4>
                            {lime_html}
                        </div>
                        """
                        components.html(styled_html, height=450, scrolling=True)
                    except Exception as e:
                        st.error("⚠️ LIME could not process this comment. It likely contains only special characters or emojis which the explainer cannot perturb.")

    st.markdown('<p class="section-title">🕸️ Interactive Network Graph (Neo4j)</p>', unsafe_allow_html=True)
    st.markdown("This 3D physics-based graph shows accounts targeting the video. Suspicious users (part of rings or high-volume) are highlighted in **Orange**.")
    
    with st.spinner("Rendering Neo4j graph..."):
        # Generate the HTML from our Neo4j database
        graph_html = generate_network_html(suspicious_authors=set(suspicious_authors))
        
        # Display it natively in Streamlit
        components.html(graph_html, height=620, scrolling=False)
