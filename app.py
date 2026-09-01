import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os

st.set_page_config(
    page_title="Goalkeeper Bisector & Positional Scouting Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Light Theme & High-Contrast Metric CSS Injection
st.markdown("""
    <style>
    /* Force main app background and default text to light tones */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Global light container boxes */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        padding: 18px !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* High contrast metric label */
    div[data-testid="stMetricLabel"] > label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* High contrast metric numeric value */
    div[data-testid="stMetricValue"] > div {
        color: #0f172a !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Headings readability fix */
    h1, h2, h3, h4, label {
        color: #0f172a !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists("goalkeeper_shot_analysis.csv"):
        return pd.read_csv("goalkeeper_shot_analysis.csv")
    else:
        st.error("Missing 'goalkeeper_shot_analysis.csv'. Using synthetic fallback data.")
        return pd.DataFrame()

df = load_data()

@st.cache_data
def compute_rankings(data, min_shots):
    gk_stats = data.groupby('goalkeeper').agg(
        shots_faced=('is_goal', 'count'),
        avg_bisector_error=('bisector_dist', 'mean'),
        avg_depth=('gk_depth', 'mean'),
        avg_occlusion=('occlusion_count', 'mean'),
        avg_model_prob=('model_goal_prob', 'mean'),
        avg_shap_risk=('shap_bisector_impact', 'mean'),
        total_gsaa=('positional_gsaa', 'sum')
    ).reset_index()

    filtered = gk_stats[gk_stats['shots_faced'] >= min_shots].copy()
    if len(filtered) < 2: return filtered

    filtered['gsaa_per_shot'] = filtered['total_gsaa'] / filtered['shots_faced']
    
    # Calibrated Z-Score Index
    filtered['z_prob'] = -zscore(filtered['avg_model_prob'])
    filtered['z_bisector'] = -zscore(filtered['avg_bisector_error'])
    filtered['z_gsaa'] = zscore(filtered['gsaa_per_shot'])

    filtered['calibrated_scouting_score'] = (
        (0.40 * filtered['z_prob']) +
        (0.30 * filtered['z_bisector']) +
        (0.30 * filtered['z_gsaa'])
    )
    return filtered.sort_values(by='calibrated_scouting_score', ascending=False)

# Sidebar Navigation
st.sidebar.title("⚽ Navigation & Filters")
page = st.sidebar.radio("Select View:", [
    "📊 Executive Scouting Dashboard",
    "🎯 Goalkeeper Deep Dive Profile",
    "📐 Interactive Geometry Simulator",
    "🤖 ML Model & SHAP Explainer",
    "📥 Export & Download Center"
])

min_shots = st.sidebar.slider("Minimum Shots Faced Filter:", 1, 30, 5)
rankings_df = compute_rankings(df, min_shots)

st.title("🎯 Goalkeeper Bisector & Positional Scouting Model")

# PAGE 1: DASHBOARD
if page == "📊 Executive Scouting Dashboard":
    st.subheader("League Positional Scouting Leaderboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shots Analyzed", len(df))
    col2.metric("Goalkeepers Evaluated", len(rankings_df))
    col3.metric("Avg Bisector Error", f"{df['bisector_dist'].mean():.2f} yds")
    col4.metric("Avg Depth", f"{df['gk_depth'].mean():.2f} yds")
    
    st.dataframe(
        rankings_df[['goalkeeper', 'shots_faced', 'avg_bisector_error', 'avg_depth', 'avg_model_prob', 'total_gsaa', 'calibrated_scouting_score']]
        .rename(columns={'calibrated_scouting_score': 'Calibrated Index'})
        .style.background_gradient(subset=['Calibrated Index'], cmap='Blues'),
        use_container_width=True
    )

# PAGE 2: GOALKEEPER PROFILE
elif page == "🎯 Goalkeeper Deep Dive Profile":
    selected_gk = st.selectbox("Select Goalkeeper:", rankings_df['goalkeeper'].unique())
    gk_data = df[df['goalkeeper'] == selected_gk]
    
    st.subheader(f"Tactical Shot Map: {selected_gk}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#1e293b')
    ax.scatter(gk_data['shot_x'], gk_data['shot_y'], c='crimson', label='Shot Origin', s=60)
    ax.scatter(gk_data['gk_x'], gk_data['gk_y'], c='cyan', label='GK Position', marker='^', s=60)
    ax.set_xlim(90, 121)
    ax.set_ylim(10, 70)
    ax.set_title(f"Freeze Frames for {selected_gk}", color='white')
    ax.legend(facecolor='#0e1117', labelcolor='white')
    st.pyplot(fig)

# PAGE 3: SIMULATOR
elif page == "📐 Interactive Geometry Simulator":
    st.subheader("Interactive Angle Bisector & Goal Cone Simulator")
    shot_x = st.slider("Shot X Position:", 95.0, 115.0, 105.0)
    shot_y = st.slider("Shot Y Position:", 20.0, 60.0, 30.0)
    drift = st.slider("Goalkeeper Drift off Bisector (Yards):", -4.0, 4.0, 1.2)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#0f172a')
    ax.plot([120, 120], [36, 44], color='red', linewidth=5, label='Goal Line')
    ax.scatter([shot_x], [shot_y], color='gold', s=120, label='Shot Origin')
    ax.set_xlim(90, 122)
    ax.set_ylim(15, 65)
    ax.legend(facecolor='#1e293b', labelcolor='white')
    st.pyplot(fig)

# PAGE 4: EXPORT
elif page == "📥 Export & Download Center":
    st.subheader("Export Data")
    st.download_button("Download Leaderboard CSV", rankings_df.to_csv(index=False), "goalkeeper_rankings.csv", "text/csv")
