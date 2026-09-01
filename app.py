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

# High-Contrast Light Theme Custom CSS
st.markdown("""
    <style>
    /* Main background and base font colors */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Typography color updates */
    h1, h2, h3, h4, label {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Light Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stMetricLabel"] > label {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #0f172a !important;
        font-weight: 700 !important;
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
    if data.empty:
        return pd.DataFrame()
        
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
    col3.metric("Avg Bisector Error", f"{df['bisector_dist'].mean():.2f} yds" if not df.empty else "N/A")
    col4.metric("Avg Depth", f"{df['gk_depth'].mean():.2f} yds" if not df.empty else "N/A")
    
    if not rankings_df.empty:
        st.dataframe(
            rankings_df[['goalkeeper', 'shots_faced', 'avg_bisector_error', 'avg_depth', 'avg_model_prob', 'total_gsaa', 'calibrated_scouting_score']]
            .rename(columns={'calibrated_scouting_score': 'Calibrated Index'})
            .style.background_gradient(subset=['Calibrated Index'], cmap='Blues'),
            use_container_width=True
        )

# PAGE 2: GOALKEEPER PROFILE
elif page == "🎯 Goalkeeper Deep Dive Profile":
    if not rankings_df.empty:
        selected_gk = st.selectbox("Select Goalkeeper:", rankings_df['goalkeeper'].unique())
        gk_data = df[df['goalkeeper'] == selected_gk]
        
        st.subheader(f"Tactical Shot Map: {selected_gk}")
        
        # Light theme Matplotlib setup
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        ax.scatter(gk_data['shot_x'], gk_data['shot_y'], c='#dc2626', label='Shot Origin', s=60)
        ax.scatter(gk_data['gk_x'], gk_data['gk_y'], c='#0284c7', label='GK Position', marker='^', s=60)
        
        ax.set_xlim(90, 121)
        ax.set_ylim(10, 70)
        ax.set_title(f"Freeze Frames for {selected_gk}", color='#0f172a', fontweight='bold')
        ax.set_xlabel("Field X Position", color='#475569')
        ax.set_ylabel("Field Y Position", color='#475569')
        ax.tick_params(colors='#0f172a')
        
        for spine in ax.spines.values():
            spine.set_color('#cbd5e1')
            
        ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a')
        st.pyplot(fig)

# PAGE 3: SIMULATOR
elif page == "📐 Interactive Geometry Simulator":
    st.subheader("Interactive Angle Bisector & Goal Cone Simulator")
    shot_x = st.slider("Shot X Position:", 95.0, 115.0, 105.0)
    shot_y = st.slider("Shot Y Position:", 20.0, 60.0, 30.0)
    drift = st.slider("Goalkeeper Drift off Bisector (Yards):", -4.0, 4.0, 1.2)
    
    # Light theme Matplotlib setup
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    
    ax.plot([120, 120], [36, 44], color='#dc2626', linewidth=5, label='Goal Line')
    ax.scatter([shot_x], [shot_y], color='#d97706', s=120, label='Shot Origin')
    
    ax.set_xlim(90, 122)
    ax.set_ylim(15, 65)
    ax.set_title("Angle Bisector & Goal Cone Simulation", color='#0f172a', fontweight='bold')
    ax.tick_params(colors='#0f172a')
    
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        
    ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a')
    st.pyplot(fig)

# PAGE 4: EXPORT
elif page == "📥 Export & Download Center":
    st.subheader("Export Data")
    if not rankings_df.empty:
        st.download_button("Download Leaderboard CSV", rankings_df.to_csv(index=False), "goalkeeper_rankings.csv", "text/csv")
