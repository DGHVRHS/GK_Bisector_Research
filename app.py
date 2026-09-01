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
    /* Main background and text colors */
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
    if len(filtered) < 2: 
        return filtered

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

# PAGE 3: SIMULATOR (GEOMETRY & xG ENGINE)
elif page == "📐 Interactive Geometry Simulator":
    st.subheader("Interactive Angle Bisector & Positional xG Simulator")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        shot_x = st.slider("Shot X Position (Yards):", 95.0, 115.0, 105.0, step=0.5)
        shot_y = st.slider("Shot Y Position (Yards):", 15.0, 65.0, 30.0, step=0.5)
    with col_ctrl2:
        depth = st.slider("GK Depth off Goal Line (Yards):", 0.5, 8.0, 3.5, step=0.25)
        drift = st.slider("GK Lateral Drift off Bisector (Yards):", -5.0, 5.0, 1.2, step=0.1)

    # 1. Geometry Calculations
    p1 = np.array([120.0, 36.0]) # Left goal post
    p2 = np.array([120.0, 44.0]) # Right goal post
    shot = np.array([shot_x, shot_y])
    
    # Distance to goal center (120, 40)
    shot_dist = np.sqrt((120.0 - shot_x)**2 + (40.0 - shot_y)**2)
    
    # Target vectors to posts
    v1 = p1 - shot
    v2 = p2 - shot
    
    # Shot Angle Calculation
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    shot_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    # Bisector Unit Vector
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    b_vec = (u1 + u2) / np.linalg.norm(u1 + u2)
    b_perp = np.array([-b_vec[1], b_vec[0]])
    
    # Intersection of bisector line with goal line (X = 120)
    t_goal = (120.0 - shot_x) / b_vec[0]
    bisector_goal_pt = shot + t_goal * b_vec
    
    # Ideal & Actual GK positions
    ideal_gk = bisector_goal_pt - depth * b_vec
    actual_gk = ideal_gk + drift * b_perp
    bisector_error = np.abs(drift)

    # 2. Dynamic xG Engine
    # Baseline xG model based on distance and shot angle
    logit_base = (0.06 * shot_angle_deg) - (0.14 * shot_dist) + 0.2
    base_xg = 1.0 / (1.0 + np.exp(-logit_base))
    
    # Positional penalty modifiers (bisector deviation & depth error)
    lateral_penalty = 0.25 * (bisector_error ** 1.5)
    depth_penalty = 0.10 * ((depth - 3.5) ** 2)
    
    logit_pos = logit_base + lateral_penalty + depth_penalty
    positional_xg = 1.0 / (1.0 + np.exp(-logit_pos))
    xg_delta = positional_xg - base_xg

    # 3. Geometry & xG Dashboard Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shot Distance", f"{shot_dist:.1f} yds")
    m2.metric("Shot Angle", f"{shot_angle_deg:.1f}°")
    m3.metric("Baseline xG", f"{base_xg:.3f}")
    m4.metric("Positional xG", f"{positional_xg:.3f}", delta=f"{xg_delta:+.3f} xG Risk", delta_color="inverse")
    m5.metric("Positional Error", f"{bisector_error:.2f} yds")

    # 4. Visualization Engine
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    
    # Pitch Markings
    ax.plot([102, 102, 120, 120], [18, 62, 62, 18], color='#cbd5e1', linestyle='--', linewidth=1.5, label='18-Yard Box')
    ax.plot([114, 114, 120, 120], [30, 50, 50, 30], color='#cbd5e1', linestyle=':', linewidth=1.2, label='6-Yard Box')
    ax.plot([120, 120], [36, 44], color='#dc2626', linewidth=6, label='Goal Line (8 yds)', zorder=4)
    
    # Goal Cone Fill & Rays
    ax.fill([shot_x, 120, 120], [shot_y, 36, 44], color='#3b82f6', alpha=0.12, label='Shot Goal Cone')
    ax.plot([shot_x, 120], [shot_y, 36], color='#2563eb', linestyle=':', linewidth=1.2)
    ax.plot([shot_x, 120], [shot_y, 44], color='#2563eb', linestyle=':', linewidth=1.2)
    
    # Angle Bisector Ray
    ax.plot([shot_x, bisector_goal_pt[0]], [shot_y, bisector_goal_pt[1]], color='#059669', linestyle='--', linewidth=2, label='Angle Bisector Ray')
    
    # Position Points
    ax.scatter([shot_x], [shot_y], color='#d97706', s=140, zorder=5, label='Shot Origin')
    ax.scatter([ideal_gk[0]], [ideal_gk[1]], color='#059669', s=130, marker='o', zorder=5, label='Ideal Position')
    ax.scatter([actual_gk[0]], [actual_gk[1]], color='#dc2626', s=130, marker='^', zorder=5, label='Actual GK Position')
    
    # Positional Error Line
    if bisector_error > 0.05:
        ax.plot([ideal_gk[0], actual_gk[0]], [ideal_gk[1], actual_gk[1]], color='#dc2626', linestyle='-', linewidth=2, label='Positional Error Vector')
    
    ax.set_xlim(88, 122)
    ax.set_ylim(10, 70)
    ax.set_title(f"Tactical Simulator — Baseline xG: {base_xg:.2f} | Positional xG: {positional_xg:.2f}", color='#0f172a', fontweight='bold', fontsize=12)
    ax.set_xlabel("Pitch X (Yards)", color='#475569')
    ax.set_ylabel("Pitch Y (Yards)", color='#475569')
    ax.tick_params(colors='#0f172a')
    ax.grid(True, linestyle='--', alpha=0.3, color='#cbd5e1')
    
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
        
    ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a', loc='upper left')
    st.pyplot(fig)

# PAGE 4: EXPORT
elif page == "📥 Export & Download Center":
    st.subheader("Export Data")
    if not rankings_df.empty:
        st.download_button("Download Leaderboard CSV", rankings_df.to_csv(index=False), "goalkeeper_rankings.csv", "text/csv")
