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
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    h1, h2, h3, h4, label {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }
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
    for fname in ["goalkeeper_shot_analysis (1).csv"]:
        if os.path.exists(fname):
            return pd.read_csv(fname)
    st.error("Missing dataset. Please ensure 'goalkeeper_shot_analysis.csv' is in the root directory.")
    return pd.DataFrame()

df = load_data()

@st.cache_data
def compute_rankings(data, min_shots, zone_filter="All"):
    if data.empty:
        return pd.DataFrame()
        
    filtered_data = data.copy()
    if zone_filter != "All":
        filtered_data = filtered_data[filtered_data['angle_zone'] == zone_filter]
        
    gk_stats = filtered_data.groupby('goalkeeper').agg(
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
    return filtered.sort_values(by='calibrated_scouting_score', ascending=False).reset_index(drop=True)

# Sidebar Navigation
st.sidebar.title("⚽ Navigation & Filters")
page = st.sidebar.radio("Select View:", [
    "📊 Executive Scouting Dashboard",
    "🎯 Goalkeeper Deep Dive Profile",
    "📐 Interactive Geometry Simulator",
    "📥 Export & Download Center"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Settings")
min_shots = st.sidebar.slider("Minimum Shots Faced Filter:", 1, 300, 20)
zone_filter = st.sidebar.selectbox("Angle Zone Filter:", ["All", "Central", "Wide/Tight"])

rankings_df = compute_rankings(df, min_shots, zone_filter)

st.title("🎯 Goalkeeper Bisector & Positional Scouting Model")

# PAGE 1: EXECUTIVE DASHBOARD
if page == "📊 Executive Scouting Dashboard":
    st.subheader("League Positional Scouting Leaderboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Shots Analyzed", len(df))
    col2.metric("Goalkeepers Evaluated", len(rankings_df))
    col3.metric("Avg Bisector Error", f"{df['bisector_dist'].mean():.2f} yds" if not df.empty else "N/A")
    col4.metric("Avg Depth", f"{df['gk_depth'].mean():.2f} yds" if not df.empty else "N/A")
    col5.metric("Central Shooting Angle %", f"{(df['angle_zone']=='Central').mean()*100:.1f}%" if not df.empty else "N/A")
    
    if not rankings_df.empty:
        st.dataframe(
            rankings_df[['goalkeeper', 'shots_faced', 'avg_bisector_error', 'avg_depth', 'avg_model_prob', 'total_gsaa', 'gsaa_per_shot', 'calibrated_scouting_score']]
            .rename(columns={
                'goalkeeper': 'Goalkeeper',
                'shots_faced': 'Shots Faced',
                'avg_bisector_error': 'Avg Bisector Error (yds)',
                'avg_depth': 'Avg Depth (yds)',
                'avg_model_prob': 'Avg Model xG',
                'total_gsaa': 'Total Positional GSAA',
                'gsaa_per_shot': 'GSAA / Shot',
                'calibrated_scouting_score': 'Calibrated Index'
            })
            .style.background_gradient(subset=['Calibrated Index'], cmap='Blues'),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Visual Analytics & Scouting Distributions")
        
        c_chart1, c_chart2 = st.columns(2)
        
        with c_chart1:
            st.markdown("##### Positional GSAA per Shot vs. Bisector Error")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')
            
            scatter = ax.scatter(
                rankings_df['avg_bisector_error'], 
                rankings_df['gsaa_per_shot'], 
                c=rankings_df['calibrated_scouting_score'], 
                cmap='Blues', 
                s=80, 
                edgecolors='#0f172a',
                linewidths=0.5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Calibrated Index", color='#0f172a')
            
            top3 = rankings_df.head(3)
            for _, row in top3.iterrows():
                ax.annotate(row['goalkeeper'].split()[-1], (row['avg_bisector_error'], row['gsaa_per_shot']),
                            xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold', color='#0284c7')
            
            ax.axhline(0, color='#94a3b8', linestyle='--', alpha=0.7)
            ax.set_xlabel("Average Bisector Error (Yards)", color='#475569')
            ax.set_ylabel("Positional GSAA per Shot", color='#475569')
            ax.set_title("Scouting Efficiency Quadrant", color='#0f172a', fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
            st.pyplot(fig)
            
        with c_chart2:
            st.markdown("##### Positional Performance by Angle Zone")
            if not df.empty:
                zone_stats = df.groupby('angle_zone').agg(
                    avg_xg=('pre_shot_xg', 'mean'),
                    avg_model_prob=('model_goal_prob', 'mean'),
                    avg_bisector=('bisector_dist', 'mean')
                ).reset_index()
                
                fig, ax = plt.subplots(figsize=(7, 4.5))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f8fafc')
                
                x_pos = np.arange(len(zone_stats))
                width = 0.35
                
                ax.bar(x_pos - width/2, zone_stats['avg_xg'], width, label='Pre-Shot xG (Baseline)', color='#94a3b8')
                ax.bar(x_pos + width/2, zone_stats['avg_model_prob'], width, label='Model Goal Prob (With Position)', color='#0284c7')
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(zone_stats['angle_zone'], fontweight='bold', color='#0f172a')
                ax.set_ylabel("Goal Probability", color='#475569')
                ax.set_title("Central (>=25°) vs Wide Shot Risk Mitigation", color='#0f172a', fontweight='bold')
                ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a')
                ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
                st.pyplot(fig)

# PAGE 2: GOALKEEPER DEEP DIVE PROFILE
elif page == "🎯 Goalkeeper Deep Dive Profile":
    if not rankings_df.empty:
        selected_gk = st.selectbox("Select Goalkeeper:", rankings_df['goalkeeper'].unique())
        gk_data = df[df['goalkeeper'] == selected_gk]
        gk_rank = rankings_df[rankings_df['goalkeeper'] == selected_gk].iloc[0]
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Shots Faced", int(gk_rank['shots_faced']))
        m2.metric("Calibrated Index", f"{gk_rank['calibrated_scouting_score']:.2f}")
        m3.metric("Total Positional GSAA", f"{gk_rank['total_gsaa']:.2f}")
        m4.metric("Avg Bisector Error", f"{gk_rank['avg_bisector_error']:.2f} yds")
        m5.metric("Avg Depth", f"{gk_rank['avg_depth']:.2f} yds")
        
        st.markdown("---")
        col_p1, col_p2 = st.columns([3, 2])
        
        with col_p1:
            st.subheader(f"Tactical Shot & Position Map: {selected_gk}")
            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')
            
            ax.plot([70, 120, 120, 70, 70], [0, 0, 80, 80, 0], color='#cbd5e1', linewidth=1.5)
            ax.plot([102, 102, 120, 120], [18, 62, 62, 18], color='#cbd5e1', linestyle='--', linewidth=1.2)
            ax.plot([120, 120], [36, 44], color='#dc2626', linewidth=5, label='Goal Line')
            
            goals = gk_data[gk_data['is_goal'] == 1]
            saves = gk_data[gk_data['is_goal'] == 0]
            
            ax.scatter(saves['shot_x'], saves['shot_y'], c='#0284c7', label='Saved Shot Origin', s=50, alpha=0.7)
            ax.scatter(goals['shot_x'], goals['shot_y'], c='#dc2626', label='Goal Shot Origin', s=70, marker='X')
            ax.scatter(gk_data['gk_x'], gk_data['gk_y'], c='#059669', label='GK Position', marker='^', s=50, alpha=0.8)
            
            ax.set_xlim(70, 122)
            ax.set_ylim(5, 75)
            ax.set_title(f"Freeze Frame Positioning & Shot Locations (N={len(gk_data)})", color='#0f172a', fontweight='bold')
            ax.set_xlabel("Pitch Length X (Yards)", color='#475569')
            ax.set_ylabel("Pitch Width Y (Yards)", color='#475569')
            ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a', loc='upper left')
            ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
            st.pyplot(fig)
            
        with col_p2:
            st.subheader("Bisector Error Distribution")
            fig, ax = plt.subplots(figsize=(6, 5.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')
            
            sns.kdeplot(df['bisector_dist'], ax=ax, color='#94a3b8', label='League Benchmark', fill=True, alpha=0.2)
            sns.kdeplot(gk_data['bisector_dist'], ax=ax, color='#0284c7', label=selected_gk, fill=True, alpha=0.4, linewidth=2)
            
            ax.axvline(gk_data['bisector_dist'].mean(), color='#0284c7', linestyle='--', label=f"GK Avg ({gk_data['bisector_dist'].mean():.2f}m)")
            ax.axvline(df['bisector_dist'].mean(), color='#64748b', linestyle=':', label=f"League Avg ({df['bisector_dist'].mean():.2f}m)")
            
            ax.set_xlabel("Bisector Distance Error (Yards)", color='#475569')
            ax.set_ylabel("Density", color='#475569')
            ax.set_title("Positioning Alignment vs League Baseline", color='#0f172a', fontweight='bold')
            ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a')
            ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
            st.pyplot(fig)

# PAGE 3: MODEL INSIGHTS & SHAP
elif page == "🤖 Model Insights & SHAP":
    st.subheader("🤖 Positional Feature Importance & Risk Drivers")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("##### Key Feature Correlations with Goal Probability")
        num_cols = ['bisector_dist', 'gk_depth', 'occlusion_count', 'shot_dist', 'shot_angle', 'pre_shot_xg', 'model_goal_prob']
        corr = df[num_cols].corr()
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor('#ffffff')
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False, square=True)
        ax.set_title("Feature Correlation Matrix", color='#0f172a', fontweight='bold')
        st.pyplot(fig)
        
    with col_s2:
        st.markdown("##### SHAP Impact: Bisector Distance vs Goal Risk")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        scatter = ax.scatter(df['bisector_dist'], df['shap_bisector_impact'], c=df['shot_dist'], cmap='viridis', alpha=0.4, s=20)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Shot Distance (Yards)", color='#0f172a')
        
        ax.axhline(0, color='#dc2626', linestyle='--', label='Neutral SHAP Impact')
        ax.set_xlabel("Bisector Distance Error (Yards)", color='#475569')
        ax.set_ylabel("SHAP Value (Goal Probability Contribution)", color='#475569')
        ax.set_title("Marginal Risk Added by Off-Bisector Positioning", color='#0f172a', fontweight='bold')
        ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0', labelcolor='#0f172a')
        ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
        st.pyplot(fig)

# PAGE 4: INTERACTIVE GEOMETRY SIMULATOR
elif page == "📐 Interactive Geometry Simulator":
    st.subheader("Interactive Angle Bisector & Positional xG Simulator")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        shot_x = st.slider("Shot X Position (Yards):", 95.0, 115.0, 105.0, step=0.5)
        shot_y = st.slider("Shot Y Position (Yards):", 15.0, 65.0, 30.0, step=0.5)
    with col_ctrl2:
        depth = st.slider("GK Depth off Goal Line (Yards):", 0.5, 8.0, 3.5, step=0.25)
        drift = st.slider("GK Lateral Drift off Bisector (Yards):", -5.0, 5.0, 1.2, step=0.1)

    p1 = np.array([120.0, 36.0])
    p2 = np.array([120.0, 44.0])
    shot = np.array([shot_x, shot_y])
    
    shot_dist = np.sqrt((120.0 - shot_x)**2 + (40.0 - shot_y)**2)
    
    v1 = p1 - shot
    v2 = p2 - shot
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    shot_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    b_vec = (u1 + u2) / np.linalg.norm(u1 + u2)
    b_perp = np.array([-b_vec[1], b_vec[0]])
    
    t_goal = (120.0 - shot_x) / b_vec[0]
    bisector_goal_pt = shot + t_goal * b_vec
    
    ideal_gk = bisector_goal_pt - depth * b_vec
    actual_gk = ideal_gk + drift * b_perp
    bisector_error = np.abs(drift)

    logit_base = (0.06 * shot_angle_deg) - (0.14 * shot_dist) + 0.2
    base_xg = 1.0 / (1.0 + np.exp(-logit_base))
    
    lateral_penalty = 0.25 * (bisector_error ** 1.5)
    depth_penalty = 0.10 * ((depth - 3.5) ** 2)
    
    logit_pos = logit_base + lateral_penalty + depth_penalty
    positional_xg = 1.0 / (1.0 + np.exp(-logit_pos))
    xg_delta = positional_xg - base_xg

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Shot Distance", f"{shot_dist:.1f} yds")
    m2.metric("Shot Angle", f"{shot_angle_deg:.1f}°")
    m3.metric("Baseline xG", f"{base_xg:.3f}")
    m4.metric("Positional xG", f"{positional_xg:.3f}", delta=f"{xg_delta:+.3f} xG Risk", delta_color="inverse")
    m5.metric("Positional Error", f"{bisector_error:.2f} yds")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    
    ax.plot([102, 102, 120, 120], [18, 62, 62, 18], color='#cbd5e1', linestyle='--', linewidth=1.5, label='18-Yard Box')
    ax.plot([114, 114, 120, 120], [30, 50, 50, 30], color='#cbd5e1', linestyle=':', linewidth=1.2, label='6-Yard Box')
    ax.plot([120, 120], [36, 44], color='#dc2626', linewidth=6, label='Goal Line (8 yds)', zorder=4)
    
    ax.fill([shot_x, 120, 120], [shot_y, 36, 44], color='#3b82f6', alpha=0.12, label='Shot Goal Cone')
    ax.plot([shot_x, 120], [shot_y, 36], color='#2563eb', linestyle=':', linewidth=1.2)
    ax.plot([shot_x, 120], [shot_y, 44], color='#2563eb', linestyle=':', linewidth=1.2)
    
    ax.plot([shot_x, bisector_goal_pt[0]], [shot_y, bisector_goal_pt[1]], color='#059669', linestyle='--', linewidth=2, label='Angle Bisector Ray')
    
    ax.scatter([shot_x], [shot_y], color='#d97706', s=140, zorder=5, label='Shot Origin')
    ax.scatter([ideal_gk[0]], [ideal_gk[1]], color='#059669', s=130, marker='o', zorder=5, label='Ideal Position')
    ax.scatter([actual_gk[0]], [actual_gk[1]], color='#dc2626', s=130, marker='^', zorder=5, label='Actual GK Position')
    
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

# PAGE 5: EXPORT CENTER
elif page == "📥 Export & Download Center":
    st.subheader("Export Data & Scouting Reports")
    if not rankings_df.empty:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("##### Download Aggregated Leaderboard")
            st.download_button("Download Leaderboard CSV", rankings_df.to_csv(index=False), "goalkeeper_rankings.csv", "text/csv")
        with col_e2:
            st.markdown("##### Download Processed Shot Dataset")
            st.download_button("Download Processed Shots CSV", df.to_csv(index=False), "goalkeeper_shot_analysis_full.csv", "text/csv")
