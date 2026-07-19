"""
app.py — DiabetesVision (with Thermal Foot modules)
====================================================
Entry point. Adds two tabs to the existing DiabetesVision app:
  - Foot Thermography (pages/thermal_foot.py)  — trained image model, AUC 0.878
  - Risk Correlation  (pages/thermal_risk.py)  — trained risk models (neuropathy 0.931, PAD 0.825)

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="DiabetesVision", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

# --- Shared styling (matches existing DiabetesVision theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    /* Hide Streamlit's auto-generated pages/ sidebar + its collapse control */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"],
    [data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] { display: none !important; }
    section[data-testid="stSidebar"] { width: 0 !important; min-width: 0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 8px;
        background: #0D1B2E; color: #7A8FA6; border: 1px solid #1E3A5F; }
    .stTabs [aria-selected="true"] { background: #00D4AA22 !important;
        color: #00D4AA !important; border-color: #00D4AA44 !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #E8EDF5; }
    .hero-badge { display:inline-block; background:#00D4AA22; color:#00D4AA;
        border:1px solid #00D4AA44; border-radius:20px; padding:4px 16px;
        font-size:0.8rem; font-weight:600; letter-spacing:0.08em;
        text-transform:uppercase; margin-bottom:1.5rem; }
    .section-header { font-size:2rem; font-weight:700; color:#E8EDF5; }
    .usecase-card, .solution-card, .problem-card, .mode-card, .stat-card {
        background:#0D1B2E; border:1px solid #1E3A5F; border-radius:12px; padding:1.5rem; }
    .usecase-title { font-size:1.1rem; font-weight:600; color:#E8EDF5; }
    .stat-number { font-size:2.5rem; font-weight:700; color:#00D4AA; }
    .stat-label { color:#7A8FA6; font-size:0.875rem; }
    .divider { border-top:1px solid #1E2D45; margin:3rem 0; }
</style>
""", unsafe_allow_html=True)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pages"))

tabs = st.tabs([
    "🏠 Home", "🙋 My Foot Check", "🟢 Patient Screening", "🔵 Doctor Dashboard",
    "🌡️ Foot Thermography", "🦶 PAD & Ulcer", "⚡ Neuropathy",
    "📊 Risk & CVD", "📋 Summary", "💚 Quality of Life", "📚 About & Research",
])

with tabs[0]:
    import home; home.run()
with tabs[1]:
    import patient_intake; patient_intake.run()
with tabs[2]:
    import patient_mode; patient_mode.run()
with tabs[3]:
    import doctor_mode; doctor_mode.run()
with tabs[4]:
    import thermal_foot; thermal_foot.run()
with tabs[5]:
    import pad_ulcer_score; pad_ulcer_score.run()
with tabs[6]:
    import neuropathy_score; neuropathy_score.run()
with tabs[7]:
    import thermal_risk; thermal_risk.run()
with tabs[8]:
    import summary_report; summary_report.run()
with tabs[9]:
    import qol_proxy; qol_proxy.run()
with tabs[10]:
    import about; about.run()
    import model_card; model_card.run()
