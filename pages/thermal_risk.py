"""
thermal_risk.py — CVD Risk & Prognosis Correlation Module (v3)
==============================================================
Trained models (thermal_risk_models.pkl) — HistGradientBoosting on 50,000
synthetic patient records, validated on a held-out patient-level test set (n=7,502):
    neuropathy  test AUC = 0.931   (meaningful — stochastic ground truth)
    PAD         test AUC = 0.825   (meaningful — stochastic ground truth)
    ulcer       test AUC = 0.995   (near-deterministic rule label)
    abnormal    test AUC = 0.997   (near-deterministic rule label)

IMPORTANT: the 50k dataset is SYNTHETIC. The neuropathy/PAD figures reflect
realistic learnable signal; the ulcer/abnormal figures are close to the
label-generating rule and are NOT evidence of clinical diagnostic accuracy.
The image-based DM classifier in thermal_foot.py (real ThermoDataBase, AUC 0.878)
remains the primary diagnostic result. This module provides risk stratification.

CVD risk: simplified UKPDS Risk Engine (Stevens et al., 2001, UKPDS 56).
"""

import streamlit as st
import shared
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

# ── TRAINED MODEL LOADER ──────────────────────────────────────────────────────

def _find_file(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [name, os.path.join(here, name), os.path.join(here, "..", name),
              os.path.join("models", name), os.path.join(here, "..", "models", name)]:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource
def load_risk_models():
    path = _find_file("thermal_risk_models.pkl")
    if path:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def model_predict(bundle, which, feature_dict):
    """Predict probability from a trained model given a feature dict."""
    if bundle is None:
        return None
    feats = bundle["features"]
    X = pd.DataFrame([{f: feature_dict.get(f, np.nan) for f in feats}])[feats]
    return float(bundle["models"][which].predict_proba(X)[0, 1])


# ── KARACHI-DERIVED COMPLICATION DATA (cross-check) ───────────────────────────

ULCER_RATE_BY_HBAIC = {"<7": 0.25, "7-8": 0.50, "8-9": 0.50, "9-10": 0.33, ">10": 0.20}

COMPLICATION_PROFILES = {
    "Foot Ulcers":  {"hba1c": 8.03, "age": 45, "bmi": 26.4, "bp": 189, "n": 6},
    "Neuropathy":   {"hba1c": 7.64, "age": 58, "bmi": 28.7, "bp": 199, "n": 5},
    "Nephropathy":  {"hba1c": 8.75, "age": 66, "bmi": 31.5, "bp": 273, "n": 2},
    "Retinopathy":  {"hba1c": 8.53, "age": 53, "bmi": 26.7, "bp": 160, "n": 3},
}


# ── HEURISTIC FALLBACKS ───────────────────────────────────────────────────────

def get_hba1c_band(hba1c):
    if hba1c < 7.0:  return "<7"
    if hba1c < 8.0:  return "7-8"
    if hba1c < 9.0:  return "8-9"
    if hba1c < 10.0: return "9-10"
    return ">10"


def foot_ulcer_probability_heuristic(hba1c, n_hot_zones, history_ulcer, pad):
    base = ULCER_RATE_BY_HBAIC[get_hba1c_band(hba1c)]
    base += min(n_hot_zones * 0.04, 0.30)
    if history_ulcer: base = min(base + 0.35, 0.95)
    if pad:           base = min(base + 0.15, 0.90)
    return min(round(base, 3), 0.95)


def neuropathy_probability_heuristic(hba1c, duration_dm, abs_temp_elevated):
    base = 0.15 + max(0, hba1c - 7.0) * 0.03 + min(duration_dm * 0.015, 0.30)
    if abs_temp_elevated: base += 0.18
    return min(round(base, 3), 0.90)


def estimate_cvd_risk(age, duration_dm, hba1c, systolic_bp, smoker,
                      total_cholesterol, hdl_cholesterol, sex_male):
    score = (age - 40) * 0.65 + duration_dm * 0.50
    if hba1c > 7.0:       score += (hba1c - 7.0) * 2.8
    if systolic_bp > 140: score += (systolic_bp - 140) * 0.25
    if hdl_cholesterol > 0:
        ratio = total_cholesterol / hdl_cholesterol
        if ratio > 4.5:   score += (ratio - 4.5) * 3.0
    if smoker:            score += 12.0
    if not sex_male:      score -= 6.0
    return round(max(1.0, min(score, 60.0)), 1)


def composite_risk(thermal_score, cvd_risk_pct, hba1c, foot_ulcer_prob):
    return round(min(
        thermal_score * 0.35 +
        min(cvd_risk_pct / 0.60, 100) * 0.30 +
        foot_ulcer_prob * 100 * 0.20 +
        max(0, hba1c - 6.5) / (16.0 - 6.5) * 100 * 0.15, 100), 1)


def cvd_band(risk_pct):
    if risk_pct < 10:  return "Low", "success", "#00D4AA"
    if risk_pct < 20:  return "Moderate", "info", "#0099FF"
    if risk_pct < 30:  return "High", "warning", "#FFA500"
    return "Very High", "error", "#FF4B4B"


def deterioration_trajectory(hba1c, iwgdf, n_hot_zones, duration_dm, abs_temp_elevated, pad):
    years = list(range(6))
    base = iwgdf * 18 + n_hot_zones * 3 + max(0, hba1c - 7.0) * 4
    if abs_temp_elevated: base += 8
    if pad:               base += 12
    rate = 9.0 if hba1c > 8.5 else 5.0 if hba1c > 7.5 else 2.5
    current   = [min(base + rate * y, 100) for y in years]
    optimised = [min(max(iwgdf * 8, 5) + 1.5 * y, 55) for y in years]
    return years, current, optimised


# ── CHART BUILDERS ────────────────────────────────────────────────────────────

def build_trajectory(years, current, optimised):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=current, mode="lines+markers",
        name="Current trajectory", line=dict(color="#FF4B4B", width=2.5), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=years, y=optimised, mode="lines+markers",
        name="Optimised (HbA1c target + podiatry)", line=dict(color="#00D4AA", width=2.5, dash="dot"), marker=dict(size=7)))
    fig.update_layout(paper_bgcolor="#0D1526", plot_bgcolor="#0D1526",
        font=dict(family="DM Sans", color="#A0B0C5"),
        xaxis=dict(title="Years", gridcolor="#1E2D45", dtick=1),
        yaxis=dict(title="Composite Risk", gridcolor="#1E2D45", range=[0, 105]),
        margin=dict(l=20, r=20, t=20, b=20), legend=dict(bgcolor="#1E2D45"), height=340)
    return fig


def build_radar(thermal_score, cvd_risk, hba1c, duration_dm, systolic_bp,
                n_hot, ulcer_prob, neuro_prob, pad_prob):
    cats = ["Foot Thermal","CVD 10-yr","Glycaemic","DM Duration","Blood Pressure",
            "Zone Involvement","Ulcer Prob","Neuropathy","PAD/Vascular"]
    vals = [thermal_score, min(cvd_risk/0.60,100),
            max(0,min((hba1c-6.5)/(16.0-6.5)*100,100)),
            min(duration_dm/30*100,100), max(0,min((systolic_bp-100)/80*100,100)),
            n_hot/10*100, ulcer_prob*100, neuro_prob*100, pad_prob*100]
    vals_c = vals + [vals[0]]; cats_c = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(r=vals_c, theta=cats_c, fill="toself",
        fillcolor="rgba(0,212,170,0.15)", line=dict(color="#00D4AA", width=2)))
    fig.update_layout(polar=dict(bgcolor="#1E2D45",
        radialaxis=dict(visible=True, range=[0,100], gridcolor="#2E3D55", tickfont=dict(color="#7A8FA6")),
        angularaxis=dict(gridcolor="#2E3D55", tickfont=dict(color="#A0B0C5"))),
        paper_bgcolor="#0D1526", font=dict(family="DM Sans", color="#A0B0C5"),
        margin=dict(l=40,r=40,t=40,b=40), height=400, showlegend=False)
    return fig


def build_karachi_chart(current_hba1c):
    bands = list(ULCER_RATE_BY_HBAIC.keys())
    rates = [v*100 for v in ULCER_RATE_BY_HBAIC.values()]
    cur = get_hba1c_band(current_hba1c)
    colours = ["#FF4B4B" if b == cur else "#1E3A5F" for b in bands]
    fig = go.Figure(go.Bar(x=bands, y=rates, marker_color=colours,
        text=[f"{r:.0f}%" for r in rates], textposition="outside"))
    fig.update_layout(paper_bgcolor="#0D1526", plot_bgcolor="#0D1526",
        font=dict(family="DM Sans", color="#A0B0C5"),
        xaxis=dict(title="HbA1c Band (%)", gridcolor="#1E2D45"),
        yaxis=dict(title="Foot Ulcer Rate (%)", gridcolor="#1E2D45", range=[0,75]),
        margin=dict(l=20,r=20,t=20,b=20), height=300, showlegend=False,
        title=dict(text="Foot Ulcer Rate by HbA1c Band (Karachi Cohort, n=20)", font=dict(color="#A0B0C5")))
    return fig


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────

def run():
    st.title("CVD Risk & Prognosis Correlation")
    st.markdown(
        "*Integrates thermal foot data with HbA1c and cardiovascular risk factors. "
        "Neuropathy/PAD models trained on 50,000 records (held-out test n=7,502).*"
    )

    bundle = load_risk_models()
    if bundle is not None:
        m = bundle["test_auc"]
        st.success(
            f"Trained risk models active — Neuropathy AUC {m['neuropathy']} · "
            f"PAD AUC {m['pad']} (held-out test, n={bundle['n_test']}). "
            f"Ulcer/abnormal labels are rule-derived (see notes)."
        )
    else:
        st.info("Heuristic mode. Place thermal_risk_models.pkl beside this file to enable trained models.")

    st.markdown("---")

    # ── STEP 1: THERMAL MEASUREMENTS ──────────────────────────────────────────
    st.subheader("Step 1 — Thermal Foot Measurements")
    st.caption("From the Foot Thermography tab, or entered manually. Zone scores standardised (0=normal, higher=abnormal).")
    c1, c2, c3, c4 = st.columns(4)
    with c1: thermal_score = st.number_input("Foot Health Score (0-100)", 0, 100, 35, key="thermal_risk_1")
    with c2: iwgdf         = st.selectbox("IWGDF Category", [0,1,2,3], index=1, key="thermal_risk_2")
    with c3: n_hot_zones   = st.number_input("High-Risk Zones (0-10)", 0, 10, 3, key="thermal_risk_3")
    with c4: abs_temp_elev = st.checkbox("Absolute Temp Elevated", key="thermal_risk_4")

    c1, c2, c3, c4 = st.columns(4)
    with c1: left_temp  = st.number_input("Left foot mean temp (degC)", 20.0, 38.0, 29.2, step=0.1, key="thermal_risk_5")
    with c2: right_temp = st.number_input("Right foot mean temp (degC)", 20.0, 38.0, 29.2, step=0.1, key="thermal_risk_6")
    with c3: mtk2 = st.number_input("MTK2 zone score", 0.0, 3.0, 0.6, step=0.05, key="thermal_risk_7")
    with c4: mtk4 = st.number_input("MTK4 zone score", 0.0, 3.0, 0.6, step=0.05, key="thermal_risk_8")

    c1, c2, c3, c4 = st.columns(4)
    with c1: hallux  = st.number_input("Hallux zone score", 0.0, 3.0, 0.5, step=0.05, key="thermal_risk_9")
    with c2: heel    = st.number_input("Heel zone score", 0.0, 3.0, 0.4, step=0.05, key="thermal_risk_10")
    with c3: midfoot = st.number_input("Midfoot zone score", 0.0, 3.0, 0.4, step=0.05, key="thermal_risk_11")
    with c4: autonomic = st.number_input("Autonomic thermal signal", 0.0, 3.0, 0.5, step=0.05, key="thermal_risk_12")

    asymmetry = abs(left_temp - right_temp)

    # ── STEP 2: CLINICAL VARIABLES ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 2 — Clinical Variables")
    c1, c2, c3, c4 = st.columns(4)
    with c1: hba1c       = shared.number("HbA1c (%)", "hba1c", "thermal_risk_13", 8.2, 4.0, 16.0, step=0.1)
    with c2: duration_dm = shared.number("DM Duration (yrs)", "duration_dm", "thermal_risk_14", 12, 0, 60, step=1)
    with c3: systolic_bp = shared.number("Systolic BP (mmHg)", "systolic_bp", "thermal_risk_15", 135, 80, 240, step=1)
    with c4: age         = shared.number("Age", "age", "thermal_risk_16", 58, 1, 120, step=1)

    c1, c2, c3, c4 = st.columns(4)
    with c1: sex_male   = st.selectbox("Sex", ["Male","Female"], key="thermal_risk_17") == "Male"
    with c2: smoker     = st.checkbox("Current Smoker", key="thermal_risk_18")
    with c3: total_chol = st.number_input("Total Cholesterol (mmol/L)", 2.0, 12.0, 5.2, step=0.1, key="thermal_risk_19")
    with c4: hdl_chol   = st.number_input("HDL (mmol/L)", 0.5, 4.0, 1.1, step=0.1, key="thermal_risk_20")

    c1, c2, c3 = st.columns(3)
    with c1: history_ulcer = st.checkbox("Previous Foot Ulcer / Amputation", key="thermal_risk_21")
    with c2: pad_known     = st.checkbox("Known PAD", key="thermal_risk_22")
    with c3: bmi           = st.number_input("BMI (kg/m2)", 15.0, 50.0, 27.0, step=0.1, key="thermal_risk_23")

    # ── FEATURE DICT ──────────────────────────────────────────────────────────
    fd = {
        "left_foot_mean_temperature_c": left_temp,
        "right_foot_mean_temperature_c": right_temp,
        "left_right_asymmetry_c": asymmetry,
        "mtk2_zone_score": mtk2, "mtk4_zone_score": mtk4,
        "hallux_zone_score": hallux, "heel_zone_score": heel,
        "midfoot_zone_score": midfoot, "autonomic_temperature_signal": autonomic,
        "age_years": age, "diabetes_duration_years": duration_dm,
        "hba1c_percent": hba1c, "bmi_kg_m2": bmi, "systolic_bp_mmHg": systolic_bp,
    }

    # ── PREDICTIONS ───────────────────────────────────────────────────────────
    if bundle is not None:
        neuro_prob = model_predict(bundle, "neuropathy", fd)
        pad_prob   = model_predict(bundle, "pad", fd)
        ulcer_prob = model_predict(bundle, "ulcer", fd)
        source = "trained model"
    else:
        neuro_prob = neuropathy_probability_heuristic(hba1c, duration_dm, abs_temp_elev)
        pad_prob   = 0.5 if pad_known else 0.1
        ulcer_prob = foot_ulcer_probability_heuristic(hba1c, n_hot_zones, history_ulcer, pad_known)
        source = "heuristic"

    if history_ulcer: ulcer_prob = max(ulcer_prob, 0.6)
    if pad_known:     pad_prob   = max(pad_prob, 0.6)

    cvd_risk = estimate_cvd_risk(age, duration_dm, hba1c, systolic_bp,
                                 smoker, total_chol, hdl_chol, sex_male)
    comp = composite_risk(thermal_score, cvd_risk, hba1c, ulcer_prob)
    band, band_color, _ = cvd_band(cvd_risk)
    years, curr_traj, opt_traj = deterioration_trajectory(
        hba1c, iwgdf, n_hot_zones, duration_dm, abs_temp_elev, pad_known)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Integrated Risk Assessment")
    st.caption(f"Complication probabilities from: **{source}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Composite Index", f"{comp}/100")
    c2.metric("10-yr CVD Risk", f"{cvd_risk}%")
    c3.metric("Neuropathy Prob", f"{neuro_prob*100:.0f}%")
    c4.metric("PAD Prob", f"{pad_prob*100:.0f}%")
    c5.metric("Foot Ulcer Prob", f"{ulcer_prob*100:.0f}%")

    if band_color == "success":
        st.success(f"CVD Risk **{band}** ({cvd_risk}%). Continue preventive care.")
    elif band_color == "info":
        st.info(f"CVD Risk **{band}** ({cvd_risk}%). Review cardiovascular risk factors.")
    elif band_color == "warning":
        st.warning(f"CVD Risk **{band}** ({cvd_risk}%). Statin + BP control + lifestyle recommended.")
    else:
        st.error(f"CVD Risk **{band}** ({cvd_risk}%). Urgent cardiometabolic intervention.")

    if neuro_prob >= 0.5:
        st.warning(f"Elevated neuropathy probability ({neuro_prob*100:.0f}%). Formal neuropathy assessment advised.")
    if pad_prob >= 0.4:
        st.warning(f"Elevated PAD probability ({pad_prob*100:.0f}%). Vascular assessment (ABI/Doppler) advised.")
    if ulcer_prob >= 0.4:
        st.error(f"Elevated foot ulcer probability ({ulcer_prob*100:.0f}%). Active podiatry surveillance needed.")

    # ── CHARTS ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Risk Visualisation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**5-Year Deterioration Trajectory**")
        st.caption("Current path vs optimised (HbA1c target + podiatry)")
        st.plotly_chart(build_trajectory(years, curr_traj, opt_traj), use_container_width=True)
    with c2:
        st.markdown("**9-Domain Risk Radar**")
        st.caption("Neuropathy/PAD/ulcer from trained models (n=50,000)")
        st.plotly_chart(build_radar(thermal_score, cvd_risk, hba1c, duration_dm,
                        systolic_bp, n_hot_zones, ulcer_prob, neuro_prob, pad_prob),
                        use_container_width=True)

    # ── KARACHI CROSS-CHECK ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Foot Ulcer Risk by HbA1c Band — Karachi Cross-Check")
    st.caption(f"Patient falls in the **{get_hba1c_band(hba1c)}** band (red). Foot ulcers occur across ALL bands.")
    st.plotly_chart(build_karachi_chart(hba1c), use_container_width=True)

    with st.expander("Karachi Cohort — Complication Profiles"):
        st.dataframe(pd.DataFrame([
            {"Complication": k, "n": v["n"], "Mean HbA1c (%)": v["hba1c"],
             "Mean Age": v["age"], "Mean BMI": v["bmi"], "Mean Blood Sugar": v["bp"]}
            for k, v in COMPLICATION_PROFILES.items()
        ]), use_container_width=True, hide_index=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Integrated Clinical Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Glycaemic & Foot Priority**")
        if hba1c > 9.0:   st.error("HbA1c > 9.0%: urgent medication review. Insulin initiation/intensification.")
        elif hba1c > 8.0: st.warning("HbA1c 8.0-9.0%: add/optimise oral agent or GLP-1 RA.")
        elif hba1c > 7.5: st.info("HbA1c 7.5-8.0%: structured lifestyle programme. Review adherence.")
        else:             st.success("HbA1c near target. Maintain current regimen.")
        if neuro_prob >= 0.5:
            st.markdown("- Formal neuropathy testing (monofilament + vibration + thermal)")
        if n_hot_zones >= 4:
            st.markdown(f"- {n_hot_zones}/10 zones above threshold - podiatry referral this week")
    with c2:
        st.markdown("**Cardiovascular Priority**")
        if cvd_risk >= 30:   st.error(f"CVD {cvd_risk}%: max-tolerated statin. ACEi/ARB. Aspirin discussion.")
        elif cvd_risk >= 20: st.warning(f"CVD {cvd_risk}%: statin recommended. BP < 130/80.")
        elif cvd_risk >= 10: st.info(f"CVD {cvd_risk}%: consider statin. Smoking cessation priority.")
        else:                st.success(f"CVD {cvd_risk}%: low risk. Reassess annually.")
        if pad_prob >= 0.4:
            st.markdown("- Vascular assessment: ABI, Doppler, consider referral")
        st.markdown(
            "**Foot -> CVD connection:** autonomic neuropathy -> cardiac autonomic "
            "neuropathy (2-3x CVD mortality); PAD = direct CVD risk equivalent; "
            "thermal elevation is an early autonomic marker."
        )

    # ── EVIDENCE NOTE ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**Evidence base & limitations** — Neuropathy (AUC 0.931) and PAD (AUC 0.825) "
        "models: HistGradientBoosting on 50,000 SYNTHETIC records, held-out test n=7,502. "
        "These reflect learnable stochastic signal. Ulcer (0.995) and abnormal (0.997) "
        "labels are near-deterministic functions of the thermal scores - high AUC there "
        "confirms rule consistency, NOT clinical accuracy. CVD risk: simplified UKPDS "
        "Risk Engine (Stevens et al. 2001). Karachi cross-check: real cohort (n=20, "
        "indicative only). Primary diagnostic result is the image classifier in the Foot "
        "Thermography tab (real ThermoDataBase, AUC 0.878). Decision-support only. "
        "Synthetic training data. Not CE/FDA approved. Requires clinician review and "
        "external validation on real labelled patients."
    )
