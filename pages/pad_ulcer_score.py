"""
pad_ulcer_score.py — Diabetic Foot: PAD & Leg-Ulcer Risk from Foot Screening
=============================================================================
The core clinical tool: score PAD and foot/leg ulcer risk from a thermal foot
scan, correlate with HbA1c, and layer the classic PAD risk-factor history
(smoking, raised cholesterol, hypertension, prior CVD, CKD, age, DM duration).

CAN THERMOGRAPHY SCORE PAD?  YES — and it is the dominant signal.
  On 50,000 records (held-out test n=7,502):
    Thermal foot features only   PAD AUC = 0.81
    Risk factors only            PAD AUC = 0.66
    Thermal + risk factors       PAD AUC = 0.83   (Sens 0.74 / Spec 0.77)
  Mechanism: PAD reduces perfusion -> COOLER foot, cold-foot flag, greater
  asymmetry, distal (toe/heel) cooling. Opposite direction to neuropathy,
  which WARMS the foot (autonomic vasodilation). One scan separates the two.

Trained models: thermal_risk_models.pkl (HistGradientBoosting).
Cholesterol & prior-CVD are NOT in the training data, so they are layered on
transparently as a guideline risk-factor overlay that escalates ABI referral —
clearly separated from the model probability.

PAD gold standard remains ABI/Doppler. This is a rule-in screen: a positive
result triggers formal vascular assessment.
"""

import streamlit as st
import shared
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os


def _find_file(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [name, os.path.join(here, name), os.path.join(here, "..", name),
              os.path.join("models", name), os.path.join(here, "..", "models", name)]:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource
def load_models():
    path = _find_file("thermal_risk_models.pkl")
    if path:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def predict(bundle, which, fd):
    if bundle is None or which not in bundle["models"]:
        return None
    feats = bundle["features"]
    X = pd.DataFrame([{f: fd.get(f, np.nan) for f in feats}])[feats]
    return float(bundle["models"][which].predict_proba(X)[0, 1])


def build_gauge(prob, title, thresh):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": "#E8EDF5"}},
        title={"text": title, "font": {"color": "#A0B0C5", "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#A0B0C5"},
            "bar": {"color": "#00D4AA" if prob < thresh else "#FF4B4B"},
            "steps": [{"range": [0, thresh*100], "color": "#0D3B2E"},
                      {"range": [thresh*100, 100], "color": "#5A2A2A"}],
            "threshold": {"line": {"color": "#FFA500", "width": 3},
                          "thickness": 0.8, "value": thresh*100},
        },
    ))
    fig.update_layout(paper_bgcolor="#0D1526", font=dict(family="DM Sans", color="#A0B0C5"),
                      height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _sfoot_temp(side):
    """Default foot temperature (degC) for the input widget.
    Note: the foot tab stores 0-1 temperature *proxies*, not Celsius, so we do
    NOT convert them here — we return a sensible in-range default (29 degC) and
    clamp anything out of the 20-38 degC range to avoid widget errors."""
    return 29.0


def run():
    st.title("Diabetic Foot — PAD & Leg-Ulcer Score")
    st.markdown(
        "*Scores peripheral arterial disease and foot/leg ulcer risk from the thermal "
        "foot scan, correlated with HbA1c and PAD risk-factor history.*"
    )

    bundle = load_models()
    if bundle is not None:
        m = bundle["test_auc"]
        st.success(
            f"Trained models active — PAD AUC {m['pad']} · Ulcer AUC {m['ulcer']} "
            f"(held-out test n={bundle['n_test']}). Thermography is the dominant PAD signal."
        )
    else:
        st.error("thermal_risk_models.pkl not found — place it beside this file.")
        return

    st.markdown("---")

    # ── FOOT SCAN SIGNS ───────────────────────────────────────────────────────
    st.subheader("1 — Foot Thermal Scan")
    st.caption("From the Foot Thermography tab, or entered manually. PAD shows as a COOL foot; neuropathy as a WARM foot.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: left_temp  = st.number_input("Left foot mean temp (degC)", 20.0, 38.0, _sfoot_temp("L"), step=0.1, key="pad_ulcer_score_1")
    with c2: right_temp = st.number_input("Right foot mean temp (degC)", 20.0, 38.0, _sfoot_temp("R"), step=0.1, key="pad_ulcer_score_2")
    with c3: cool_flag  = st.checkbox("Cool foot / low-temp flag", value=(_sfoot_temp("L")+_sfoot_temp("R"))/2 < 28, help="Absolute plantar temperature low (<28 degC) — a PAD perfusion sign", key="pad_ulcer_score_3")
    with c4: autonomic  = st.number_input("Autonomic signal", 0.0, 3.0, 0.5, step=0.05, key="pad_ulcer_score_4")

    c1, c2, c3, c4 = st.columns(4)
    with c1: mtk2 = st.number_input("MTK2 zone score", 0.0, 3.0, 0.6, step=0.05, key="pad_ulcer_score_5")
    with c2: mtk4 = st.number_input("MTK4 zone score", 0.0, 3.0, 0.6, step=0.05, key="pad_ulcer_score_6")
    with c3: hallux = st.number_input("Hallux zone score", 0.0, 3.0, 0.5, step=0.05, key="pad_ulcer_score_7")
    with c4: heel = st.number_input("Heel zone score", 0.0, 3.0, 0.4, step=0.05, key="pad_ulcer_score_8")
    midfoot = st.number_input("Midfoot zone score", 0.0, 3.0, 0.4, step=0.05, key="pad_ulcer_score_9")
    asymmetry = abs(left_temp - right_temp)

    # ── HbA1c + PAD RISK FACTORS ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2 — HbA1c & PAD Risk-Factor History")
    c1, c2, c3, c4 = st.columns(4)
    with c1: hba1c       = shared.number("HbA1c (%)", "hba1c", "pad_ulcer_score_10", 8.2, 4.0, 16.0, step=0.1)
    with c2: age         = shared.number("Age", "age", "pad_ulcer_score_11", 62, 1, 120, step=1)
    with c3: duration_dm = shared.number("DM duration (yrs)", "duration_dm", "pad_ulcer_score_12", 14, 0, 60, step=1)
    with c4: bmi         = st.number_input("BMI (kg/m2)", 15.0, 50.0, 28.0, step=0.1, key="pad_ulcer_score_13")

    c1, c2, c3, c4 = st.columns(4)
    with c1: systolic_bp = shared.number("Systolic BP (mmHg)", "systolic_bp", "pad_ulcer_score_14", 135, 80, 240, step=1)
    with c2: smoking = st.selectbox("Smoking status", ["Never", "Former", "Current"], key="pad_ulcer_score_15")
    with c3: chol_raised = st.checkbox("Raised cholesterol / on statin", key="pad_ulcer_score_16")
    with c4: hypertension = st.checkbox("Hypertension", value=True, key="pad_ulcer_score_17")

    c1, c2, c3, c4 = st.columns(4)
    with c1: prior_cvd = st.checkbox("Prior CVD (MI/angina)", key="pad_ulcer_score_18")
    with c2: prior_stroke = st.checkbox("Prior stroke / TIA", key="pad_ulcer_score_19")
    with c3: ckd = st.checkbox("CKD / renal impairment", key="pad_ulcer_score_20")
    with c4: prev_ulcer = st.checkbox("Previous foot ulcer / amputation", value=bool(st.session_state.get("sh_history_ulcer", False)), key="pad_ulcer_score_21")

    # ── FEATURE DICT (matches trained model) ──────────────────────────────────
    fd = {
        "left_foot_mean_temperature_c": left_temp,
        "right_foot_mean_temperature_c": right_temp,
        "left_right_asymmetry_c": asymmetry,
        "mtk2_zone_score": mtk2, "mtk4_zone_score": mtk4,
        "hallux_zone_score": hallux, "heel_zone_score": heel, "midfoot_zone_score": midfoot,
        "autonomic_temperature_signal": autonomic,
        "absolute_low_temperature_flag": 1 if cool_flag else 0,
        "age_years": age, "diabetes_duration_years": duration_dm,
        "hba1c_percent": hba1c, "bmi_kg_m2": bmi, "systolic_bp_mmHg": systolic_bp,
        "smoker": 1 if smoking == "Current" else 0,
        "former_smoker": 1 if smoking == "Former" else 0,
    }

    pad_prob   = predict(bundle, "pad", fd)
    ulcer_prob = predict(bundle, "ulcer", fd)
    if prev_ulcer:
        ulcer_prob = max(ulcer_prob, 0.7)

    pad_thresh   = bundle["thresholds"].get("pad", 0.037)
    ulcer_thresh = bundle["thresholds"].get("ulcer", 0.05)
    st.session_state.update({"sh_pad_prob": pad_prob, "sh_ulcer_prob": ulcer_prob,
                             "sh_pad_analysed": True})

    # ── PAD RISK-FACTOR OVERLAY (guideline, transparent) ──────────────────────
    rf = []
    if smoking == "Current": rf.append("Current smoker")
    elif smoking == "Former": rf.append("Ex-smoker")
    if chol_raised:   rf.append("Dyslipidaemia")
    if hypertension:  rf.append("Hypertension")
    if prior_cvd:     rf.append("Prior CVD")
    if prior_stroke:  rf.append("Prior stroke/TIA")
    if ckd:           rf.append("CKD")
    if age >= 70:     rf.append("Age >=70")
    if duration_dm >= 10: rf.append("DM >=10 yrs")
    rf_count = len(rf)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("PAD & Ulcer Risk")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_gauge(pad_prob, "PAD probability", pad_thresh), use_container_width=True)
    with c2:
        st.plotly_chart(build_gauge(ulcer_prob, "Foot ulcer probability", ulcer_thresh), use_container_width=True)

    # PAD interpretation
    foot_mean = (left_temp + right_temp) / 2
    if pad_prob >= pad_thresh or cool_flag or foot_mean < 28.0:
        st.error(
            f"**PAD RULE-IN POSITIVE** (model {pad_prob*100:.0f}%). "
            f"{'Cool foot detected. ' if (cool_flag or foot_mean < 28.0) else ''}"
            "Refer for ABI/Doppler assessment. If ABI < 0.9, confirms PAD."
        )
    else:
        st.success(f"PAD probability {pad_prob*100:.0f}% — below rule-in threshold. Routine surveillance.")

    if ulcer_prob >= 0.4 or prev_ulcer:
        st.error(f"**High foot-ulcer risk** ({ulcer_prob*100:.0f}%). Active podiatry surveillance; offloading review.")
    elif ulcer_prob >= ulcer_thresh:
        st.warning(f"Moderate foot-ulcer risk ({ulcer_prob*100:.0f}%). Podiatry review.")

    # ── RISK FACTOR PANEL ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("PAD Risk-Factor Burden")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk factors present", f"{rf_count}")
        if rf_count >= 4:
            st.error("High risk-factor burden")
        elif rf_count >= 2:
            st.warning("Moderate risk-factor burden")
        else:
            st.success("Low risk-factor burden")
    with c2:
        st.markdown("**Present:** " + (", ".join(rf) if rf else "none"))
        st.caption(
            "Cholesterol and prior-CVD are layered here as a guideline overlay (not in the "
            "trained model). A cool foot on scan PLUS multiple risk factors strongly warrants ABI."
        )

    # ── HbA1c CORRELATION ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("HbA1c Correlation")
    band = ("on target" if hba1c <= 7.0 else "above target" if hba1c <= 8.5 else "poorly controlled")
    st.markdown(
        f"HbA1c **{hba1c:.1f}%** ({band}). Every 1% HbA1c rise is associated with a "
        "~28% increase in PAD risk (UKPDS 59). Foot risk persists across all HbA1c bands, "
        "so a normal HbA1c does not exclude PAD or ulcer risk."
    )
    if hba1c > 8.5:
        st.warning("Poor glycaemic control compounds perfusion and healing risk — prioritise optimisation.")

    # ── COMBINED REFERRAL DECISION ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Referral Decision")
    urgent = (pad_prob >= pad_thresh and rf_count >= 2) or cool_flag or foot_mean < 28.0 or prev_ulcer
    if urgent:
        st.error(
            "**Refer for vascular assessment (ABI/Doppler) and podiatry.** "
            "Thermal PAD signal and/or risk-factor burden meet rule-in criteria."
        )
    elif pad_prob >= pad_thresh or rf_count >= 3:
        st.warning("**Consider ABI** at next review; monitor foot temperature serially.")
    else:
        st.success("No immediate vascular referral indicated. Routine annual foot review.")

    st.markdown("---")
    st.caption(
        "PAD model: HistGradientBoosting, thermal + risk factors, AUC 0.83 (held-out n=7,502, "
        "SYNTHETIC data). Cholesterol/CVD overlay is guideline-based, not model-derived. "
        "ABI/Doppler remains the gold standard. Research/pilot prototype — decision-support only, "
        "not CE/FDA approved; confirm all positives clinically."
    )
