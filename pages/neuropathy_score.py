"""
neuropathy_score.py — Diabetic Peripheral Neuropathy Score
===========================================================
Combines three streams into one neuropathy likelihood:
  1. TRAINED MODEL (thermal + clinical), PAD-distinct warm-foot signal — AUC 0.931
     (held-out test n=7,502, synthetic data)
  2. SYMPTOM questionnaire — MNSI-style patient-reported items
     (burning feet, numbness, tingling, night pain, sensitivity...) — validated,
     cut-point >=7 positive (Michigan Neuropathy Screening Instrument)
  3. Bedside exam options (monofilament sites felt, vibration, ankle reflexes)

Thermal signal: neuropathy WARMS the foot (autonomic vasodilation) — opposite to
the cool PAD foot. So the same scan discriminates neuropathy from ischaemia.

MNSI reference: Feldman et al., Diabetes Care 1994; Herman et al. 2012.
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


# MNSI-style symptom questionnaire, reworded so YES = abnormal for every item
# (13 scored items; 2 items excluded from score per MNSI convention).
# direction: 1 = scored (yes adds 1); 0 = not scored toward neuropathy.
MNSI_ITEMS = [
    ("Are your legs and/or feet numb?", 1),
    ("Do you have burning pain in your legs and/or feet?", 1),
    ("Are your feet too sensitive to touch?", 1),
    ("Do you get muscle cramps in your legs and/or feet?", 1),
    ("Do you have prickling/tingling feelings in your legs or feet?", 1),
    ("Does it hurt when the bed covers touch your skin?", 1),
    ("In the bath, are you UNABLE to tell hot water from cold?", 1),
    ("Have you ever had an open sore on your foot?", 1),
    ("Has a doctor told you that you have diabetic neuropathy?", 1),
    ("Do you feel weak all over most of the time?", 0),      # excluded (asthenia, non-specific)
    ("Are your symptoms worse at night?", 1),
    ("Do your legs hurt when you walk?", 0),                  # excluded (claudication, vascular)
    ("When walking, do your feet feel numb / hard to sense?", 1),
    ("Is the skin on your feet so dry it cracks open?", 1),
    ("Have you ever had an amputation?", 1),
]


def mnsi_score(answers):
    """answers: list of bool (yes=True). Returns 0-13 score (yes=abnormal)."""
    return sum(1 for (q, d), a in zip(MNSI_ITEMS, answers) if d == 1 and a)


def build_combined_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": "#E8EDF5"}},
        title={"text": "Combined neuropathy likelihood", "font": {"color": "#A0B0C5", "size": 14}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#A0B0C5"},
               "bar": {"color": "#00D4AA" if prob < 0.5 else "#FFA500" if prob < 0.7 else "#FF4B4B"},
               "steps": [{"range": [0, 40], "color": "#0D3B2E"},
                         {"range": [40, 70], "color": "#5A4A1E"},
                         {"range": [70, 100], "color": "#5A2A2A"}]},
    ))
    fig.update_layout(paper_bgcolor="#0D1526", font=dict(family="DM Sans", color="#A0B0C5"),
                      height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def run():
    st.title("Diabetic Neuropathy Score")
    st.markdown(
        "*Combines the thermal+clinical model with a validated symptom questionnaire "
        "(burning feet, numbness, tingling...) and optional bedside exam.*"
    )

    bundle = load_models()
    if bundle is not None:
        st.success(
            f"Trained neuropathy model active — AUC {bundle['test_auc']['neuropathy']} "
            f"(held-out test n={bundle['n_test']}). Thermal signal = warm autonomic foot."
        )
    else:
        st.error("thermal_risk_models.pkl not found — place it beside this file.")
        return
    st.markdown("---")

    # ── THERMAL + CLINICAL ────────────────────────────────────────────────────
    st.subheader("1 — Foot Scan & Clinical")
    c1, c2, c3, c4 = st.columns(4)
    with c1: left_temp  = st.number_input("Left foot temp (degC)", 20.0, 38.0, 30.0, step=0.1, key="neuropathy_score_1")
    with c2: right_temp = st.number_input("Right foot temp (degC)", 20.0, 38.0, 30.0, step=0.1, key="neuropathy_score_2")
    with c3: autonomic  = st.number_input("Autonomic signal", 0.0, 3.0, 1.0, step=0.05, key="neuropathy_score_3")
    with c4: hba1c      = shared.number("HbA1c (%)", "hba1c", "neuropathy_score_4", 8.5, 4.0, 16.0, step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    with c1: age = shared.number("Age", "age", "neuropathy_score_5", 60, 1, 120, step=1)
    with c2: duration_dm = shared.number("DM duration (yrs)", "duration_dm", "neuropathy_score_6", 15, 0, 60, step=1)
    with c3: mtk2 = st.number_input("MTK2 zone score", 0.0, 3.0, 0.9, step=0.05, key="neuropathy_score_7")
    with c4: mtk4 = st.number_input("MTK4 zone score", 0.0, 3.0, 0.9, step=0.05, key="neuropathy_score_8")

    c1, c2, c3, c4 = st.columns(4)
    with c1: hallux = st.number_input("Hallux zone", 0.0, 3.0, 0.7, step=0.05, key="neuropathy_score_9")
    with c2: heel = st.number_input("Heel zone", 0.0, 3.0, 0.5, step=0.05, key="neuropathy_score_10")
    with c3: midfoot = st.number_input("Midfoot zone", 0.0, 3.0, 0.5, step=0.05, key="neuropathy_score_11")
    with c4: bmi = st.number_input("BMI", 15.0, 50.0, 28.0, step=0.1, key="neuropathy_score_12")

    systolic_bp = shared.number("Systolic BP (mmHg)", "systolic_bp", "neuropathy_score_13", 135, 80, 240, step=1)
    asymmetry = abs(left_temp - right_temp)

    fd = {
        "left_foot_mean_temperature_c": left_temp, "right_foot_mean_temperature_c": right_temp,
        "left_right_asymmetry_c": asymmetry, "mtk2_zone_score": mtk2, "mtk4_zone_score": mtk4,
        "hallux_zone_score": hallux, "heel_zone_score": heel, "midfoot_zone_score": midfoot,
        "autonomic_temperature_signal": autonomic, "absolute_low_temperature_flag": 0,
        "age_years": age, "diabetes_duration_years": duration_dm, "hba1c_percent": hba1c,
        "bmi_kg_m2": bmi, "systolic_bp_mmHg": systolic_bp, "smoker": 0, "former_smoker": 0,
    }
    model_prob = predict(bundle, "neuropathy", fd)

    # ── SYMPTOM QUESTIONNAIRE ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2 — Symptom Questionnaire (MNSI)")
    st.caption("Patient-reported. 'Burning feet' and night-worse pain are hallmarks of painful/small-fibre neuropathy.")
    answers = []
    cols = st.columns(2)
    for i, (q, _) in enumerate(MNSI_ITEMS):
        with cols[i % 2]:
            answers.append(st.checkbox(q, key=f"mnsi_{i}"))
    sx = mnsi_score(answers)
    sx_positive = sx >= 7
    burning = answers[1]   # burning pain item

    # ── OPTIONAL EXAM ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3 — Bedside Exam (optional)")
    c1, c2, c3 = st.columns(3)
    with c1: mono = st.slider("Monofilament sites felt (/10)", 0, 10, 10, key="neuropathy_score_14")
    with c2: vibration = st.checkbox("Reduced/absent vibration (128Hz)", key="neuropathy_score_15")
    with c3: reflexes = st.checkbox("Absent ankle reflexes", key="neuropathy_score_16")

    # ── COMBINE ───────────────────────────────────────────────────────────────
    # Weighted blend: model 0.5, symptoms 0.3, exam 0.2
    sx_norm = min(sx / 10.0, 1.0)
    exam_score = 0.0
    exam_items = 0
    if mono < 8: exam_score += (8 - mono) / 8; 
    exam_items += 1
    if vibration: exam_score += 1
    if reflexes: exam_score += 1
    exam_norm = min(exam_score / 3.0, 1.0)

    combined = 0.5 * model_prob + 0.3 * sx_norm + 0.2 * exam_norm
    combined = min(combined, 1.0)
    st.session_state.update({"sh_neuro_model": model_prob, "sh_neuro_combined": combined,
                             "sh_mnsi": sx, "sh_neuro_analysed": True})

    st.markdown("---")
    st.subheader("Neuropathy Likelihood")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(build_combined_gauge(combined), use_container_width=True)
    with c2:
        st.metric("Model probability (thermal+clinical)", f"{model_prob*100:.0f}%")
        st.metric("MNSI symptom score", f"{sx} / 13", delta="Positive (>=7)" if sx_positive else "Below cut-point",
                  delta_color="inverse" if sx_positive else "normal")
        st.metric("Monofilament", f"{mono}/10 felt")

    if combined >= 0.7:
        st.error("**High neuropathy likelihood.** Formal assessment; foot-protection education; review painful-neuropathy treatment if symptomatic.")
    elif combined >= 0.4:
        st.warning("**Moderate neuropathy likelihood.** Monofilament + vibration testing; 6-monthly foot review.")
    else:
        st.success("**Low neuropathy likelihood.** Routine annual foot screening.")

    if burning:
        st.info(
            "Burning feet reported — a hallmark of painful/small-fibre diabetic neuropathy. "
            "Consider small-fibre-specific assessment and neuropathic-pain management "
            "(e.g. duloxetine, amitriptyline, pregabalin/gabapentin per local guidance)."
        )
    if model_prob >= 0.5 and sx < 7:
        st.info("Objective thermal/clinical signal without many symptoms — suggests subclinical neuropathy. Early foot protection matters most here.")

    # ── HbA1c LINK ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"**HbA1c {hba1c:.1f}%.** Neuropathy risk rises steeply with cumulative glycaemic "
        "exposure and DM duration; tight control slows progression (DCCT/EDIC). "
        "Symptoms can persist or worsen transiently during rapid glucose lowering."
    )

    st.markdown("---")
    st.caption(
        "Model: HistGradientBoosting (thermal+clinical), AUC 0.931, held-out n=7,502, SYNTHETIC data. "
        "MNSI: Feldman et al. 1994. Combined index is a transparent weighted blend "
        "(model 0.5 / symptoms 0.3 / exam 0.2), not an independently validated score. "
        "Research/pilot prototype — decision-support only."
    )
