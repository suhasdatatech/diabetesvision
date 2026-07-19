"""
thermal_foot.py — Plantar Thermography Foot Health Module (v3)
==============================================================
Evidence base:
  - ThermoDataBase: 1,866 labelled plantar thermal images (Control vs DM Group)
  - 91 paired bilateral Excel records (temperature time-series)
  - Key finding: DM patients show metatarsal temps ~3 degC HIGHER than controls
    (early autonomic neuropathy -> vasodilation -> increased plantar skin blood flow)
  - TRAINED MODEL (thermal_img_model.pkl): Logistic model on 16 HSV zone
    features from 333 original thermal images (167 patients).
    Patient-grouped 5-fold CV AUC = 0.878  (Sens 0.70 / Spec 0.92).
    A >0.26 AUC improvement over bilateral deltaT alone (0.617).

Analysis modes (auto-selected):
  1. TRAINED MODEL MODE (if thermal_img_model.pkl present) — DEFAULT
     Lightweight scikit-learn model (2 KB, no TensorFlow). AUC 0.878.
  2. TensorFlow CNN MODE (if models/thermal_foot_model.h5 present)
     Optional heavier MobileNetV2 from train_thermal_cnn.py.
  3. ALGORITHMIC MODE (fallback)
     Zone temperature proxies scored against empirical thresholds.
"""

import streamlit as st
import shared
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import os

# ── EMPIRICALLY-DERIVED CONSTANTS ─────────────────────────────────────────────

ZONE_THRESHOLDS = {
    "Hallux":            0.39,
    "2nd-3rd Toe":       0.50,
    "4th-5th Toe":       0.50,
    "1st Met Head":      0.84,
    "2nd Met Head":      0.84,   # strongest single-zone bilateral predictor
    "3rd Met Head":      0.28,
    "4th-5th Met Head":  0.92,
    "Midfoot (Medial)":  0.84,
    "Midfoot (Lateral)": 1.40,
    "Heel":              0.84,
}

ABSOLUTE_TEMP_ELEVATED = 0.62   # proxy value ~ 28 degC (elevated DM range)

ZONE_LABELS = list(ZONE_THRESHOLDS.keys())

ZONE_COORDS = [
    (0.00, 0.12, 0.35, 0.65),
    (0.00, 0.12, 0.15, 0.35),
    (0.00, 0.12, 0.65, 0.85),
    (0.12, 0.28, 0.30, 0.50),
    (0.12, 0.28, 0.45, 0.60),
    (0.12, 0.28, 0.55, 0.68),
    (0.12, 0.28, 0.60, 0.80),
    (0.28, 0.65, 0.20, 0.50),
    (0.28, 0.65, 0.50, 0.80),
    (0.65, 1.00, 0.25, 0.75),
]

IWGDF_CATEGORIES = {
    0: {"label": "IWGDF Category 0 - No Risk",      "color": "success",
        "description": "No neuropathy, no PAD, no previous ulcer.",
        "review": "Annual", "action": "Standard care. Annual thermography baseline."},
    1: {"label": "IWGDF Category 1 - Low Risk",     "color": "info",
        "description": "Neuropathy OR PAD, without deformity or previous lesion.",
        "review": "6-12 months", "action": "Podiatry review. Therapeutic footwear assessment."},
    2: {"label": "IWGDF Category 2 - Moderate Risk","color": "warning",
        "description": "Neuropathy + PAD or foot deformity. Elevated ulceration risk.",
        "review": "3-6 months", "action": "Podiatry referral within 4 weeks. Vascular Doppler."},
    3: {"label": "IWGDF Category 3 - High Risk",    "color": "error",
        "description": "Previous ulcer or amputation. Highest risk.",
        "review": "1-3 months", "action": "Urgent MDT referral. Offloading device."},
}


# ── MODEL LOADER ──────────────────────────────────────────────────────────────

IMG_MODEL_FEATURES = [
    "zone0","zone1","zone2","zone3","zone4","zone5","zone6","zone7","zone8","zone9",
    "whole_mean","whole_max","whole_std","forefoot","heel","toe_heel_grad",
]

def _find_file(name):
    """Search common locations: cwd, this file's dir, parent, models/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [name, os.path.join(here, name), os.path.join(here, "..", name),
              os.path.join("models", name), os.path.join(here, "..", "models", name)]:
        if os.path.exists(p):
            return p
    return None


@st.cache_resource
def load_thermal_model():
    """Returns (obj, mode): mode in {'sklearn','tensorflow','none'}."""
    pkl = _find_file("thermal_img_model.pkl")
    if pkl:
        import pickle
        with open(pkl, "rb") as f:
            return pickle.load(f), "sklearn"
    h5 = _find_file("thermal_foot_model.h5")
    if h5:
        import tensorflow as tf
        return tf.keras.models.load_model(h5), "tensorflow"
    return None, "none"


# ── IMAGE PROCESSING ──────────────────────────────────────────────────────────

def extract_temp_proxy(img_array):
    """JET/Rainbow thermogram -> [0,1] temp proxy (high=hot). ~0.62 ≈ 28 degC."""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1] / 255.0
    raw = np.clip(1.0 - hue / 120.0, 0, 1)
    return raw * np.clip(sat * 2.0, 0, 1)


def _zone_vals(tp):
    h, w = tp.shape
    z = []
    for ys, ye, xs, xe in ZONE_COORDS:
        r0, r1 = int(ys*h), max(int(ye*h), int(ys*h)+1)
        c0, c1 = int(xs*w), max(int(xe*w), int(xs*w)+1)
        patch = tp[r0:r1, c0:c1]
        z.append(float(patch.mean()) if patch.size else 0.0)
    return z


def extract_model_features(img_array):
    """16 HSV zone features matching training order."""
    tp = extract_temp_proxy(img_array)
    z = _zone_vals(tp)
    feats = {
        **{f"zone{i}": z[i] for i in range(10)},
        "whole_mean": float(tp.mean()),
        "whole_max":  float(tp.max()),
        "whole_std":  float(tp.std()),
        "forefoot":   float(np.mean(z[3:7])),
        "heel":       z[9],
        "toe_heel_grad": float(np.mean(z[:3]) - z[9]),
    }
    return np.array([[feats[k] for k in IMG_MODEL_FEATURES]])


def predict_dm_probability(bundle, mode, img_array):
    if mode == "sklearn":
        return float(bundle["model"].predict_proba(extract_model_features(img_array))[0, 1])
    if mode == "tensorflow":
        img = cv2.resize(img_array, (128, 128)).astype(np.float32) / 255.0
        return float(bundle.predict(np.expand_dims(img, 0), verbose=0)[0][0])
    return None


def zone_mean(temp_half, zone_idx):
    h, w = temp_half.shape
    ys, ye, xs, xe = ZONE_COORDS[zone_idx]
    r0, r1 = int(ys*h), max(int(ye*h), int(ys*h)+1)
    c0, c1 = int(xs*w), max(int(xe*w), int(xs*w)+1)
    patch = temp_half[r0:r1, c0:c1]
    return float(patch.mean()) if patch.size > 0 else 0.0


def analyse_zones(img_array):
    tp = extract_temp_proxy(img_array)
    mid = tp.shape[1] // 2
    left_map, right_map = tp[:, :mid], tp[:, mid:]
    lt, rt, dl = [], [], []
    for i in range(len(ZONE_LABELS)):
        a = zone_mean(left_map, i)
        b = zone_mean(right_map, i)
        lt.append(a); rt.append(b); dl.append(abs(a - b))
    return lt, rt, dl


def compute_foot_score(deltas, abs_temp_proxy, history_ulcer, neuropathy, pad, dm_prob=None):
    """Composite score (0-100) + IWGDF category. Uses model prob if available."""
    thresholds = list(ZONE_THRESHOLDS.values())
    n_hot = sum(1 for d, t in zip(deltas, thresholds) if d >= t)
    score = min(int((n_hot / len(ZONE_LABELS)) * 45), 45)

    # Trained-model contribution (image-level DM likelihood)
    if dm_prob is not None:
        score += int(dm_prob * 20)

    if abs_temp_proxy > ABSOLUTE_TEMP_ELEVATED: score += 8
    if neuropathy:    score += 8
    if pad:           score += 18
    if history_ulcer: score += 18
    score = min(score, 100)

    if history_ulcer or score >= 85:
        iwgdf = 3
    elif (neuropathy and pad) or score >= 60:
        iwgdf = 2
    elif neuropathy or pad or score >= 30:
        iwgdf = 1
    else:
        iwgdf = 0
    return score, iwgdf


def overlay_risk_map(img_array, deltas):
    h, w = img_array.shape[:2]
    mid = w // 2
    risk = np.zeros((h, w), dtype=np.float32)
    thresholds = list(ZONE_THRESHOLDS.values())
    for i in range(len(ZONE_LABELS)):
        ys, ye, xs, xe = ZONE_COORDS[i]
        weight = min(deltas[i] / (thresholds[i] + 1e-6), 1.0)
        r0, r1 = int(ys*h), max(int(ye*h), int(ys*h)+1)
        c0l, c1l = int(xs*mid), max(int(xe*mid), int(xs*mid)+1)
        risk[r0:r1, c0l:c1l] = np.maximum(risk[r0:r1, c0l:c1l], weight)
        c0r = mid + int(xs*mid); c1r = mid + max(int(xe*mid), int(xs*mid)+1)
        risk[r0:r1, c0r:c1r] = np.maximum(risk[r0:r1, c0r:c1r], weight)
    hm = cv2.applyColorMap(np.uint8(255 * risk), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return (img_array * 0.55 + hm * 0.45).astype(np.uint8)


def build_zone_chart(deltas):
    thresholds = list(ZONE_THRESHOLDS.values())
    colours = ["#FF4B4B" if d >= t else "#FFA500" if d >= t*0.6 else "#00D4AA"
               for d, t in zip(deltas, thresholds)]
    fig = go.Figure(go.Bar(x=deltas, y=ZONE_LABELS, orientation="h",
        marker_color=colours, text=[f"{d:.3f}" for d in deltas], textposition="outside"))
    for i, thresh in enumerate(thresholds):
        fig.add_shape(type="line", x0=thresh, x1=thresh, y0=i-0.4, y1=i+0.4,
                      line=dict(color="#FF4B4B", width=2, dash="dot"))
    fig.update_layout(paper_bgcolor="#0D1526", plot_bgcolor="#0D1526",
        font=dict(family="DM Sans", color="#A0B0C5"),
        xaxis=dict(title="deltaT (proxy units)", gridcolor="#1E2D45",
                   range=[0, max(max(deltas)*1.3, 1.2)]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=60, t=20, b=20), showlegend=False, height=420)
    return fig


def build_zone_temp_chart(left_temps, right_temps):
    """Grouped bar chart of left vs right temperature proxy per zone."""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Left foot", x=ZONE_LABELS, y=left_temps, marker_color="#00D4AA"))
    fig.add_trace(go.Bar(name="Right foot", x=ZONE_LABELS, y=right_temps, marker_color="#0099FF"))
    fig.update_layout(barmode="group", paper_bgcolor="#0D1526", plot_bgcolor="#0D1526",
        font=dict(family="DM Sans", color="#A0B0C5"),
        xaxis=dict(gridcolor="#1E2D45", tickangle=-35),
        yaxis=dict(title="Temperature proxy (0-1)", gridcolor="#1E2D45"),
        legend=dict(bgcolor="#1E2D45"), margin=dict(l=20, r=20, t=20, b=80), height=380)
    return fig


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────

def run():
    st.title("Plantar Thermography - Foot Health Assessment")
    st.markdown(
        "*Zone-level thermal analysis with a trained classifier. "
        "Evidence: ThermoDataBase (1,866 images) + bilateral temperature records.*"
    )

    model, model_mode = load_thermal_model()
    if model_mode == "sklearn":
        m = model
        st.success(
            f"Trained model active. Validation AUC {m['auc']} · "
            f"Sensitivity {m['sensitivity']} · Specificity {m['specificity']} "
            f"(patient-grouped 5-fold CV, n=333 images)."
        )
    elif model_mode == "tensorflow":
        st.success("TensorFlow CNN model active (models/thermal_foot_model.h5).")
    else:
        st.info(
            "Algorithmic mode: empirical zone thresholds. Place thermal_img_model.pkl "
            "beside this file to enable the trained model (AUC 0.878)."
        )

    st.markdown("---")

    # ── PATIENT INFO ──────────────────────────────────────────────────────────
    st.subheader("Patient Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1: patient_id  = shared.text("Patient ID", "name", "thermal_foot_1", placeholder="PT-2025-001")
    with c2: patient_age = shared.number("Age", "age", "thermal_foot_2", 55, 1, 120, step=1)
    with c3: duration_dm = shared.number("DM Duration (years)", "duration_dm", "thermal_foot_3", 10, 0, 60, step=1)
    with c4: hba1c       = shared.number("HbA1c (%)", "hba1c", "thermal_foot_4", 8.0, 4.0, 16.0, step=0.1)

    c1, c2, c3 = st.columns(3)
    with c1: neuropathy    = st.checkbox("Known Peripheral Neuropathy", key="thermal_foot_5")
    with c2: pad           = st.checkbox("Known PAD / Poor Circulation", key="thermal_foot_6")
    with c3: history_ulcer = st.checkbox("Previous Foot Ulcer or Amputation", key="thermal_foot_7")

    clinical_notes = st.text_area("Clinical Notes",
        placeholder="Medications, footwear, relevant history...", key="thermal_foot_8")

    # ── IMAGE UPLOAD ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Upload Plantar Thermal Image")
    st.info(
        "Bilateral plantar thermogram: LEFT foot on left half, RIGHT foot on right half. "
        "JET/Rainbow colormap. Allow 10 min acclimatisation at 22-24 degC before capture."
    )
    uploaded = st.file_uploader(
        "Upload thermogram(s) — you can add 2-3 views (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="thermal_foot_9",
    )

    if uploaded:
        # Process every uploaded image
        per_image = []
        for uf in uploaded:
            arr = np.array(Image.open(uf).convert("RGB"))
            lt, rt, dl = analyse_zones(arr)
            atp = float(extract_temp_proxy(arr).mean())
            p = predict_dm_probability(model, model_mode, arr)
            per_image.append({"name": uf.name, "arr": arr, "lt": lt, "rt": rt,
                              "deltas": dl, "abs_tp": atp, "dm_prob": p})

        n_imgs = len(per_image)
        st.caption(f"{n_imgs} image(s) uploaded — results aggregated across all views.")

        # Show each image with its own risk overlay
        cols = st.columns(min(n_imgs, 3))
        for i, rec in enumerate(per_image):
            with cols[i % len(cols)]:
                ov = overlay_risk_map(rec["arr"], rec["deltas"])
                cap = rec["name"][:18]
                if rec["dm_prob"] is not None:
                    cap += f" — DM {rec['dm_prob']*100:.0f}%"
                st.image(ov, caption=cap, use_container_width=True)

        # Aggregate across views: mean per-zone delta, mean abs temp,
        # and mean DM probability (most robust vs a single noisy shot)
        deltas = [float(np.mean([r["deltas"][z] for r in per_image]))
                  for z in range(len(ZONE_LABELS))]
        left_temps  = [float(np.mean([r["lt"][z] for r in per_image])) for z in range(len(ZONE_LABELS))]
        right_temps = [float(np.mean([r["rt"][z] for r in per_image])) for z in range(len(ZONE_LABELS))]
        abs_temp_proxy = float(np.mean([r["abs_tp"] for r in per_image]))
        probs = [r["dm_prob"] for r in per_image if r["dm_prob"] is not None]
        dm_prob = float(np.mean(probs)) if probs else None

        thermal_score, iwgdf = compute_foot_score(
            deltas, abs_temp_proxy, history_ulcer, neuropathy, pad, dm_prob)
        n_hot = sum(1 for d, t in zip(deltas, ZONE_THRESHOLDS.values()) if d >= t)

        # Publish to shared state so other tabs auto-populate
        st.session_state.update({
            "sh_patient_id": patient_id, "sh_age": patient_age,
            "sh_duration_dm": duration_dm, "sh_hba1c": hba1c,
            "sh_neuropathy": neuropathy, "sh_pad": pad, "sh_history_ulcer": history_ulcer,
            "sh_thermal_score": thermal_score, "sh_iwgdf": iwgdf, "sh_n_hot": n_hot,
            "sh_dm_prob": dm_prob,
            "sh_abs_elev": bool(abs_temp_proxy > ABSOLUTE_TEMP_ELEVATED),
            "sh_left_temps": left_temps, "sh_right_temps": right_temps,
            "sh_asymmetry": float(abs(np.mean(left_temps) - np.mean(right_temps))),
            "sh_foot_analysed": True,
        })

        # Per-image probability table when more than one view
        if n_imgs > 1 and probs:
            st.markdown("**Per-image DM probability** (aggregated result uses the mean):")
            st.dataframe(pd.DataFrame({
                "Image": [r["name"] for r in per_image],
                "DM probability": [f"{r['dm_prob']*100:.1f}%" if r['dm_prob'] is not None else "n/a"
                                   for r in per_image],
            }), use_container_width=True, hide_index=True)

        # ── METRICS ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Assessment Results")

        if dm_prob is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            c5.metric("Model DM Probability", f"{dm_prob*100:.1f}%",
                      delta="Diabetic pattern" if dm_prob > 0.611 else "Control pattern",
                      delta_color="inverse" if dm_prob > 0.611 else "normal",
                      help="Trained model. Decision threshold 0.611 (Youden-optimal; Sens 0.70/Spec 0.92).")
        else:
            c1, c2, c3, c4 = st.columns(4)

        c1.metric("Foot Health Score", f"{thermal_score}/100")
        c2.metric("IWGDF Category",    str(iwgdf))
        c3.metric("Zones Above Threshold", f"{n_hot}/{len(ZONE_LABELS)}")
        c4.metric("Absolute Temp Signal",
                  "Elevated" if abs_temp_proxy > ABSOLUTE_TEMP_ELEVATED else "Normal",
                  delta="~+3 degC (autonomic)" if abs_temp_proxy > ABSOLUTE_TEMP_ELEVATED else "Within range",
                  delta_color="inverse" if abs_temp_proxy > ABSOLUTE_TEMP_ELEVATED else "normal")

        info = IWGDF_CATEGORIES[iwgdf]
        {"success": st.success, "info": st.info,
         "warning": st.warning, "error": st.error}[info["color"]](
            f"**{info['label']}** — {info['description']}")
        st.markdown(f"**Review frequency:** {info['review']}  |  **Action:** {info['action']}")

        # ── ZONE CHART ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Zone-by-Zone Bilateral Temperature Asymmetry")
        st.caption(
            "Dashed markers = empirical zone thresholds. The trained classifier uses all "
            "16 zone/whole-foot features — validation AUC 0.878 vs 0.617 for asymmetry alone."
        )
        st.plotly_chart(build_zone_chart(deltas), use_container_width=True)

        st.markdown("**Zone temperatures — left vs right foot**")
        st.caption("Absolute zone temperature proxy per foot. Large left/right gaps flag asymmetry; overall warmth suggests autonomic (neuropathy) change, coolness suggests PAD.")
        st.plotly_chart(build_zone_temp_chart(left_temps, right_temps), use_container_width=True)

        with st.expander("Full Zone Data Table"):
            thresholds = list(ZONE_THRESHOLDS.values())
            st.dataframe(pd.DataFrame({
                "Zone": ZONE_LABELS,
                "Left (proxy)":  [f"{t:.3f}" for t in left_temps],
                "Right (proxy)": [f"{t:.3f}" for t in right_temps],
                "DeltaT":        [f"{d:.3f}" for d in deltas],
                "Threshold":     [f"{t:.3f}" for t in thresholds],
                "Status": ["HIGH RISK" if d >= t else "MONITOR" if d >= t*0.6 else "NORMAL"
                           for d, t in zip(deltas, thresholds)],
            }), use_container_width=True, hide_index=True)

        # ── TEMPERATURE INTERPRETATION ────────────────────────────────────────
        st.markdown("---")
        st.subheader("Temperature Pattern Interpretation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model & Asymmetry**")
            st.markdown("""
- Trained on 333 original thermal images (167 patients)
- Patient-grouped 5-fold CV AUC: **0.878**
- Strongest features: lateral/MTK1/MTK3 absolute temps, whole-foot max temp
- 2nd Met Head is the best single bilateral predictor
            """)
        with col2:
            st.markdown("**Absolute Temperature Elevation**")
            st.markdown("""
DM patients showed metatarsal temps **~3 degC higher** than controls:
- Control mean MTK2: 25.4 degC
- DM mean MTK2: **28.6 degC**

Mechanism: **early autonomic neuropathy** -> vasodilation -> raised plantar blood flow.
Precedes the cool, ischaemic foot of late-stage PAD.
            """)
        if abs_temp_proxy > ABSOLUTE_TEMP_ELEVATED:
            st.warning(
                "Elevated absolute plantar temperature detected: early autonomic neuropathy pattern. "
                "Consider autonomic function testing (HRV, sudomotor)."
            )

        # ── MONOFILAMENT COMPARISON ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("Thermal vs 10g Monofilament")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**10g Monofilament limitations:**
- Detects neuropathy at late (clinical) stage only
- High inter-rater variability (~30% missed cases)
- Operator-dependent
- No vascular information
- Cannot track longitudinal change
            """)
        with col2:
            st.markdown("""
**Plantar Thermography (this tool):**
- Detects pre-clinical thermal asymmetry
- Fully automated, reproducible
- 10-zone bilateral coverage
- Captures absolute temperature elevation (autonomic signal)
- Trained classifier AUC 0.878 (Sens 0.70 / Spec 0.92)
            """)

        # ── RECOMMENDATIONS ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Clinical Recommendations")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Immediate Actions**")
            if iwgdf == 0:
                st.markdown("- Annual foot review with thermography baseline\n"
                            "- Self-inspection education\n"
                            "- Maintain HbA1c < 7.0%, BP < 130/80")
            elif iwgdf == 1:
                st.markdown("- Podiatry review every 6 months\n"
                            "- Therapeutic footwear assessment\n"
                            "- Intensify glycaemic control (HbA1c < 7.5%)\n"
                            "- Repeat thermography in 3 months")
            elif iwgdf == 2:
                st.markdown("- **Podiatry referral within 4 weeks**\n"
                            "- Vascular assessment (Doppler ABI)\n"
                            "- Bespoke moulded footwear\n"
                            "- HbA1c optimisation and medication review")
            else:
                st.markdown("- **Urgent MDT referral (within 1 week)**\n"
                            "- Vascular surgery / PAD assessment\n"
                            "- Offloading device (total contact cast)\n"
                            "- Daily wound surveillance")
        with col2:
            st.markdown("**Patient Education**")
            st.markdown("- Inspect both feet daily (mirror for sole)\n"
                        "- Report new blisters, redness or numbness immediately\n"
                        "- Never walk barefoot\n"
                        "- Avoid heat sources near feet\n"
                        "- Correct footwear: deep toe box, cushioned sole")
            if hba1c > 8.5:
                st.warning(f"HbA1c {hba1c:.1f}%: urgent medication review.")
            if duration_dm > 15:
                st.info(f"DM duration {duration_dm} years: cumulative autonomic neuropathy risk elevated.")

        if clinical_notes:
            st.markdown("---")
            st.info(f"**Clinical notes:** {clinical_notes}")
