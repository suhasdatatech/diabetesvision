"""
patient_intake.py — Patient-facing self-assessment & advice
============================================================
A simple, plain-language window a patient can fill in themselves:
  - Name, age, blood pressure, last-known HbA1c, weight, ethnicity, diabetes control
  - Upload 3-4 foot photos (thermal if available; ordinary photos still stored)
  - Runs the trained foot model on thermal images
  - Gives plain-English self-advice AND a summary to share with a clinician

Not a diagnosis. Encourages professional review for anything concerning.
"""

import streamlit as st
import shared
import numpy as np
from PIL import Image
from datetime import date

# Reuse the trained foot model + image analysis from thermal_foot
try:
    import thermal_foot as tf
except Exception:
    tf = None


def _bmi(weight_kg, height_cm):
    if weight_kg and height_cm:
        m = height_cm / 100.0
        return round(weight_kg / (m * m), 1)
    return None


def run():
    st.title("My Foot Check")
    st.markdown(
        "*Fill this in yourself. We'll give you simple advice and a summary you can "
        "show your doctor or nurse. This is not a diagnosis.*"
    )
    st.markdown("---")

    # ── ABOUT YOU ─────────────────────────────────────────────────────────────
    st.subheader("1 — About you")
    c1, c2, c3 = st.columns(3)
    with c1:
        name = shared.text("Your name", "name", "pi_name")
        age  = shared.number("Age", "age", "pi_age", 55, 1, 120, step=1)
    with c2:
        weight = shared.number("Weight (kg)", "weight", "pi_weight", 80.0, 20.0, 250.0, step=0.5)
        height = shared.number("Height (cm)", "height", "pi_height", 170.0, 100.0, 220.0, step=0.5)
    with c3:
        ethnicity = st.selectbox("Ethnicity",
            ["Prefer not to say", "South Asian", "White", "Black / African / Caribbean",
             "East Asian", "Middle Eastern", "Mixed / other"], key="pi_eth")
        control = st.selectbox("How well controlled is your diabetes?",
            ["Not sure", "Well controlled", "Fairly controlled", "Poorly controlled"], key="pi_ctrl")

    c1, c2, c3 = st.columns(3)
    with c1:
        bp_sys = shared.number("Blood pressure — top number (systolic)", "systolic_bp", "pi_bps", 135, 80, 240, step=1)
    with c2:
        bp_dia = shared.number("Blood pressure — bottom number (diastolic)", "diastolic_bp", "pi_bpd", 80, 40, 140, step=1)
    with c3:
        hba1c = shared.number("Last known HbA1c (%)", "hba1c", "pi_hba1c", 7.5, 4.0, 16.0, step=0.1)

    symptoms = st.multiselect(
        "Any of these in your feet? (tick all that apply)",
        ["Burning", "Numbness", "Tingling / pins and needles", "Pain",
         "A sore or ulcer that won't heal", "Colour change", "Swelling", "None"],
        key="pi_sx")

    # ── PHOTOS ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2 — Add photos of your feet")
    st.caption("Add 3-4 clear photos (both feet, soles, and any area you're worried about). Thermal images give the most detail, but ordinary photos are fine too.")
    files = st.file_uploader("Upload foot photos (JPG or PNG)",
                             type=["jpg", "jpeg", "png", "bmp"],
                             accept_multiple_files=True, key="pi_files")

    if not st.button("Get my advice", key="pi_go"):
        st.info("Fill in the form, add your photos, then press **Get my advice**.")
        return

    # ── ANALYSE ───────────────────────────────────────────────────────────────
    bmi = _bmi(weight, height)
    probs = []
    if files:
        st.markdown("---")
        st.subheader("Your photos")
        cols = st.columns(min(len(files), 4))
        model = mode = None
        if tf is not None:
            model, mode = tf.load_thermal_model()
        for i, f in enumerate(files):
            arr = np.array(Image.open(f).convert("RGB"))
            with cols[i % len(cols)]:
                st.image(arr, caption=f.name[:16], use_container_width=True)
            if tf is not None and mode in ("sklearn", "tensorflow"):
                try:
                    p = tf.predict_dm_probability(model, mode, arr)
                    if p is not None:
                        probs.append(p)
                except Exception:
                    pass

    mean_prob = float(np.mean(probs)) if probs else None

    # ── SELF-ADVICE ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Your advice")

    urgent = False
    tips = []

    if "A sore or ulcer that won't heal" in symptoms:
        urgent = True
        st.error("You mentioned a sore or ulcer that won't heal. Please contact your doctor, "
                 "diabetes nurse, or podiatrist **as soon as possible** — foot sores in diabetes "
                 "need prompt attention.")
    if any(s in symptoms for s in ["Colour change", "Swelling"]):
        urgent = True
        st.warning("Colour change or swelling in a foot should be checked by a clinician soon.")

    if any(s in symptoms for s in ["Burning", "Numbness", "Tingling / pins and needles", "Pain"]):
        tips.append("The feelings you described (burning, numbness, tingling, or pain) can be "
                    "signs of nerve changes. Mention them at your next diabetes review.")

    if hba1c and hba1c >= 8.5:
        tips.append(f"Your last HbA1c ({hba1c:.1f}%) is above the usual target of around 7%. "
                    "Better glucose control lowers foot and heart risk over time — worth discussing "
                    "your treatment with your team.")
    elif hba1c and hba1c >= 7.5:
        tips.append(f"Your last HbA1c ({hba1c:.1f}%) is a little above target. Small improvements help.")
    elif hba1c and 0 < hba1c < 7.5:
        tips.append(f"Your last HbA1c ({hba1c:.1f}%) looks close to target — keep it up.")

    if bp_sys >= 140 or bp_dia >= 90:
        tips.append(f"Your blood pressure ({bp_sys}/{bp_dia}) is on the high side. High blood "
                    "pressure adds to circulation risk — ask about having it rechecked.")

    if bmi and bmi >= 30:
        tips.append(f"Your BMI is about {bmi}. Gradual weight loss improves diabetes control and "
                    "reduces pressure on your feet.")

    if mean_prob is not None:
        pct = mean_prob * 100
        if pct >= 61:
            tips.append(f"The photo analysis flagged a pattern worth a professional look "
                        f"(score {pct:.0f}%). This is only a screening hint, not a diagnosis.")
        else:
            tips.append(f"The photo analysis did not flag a concerning pattern (score {pct:.0f}%). "
                        "Keep up regular foot checks.")

    if not urgent:
        st.success("Nothing here needs emergency action based on what you entered. "
                   "Keep up your routine care and the everyday tips below.")

    for t in tips:
        st.markdown(f"- {t}")

    st.markdown("**Everyday foot care**")
    st.markdown("""
- Look at both feet every day (use a mirror for the soles)
- Wash and dry gently, especially between the toes
- Never walk barefoot; check inside shoes before wearing
- Keep skin moisturised, but not between the toes
- See a professional for corns, calluses, or nail problems — don't cut them yourself
- Go to your annual diabetes foot review
    """)

    # ── SHAREABLE SUMMARY ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Summary to show your clinician")
    prob_txt = f"{mean_prob*100:.0f}% (screening hint only)" if mean_prob is not None else "no thermal analysis"
    sx_txt = ", ".join(s for s in symptoms if s != "None") or "none reported"
    st.markdown(f"""
<div style="background:#fff;color:#111;border:1px solid #ccc;border-radius:8px;padding:16px;font-family:sans-serif">
  <h3 style="margin:0 0 8px">Patient foot self-check</h3>
  <table style="border-collapse:collapse;font-size:14px">
    <tr><td style="padding:3px 10px;color:#555">Name</td><td style="padding:3px 10px;font-weight:600">{name or '—'}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Date</td><td style="padding:3px 10px">{date.today().isoformat()}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Age</td><td style="padding:3px 10px">{age}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">BP</td><td style="padding:3px 10px">{bp_sys}/{bp_dia} mmHg</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Last HbA1c</td><td style="padding:3px 10px">{(str(hba1c)+'%') if hba1c else 'unknown'}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Weight / BMI</td><td style="padding:3px 10px">{weight} kg / {bmi if bmi else '—'}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Ethnicity</td><td style="padding:3px 10px">{ethnicity}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Diabetes control (self-report)</td><td style="padding:3px 10px">{control}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Foot symptoms</td><td style="padding:3px 10px">{sx_txt}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Photos analysed</td><td style="padding:3px 10px">{len(files) if files else 0}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Thermal screening hint</td><td style="padding:3px 10px">{prob_txt}</td></tr>
    <tr><td style="padding:3px 10px;color:#555">Flagged urgent</td><td style="padding:3px 10px;font-weight:600">{'YES' if urgent else 'No'}</td></tr>
  </table>
  <p style="font-size:11px;color:#888;margin-top:10px">Patient self-assessment. Not a diagnosis. For clinician review.</p>
</div>
""", unsafe_allow_html=True)
    st.caption("Tip: press Ctrl/Cmd+P to save this as a PDF, or screenshot it to send to your clinic.")

    # Publish to shared state so clinician tabs can reuse
    st.session_state.update({
        "sh_patient_id": name or "Patient", "sh_age": age, "sh_hba1c": hba1c or None,
        "sh_from_patient": True,
    })

    if urgent:
        st.error("Because something you entered may need prompt attention, please don't wait for "
                 "your next routine appointment — contact your care team.")
