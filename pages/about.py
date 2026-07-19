"""
about.py — About & Research (global, self-contained styling)
Detailed model table is appended by model_card.py under the same tab.
"""
import streamlit as st


def run():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
      .ab-badge { display:inline-block; background:#00D4AA22; color:#00D4AA;
        border:1px solid #00D4AA44; border-radius:20px; padding:4px 16px; font-size:0.75rem;
        font-weight:600; letter-spacing:0.09em; text-transform:uppercase;
        font-family:'DM Sans',sans-serif; }
      .ab-title { font-family:'DM Serif Display',serif; font-size:2.6rem; color:#E8EDF5; margin:0.8rem 0 1.4rem; }
      .ab-h { font-family:'DM Serif Display',serif; font-size:1.6rem; color:#E8EDF5; margin:0 0 0.8rem; }
      .ab-card { background:#0D1B2E; border:1px solid #1E3A5F; border-radius:12px; padding:1.4rem;
        font-family:'DM Sans',sans-serif; }
      .ab-card p, .ab-card li { color:#B0BEC5; line-height:1.75; font-size:0.92rem; }
      .ab-divider { border-top:1px solid #1E2D45; margin:2.2rem 0; }
    </style>
    <div class="ab-badge">Research &amp; methodology</div>
    <div class="ab-title">About DiabetesVision</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ab-h">What it does</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ab-card"><p>
    DiabetesVision screens for the three most common diabetic complications that benefit from
    early detection — <b>retinopathy</b> (eye), <b>skin</b> changes, and <b>foot</b> disease
    (neuropathy, peripheral arterial disease and ulceration). It pairs deep-learning image
    models with foot thermography and validated clinical risk scoring, and presents the results
    two ways: detailed dashboards for clinicians, and a plain-language self-check for the public.
    </p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ab-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ab-h">The science of foot thermography</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ab-card"><p>
    Skin temperature reflects what is happening underneath. In diabetes:
    </p>
    <ul>
      <li><b>Warm foot →</b> early autonomic neuropathy causes blood vessels to dilate, raising
      plantar skin temperature (our data showed diabetic feet ~3&deg;C warmer at the metatarsals).</li>
      <li><b>Cool foot →</b> peripheral arterial disease reduces perfusion, cooling the foot.</li>
      <li><b>Local asymmetry →</b> a hot spot on one foot but not the other can precede ulceration
      by weeks, before any visible break in the skin.</li>
    </ul>
    <p>The tool divides each sole into ten anatomical zones, compares left vs right, and reads both
    the pattern and the absolute temperature — extracting far more than the standard 10&nbsp;g
    monofilament, which only detects neuropathy once it is already advanced.</p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ab-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ab-h">Explainability</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ab-card"><p>
    Every image result carries a visual explanation — a Grad-CAM heatmap for the retinal and skin
    models, and a zone-risk overlay for the foot scan — so a clinician can verify <i>why</i> the
    model reached its conclusion rather than trusting a black box. Risk scores show the
    contributing factors, not just a number.
    </p></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ab-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ab-h">Datasets &amp; references</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="ab-card"><ul>
      <li><b>APTOS 2019</b> — retinal fundus images graded 0-4 for diabetic retinopathy.</li>
      <li><b>HAM10000</b> — 10,015 dermatoscopic skin-lesion images.</li>
      <li><b>ThermoDataBase</b> — labelled plantar thermal images (control vs diabetic).</li>
      <li><b>Synthetic thermal cohort (50,000)</b> — for risk-model prototyping and calibration.</li>
      <li>IWGDF foot-risk guidance (Bus et al. 2020); UKPDS Risk Engine (Stevens et al. 2001);
      MNSI (Feldman et al. 1994); FINDRISC (Lindström &amp; Tuomilehto 2003);
      EQ-5D utilities (Beaudet et al. 2014); Grad-CAM (Selvaraju et al. 2017).</li>
    </ul></div>
    """, unsafe_allow_html=True)
