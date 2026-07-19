"""
home.py — DiabetesVision landing page (global, self-contained styling)
"""
import streamlit as st


def run():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');
      .dv-hero { text-align:center; padding:2.5rem 0 1rem; }
      .dv-badge { display:inline-block; background:#00D4AA22; color:#00D4AA;
        border:1px solid #00D4AA44; border-radius:20px; padding:5px 18px;
        font-size:0.78rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
        font-family:'DM Sans',sans-serif; margin-bottom:1.8rem; }
      .dv-title { font-family:'DM Serif Display',serif; font-size:3.6rem; line-height:1.1;
        color:#E8EDF5; margin:0 0 1.4rem; }
      .dv-title .accent { color:#00D4AA; font-style:italic; display:block; }
      .dv-sub { color:#7A8FA6; font-size:1.1rem; max-width:640px; margin:0 auto 1rem;
        line-height:1.7; font-family:'DM Sans',sans-serif; }
      .dv-stat { background:#0D1B2E; border:1px solid #1E3A5F; border-radius:14px;
        padding:1.6rem 1rem; text-align:center; }
      .dv-stat .n { font-family:'DM Serif Display',serif; font-size:2.4rem; color:#00D4AA; }
      .dv-stat .l { color:#7A8FA6; font-size:0.85rem; margin-top:0.4rem; font-family:'DM Sans',sans-serif; }
      .dv-h { font-family:'DM Serif Display',serif; font-size:2rem; color:#E8EDF5; margin:0 0 0.3rem; }
      .dv-hs { color:#7A8FA6; font-size:0.95rem; margin-bottom:1.5rem; font-family:'DM Sans',sans-serif; }
      .dv-card { background:#0D1B2E; border:1px solid #1E3A5F; border-radius:14px;
        padding:1.5rem; height:100%; font-family:'DM Sans',sans-serif; }
      .dv-card h4 { color:#E8EDF5; margin:0 0 0.6rem; font-size:1.1rem; font-weight:600; }
      .dv-card p { color:#B0BEC5; line-height:1.7; margin:0; font-size:0.92rem; }
      .dv-divider { border-top:1px solid #1E2D45; margin:3rem 0; }
    </style>

    <div class="dv-hero">
      <div class="dv-badge">AI diabetic screening</div>
      <div class="dv-title">Early Detection.<span class="accent">Clearer Vision. Healthier Feet.</span></div>
      <p class="dv-sub">DiabetesVision brings specialist-level screening for the eye, skin and
      foot into one place — combining deep-learning image models with foot thermography to
      catch diabetic complications earlier, anywhere in the world.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Global stats ──────────────────────────────────────────────────────────
    stats = [
        ("537M", "Adults living with diabetes worldwide"),
        ("1 in 3", "Will develop diabetic retinopathy"),
        ("~50%", "Develop peripheral neuropathy"),
        ("~85%", "Of amputations begin as a foot ulcer"),
    ]
    cols = st.columns(4)
    for c, (n, l) in zip(cols, stats):
        c.markdown(f'<div class="dv-stat"><div class="n">{n}</div><div class="l">{l}</div></div>',
                   unsafe_allow_html=True)

    st.markdown('<div class="dv-divider"></div>', unsafe_allow_html=True)

    # ── The technology ────────────────────────────────────────────────────────
    st.markdown('<div class="dv-h">The technology</div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-hs">Three complementary screening methods in one tool.</div>', unsafe_allow_html=True)
    tech = [
        ("Retinal & skin AI",
         "MobileNetV2 deep-learning models grade diabetic retinopathy (0-4) from fundus "
         "scans and flag concerning skin lesions, each with a Grad-CAM heatmap so clinicians "
         "can see what drove the result."),
        ("Foot thermography",
         "Temperature-zone analysis of a foot photo. A warm foot points to autonomic "
         "neuropathy; a cool foot points to poor circulation (PAD). One scan separates the "
         "two — trained model AUC 0.88 for diabetic-foot detection."),
        ("Risk & ulcer scoring",
         "Thermal signs combine with HbA1c and risk factors to score neuropathy (AUC 0.93), "
         "PAD (AUC 0.83), foot-ulcer risk and cardiovascular risk — turning a picture into "
         "an action plan."),
    ]
    cols = st.columns(3)
    for c, (h, p) in zip(cols, tech):
        c.markdown(f'<div class="dv-card"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="dv-divider"></div>', unsafe_allow_html=True)

    # ── Two audiences ─────────────────────────────────────────────────────────
    st.markdown('<div class="dv-h">Built for two audiences</div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-hs">The same evidence, presented for who is looking at it.</div>', unsafe_allow_html=True)
    aud = [
        ("For clinicians",
         "Dedicated dashboards for retinal grading, foot thermography, PAD, neuropathy and "
         "ulcer scoring, cardiovascular risk and a printable patient summary — with the "
         "model confidence and reasoning shown alongside every result."),
        ("For the public",
         "A plain-language <b>My Foot Check</b>: enter a few details, add a photo or two, and "
         "get simple advice plus a summary you can share with your care team. No medical "
         "jargon, clear next steps."),
    ]
    cols = st.columns(2)
    for c, (h, p) in zip(cols, aud):
        c.markdown(f'<div class="dv-card"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="dv-divider"></div>', unsafe_allow_html=True)
    st.caption("Research / pilot prototype. Decision-support only — not a diagnostic device, "
               "and not a replacement for professional medical assessment.")
