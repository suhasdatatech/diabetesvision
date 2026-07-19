"""
qol_proxy.py — Quality-of-Life Proxy Index (complication-burden)
=================================================================
IMPORTANT — this is a PROXY, not a measured or predicted quality of life.
No validated QoL outcome data (EQ-5D, SF-36, ADDQoL) exists in the training sets,
so no QoL model could be trained. Instead this computes a transparent
health-utility estimate from published EQ-5D disutility weights for diabetes
complications. It is literature-informed and reproducible, but it is NOT a
substitute for administering a validated QoL instrument.

Utility weights (decrement from baseline) — approximate published EQ-5D values:
  Baseline utility (uncomplicated T2DM): 0.785
  Sources: Beaudet et al., Value in Health 2014; UKPDS 62 (Clarke 2002);
           Bagust & Beale 2005.
    Neuropathy (symptomatic)   -0.084
    PAD                         -0.061
    Foot ulcer (current)        -0.170
    Amputation history          -0.280
    Retinopathy / vision loss   -0.070
    Nephropathy / CKD           -0.078   (dialysis far higher)
    CVD event history (MI/CVA)  -0.090
    Severe obesity (per BMI>30) -0.006 per unit (capped)
    Poor glycaemic control      up to -0.050 (HbA1c-scaled)

Output:
  - Estimated health utility (0-1)
  - QoL proxy score 0-100 (utility x 100)
  - Domain-by-domain waterfall of what is pulling QoL down
  - Modelled gain from correcting modifiable factors
"""

import streamlit as st
import plotly.graph_objects as go

BASELINE_UTILITY = 0.785

WEIGHTS = {
    "Symptomatic neuropathy": 0.084,
    "Peripheral arterial disease": 0.061,
    "Active foot ulcer": 0.170,
    "Amputation history": 0.280,
    "Retinopathy / vision loss": 0.070,
    "Nephropathy / CKD": 0.078,
    "CVD event history": 0.090,
}


def compute_utility(flags, bmi, hba1c):
    """Return (utility, contributions dict)."""
    contrib = {}
    for k, present in flags.items():
        if present:
            contrib[k] = -WEIGHTS[k]
    # Obesity decrement (BMI over 30, capped at 10 units)
    if bmi > 30:
        contrib["Obesity (BMI>30)"] = -min((bmi - 30) * 0.006, 0.06)
    # Glycaemic decrement (HbA1c over 7.5%, scaled, capped)
    if hba1c > 7.5:
        contrib["Poor glycaemic control"] = -min((hba1c - 7.5) * 0.012, 0.05)

    utility = BASELINE_UTILITY + sum(contrib.values())
    utility = max(0.0, min(utility, 1.0))
    return utility, contrib


def build_waterfall(contrib):
    labels = ["Baseline"] + list(contrib.keys()) + ["Net QoL"]
    measures = ["absolute"] + ["relative"] * len(contrib) + ["total"]
    values = [BASELINE_UTILITY] + list(contrib.values()) + [None]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels,
        y=[v if v is not None else 0 for v in values],
        text=[f"{v:+.3f}" if (v is not None and m != 'absolute') else (f"{v:.3f}" if v is not None else "")
              for v, m in zip(values, measures)],
        textposition="outside",
        connector={"line": {"color": "#2E3D55"}},
        increasing={"marker": {"color": "#00D4AA"}},
        decreasing={"marker": {"color": "#FF4B4B"}},
        totals={"marker": {"color": "#0099FF"}},
    ))
    fig.update_layout(paper_bgcolor="#0D1526", plot_bgcolor="#0D1526",
        font=dict(family="DM Sans", color="#A0B0C5"),
        xaxis=dict(tickangle=-30), yaxis=dict(title="Health utility", gridcolor="#1E2D45", range=[0, 1]),
        margin=dict(l=20, r=20, t=20, b=80), height=420, showlegend=False)
    return fig


def run():
    st.title("Quality-of-Life Proxy Index")
    st.markdown(
        "*Estimates a health-utility score from complication burden using published "
        "EQ-5D disutility weights. A transparent proxy — not a measured or predicted QoL.*"
    )
    st.warning(
        "This is a literature-weighted PROXY, not a validated quality-of-life measurement. "
        "No QoL outcome data (EQ-5D/SF-36/ADDQoL) exists in the datasets, so no QoL model "
        "could be trained. For real QoL, administer a validated instrument. Research/pilot use only."
    )
    st.markdown("---")

    # ── INPUTS ────────────────────────────────────────────────────────────────
    st.subheader("Complication & Clinical Profile")
    st.caption("Tick present complications. These can be auto-populated from the Foot Thermography and Risk tabs.")
    c1, c2, c3 = st.columns(3)
    with c1:
        neuro  = st.checkbox("Symptomatic neuropathy", key="qol_proxy_1")
        pad    = st.checkbox("Peripheral arterial disease", key="qol_proxy_2")
        ulcer  = st.checkbox("Active foot ulcer", key="qol_proxy_3")
    with c2:
        amput  = st.checkbox("Amputation history", key="qol_proxy_4")
        retino = st.checkbox("Retinopathy / vision loss", key="qol_proxy_5")
        nephro = st.checkbox("Nephropathy / CKD", key="qol_proxy_6")
    with c3:
        cvd    = st.checkbox("CVD event history (MI/stroke)", key="qol_proxy_7")
        bmi    = st.number_input("BMI (kg/m2)", 15.0, 50.0, 29.0, step=0.1, key="qol_proxy_8")
        hba1c  = st.number_input("HbA1c (%)", 4.0, 16.0, 8.2, step=0.1, key="qol_proxy_9")

    flags = {
        "Symptomatic neuropathy": neuro,
        "Peripheral arterial disease": pad,
        "Active foot ulcer": ulcer,
        "Amputation history": amput,
        "Retinopathy / vision loss": retino,
        "Nephropathy / CKD": nephro,
        "CVD event history": cvd,
    }

    utility, contrib = compute_utility(flags, bmi, hba1c)
    qol_score = round(utility * 100, 1)

    # ── RESULT ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Estimated Health Utility")
    c1, c2, c3 = st.columns(3)
    c1.metric("Health Utility (0-1)", f"{utility:.3f}")
    c2.metric("QoL Proxy Score", f"{qol_score}/100")
    baseline_pct = round(BASELINE_UTILITY * 100, 1)
    c3.metric("vs Uncomplicated Baseline", f"{qol_score - baseline_pct:+.1f}",
              delta=f"baseline {baseline_pct}", delta_color="off")

    if utility >= 0.75:
        st.success("Relatively preserved estimated QoL. Focus on prevention and maintenance.")
    elif utility >= 0.60:
        st.info("Moderate QoL burden. Address modifiable complications and glycaemic control.")
    elif utility >= 0.45:
        st.warning("Substantial QoL burden. Multidisciplinary review recommended.")
    else:
        st.error("Severe QoL burden. Prioritise complications with the largest impact below.")

    # ── WATERFALL ─────────────────────────────────────────────────────────────
    if contrib:
        st.markdown("---")
        st.subheader("What is pulling QoL down")
        st.caption("Each bar is the estimated utility decrement from that factor (published EQ-5D weights).")
        st.plotly_chart(build_waterfall(contrib), use_container_width=True)
    else:
        st.info("No complications entered — estimated utility equals the uncomplicated baseline (0.785).")

    # ── MODIFIABLE GAIN ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Modelled Gain from Modifiable Factors")
    modifiable = {}
    if "Poor glycaemic control" in contrib:
        modifiable["Bring HbA1c to <=7.5%"] = -contrib["Poor glycaemic control"]
    if "Obesity (BMI>30)" in contrib:
        modifiable["Weight loss to BMI 30"] = -contrib["Obesity (BMI>30)"]
    if ulcer:
        modifiable["Heal active ulcer"] = WEIGHTS["Active foot ulcer"]
    if modifiable:
        total_gain = sum(modifiable.values())
        for k, v in modifiable.items():
            st.markdown(f"- {k}: **+{v:.3f}** utility (+{v*100:.1f} QoL points)")
        st.success(f"Potential recoverable QoL from modifiable factors: "
                   f"**+{total_gain:.3f} utility (+{total_gain*100:.1f} points)** → "
                   f"projected {min((utility+total_gain)*100,100):.1f}/100.")
    else:
        st.info("No modelled modifiable QoL gains from current inputs (glycaemia, weight, active ulcer).")

    st.markdown("---")
    st.caption(
        "Utility weights: Beaudet et al. Value in Health 2014; UKPDS 62 (Clarke 2002); "
        "Bagust & Beale 2005. Baseline uncomplicated T2DM utility 0.785. "
        "PROXY INDEX — not a validated QoL measurement. To train a real QoL model, collect "
        "EQ-5D-5L or ADDQoL scores on real patients. Research/pilot prototype only."
    )
