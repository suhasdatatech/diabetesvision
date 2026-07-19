"""
summary_report.py — Combined Patient Summary (printable)
=========================================================
Pulls together the results published to st.session_state by the Foot
Thermography, PAD & Ulcer, and Neuropathy tabs into one view, with an
overall foot-risk impression and a printable HTML block.
"""

import streamlit as st
from datetime import date


def g(key, default=None):
    return st.session_state.get(key, default)


def run():
    st.title("Patient Summary")
    st.caption("Consolidated view of results entered across the Foot Thermography, PAD & Ulcer and Neuropathy tabs. Use your browser's Print (Ctrl/Cmd+P) to save as PDF.")

    if not g("sh_foot_analysed"):
        st.info(
            "No foot scan analysed yet. Go to the **Foot Thermography** tab and analyse an image; "
            "then visit **PAD & Ulcer** and **Neuropathy**. Their results will appear here automatically."
        )
        return

    # ── Header / demographics ─────────────────────────────────────────────────
    pid = g("sh_patient_id") or "—"
    st.markdown(f"### {pid}  ·  {date.today().isoformat()}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Age", g("sh_age", "—"))
    c2.metric("DM duration (yrs)", g("sh_duration_dm", "—"))
    c3.metric("HbA1c (%)", f"{g('sh_hba1c'):.1f}" if g("sh_hba1c") is not None else "—")
    c4.metric("IWGDF category", g("sh_iwgdf", "—"))

    # ── Scores ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Foot risk scores")
    c1, c2, c3, c4 = st.columns(4)
    dm = g("sh_dm_prob")
    c1.metric("Foot health score", f"{g('sh_thermal_score','—')}/100")
    c2.metric("DM image probability", f"{dm*100:.0f}%" if dm is not None else "n/a")
    pad = g("sh_pad_prob")
    c3.metric("PAD probability", f"{pad*100:.0f}%" if pad is not None else "not run")
    neu = g("sh_neuro_combined")
    c4.metric("Neuropathy likelihood", f"{neu*100:.0f}%" if neu is not None else "not run")

    ulcer = g("sh_ulcer_prob")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ulcer probability", f"{ulcer*100:.0f}%" if ulcer is not None else "not run")
    c2.metric("High-risk zones", f"{g('sh_n_hot','—')}/10")
    c3.metric("Bilateral asymmetry", f"{g('sh_asymmetry'):.2f}" if g("sh_asymmetry") is not None else "—")

    # ── Overall impression ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Overall impression")
    flags = []
    if pad is not None and pad >= 0.30: flags.append("PAD rule-in positive — vascular assessment (ABI/Doppler)")
    if g("sh_abs_elev"): flags.append("Elevated absolute foot temperature — autonomic/neuropathy pattern")
    if neu is not None and neu >= 0.7: flags.append("High neuropathy likelihood")
    elif neu is not None and neu >= 0.4: flags.append("Moderate neuropathy likelihood")
    if ulcer is not None and ulcer >= 0.4: flags.append("Elevated foot-ulcer risk — podiatry surveillance")
    if g("sh_history_ulcer"): flags.append("Previous ulcer/amputation — IWGDF high-risk")
    if g("sh_hba1c") is not None and g("sh_hba1c") > 8.5: flags.append("Poor glycaemic control — optimise")

    if not flags:
        st.success("No high-risk flags from the entered data. Continue routine annual foot review.")
    else:
        for f in flags:
            st.warning(f)

    # ── Printable block ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Printable summary")
    def fmt(v, suf="", pct=False):
        if v is None: return "not assessed"
        if pct: return f"{v*100:.0f}%"
        return f"{v}{suf}"
    rows = [
        ("Patient", pid),
        ("Date", date.today().isoformat()),
        ("Age / DM duration", f"{g('sh_age','—')} / {g('sh_duration_dm','—')} yrs"),
        ("HbA1c", fmt(g('sh_hba1c'), '%')),
        ("Foot health score", f"{g('sh_thermal_score','—')}/100 (IWGDF {g('sh_iwgdf','—')})"),
        ("DM image probability", fmt(dm, pct=True)),
        ("PAD probability", fmt(pad, pct=True)),
        ("Neuropathy likelihood", fmt(neu, pct=True)),
        ("MNSI symptoms", f"{g('sh_mnsi','—')}/13" if g('sh_mnsi') is not None else "not assessed"),
        ("Foot ulcer probability", fmt(ulcer, pct=True)),
    ]
    trs = "".join(
        f"<tr><td style='padding:4px 10px;color:#555'>{k}</td>"
        f"<td style='padding:4px 10px;font-weight:600'>{v}</td></tr>" for k, v in rows)
    flag_html = "".join(f"<li>{f}</li>" for f in flags) or "<li>No high-risk flags</li>"
    st.markdown(f"""
<div style="background:#fff;color:#111;border:1px solid #ccc;border-radius:8px;padding:18px;font-family:sans-serif">
  <h3 style="margin:0 0 8px">DiabetesVision — Foot Assessment Summary</h3>
  <table style="border-collapse:collapse;font-size:14px">{trs}</table>
  <p style="margin:12px 0 4px;font-weight:600">Flags &amp; actions</p>
  <ul style="font-size:14px;margin:0">{flag_html}</ul>
  <p style="font-size:11px;color:#888;margin-top:12px">
    Research/pilot prototype. Decision-support only; not a diagnostic device. Models partly
    trained on synthetic data. Confirm all findings clinically (ABI/Doppler, monofilament, labs).
  </p>
</div>
""", unsafe_allow_html=True)
    st.caption("Tip: Ctrl/Cmd+P → Save as PDF to file this summary.")
