"""
model_card.py — Models & Validation documentation (rendered under About tab)
"""
import streamlit as st
import pandas as pd


def run():
    st.markdown("---")
    st.markdown('<div class="section-header">Thermal Foot Models &amp; Validation</div>', unsafe_allow_html=True)
    st.caption("Documentation of the models added to DiabetesVision, with honest performance and limitations.")

    rows = [
        ["Skin lesion (original)", "MobileNetV2", "HAM10000 (10,015 imgs)", "85.1% acc", "Concerning vs not"],
        ["Retinal DR (original)", "MobileNetV2", "APTOS 2019 (3,662 imgs)", "70.6% acc", "DR grade 0-4"],
        ["Foot thermography", "Logistic (HSV zones)", "ThermoDataBase (333 imgs, 167 pts)", "AUC 0.878", "Control vs DM"],
        ["Neuropathy", "HistGradientBoosting", "50k synthetic (test 7,502)", "AUC 0.931", "Neuropathy present"],
        ["PAD", "HistGradientBoosting", "50k synthetic (test 7,502)", "AUC 0.825", "PAD present"],
        ["Foot ulcer", "HistGradientBoosting", "50k synthetic (test 7,502)", "AUC 0.995*", "Active ulcer"],
    ]
    df = pd.DataFrame(rows, columns=["Model", "Architecture", "Dataset", "Performance", "Target"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### How the foot image model was built")
    st.markdown("""
The foot thermography classifier improved across three iterations, each validated with
**patient-grouped cross-validation** (no patient in both train and test):

- v1 — bilateral temperature asymmetry only: **AUC 0.617**
- v2 — full temperature feature set (absolute temps + gradients): **AUC 0.865**
- v3 — 16 HSV zone features on real images: **AUC 0.878** (Sens 0.70 / Spec 0.92)

Key finding from the temperature data: diabetic feet run **~3°C warmer** at the
metatarsals (early autonomic neuropathy → vasodilation). PAD reverses this — a
**cool** foot from reduced perfusion. One scan separates the two.
    """)

    st.markdown("#### PAD is detectable from the scan")
    st.markdown("""
On 50,000 records, thermography was the **dominant** PAD signal:
thermal-only AUC **0.81**, risk-factors-only 0.66, combined **0.83**.
PAD+ feet were cooler (28.7°C vs 29.2°C) with more asymmetry — a genuine,
interpretable signature.
    """)

    st.markdown("#### Prediabetes — why it was dropped")
    st.info(
        "Thermography could not distinguish prediabetes from normoglycaemia "
        "(AUC 0.51, i.e. chance). A screen there would only generate false positives, "
        "so no prediabetes model is shipped."
    )

    st.markdown("#### Limitations")
    st.warning(
        "The neuropathy, PAD and ulcer models were trained on SYNTHETIC data. Neuropathy "
        "(0.931) and PAD (0.825) reflect learnable signal; *ulcer (0.995) and the abnormal "
        "rule label are near-deterministic functions of the inputs — high AUC there confirms "
        "rule consistency, not clinical accuracy. The foot image model (0.878) uses real "
        "images but only 167 patients. Everything here is a research/pilot prototype requiring "
        "prospective validation on real, independently-labelled patients. Not CE/FDA approved."
    )

    st.caption(
        "References: FINDRISC — Lindström & Tuomilehto 2003; MNSI — Feldman et al. 1994; "
        "UKPDS Risk Engine — Stevens et al. 2001; EQ-5D utilities — Beaudet et al. 2014; "
        "IWGDF foot risk — Bus et al. 2020."
    )
