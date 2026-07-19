# 🩺 DiabetesVision — AI Screening for Eye, Skin & Foot

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Early Detection. Clearer Vision. Healthier Feet.**

DiabetesVision brings specialist-level screening for the three most common diabetic
complications — **retinopathy** (eye), **skin** changes, and **foot** disease
(neuropathy, peripheral arterial disease and ulceration) — into one tool, combining
deep-learning image models with foot thermography and validated clinical risk scoring.

🔗 **Live app:** https://diabetesvision-cyixsu4gyaaufvpgclrbfp.streamlit.app/

---

## ✨ What's inside

Ten tabs, built for two audiences — clinicians and the public:

| Tab | What it does |
|-----|--------------|
| 🏠 Home | Overview of the technology and who it's for |
| 🙋 My Foot Check | Patient-facing self-check: enter details, add photos, get plain-language advice + a summary to share with a clinician |
| 🟢 Patient Screening | Skin-lesion risk assessment with Grad-CAM heatmap |
| 🔵 Doctor Dashboard | Retinal fundus grading (DR 0–4) with clinical recommendations |
| 🌡️ Foot Thermography | Thermal foot analysis (multi-image); trained model flags diabetic-foot pattern |
| 🦶 PAD & Ulcer | Peripheral arterial disease + foot-ulcer scoring with ABI referral logic |
| ⚡ Neuropathy | Neuropathy model + MNSI symptom questionnaire + bedside exam |
| 📊 Risk & CVD | Cardiovascular risk, complication radar, deterioration trajectory |
| 📋 Summary | Consolidated, printable patient summary |
| 💚 Quality of Life | Health-utility proxy from complication burden |

Enter a patient's details once (age, HbA1c, blood pressure…) and they **auto-fill across every tab**.

---

## 🧠 The technology

**Deep-learning image models (MobileNetV2 + Grad-CAM).** Retinal DR grading and skin-lesion
classification, each with a heatmap so clinicians can see what drove the result.

**Foot thermography.** Temperature-zone analysis of a foot image. A *warm* foot points to
early autonomic neuropathy (vasodilation); a *cool* foot points to poor circulation (PAD).
One scan separates the two.

**Risk scoring.** Thermal signs combine with HbA1c and clinical risk factors to score
neuropathy, PAD, foot-ulcer risk and cardiovascular risk — turning an image into an action plan.

### Model performance

| Model | Task | Data | Result |
|-------|------|------|--------|
| Skin lesion | Concerning vs not | HAM10000 (10,015 imgs) | 85.1% acc |
| Retinal DR | DR grade 0–4 | APTOS 2019 | 70.6% acc |
| Foot thermography | Control vs diabetic | ThermoDataBase (333 imgs, 167 patients) | **AUC 0.878** |
| Neuropathy | Neuropathy present | 50k synthetic (test 7,502) | **AUC 0.931** |
| PAD | PAD present | 50k synthetic (test 7,502) | **AUC 0.825** |

The foot image model improved across three iterations (bilateral asymmetry 0.617 →
full temperature features 0.865 → HSV zone features **0.878**), all with patient-grouped
cross-validation.

---

## 🚀 Run locally

```bash
git clone https://github.com/065010-AmanMalhi/diabetesvision.git
cd diabetesvision

python -m venv venv          # use Python 3.12 (TensorFlow has no 3.13/3.14 builds yet)
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. The retinal/skin `.h5` models download on first run;
the thermal `.pkl` models ship in the repo.

---

## 📁 Project structure

```
diabetesvision/
├── app.py                       # 10-tab entry point
├── .streamlit/config.toml       # dark theme
├── pages/
│   ├── home.py  about.py  patient_mode.py  doctor_mode.py
│   ├── patient_intake.py        # My Foot Check (patient-facing)
│   ├── thermal_foot.py          # foot thermography (AUC 0.878)
│   ├── pad_ulcer_score.py       # PAD + ulcer scoring
│   ├── neuropathy_score.py      # neuropathy + MNSI
│   ├── thermal_risk.py          # CVD / prognosis
│   ├── summary_report.py        # printable summary
│   ├── qol_proxy.py             # quality-of-life proxy
│   ├── model_card.py            # models & validation (under About)
│   └── shared.py                # cross-tab autofill helper
├── thermal_img_model.pkl
├── thermal_risk_models.pkl
└── requirements.txt
```

---

## 🔬 Datasets & references

APTOS 2019 (retinal) · HAM10000 (skin) · ThermoDataBase (plantar thermal) ·
synthetic thermal cohort (50,000, for risk-model prototyping). Clinical scores:
IWGDF foot risk (Bus et al. 2020), UKPDS Risk Engine (Stevens et al. 2001),
MNSI (Feldman et al. 1994), FINDRISC (Lindström & Tuomilehto 2003),
EQ-5D utilities (Beaudet et al. 2014), Grad-CAM (Selvaraju et al. 2017).

---

## ⚠️ Disclaimer

Research / pilot prototype. **Decision-support only — not a diagnostic device, not a
replacement for professional medical assessment, and not CE/FDA approved.** The
neuropathy, PAD and ulcer models are trained on synthetic data; the foot image model
uses real images but a modest patient count. All outputs require clinician review and
external validation on real, independently-labelled patients before any clinical use.

---

## 👨‍💻 Author

**Aman Malhi** — extended with foot thermography, PAD/neuropathy/ulcer scoring,
a patient self-check, cross-tab autofill, and a printable summary.