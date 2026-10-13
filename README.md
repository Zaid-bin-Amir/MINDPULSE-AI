# MindPulse AI: Prescriptive Health & Habit Engine

An end-to-end machine learning pipeline and interactive dashboard that assesses depression risk from daily habits and demographic factors — built on a real-world dataset of 140,700+ records. The engine pairs an **XGBoost classifier** with **SHAP** explainability to give transparent, personalized risk assessments instead of an opaque score.

> **Disclaimer:** MindPulse AI is a lifestyle-based screening tool, not a diagnostic instrument. It does not replace professional medical or psychological evaluation.

## Features

- **Leakage-free training pipeline** — data is split before imputation, and scaling is isolated inside each cross-validation fold
- **Robust feature engineering** — normalized degree standardization, whitelist filtering for corrupted categorical entries, and merged mutually-exclusive columns (student vs. professional metrics)
- **Class-imbalance handling** — `scale_pos_weight` tuned for the dataset's ~18% positive class
- **Overfitting control** — `min_child_weight`, `subsample`, and `colsample_bytree` tuned to prevent the model from overfitting on rare/out-of-distribution age groups while preserving natural risk variation
- **Explainable predictions** — SHAP (TreeExplainer) breaks down exactly which habits drove each individual risk score
- **Interactive dashboard** — a Streamlit app with live risk scoring, feature attribution charts, and a personalized habit action plan

## Tech Stack

| Layer | Tools |
|---|---|
| Data & modeling | pandas, numpy, scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |

## Project Structure

```
mindpulse-ai/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── MindPulse_Training.ipynb   # Full data pipeline, model training & evaluation
└── app.py                     # Streamlit dashboard
```

## Installation

```bash
git clone https://github.com/Mohsin801/mindpulse-ai.git
cd mindpulse-ai
pip install -r requirements.txt
```

Place a `train.csv` file (depression survey dataset) in the project root before running the notebook. See [Dataset](#dataset) below for where to get it.

## Usage

1. Run the training pipeline end to end:
   ```bash
   jupyter notebook MindPulse_Training.ipynb
   ```
   This cleans the data, compares seven models via 5-fold cross-validation, trains the final XGBoost model, and exports three artifacts: `mindpulse_model.pkl`, `mindpulse_explainer.pkl`, `mindpulse_encoders.pkl`.

2. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Model Performance

Evaluated on a held-out 20% test set:

| Metric | Score |
|---|---|
| ROC-AUC | 0.963 |
| Recall (Depression class) | 91% |
| Precision (Depression class) | 65% |

Recall is prioritized over precision by design — for a mental-health screening tool, missing an at-risk individual is costlier than flagging someone who turns out not to be.

## Dataset

This project uses the ["Exploring Mental Health Data" dataset by Adil Shamim on Kaggle](https://www.kaggle.com/datasets/adilshamim8/exploring-mental-health-data), covering demographics, academic/work pressure, sleep, diet, financial stress, and satisfaction. It is not included in this repository — download `train.csv` from the link and place it in the project root before running the notebook.

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

## Author

**Zaid Bin Aamir**
BSCS, University of Lahore — Sargodha Campus
AI/ML Internship, Zynvex