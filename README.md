# Chart Retention Analysis: Identifying Short-Lived Music Tracks

## Overview

This project analyzes chart retention dynamics in music streaming data, with a
focus on understanding **why most tracks exit charts quickly while a small
minority persist for long periods**. Rather than attempting to predict viral
hits, the analysis reframes the problem as a **risk-filtering task**: identifying
tracks that are unlikely to sustain long-term engagement.

The project mirrors retention and churn dynamics commonly observed in
media platforms and games, where outcomes follow heavy-tailed distributions and
precise prediction is inherently difficult.

---

## Data

The analysis uses Spotify Top 200 chart data enriched with audio features
retrieved via the Spotify Web API. Key variables include:

- Chart retention (`weeks_on_chart`)
- Streaming volume
- Audio features such as danceability, energy, valence, loudness, speechiness,
  acousticness, tempo, and liveness

Only **audio-level features** are used for modeling. Artist identity, marketing,
playlist placement, and other external drivers are intentionally excluded to
avoid leakage.

---

## Key Observations (EDA)

- Chart retention follows a **heavy-tailed distribution**: most tracks churn
  quickly, while a small fraction remain for extended periods.
- Mean-based analysis is misleading; log-scaled targets and rank-based methods
  are more appropriate.
- Univariate comparisons show overlapping distributions across retention
  buckets, indicating weak separability.
- Audio features appear to **constrain outcomes** rather than determine them.

These patterns closely resemble retention dynamics in games and digital media.

---

## Statistical Analysis

To assess whether audio features contain real signal:

- **Spearman rank correlations** were used to quantify monotonic associations
  with log-transformed retention.
- **Permutation tests** validated that several observed correlations exceed
  what would be expected by chance.

Results show that:
- Valence and loudness have the strongest (but still weak) positive associations
  with retention.
- Speechiness and liveness are associated with higher early-exit risk.
- Acousticness, instrumentalness, and tempo exhibit negligible effects.

All effects are small (|ρ| ≈ 0.05–0.12), confirming that audio features alone
cannot explain longevity.

---

## Modeling Approach

The modeling task is framed as **binary classification**:

**Short-lived tracks (≤ 5 weeks on chart) vs. Others**

This reflects a practical use case: early identification of tracks at high risk
of rapid exit.

Models used:
- Logistic Regression (interpretable baseline)
- Random Forest (nonlinear interactions)

Performance is intentionally modest:
- Logistic Regression ROC-AUC ≈ 0.58
- Random Forest ROC-AUC ≈ 0.61

These results are consistent with the statistical findings and indicate
non-random but limited signal.

---

## Interpretation & Practical Implications

The analysis demonstrates that:

- Audio features provide **real but weak signal** regarding chart retention.
- Precise hit prediction is not feasible using audio features alone.
- However, models can support **risk filtering**, ranking, and prioritization by
  flagging tracks unlikely to sustain engagement.

This aligns with real-world practice in music, gaming, and media analytics, where
models are used to reduce poor bets and focus human judgment rather than to make
deterministic predictions.

---

## Project Structure

notebooks/
- 01_eda.ipynb
- 02_statistical_tests.ipynb
- 03_binary_modeling.ipynb

data/
- spotify_top_songs_audio_features.csv

scripts/
- data_ingest.py
- preprocessing.py
- train.py

models/
- retention_model.joblib

.github/
- workflows/
  - train.yml

run_training.py
requirements.txt
README.md

---

## Tools Used

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- SciPy (statistical testing)

---

## Status

Complete.  
This project emphasizes methodological rigor, honest interpretation, and
business-relevant framing, alongside a reproducible machine learning workflow.

---

## ML Pipeline, CI/CD

This project includes a structured machine learning workflow for chart retention modeling.

On every push to the main branch:
- Dependencies are installed in a clean environment  
- Data ingestion and preprocessing steps are executed  
- The binary classification training pipeline (`run_training.py`) runs end-to-end  
- Model performance is evaluated  
- The trained retention model is saved  

This ensures the full workflow (ingestion → preprocessing → classification → evaluation → artifact generation) 
remains reproducible and environment-independent.

In addition, each successful run automatically publishes a versioned retention classification model
to GitHub Releases. This provides traceability of model iterations and enables automated delivery
of updated model versions.

