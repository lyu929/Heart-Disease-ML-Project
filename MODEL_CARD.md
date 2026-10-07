# Model card: heartrisk logistic regression

| | |
|---|---|
| Model | L2 logistic regression (C tuned by inner CV) on 11 clinical variables, one-hot encoded categorical variables, median imputation without missing indicators (they encode the recording hospital; see the report, section 4) |
| Output | probability of angiographic coronary artery disease, a binary call at a frozen threshold, and a risk band |
| Threshold | F1-optimal on 5-fold out-of-fold predictions of the training data (0.42 for the current bundle; see `GET /v1/model`) |
| Version | `heartrisk train` writes `models/heartrisk.joblib` plus `heartrisk_card.json` (data SHA-256, library versions, git commit) |
| Owner | Haolin Lyu (post-course extension of the Group 10 course project) |

## Intended use

Teaching and research: it demonstrates a leakage-free clinical-prediction workflow. It is **not** a
medical device, has not been prospectively validated, and must not be used for diagnosis or triage.

## Training data

`data/heart.csv`: 918 adults referred for coronary angiography at four hospitals in the 1980s
(Cleveland, Budapest, Zürich/Basel, Long Beach). Prevalence is 55 %, and 79 % of patients are male.
See [docs/data.md](docs/data.md).

## Performance

These figures are for the deployed recipe under 5×5 nested cross-validation (mean, with the corrected
95 % CI); see
[reports/REPORT.md](reports/REPORT.md).

| Metric | Value |
|---|---|
| ROC-AUC | 0.923 (0.902–0.944) |
| Brier score | 0.107 |
| Calibration slope / ECE | 1.03 / 0.06 |
| Sensitivity / specificity at threshold | 0.91 / 0.79 |
| Leave-one-hospital-out AUC (evaluation recipe) | 0.79 (Switzerland) – 0.96 (Hungary) |

## Factors and limitations

* **Population shift.** Performance and calibration vary substantially between hospitals. The
  calibration intercept on the Swiss cohort is +2.2, so absolute risks need local recalibration before
  any use.
* **Missing data.** Cholesterol is missing for 19 % of patients, and the missingness is informative
  (site-dependent). The API flags imputed values in its warnings.
* **Sex imbalance.** 21 % of patients are women. Performance by sex has not been separately validated
  with adequate precision.
* **Old, small, referral-based data.** The data does not represent screening populations, and there are
  no sample weights.

## Explanations

`POST /v1/explain` returns exact interventional Shapley values over the 11 raw variables (all 2^11
coalitions) against a stratified 32-patient reference sample. The values add up exactly to the
predicted probability minus the reference mean. They explain the model, not causal effects.
