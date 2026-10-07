# From the course project to `heartrisk`

The original submission (Group 10, Machine Learning final project) lives unchanged in `legacy/`.
This table maps every legacy component to its replacement and records the defects that motivated the
rewrite.

| Legacy file | Replaced by | What changed |
|---|---|---|
| `data_loader.py` (`DATASET` constant) | `heartrisk.data` (`DATASETS` registry, `DatasetSpec`) | Per-dataset schema (numeric / binary / categorical / target); file validation; `infer_sites()` recovers the four source hospitals; data card. |
| `preprocess.py` | `heartrisk.features` | Same median / most-frequent imputation, scaling and one-hot encoding, plus `ZeroAsMissing` (Cholesterol = 0 and RestingBP = 0 are missing, not values) and missing-value indicators. Everything is inside the fitted pipeline. |
| `evaluate.py` (`compute_metrics`, `find_best_threshold`, `paired_test`) | `heartrisk.metrics`, `heartrisk.thresholds`, `heartrisk.stats` | Exact threshold optimisation (all cut-points instead of a 17-step grid); F1 / Youden / target-sensitivity / fixed strategies; calibration intercept & slope; corrected resampled t-test instead of `ttest_rel` on overlapping folds; Holm correction; bootstrap CIs. |
| `model.py` (`train_model_suite`, `run_single_dataset_cv`) | `heartrisk.models`, `heartrisk.evaluation` | One registry of full pipelines; 5×5 repeated CV with identical splits for all models; inner-CV grid search (nested CV); **threshold chosen on inner out-of-fold predictions of the training part**; per-fold timing and parameters saved. |
| `model.py` DNN (`FocalLoss`, `DNNClassifier`) | `heartrisk.dnn.TorchMLPClassifier` | scikit-learn compatible (clone / Pipeline / pickle), early stopping on a validation split, deterministic seeding. |
| `model.py` SMOTE / ADASYN RF | `smote_rf` in `heartrisk.models` | SMOTE runs inside the imbalanced-learn pipeline, so synthetic samples are generated from training folds only. Evaluated where imbalance actually exists (Framingham, 15 % events) — `heart.csv` is 55 % positive. |
| `model.py` stacking | `stacking` in `heartrisk.models` | Same LR + RF + XGB base learners, stratified internal CV. |
| `model.py` `train_deployment_models` + `advisor.py` | `heartrisk.train`, `heartrisk.bundle`, `heartrisk.predict` | One versioned bundle (pipeline, calibrated probabilities, frozen threshold, reference sample, training ranges, data hash, library versions, model card). The legacy advisor classified at 0.5 although evaluation used a tuned threshold; now the same threshold is used everywhere. |
| `advisor.py` console prompts | `heartrisk.schema`, `heartrisk.api`, `app/streamlit_app.py` | Pydantic validation (ranges, enums, `null` for unmeasured values), REST API with batch scoring and explanations, Streamlit UI. |
| `visualize.py` | `heartrisk.plots` | ROC/PR, reliability with Wilson CIs, decision curves, forest plots, leakage audit, leave-one-hospital-out, explanation waterfall. |
| `additional_experiments.py` | `heartrisk.study` (benchmark, ablations) | Same nested protocol on every cohort (the legacy comparison used a fixed 0.5 threshold and no tuning); cleaning and calibration ablations; leave-one-hospital-out validation. |
| `app.py` menu | `heartrisk` CLI | `study`, `evaluate`, `train`, `predict`, `explain`, `serve`, `info`. |
| `setup.py`, committed `*.egg-info` | `pyproject.toml`, `.gitignore` | src layout, optional extras (`api`, `ui`, `torch`, `shap`, `dev`), build artefacts no longer committed. |

## Defects found in the legacy evaluation

1. **Threshold tuned on the test fold.** `fit_and_evaluate_pipeline` called
   `find_best_threshold(y_test, y_prob)` and then `compute_metrics(y_test, y_prob, threshold)`, for every
   model. The reported F1 / recall / precision were therefore optimistic. `heartrisk` records both numbers on
   the same fitted models (`leaky_*` columns) so the optimism is measured, not guessed — see the leakage
   audit in `reports/REPORT.md`.
2. **Impossible zeros used as values.** 172 patients have Cholesterol = 0 (all of the Switzerland cohort);
   they have an 88 % disease rate, so a model can learn "cholesterol = 0 → disease", a recording artefact.
3. **Paired t-test on CV folds.** Overlapping training sets make fold scores dependent; the plain paired
   t-test understates the variance. Replaced by the Nadeau–Bengio correction.
4. **Deployment ≠ evaluation.** The advisor classified at a fixed 0.5 threshold, not the evaluated one.
