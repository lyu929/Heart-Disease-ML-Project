# Data provenance and caveats

| File | Rows | Outcome | Used for |
|---|---|---|---|
| `heart.csv` | 918 | angiographic CAD (> 50 % narrowing), prevalence 55.3 % | all modelling, deployment |
| `cleveland_processed.csv` | 303 | same outcome, prevalence 45.9 % | within-cohort benchmark only |
| `framingham_processed.csv` | 4 240 | 10-year incident CHD, 15.2 % | within-cohort benchmark only |
| `kaggle_processed.csv` | 918 | = `heart.csv`, z-scored | not used (redundant) |

## heart.csv

This is the Kaggle "Heart Failure Prediction Dataset" (fedesoriano, 2021). It combines the UCI Heart
Disease cohorts and drops duplicate patients. Check the original source pages for licence terms before
redistributing it.

The rows are stored hospital by hospital. `heartrisk.data.infer_sites()` restores the hospital labels
from the published cohort sizes. It then checks them against two facts:

* the published number of positives in each cohort, and
* the absence of cholesterol values in the Swiss cohort.

If the file has been reordered, the function raises an error rather than guessing.

| hospital | rows | positives | cholesterol recorded as 0 |
|---|---|---|---|
| Hungarian Institute of Cardiology, Budapest | 293 | 106 | 0 |
| University Hospitals Zürich & Basel | 123 | 115 | 123 (all) |
| VA Medical Center, Long Beach | 200 | 149 | 49 |
| Cleveland Clinic Foundation | 302 | 138 | 0 |

Two of these counts are one lower than the UCI originals: Hungarian has 293 rows (one negative duplicate
removed) and Cleveland has 302 (one positive duplicate removed).

**Zeros.** `Cholesterol == 0` (172 rows) and `RestingBP == 0` (1 row) are physiologically impossible,
so they are missing values. Because the missingness is concentrated in high-prevalence hospitals,
patients with "cholesterol = 0" have an 88 % disease rate. A model fed the raw zeros can therefore learn
the recording site instead of physiology. `heartrisk` handles this in three steps:

1. it converts the zeros to missing values inside the pipeline;
2. it imputes them with the median of the training fold;
3. it adds a missing-value indicator.

The report quantifies both the zeros-as-values variant and the variant without the indicator.

## Processed teaching extracts

The two `*_processed.csv` files were imputed and z-scored on all rows **before** they were committed.
Learned models cannot undo that, so they get a little information from the test rows. This is a mild
form of leakage, and they are therefore only benchmarked within themselves. The Cleveland patients are
also contained in `heart.csv`, so evaluating a `heart.csv` model on them would not be external
validation. Framingham measures a different, prognostic outcome and is never pooled with the diagnostic
cohorts.

## Privacy

All files are public, de-identified research datasets. No patient-level data from any other source is
included in this repository.
