# Heart Disease Robustness Project

---

## 1. Project Overview

This project focuses on heart disease prediction using machine learning, with an emphasis on robustness, model comparison, and real-world usability through an interactive advisor system.

The system includes:

- Multiple machine learning models
- Imbalance handling strategies
- Cross-validation based evaluation
- Deployment-ready prediction interface
- Interactive patient risk advisor

---

## 2. Main Objectives

1. Build multiple machine learning models for heart disease prediction.
2. Compare model performance using 5-fold cross-validation.
3. Improve performance under class imbalance.
4. Evaluate models using Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, Brier Score, and ECE.
5. Provide real-time prediction for new patient input through an advisor system.

---

## 3. Models Included

### Core Models

- Logistic Regression
- Random Forest
- Tuned Random Forest
- XGBoost
- MLP Classifier
- Deep Neural Network

### Robustness and Imbalance Methods

- SMOTE Random Forest
- ADASYN Random Forest
- Domain-Weighted XGBoost
- Stacking Ensemble

---

## 4. Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score
- Expected Calibration Error

---

## 5. Project Structure

    Heart_Disease_Project/
    ├── app.py
    ├── advisor.py
    ├── data_loader.py
    ├── evaluate.py
    ├── model.py
    ├── preprocess.py
    ├── visualize.py
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── data/
    │   └── heart.csv
    ├── outputs/
    └── models/

---

## 6. Dataset

Place the dataset file in this location:

    data/heart.csv

The dataset should contain the following columns:

| Feature | Description |
|---|---|
| Age | Patient age |
| Sex | M or F |
| ChestPainType | ATA, NAP, ASY, or TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | 0 or 1 |
| RestingECG | Normal, ST, or LVH |
| MaxHR | Maximum heart rate |
| ExerciseAngina | Y or N |
| Oldpeak | ST depression |
| ST_Slope | Up, Flat, or Down |
| HeartDisease | Target label, 0 or 1 |

---

## 7. Required Folder Setup

Before running the project, create these folders:

    mkdir data models outputs

The required folder structure is:

    data/
    models/
    outputs/

---

## 8. Installation

Create a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

---

## 9. Run the Project

Run the project directly:

    python app.py

Or install it as a command-line app:

    python setup.py install
    heart-disease-app

---

## 10. Main Menu

After running the project, the system shows:

    ===== Main Menu =====
    1 - Run full training pipeline
    2 - Run advisor demo
    q - Quit

---

## 11. Training Pipeline

Choose option 1:

    1

The training pipeline will:

1. Load the dataset.
2. Inspect the dataset.
3. Preprocess numeric and categorical features.
4. Train all models.
5. Run 5-fold cross-validation.
6. Evaluate model performance.
7. Save result files.
8. Save trained models.

Main output files:

    outputs/experiment_results.csv
    outputs/summary_results.csv
    models/advisor_bundle.pkl
    models/rf_model.pkl
    models/tuned_rf_model.pkl
    models/xgb_model.pkl
    models/mlp_model.pkl

---

## 12. Advisor System

Choose option 2:

    2

The advisor system allows users to enter patient information and receive a heart disease risk prediction.

Available model choices may include:

    1 - Logistic Regression
    2 - Random Forest
    3 - Tuned Random Forest
    4 - MLP Classifier
    5 - Stacking Ensemble
    6 - XGBoost
    7 - Domain-Weighted XGBoost

Example patient input:

    Age: 50
    Sex: F
    ChestPainType: TA
    RestingBP: 180
    Cholesterol: 200
    FastingBS: 0
    RestingECG: ST
    MaxHR: 100
    ExerciseAngina: N
    Oldpeak: 4
    ST_Slope: Flat

Example prediction output:

    Model: XGBoost
    Predicted Label: 1
    Probability: 0.8123
    Risk Level: Very High
    Advice: Very high predicted risk. Immediate medical evaluation is strongly recommended.

---

## 13. Advisor Output Files

The advisor saves results to:

    outputs/advisor_result.txt
    outputs/patient_predictions.csv

The text file stores the latest prediction result.

The CSV file stores the history of patient predictions.

---

## 14. Dependencies

Required packages:

- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- xgboost
- imbalanced-learn
- torch
- scipy

---

## 15. Recommended Workflow

1. Create the required folders.
2. Put heart.csv into the data folder.
3. Create and activate the virtual environment.
4. Install dependencies.
5. Run the project.
6. Choose option 1 to train models.
7. Choose option 2 to use the advisor.

Commands:

    mkdir data models outputs
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python app.py

---

## 16. Troubleshooting

### xgboost not found

Install XGBoost:

    pip install xgboost

### imblearn not found

Install imbalanced-learn:

    pip install imbalanced-learn

### Mac Python environment issue

Use a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Advisor cannot run

Run the training pipeline first so that this file is created:

    models/advisor_bundle.pkl

### VS Code Run button does not work

Make sure VS Code uses the project virtual environment:

    Heart_Disease_Project/venv/bin/python

Also make sure the working directory is the project root folder.

---

## 17. Example Commands

Install dependencies:

    pip install -r requirements.txt

Run the project:

    python app.py

Install command-line app:

    python setup.py install

Run installed app:

    heart-disease-app

---

## 18. Project Highlights

- Multiple machine learning models
- Class imbalance handling
- SMOTE and ADASYN support
- XGBoost and neural network support
- Stacking ensemble
- Interactive advisor system
- Real-time patient prediction
- Saved prediction history

---

## 19. Author and Course Context

Machine Learning Final Project

Heart Disease Prediction System

Focus: robustness, model comparison, and deployment-ready advisor usage