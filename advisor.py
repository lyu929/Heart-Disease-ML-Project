import os
import joblib
import pandas as pd


MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

MODEL_MAP = {
    "1": "Logistic Regression",
    "2": "Random Forest",
    "3": "Tuned Random Forest",
    "4": "MLP Classifier",
    "5": "Stacking Ensemble",
    "6": "XGBoost",
    "7": "Domain-Weighted XGBoost",
}

FEATURE_HINTS = {
    "Sex": "M = male, F = female",
    "FastingBS": "0 = no, 1 = yes",
    "ChestPainType": "ATA / NAP / ASY / TA",
    "RestingECG": "Normal / ST / LVH",
    "ExerciseAngina": "N = no, Y = yes",
    "ST_Slope": "Up / Flat / Down",
}


def load_advisor_bundle():
    file_path = os.path.join(MODELS_DIR, "advisor_bundle.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Advisor models not found. Please run the full training pipeline first."
        )
    return joblib.load(file_path)


def get_available_model_choices():
    bundle = load_advisor_bundle()
    available_model_names = bundle.get("models", {}).keys()

    filtered = {}
    for key, model_name in MODEL_MAP.items():
        if model_name in available_model_names:
            filtered[key] = model_name

    if not filtered:
        raise ValueError("No trained advisor models are available.")

    return filtered


def get_numeric_input(prompt, min_value=None, max_value=None, integer_only=False):
    while True:
        value = input(prompt).strip().lower()

        if value in ["q", "quit"]:
            return None

        try:
            if integer_only:
                value = int(value)
            else:
                value = float(value)

            if min_value is not None and value < min_value:
                print(f"Invalid input. Please enter a value >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Invalid input. Please enter a value <= {max_value}.")
                continue

            return value

        except ValueError:
            if integer_only:
                print("Invalid input. Please enter a valid integer.")
            else:
                print("Invalid input. Please enter a valid number.")


def get_categorical_input(prompt, allowed_values):
    allowed_values_str = [str(v) for v in allowed_values]

    while True:
        value = input(prompt).strip()

        if value.lower() in ["q", "quit"]:
            return None

        if value in allowed_values_str:
            return value

        print(f"Invalid input. Allowed values: {allowed_values_str}")


def get_patient_input():
    bundle = load_advisor_bundle()
    feature_info = bundle.get("feature_info", [])

    print("\n===== Enter Patient Information =====")
    print("Type 'q' or 'quit' at any time to exit.")
    print("Hints and recommended ranges are shown when available.\n")

    patient_data = {}

    for feature in feature_info:
        name = feature["name"]
        feature_type = feature["type"]
        unique_values = feature.get("unique_values")
        min_value = feature.get("min")
        max_value = feature.get("max")
        hint = FEATURE_HINTS.get(name, "")

        # 这些字段最好按整数输入
        integer_fields = {"Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"}
        integer_only = name in integer_fields

        if feature_type == "numeric":
            if min_value is not None and max_value is not None:
                if hint:
                    prompt = f"{name} ({hint}) [range: {min_value:.4f} ~ {max_value:.4f}]: "
                else:
                    prompt = f"{name} [range: {min_value:.4f} ~ {max_value:.4f}]: "
            else:
                if hint:
                    prompt = f"{name} ({hint}): "
                else:
                    prompt = f"{name}: "

            value = get_numeric_input(
                prompt,
                min_value=min_value,
                max_value=max_value,
                integer_only=integer_only
            )
            if value is None:
                return None
            patient_data[name] = value

        else:
            if hint:
                prompt = f"{name} ({hint}) (allowed: {unique_values}): "
            else:
                prompt = f"{name} (allowed: {unique_values}): "

            value = get_categorical_input(prompt, unique_values)
            if value is None:
                return None
            patient_data[name] = value

    return patient_data


def get_risk_level(probability):
    if probability >= 0.80:
        return "Very High"
    elif probability >= 0.60:
        return "High"
    elif probability >= 0.40:
        return "Moderate"
    elif probability >= 0.20:
        return "Mild"
    return "Low"


def generate_advice(probability):
    if probability >= 0.80:
        return "Very high predicted risk. Immediate medical evaluation is strongly recommended."
    elif probability >= 0.60:
        return "High predicted risk. Clinical follow-up and further testing are recommended."
    elif probability >= 0.40:
        return "Moderate predicted risk. Monitor health indicators and consider preventive care."
    elif probability >= 0.20:
        return "Mild predicted risk. Maintain healthy lifestyle habits and routine checkups."
    else:
        return "Low predicted risk. Continue healthy habits and regular medical monitoring."


def run_advisor(patient_data, model_choice="1"):
    bundle = load_advisor_bundle()
    available_choices = get_available_model_choices()

    if model_choice not in available_choices:
        raise ValueError("Invalid model choice.")

    model_name = available_choices[model_choice]
    model = bundle["models"][model_name]
    shared_features = bundle["shared_features"]

    patient_df = pd.DataFrame([patient_data])

    for col in shared_features:
        if col not in patient_df.columns:
            raise ValueError(f"Missing feature: {col}")

    patient_df = patient_df[shared_features]

    probability = model.predict_proba(patient_df)[:, 1][0]
    predicted_label = int(probability >= 0.5)

    result = {
        "Model": model_name,
        "PredictedLabel": predicted_label,
        "Probability": float(probability),
        "RiskLevel": get_risk_level(probability),
        "Advice": generate_advice(probability),
    }

    return result


def save_advisor_result_txt(result, patient_data):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUTS_DIR, "advisor_result.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Heart Disease Risk Advisor Result\n")
        f.write("=" * 50 + "\n\n")

        f.write("Patient Information:\n")
        for key, value in patient_data.items():
            f.write(f"{key}: {value}\n")

        f.write("\nPrediction Result:\n")
        f.write(f"Model: {result['Model']}\n")
        f.write(f"PredictedLabel: {result['PredictedLabel']}\n")
        f.write(f"Probability: {result['Probability']:.4f}\n")
        f.write(f"RiskLevel: {result['RiskLevel']}\n")
        f.write(f"Advice: {result['Advice']}\n")


def append_advisor_result_csv(patient_data, result):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUTS_DIR, "patient_predictions.csv")

    row = {}
    row.update(patient_data)
    row.update(result)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()

    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)

    record_id = len(df)
    total_records = len(df)
    return record_id, total_records