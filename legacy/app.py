from data_loader import load_dataset, inspect_dataset
from model import train_full_project_pipeline
from visualize import save_all_visualizations
from advisor import (
    run_advisor,
    save_advisor_result_txt,
    append_advisor_result_csv,
    get_patient_input,
    get_available_model_choices,
)


def run_training_pipeline():
    print("===== Heart Disease Prediction Project =====")

    print("\n[1] Loading dataset...")
    df = load_dataset()

    print("\n[2] Inspecting dataset...")
    inspect_dataset(df)

    print("\n[3] Running training pipeline...")
    summary_df, artifacts = train_full_project_pipeline(df)

    print("\n[4] Saving visualizations...")
    save_all_visualizations(artifacts)

    print("\n===== Final Summary Results =====")
    print(summary_df)

    if "best_rf_params" in artifacts and artifacts["best_rf_params"]:
        print("\nBest parameters for Tuned Random Forest:")
        print(artifacts["best_rf_params"])

    if "errors" in artifacts and artifacts["errors"]:
        print("\nModels skipped due to errors:")
        for model_name, error_msg in artifacts["errors"].items():
            print(f"- {model_name}: {error_msg}")

    print("\nSaved files:")
    print("- outputs/experiment_results.csv")
    print("- outputs/summary_results.csv")
    print("- outputs/")
    print("- models/")
    print("\nSaved model files may include:")
    print("- models/rf_model.pkl")
    print("- models/tuned_rf_model.pkl")
    print("- models/xgb_model.pkl")
    print("- models/mlp_model.pkl")
    print("- models/advisor_bundle.pkl")


def run_advisor_loop():
    print("\n===== Heart Disease Risk Advisor =====")
    print("Please run the training pipeline first so the advisor models are available.")
    print("Type 'q' or 'quit' at the model selection step to exit.\n")

    while True:
        try:
            available_models = get_available_model_choices()
        except Exception as e:
            print(f"\nError: {e}")
            print("Please run the full training pipeline first.\n")
            return

        print("Select model for prediction:")
        for key, label in available_models.items():
            print(f"{key} - {label}")
        print("q - Quit advisor")

        allowed_choices = list(available_models.keys())
        model_choice = input(
            f"\nEnter your model choice ({'/'.join(allowed_choices)} or q): "
        ).strip().lower()

        if model_choice in ["q", "quit"]:
            print("\nExiting advisor.")
            break

        if model_choice not in allowed_choices:
            print("\nInvalid model choice. Please try again.\n")
            continue

        try:
            patient_data = get_patient_input()

            if patient_data is None:
                print("\nExiting advisor.")
                break

            result = run_advisor(patient_data, model_choice=model_choice)

            save_advisor_result_txt(result, patient_data)
            record_id, total_records = append_advisor_result_csv(patient_data, result)

            print("\nPatient Information:")
            for key, value in patient_data.items():
                print(f"  {key}: {value}")

            print("\nPrediction Result:")
            print(f"Model           : {result['Model']}")
            print(f"Predicted Label : {result['PredictedLabel']}")
            print(f"Probability     : {result['Probability']:.4f}")
            print(f"Risk Level      : {result['RiskLevel']}")
            print(f"Advice          : {result['Advice']}")

            print("\nSaved to:")
            print("- outputs/advisor_result.txt")
            print("- outputs/patient_predictions.csv")

            print(f"\nRecord ID       : {record_id}")
            print(f"Total Records   : {total_records}")
            print("\nYou can continue entering another patient, or type q to quit.\n")

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


def main():
    while True:
        print("\n===== Main Menu =====")
        print("1 - Run full training pipeline")
        print("2 - Run advisor demo")
        print("q - Quit")

        choice = input("\nEnter your choice (1/2 or q): ").strip().lower()

        if choice == "1":
            run_training_pipeline()
        elif choice == "2":
            run_advisor_loop()
        elif choice in ["q", "quit"]:
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or q.")


if __name__ == "__main__":
    main()