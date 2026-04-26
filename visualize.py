import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_barplot(summary_df, metric, filename):
    if summary_df is None or summary_df.empty:
        print(f"Skipping {metric} bar plot: empty summary results.")
        return

    if metric not in summary_df.columns:
        print(f"Skipping {metric} bar plot: metric column not found.")
        return

    if "Model" not in summary_df.columns or "Experiment" not in summary_df.columns:
        print(f"Skipping {metric} bar plot: required columns missing.")
        return

    pivot_df = summary_df.pivot(index="Model", columns="Experiment", values=metric)

    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", ax=plt.gca())

    plt.title(f"{metric} by Model and Experiment")
    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_dnn_loss(loss_history):
    if not loss_history:
        print("Skipping DNN loss curve: no loss history.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")

    plt.title("Deep Neural Network Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    output_path = os.path.join(OUTPUTS_DIR, "dnn_loss_curve.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_degradation_plot(summary_df):
    if summary_df is None or summary_df.empty:
        print("Skipping degradation plot: empty summary results.")
        return

    if "F1_Degradation_%" not in summary_df.columns:
        print("Skipping degradation plot: F1_Degradation_% column not found.")
        return

    df = summary_df.dropna(subset=["F1_Degradation_%"]).copy()

    if df.empty:
        print("Skipping degradation plot: no degradation values.")
        return

    plt.figure(figsize=(12, 6))

    for exp_name in df["Experiment"].unique():
        sub = df[df["Experiment"] == exp_name]
        plt.plot(sub["Model"], sub["F1_Degradation_%"], marker="o", label=exp_name)

    plt.title("Average F1 Degradation Across Experiments")
    plt.ylabel("F1 Degradation (%)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUTS_DIR, "f1_degradation_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_roc_curve_comparison(curve_data):
    """
    Save true ROC curve line plot:
    x-axis = False Positive Rate
    y-axis = True Positive Rate

    curve_data format:
    {
        "Model Name": {
            "y_true": [...],
            "y_prob": [...]
        }
    }
    """
    if not curve_data:
        print("Skipping ROC curve comparison: no curve data.")
        return

    plt.figure(figsize=(9, 7))

    plotted_any = False

    for model_name, data in curve_data.items():
        y_true = np.array(data.get("y_true", []))
        y_prob = np.array(data.get("y_prob", []))

        if len(y_true) == 0 or len(y_prob) == 0:
            continue

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

        plotted_any = True

    if not plotted_any:
        plt.close()
        print("Skipping ROC curve comparison: no valid model curve.")
        return

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Guess"
    )

    plt.title("ROC Curve Comparison Across Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(OUTPUTS_DIR, "roc_curve_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_pr_curve_comparison(curve_data):
    """
    Save true Precision-Recall curve line plot:
    x-axis = Recall
    y-axis = Precision
    """
    if not curve_data:
        print("Skipping PR curve comparison: no curve data.")
        return

    plt.figure(figsize=(9, 7))

    plotted_any = False

    for model_name, data in curve_data.items():
        y_true = np.array(data.get("y_true", []))
        y_prob = np.array(data.get("y_prob", []))

        if len(y_true) == 0 or len(y_prob) == 0:
            continue

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (AP = {avg_precision:.3f})"
        )

        plotted_any = True

    if not plotted_any:
        plt.close()
        print("Skipping PR curve comparison: no valid model curve.")
        return

    plt.title("Precision-Recall Curve Comparison Across Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()

    output_path = os.path.join(OUTPUTS_DIR, "pr_curve_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_all_visualizations(artifacts):
    if artifacts is None:
        print("No artifacts found. Skipping all visualizations.")
        return

    summary_df = artifacts.get("summary_results", None)

    if summary_df is None or summary_df.empty:
        print("No summary results found. Skipping metric bar plots.")
    else:
        save_barplot(summary_df, "Recall", "recall_comparison.png")
        save_barplot(summary_df, "F1", "f1_comparison.png")
        save_barplot(summary_df, "ROC_AUC", "roc_auc_comparison.png")
        save_barplot(summary_df, "PR_AUC", "pr_auc_comparison.png")
        save_barplot(summary_df, "ECE", "ece_comparison.png")
        save_degradation_plot(summary_df)

    save_dnn_loss(artifacts.get("dnn_loss_history", []))

    curve_data = artifacts.get("curve_data", {})
    save_roc_curve_comparison(curve_data)
    save_pr_curve_comparison(curve_data)