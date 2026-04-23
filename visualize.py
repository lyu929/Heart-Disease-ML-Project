import os
import matplotlib.pyplot as plt


OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_barplot(summary_df, metric, filename):
    pivot_df = summary_df.pivot(index="Model", columns="Experiment", values=metric)

    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.title(f"{metric} by Model and Experiment")
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, filename))
    plt.close()


def save_dnn_loss(loss_history):
    if not loss_history:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker="o")
    plt.title("Deep Neural Network Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_loss_curve.png"))
    plt.close()


def save_degradation_plot(summary_df):
    if "F1_Degradation_%" not in summary_df.columns:
        return

    df = summary_df.dropna(subset=["F1_Degradation_%"]).copy()
    if df.empty:
        return

    plt.figure(figsize=(12, 6))
    for exp_name in df["Experiment"].unique():
        sub = df[df["Experiment"] == exp_name]
        plt.plot(sub["Model"], sub["F1_Degradation_%"], marker="o", label=exp_name)

    plt.title("Average F1 Degradation Across Experiments")
    plt.ylabel("F1 Degradation (%)")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "f1_degradation_plot.png"))
    plt.close()


def save_all_visualizations(artifacts):
    summary_df = artifacts["summary_results"]

    save_barplot(summary_df, "Recall", "recall_comparison.png")
    save_barplot(summary_df, "F1", "f1_comparison.png")
    save_barplot(summary_df, "ROC_AUC", "roc_auc_comparison.png")
    save_barplot(summary_df, "PR_AUC", "pr_auc_comparison.png")
    save_barplot(summary_df, "ECE", "ece_comparison.png")
    save_degradation_plot(summary_df)
    save_dnn_loss(artifacts.get("dnn_loss_history", []))