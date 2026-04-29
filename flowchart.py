import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_OUTPUT = PROJECT_DIR / "flowchart.png"
OUTPUTS_OUTPUT = PROJECT_DIR / "outputs" / "flowchart.png"


def add_box(ax, xy, text, width=3.0, height=0.72, facecolor="#e8f1fb"):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.4,
        edgecolor="#2f4f6f",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#1f2933",
        wrap=True,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.3,
        color="#2f4f6f",
        shrinkA=12,
        shrinkB=12,
    )
    ax.add_patch(arrow)


def build_flowchart():
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    nodes = {
        "data": ((5, 9.2), "Heart Disease Dataset\nheart.csv"),
        "load": ((5, 8.0), "Data Loading\nand Inspection"),
        "preprocess": ((5, 6.8), "Preprocessing\nImputation, Scaling,\nOne-Hot Encoding"),
        "cv": ((5, 5.6), "5-Fold Stratified\nCross-Validation"),
        "models": ((2.7, 4.25), "Model Training\nLR, RF, Tuned RF,\nXGBoost, MLP,\nSMOTE/ADASYN RF,\nStacking, NN"),
        "metrics": ((7.3, 4.25), "Evaluation Metrics\nAccuracy, Precision,\nRecall, F1,\nROC-AUC, PR-AUC,\nBrier, ECE"),
        "outputs": ((7.3, 2.9), "Saved Outputs\nCSV Results and\nComparison Plots"),
        "saving": ((2.7, 2.9), "Model Saving\nadvisor_bundle.pkl\nand model files"),
        "advisor": ((5, 1.6), "Interactive Advisor\nModel Selection and\nPatient Input"),
        "prediction": ((5, 0.45), "Risk Prediction\nProbability, Risk Level,\nAdvice, History CSV"),
    }

    colors = {
        "data": "#f8ead8",
        "load": "#e8f1fb",
        "preprocess": "#e8f1fb",
        "cv": "#e9f7ef",
        "models": "#fff4cc",
        "metrics": "#fff4cc",
        "outputs": "#f3e8ff",
        "saving": "#f3e8ff",
        "advisor": "#e6fffb",
        "prediction": "#e6fffb",
    }

    for key, (xy, text) in nodes.items():
        add_box(ax, xy, text, facecolor=colors[key])

    add_arrow(ax, (5, 8.84), (5, 8.36))
    add_arrow(ax, (5, 7.64), (5, 7.16))
    add_arrow(ax, (5, 6.44), (5, 5.96))
    add_arrow(ax, (4.45, 5.25), (3.25, 4.62))
    add_arrow(ax, (5.55, 5.25), (6.75, 4.62))
    add_arrow(ax, (3.55, 4.0), (6.45, 4.0))
    add_arrow(ax, (7.3, 3.9), (7.3, 3.25))
    add_arrow(ax, (2.7, 3.9), (2.7, 3.25))
    add_arrow(ax, (3.4, 2.55), (4.35, 1.95))
    add_arrow(ax, (6.6, 2.55), (5.65, 1.95))
    add_arrow(ax, (5, 1.24), (5, 0.82))

    ax.text(
        5,
        9.82,
        "End-to-End Heart Disease Prediction Framework",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#172033",
    )

    fig.tight_layout(pad=0.4)
    OUTPUTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUTS_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_flowchart()
    print(f"Saved: {ROOT_OUTPUT}")
    print(f"Saved: {OUTPUTS_OUTPUT}")
