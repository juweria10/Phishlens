"""Readers for the files the model-training notebook already produced."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "ML_results"
MODEL_DIR = ROOT / "saved_model"
EDA_IMAGES = ROOT / "EDA_images"
GRAPHS = RESULTS / "graphs"

GRAPH_TITLES = {
    "01_confusion_matrices.png": "Confusion matrices, all features",
    "02_metric_comparison.png": "Metric comparison across models",
    "03_roc_curves.png": "ROC curves",
    "04_baseline_vs_tuned.png": "Baseline vs tuned models",
    "05_train_vs_test_accuracy.png": "Train vs test accuracy",
    "06_learning_curves.png": "Learning curves",
    "07_decision_tree_depth_curve.png": "Decision tree depth curve",
    "08_logreg_C_curve.png": "Logistic regression C curve",
    "09_rf_feature_importance.png": "Random forest feature importance",
    "10_url_only_confusion_matrices.png": "Confusion matrices, URL-only features",
    "11_feature_setting_comparison.png": "Feature setting comparison",
    "12_url_only_train_vs_test.png": "URL-only train vs test",
}
EDA_TITLES = {
    "class_distribution.png": "Class distribution",
    "URLLength_histogram.png": "URL length",
    "NoOfSubDomain_histogram.png": "Number of subdomains",
    "NoOfLettersInURL_histogram.png": "Letters in URL",
    "NoOfExternalRef_histogram.png": "External references",
    "pearson_heatmap.png": "Pearson correlation",
    "spearman_heatmap.png": "Spearman correlation",
}


def metadata() -> dict:
    return json.load(open(MODEL_DIR / "model_metadata.json"))


def run_summary() -> dict:
    return json.load(open(RESULTS / "run_summary.json"))


def feature_settings() -> pd.DataFrame:
    """Tuned test scores for the three feature settings the team compared."""
    full = pd.read_csv(RESULTS / "model_comparison_table.csv")
    full = full[full["Version"] == "Tuned"]
    s1 = pd.read_csv(RESULTS / "sensitivity_S1_without_URLSimilarityIndex.csv")
    s2 = pd.read_csv(RESULTS / "sensitivity_S2_url_only_features.csv")
    s2 = s2[s2["Version"] == "Tuned"]
    frames = [
        pd.DataFrame({"setting": "All 50 features", "model": full["Model"], "accuracy": full["Test Accuracy"],
                      "f1": full["Test F1-score"], "precision": full["Test Precision"], "recall": full["Test Recall"]}),
        pd.DataFrame({"setting": "Without URLSimilarityIndex", "model": s1["Model"], "accuracy": s1["Accuracy"],
                      "f1": s1["F1-score"], "precision": s1["Precision"], "recall": s1["Recall"]}),
        pd.DataFrame({"setting": "URL features only (21)", "model": s2["Model"], "accuracy": s2["Test Accuracy"],
                      "f1": s2["Test F1-score"], "precision": s2["Test Precision"], "recall": s2["Test Recall"]}),
    ]
    return pd.concat(frames, ignore_index=True)


def confusion_matrices(url_only: bool) -> dict[str, tuple[int, int, int, int]]:
    """model -> (true phishing predicted phishing, missed phishing, false alarms, true legitimate)."""
    path = RESULTS / ("sensitivity_S2_confusion_matrices.csv" if url_only else "confusion_matrices.csv")
    df = pd.read_csv(path)
    df.columns = ["model", "true", "pred_phish", "pred_legit"]
    out = {}
    for model, g in df.groupby("model", sort=False):
        ph = g[g["true"] == "True Phishing"].iloc[0]
        lg = g[g["true"] == "True Legitimate"].iloc[0]
        out[model] = (int(ph.pred_phish), int(ph.pred_legit), int(lg.pred_phish), int(lg.pred_legit))
    return out


def rf_importance(top: int = 10) -> list[tuple[str, float]]:
    df = pd.read_csv(RESULTS / "random_forest_feature_importance.csv")
    df.columns = ["feature", "importance"]
    return [(r.feature, float(r.importance)) for r in df.head(top).itertuples()]
