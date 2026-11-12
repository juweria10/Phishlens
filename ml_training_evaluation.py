"""
AI-Based Phishing Website and URL Detection System
Member 3 - Machine Learning & Model Comparison

Trains and evaluates Logistic Regression, Decision Tree and Random Forest on the
cleaned dataset produced by the EDA step (dw&eda.py).

Run from the PAI_Project folder:
    python ml_training_evaluation.py

Label convention (from the EDA script): 0 = Phishing, 1 = Legitimate.
Phishing is treated as the POSITIVE class for precision / recall / F1, because
finding phishing pages is the purpose of the system.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, make_scorer,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "EDA result" / "cleaned_URL_Dataset.csv"
OUT_DIR = BASE / "ML_results"
FIG_DIR = OUT_DIR / "graphs"
MODEL_DIR = BASE / "saved_model"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
POS = 0                       # phishing = positive class
CLASS_NAMES = ["Phishing", "Legitimate"]
NON_FEATURE_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title", "label"]

PHISH_F1 = make_scorer(f1_score, pos_label=POS)   # F1 of the phishing class
# Features that can be computed from the URL string alone (no page download needed)
URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate", "TLDLegitimateProb",
    "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar",
    "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
    "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS"]
STRONGEST_SINGLE = "URLSimilarityIndex"   # alone gives ~99.6% accuracy (see report)

CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
sns.set_theme(style="whitegrid", rc={"axes.grid.axis": "y"})


def legend_below(ax, ncol=3):
    """Put the legend under the plot so it never hides bars."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False)

COLORS = {"Logistic Regression": "#4C72B0",
          "Decision Tree": "#DD8452",
          "Random Forest": "#55A868"}


# ----------------------------------------------------------------------------
# 1. Load data and prepare it
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    log = {"rows_in_cleaned_file": len(df)}

    # Rows without a label are unusable
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # The EDA step only removed rows that were *identical*. Many pages still appear
    # more than once (same URL, differing only in a few missing cells). If such
    # copies fall on both sides of a train/test split the test score is inflated,
    # so keep one row per URL (the one with the fewest missing values).
    df["_n_missing"] = df.isnull().sum(axis=1)
    with_url = df[df["URL"].notna()].sort_values("_n_missing")
    no_url = df[df["URL"].isna()]
    log["rows_with_missing_url"] = len(no_url)
    with_url = with_url.drop_duplicates(subset="URL", keep="first")
    df = pd.concat([with_url, no_url]).drop(columns="_n_missing")
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].astype(float)
    y = df["label"]

    log["rows_after_url_dedup"] = len(df)
    log["n_features"] = len(feature_cols)
    log["missing_cells_in_features"] = int(X.isnull().sum().sum())
    log["class_counts"] = {"Phishing(0)": int((y == 0).sum()),
                           "Legitimate(1)": int((y == 1).sum())}
    return X, y, feature_cols, log


def make_pipeline(model, scale):
    """Preprocessing (median imputation, optional scaling) + classifier.
    Imputer / scaler are fitted ONLY on the training data inside each fit()."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# 2. Metrics helpers
# ----------------------------------------------------------------------------
def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, pos_label=POS),
        "Recall": recall_score(y, pred, pos_label=POS),
        "F1-score": f1_score(y, pred, pos_label=POS),
    }, pred


def phishing_score(pipe, X):
    """Probability that a URL is phishing (class 0)."""
    idx = list(pipe.classes_).index(POS)
    return pipe.predict_proba(X)[:, idx]


def tune_models(baselines, param_grids, X_train, y_train):
    """GridSearchCV (5-fold, scored on phishing-class F1) for every model."""
    tuned, best_params, grid_tables = {}, {}, {}
    for name, pipe in baselines.items():
        print(f"Tuning {name} ...")
        gs = GridSearchCV(clone(pipe), param_grids[name], scoring=PHISH_F1, cv=CV,
                          n_jobs=-1, refit=True, return_train_score=True)
        gs.fit(X_train, y_train)
        tuned[name] = gs.best_estimator_
        best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        best_params[name]["cv_f1_phishing"] = round(float(gs.best_score_), 5)
        grid_tables[name] = pd.DataFrame(gs.cv_results_)[
            ["params", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        print("   best:", best_params[name])
    return tuned, best_params, grid_tables


# ----------------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    X, y, feature_cols, data_log = load_data()
    print("Data summary:", json.dumps(data_log, indent=2))

    # One split, reused for every model -> fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    data_log.update({"train_rows": len(X_train), "test_rows": len(X_test)})
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # ---------------- model definitions -------------------------------------
    baselines = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), scale=True),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False),
        "Random Forest": make_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            scale=False),
    }
    param_grids = {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__min_samples_split": [2, 10, 30],
        },
        "Random Forest": {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2"],
        },
    }

    # ---------------- baseline training + CV --------------------------------
    rows, cv_rows = [], []
    fitted_base = {}
    for name, pipe in baselines.items():
        pipe.fit(X_train, y_train)
        fitted_base[name] = pipe
        tr, _ = evaluate(pipe, X_train, y_train)
        te, _ = evaluate(pipe, X_test, y_test)
        rows.append({"Model": name, "Version": "Baseline (default)",
                     **{f"Train {k}": v for k, v in tr.items()},
                     **{f"Test {k}": v for k, v in te.items()}})
        print(f"[baseline] {name:20s} train acc={tr['Accuracy']:.4f}  test acc={te['Accuracy']:.4f}")

    # ---------------- hyper-parameter tuning --------------------------------
    tuned, best_params, grid_tables = tune_models(baselines, param_grids, X_train, y_train)

    # ---------------- test-set + CV evaluation of tuned models --------------
    preds, scores = {}, {}
    cm_data = {}
    reports = {}
    for name, pipe in tuned.items():
        tr, _ = evaluate(pipe, X_train, y_train)
        te, pred = evaluate(pipe, X_test, y_test)
        preds[name] = pred
        scores[name] = phishing_score(pipe, X_test)
        cm_data[name] = confusion_matrix(y_test, pred, labels=[POS, 1])
        reports[name] = classification_report(
            y_test, pred, labels=[POS, 1], target_names=CLASS_NAMES, digits=4)
        rows.append({"Model": name, "Version": "Tuned",
                     **{f"Train {k}": v for k, v in tr.items()},
                     **{f"Test {k}": v for k, v in te.items()}})

        # 5-fold CV on the training data for stability (mean +/- std)
        cvres = cross_validate(
            pipe, X_train, y_train, cv=CV, n_jobs=-1, return_train_score=True,
            scoring={"accuracy": "accuracy", "f1": PHISH_F1})
        cv_rows.append({
            "Model": name,
            "CV train accuracy": cvres["train_accuracy"].mean(),
            "CV val accuracy (mean)": cvres["test_accuracy"].mean(),
            "CV val accuracy (std)": cvres["test_accuracy"].std(),
            "CV val F1 (mean)": cvres["test_f1"].mean(),
            "CV val F1 (std)": cvres["test_f1"].std(),
        })

    # Baseline confusion matrices / ROC too (for the report)
    base_cm = {n: confusion_matrix(y_test, p.predict(X_test), labels=[POS, 1])
               for n, p in fitted_base.items()}

    results = pd.DataFrame(rows)
    results["Train-Test accuracy gap"] = results["Train Accuracy"] - results["Test Accuracy"]
    results["Train-Test F1 gap"] = results["Train F1-score"] - results["Test F1-score"]
    cv_df = pd.DataFrame(cv_rows)

    # ROC-AUC (tuned)
    roc = {}
    for name in tuned:
        fpr, tpr, _ = roc_curve((y_test == POS).astype(int), scores[name])
        roc[name] = (fpr, tpr, auc(fpr, tpr))
    results.loc[results.Version == "Tuned", "Test ROC-AUC"] = \
        [roc[m][2] for m in results.loc[results.Version == "Tuned", "Model"]]

    # ---------------- model selection (by CV, NOT by test set) --------------
    best_name = cv_df.sort_values(["CV val F1 (mean)", "CV val F1 (std)"],
                                  ascending=[False, True]).iloc[0]["Model"]
    print("Selected model (highest CV F1):", best_name)

    # ---------------- save tables -------------------------------------------
    results.round(5).to_csv(OUT_DIR / "model_comparison_table.csv", index=False)
    cv_df.round(5).to_csv(OUT_DIR / "cross_validation_results.csv", index=False)
    for n, t in grid_tables.items():
        t.to_csv(OUT_DIR / f"gridsearch_{n.lower().replace(' ', '_')}.csv", index=False)
    with open(OUT_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    with open(OUT_DIR / "classification_reports.txt", "w") as f:
        for n, r in reports.items():
            f.write(f"===== {n} (tuned, test set) =====\n{r}\n")
    cm_df = pd.concat({n: pd.DataFrame(m, index=[f"True {c}" for c in CLASS_NAMES],
                                       columns=[f"Pred {c}" for c in CLASS_NAMES])
                       for n, m in cm_data.items()})
    cm_df.to_csv(OUT_DIR / "confusion_matrices.csv")

    # ======================= GRAPHS ==========================================
    names = list(tuned)

    # (a) Confusion matrices (tuned)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, n in zip(axes, names):
        sns.heatmap(cm_data[n], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
        ax.set_title(n); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices - tuned models (test set)", fontsize=14)
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_confusion_matrices.png", dpi=150); plt.close(fig)

    # (b) Metric comparison bars (tuned)
    tuned_res = results[results.Version == "Tuned"].set_index("Model")
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    w = 0.25
    for i, n in enumerate(names):
        vals = [tuned_res.loc[n, f"Test {m}"] for m in metrics]
        bars = ax.bar(np.arange(len(metrics)) + (i - 1) * w, vals, w, label=n, color=COLORS[n])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
    lo = min(tuned_res[[f"Test {m}" for m in metrics]].min()) - 0.02
    ax.set_ylim(max(0, lo), 1.012)
    ax.set_xticks(np.arange(len(metrics))); ax.set_xticklabels(metrics)
    ax.set_ylabel("Score"); ax.set_title("Test-set performance of tuned models (phishing = positive class)")
    legend_below(ax); fig.tight_layout()
    fig.savefig(FIG_DIR / "02_metric_comparison.png", dpi=150); plt.close(fig)

    # (c) ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in names:
        fpr, tpr, a = roc[n]
        ax.plot(fpr, tpr, label=f"{n} (AUC = {a:.4f})", color=COLORS[n], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (phishing detected)")
    ax.set_title("ROC curves - tuned models"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_roc_curves.png", dpi=150); plt.close(fig)

    # (d) Baseline vs tuned F1
    fig, ax = plt.subplots(figsize=(8, 4.8))
    base_res = results[results.Version.str.startswith("Baseline")].set_index("Model")
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.18, [base_res.loc[n, "Test F1-score"] for n in names], 0.36,
                label="Baseline (default)", color="#B0B0B0")
    b2 = ax.bar(x + 0.18, [tuned_res.loc[n, "Test F1-score"] for n in names], 0.36,
                label="Tuned", color="#2E7D32")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0004,
                    f"{b.get_height():.4f}", ha="center", fontsize=8)
    lo = min(base_res["Test F1-score"].min(), tuned_res["Test F1-score"].min()) - 0.01
    ax.set_ylim(lo, 1.005)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Test F1-score (phishing)")
    ax.set_title("Effect of hyper-parameter tuning"); legend_below(ax, 2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_baseline_vs_tuned.png", dpi=150); plt.close(fig)

    # (e) Overfitting: train vs test accuracy, baseline & tuned
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, tr_v, te_v = [], [], []
    for _, r in results.iterrows():
        labels.append(f"{r['Model']}\n({'baseline' if r['Version'].startswith('Base') else 'tuned'})")
        tr_v.append(r["Train Accuracy"]); te_v.append(r["Test Accuracy"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, tr_v, 0.4, label="Train accuracy", color="#4C72B0")
    ax.bar(x + 0.2, te_v, 0.4, label="Test accuracy", color="#DD8452")
    for i, (a, b) in enumerate(zip(tr_v, te_v)):
        ax.text(i - 0.2, a + 0.001, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.001, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_ylim(min(te_v) - 0.03, 1.012)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("Accuracy")
    ax.set_title("Overfitting check: training vs test accuracy"); legend_below(ax, 2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_train_vs_test_accuracy.png", dpi=150); plt.close(fig)

    # (f) Learning curves (tuned models)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    lc_summary = {}
    for ax, n in zip(axes, names):
        sizes, trs, vas = learning_curve(
            tuned[n], X_train, y_train, cv=CV, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=RANDOM_STATE,
            shuffle=True)
        ax.plot(sizes, trs.mean(1), "o-", color="#4C72B0", label="Training")
        ax.fill_between(sizes, trs.mean(1) - trs.std(1), trs.mean(1) + trs.std(1), alpha=.15, color="#4C72B0")
        ax.plot(sizes, vas.mean(1), "o-", color="#DD8452", label="Cross-validation")
        ax.fill_between(sizes, vas.mean(1) - vas.std(1), vas.mean(1) + vas.std(1), alpha=.15, color="#DD8452")
        ax.set_title(n); ax.set_xlabel("Training examples"); ax.legend(loc="lower right")
        lc_summary[n] = {"final_train": float(trs.mean(1)[-1]), "final_cv": float(vas.mean(1)[-1])}
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Learning curves (tuned models, 5-fold CV)", fontsize=14)
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_learning_curves.png", dpi=150); plt.close(fig)

    # (g) Validation curve - Decision Tree depth (shows overfitting directly)
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
    dt_pipe = make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False)
    tr_s, va_s = validation_curve(dt_pipe, X_train, y_train, param_name="clf__max_depth",
                                  param_range=depths, cv=CV, scoring="accuracy", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(depths, tr_s.mean(1), "o-", label="Training", color="#4C72B0")
    ax.plot(depths, va_s.mean(1), "o-", label="Cross-validation", color="#DD8452")
    ax.fill_between(depths, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=.15, color="#DD8452")
    ax.set_xlabel("Decision Tree max_depth"); ax.set_ylabel("Accuracy")
    ax.set_title("Validation curve: Decision Tree depth")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "07_decision_tree_depth_curve.png", dpi=150); plt.close(fig)
    pd.DataFrame({"max_depth": depths, "train_acc": tr_s.mean(1), "cv_acc": va_s.mean(1),
                  "cv_std": va_s.std(1)}).round(5).to_csv(OUT_DIR / "decision_tree_depth_curve.csv", index=False)

    # (h) Validation curve - Logistic Regression C
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lr_pipe = make_pipeline(LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), scale=True)
    tr_c, va_c = validation_curve(lr_pipe, X_train, y_train, param_name="clf__C",
                                  param_range=Cs, cv=CV, scoring="accuracy", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogx(Cs, tr_c.mean(1), "o-", label="Training", color="#4C72B0")
    ax.semilogx(Cs, va_c.mean(1), "o-", label="Cross-validation", color="#DD8452")
    ax.set_xlabel("Logistic Regression C (inverse regularisation)"); ax.set_ylabel("Accuracy")
    ax.set_title("Validation curve: Logistic Regression C"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "08_logreg_C_curve.png", dpi=150); plt.close(fig)

    # (i) Random-forest feature importance (top 15) + LR coefficients
    rf = tuned["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    imp.round(5).to_csv(OUT_DIR / "random_forest_feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(15)[::-1].plot.barh(ax=ax, color=COLORS["Random Forest"])
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Random Forest - top 15 features"); fig.tight_layout()
    fig.savefig(FIG_DIR / "09_rf_feature_importance.png", dpi=150); plt.close(fig)

    # ---------------- sensitivity experiments --------------------------------
    # S1: remove the single strongest feature (re-fit tuned pipelines)
    s1_rows = []
    keep = [c for c in feature_cols if c != STRONGEST_SINGLE]
    for name, pipe in tuned.items():
        p = clone(pipe).fit(X_train[keep], y_train)
        m, _ = evaluate(p, X_test[keep], y_test)
        s1_rows.append({"Setting": f"All features minus {STRONGEST_SINGLE}", "Model": name, **m})

    # S2: URL-only features, full re-tuning (the realistic "just a URL string" case)
    print("\n--- Sensitivity S2: URL-only features (%d) ---" % len(URL_ONLY_FEATURES))
    Xtr_u, Xte_u = X_train[URL_ONLY_FEATURES], X_test[URL_ONLY_FEATURES]
    base_u = {n: clone(p) for n, p in baselines.items()}
    s2_rows, s2_tuned_rows = [], []
    for n, p in base_u.items():
        p.fit(Xtr_u, y_train)
        tr, _ = evaluate(p, Xtr_u, y_train); te, _ = evaluate(p, Xte_u, y_test)
        s2_rows.append({"Setting": "URL-only features", "Model": n, "Version": "Baseline (default)",
                        "Train Accuracy": tr["Accuracy"], "Test Accuracy": te["Accuracy"],
                        "Test Precision": te["Precision"], "Test Recall": te["Recall"],
                        "Test F1-score": te["F1-score"]})
    tuned_u, best_params_u, _ = tune_models(base_u, param_grids, Xtr_u, y_train)
    cm_u, cv_u_rows, roc_u = {}, [], {}
    for n, p in tuned_u.items():
        tr, _ = evaluate(p, Xtr_u, y_train); te, pr = evaluate(p, Xte_u, y_test)
        cm_u[n] = confusion_matrix(y_test, pr, labels=[POS, 1])
        sc = phishing_score(p, Xte_u)
        fpr, tpr, _ = roc_curve((y_test == POS).astype(int), sc); roc_u[n] = (fpr, tpr, auc(fpr, tpr))
        s2_rows.append({"Setting": "URL-only features", "Model": n, "Version": "Tuned",
                        "Train Accuracy": tr["Accuracy"], "Test Accuracy": te["Accuracy"],
                        "Test Precision": te["Precision"], "Test Recall": te["Recall"],
                        "Test F1-score": te["F1-score"], "Test ROC-AUC": roc_u[n][2]})
        cvr = cross_validate(p, Xtr_u, y_train, cv=CV, n_jobs=-1, scoring={"f1": PHISH_F1})
        cv_u_rows.append({"Model": n, "CV val F1 (mean)": cvr["test_f1"].mean(),
                          "CV val F1 (std)": cvr["test_f1"].std()})
    s2 = pd.DataFrame(s2_rows)
    s2["Train-Test accuracy gap"] = s2["Train Accuracy"] - s2["Test Accuracy"]
    cv_u = pd.DataFrame(cv_u_rows)
    best_u = cv_u.sort_values("CV val F1 (mean)", ascending=False).iloc[0]["Model"]
    s1 = pd.DataFrame(s1_rows)
    s1.round(5).to_csv(OUT_DIR / "sensitivity_S1_without_URLSimilarityIndex.csv", index=False)
    s2.round(5).to_csv(OUT_DIR / "sensitivity_S2_url_only_features.csv", index=False)
    cv_u.round(5).to_csv(OUT_DIR / "sensitivity_S2_cv_results.csv", index=False)
    with open(OUT_DIR / "sensitivity_S2_best_hyperparameters.json", "w") as f:
        json.dump(best_params_u, f, indent=2, default=str)
    cm_u_df = pd.concat({n: pd.DataFrame(m, index=[f"True {c}" for c in CLASS_NAMES],
                                         columns=[f"Pred {c}" for c in CLASS_NAMES])
                         for n, m in cm_u.items()})
    cm_u_df.to_csv(OUT_DIR / "sensitivity_S2_confusion_matrices.csv")

    # graph 10: confusion matrices, URL-only
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, n in zip(axes, names):
        sns.heatmap(cm_u[n], annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
        ax.set_title(n); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices - URL-only features, tuned (test set)", fontsize=14)
    fig.tight_layout(); fig.savefig(FIG_DIR / "10_url_only_confusion_matrices.png", dpi=150); plt.close(fig)

    # graph 11: test F1 across the three feature settings
    s2t = s2[s2.Version == "Tuned"].set_index("Model")
    s1i = s1.set_index("Model")
    fig, ax = plt.subplots(figsize=(9, 5))
    settings = [("All 50 features", [tuned_res.loc[n, "Test F1-score"] for n in names]),
                (f"Minus {STRONGEST_SINGLE}", [s1i.loc[n, "F1-score"] for n in names]),
                ("URL-only (21 features)", [s2t.loc[n, "Test F1-score"] for n in names])]
    for i, (lab, vals) in enumerate(settings):
        bars = ax.bar(np.arange(3) + (i - 1) * 0.26, vals, 0.26, label=lab,
                      color=["#2E7D32", "#7CB342", "#F9A825"][i])
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(names); ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Test F1-score (phishing)"); legend_below(ax)
    ax.set_title("Model performance under three feature settings")
    fig.tight_layout(); fig.savefig(FIG_DIR / "11_feature_setting_comparison.png", dpi=150); plt.close(fig)

    # graph 12: URL-only train vs test (overfitting visible here)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r['Model']}\n({'baseline' if r['Version'].startswith('Base') else 'tuned'})" for _, r in s2.iterrows()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, s2["Train Accuracy"], 0.4, label="Train accuracy", color="#4C72B0")
    ax.bar(x + 0.2, s2["Test Accuracy"], 0.4, label="Test accuracy", color="#DD8452")
    for i, (a_, b_) in enumerate(zip(s2["Train Accuracy"], s2["Test Accuracy"])):
        ax.text(i - 0.2, a_ + 0.002, f"{a_:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b_ + 0.002, f"{b_:.3f}", ha="center", fontsize=8)
    ax.set_ylim(s2["Test Accuracy"].min() - 0.03, 1.02)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("Accuracy")
    ax.set_title("Overfitting check with URL-only features: training vs test accuracy")
    legend_below(ax, 2); fig.tight_layout()
    fig.savefig(FIG_DIR / "12_url_only_train_vs_test.png", dpi=150); plt.close(fig)

    # secondary deployable model for URL-only deployments
    joblib.dump(tuned_u[best_u], MODEL_DIR / "phishing_model_url_only_pipeline.joblib")
    with open(MODEL_DIR / "feature_columns_url_only.json", "w") as f:
        json.dump(URL_ONLY_FEATURES, f, indent=2)
    sens = s1  # kept for console summary

    # ---------------- save selected model + preprocessing --------------------
    best_pipe = tuned[best_name]
    joblib.dump(best_pipe, MODEL_DIR / "phishing_model_pipeline.joblib")      # imputer + (scaler) + classifier
    joblib.dump(best_pipe.named_steps["imputer"], MODEL_DIR / "preprocessing_imputer.joblib")
    if "scaler" in best_pipe.named_steps:
        joblib.dump(best_pipe.named_steps["scaler"], MODEL_DIR / "preprocessing_scaler.joblib")
    joblib.dump(best_pipe.named_steps["clf"], MODEL_DIR / "classifier_only.joblib")
    meta = {
        "selected_model": best_name,
        "selection_rule": "highest mean 5-fold CV F1 (phishing) on the training split",
        "hyperparameters": best_params[best_name],
        "feature_columns": feature_cols,
        "label_mapping": {"0": "Phishing", "1": "Legitimate"},
        "class_order_in_model": [int(c) for c in best_pipe.classes_],
        "test_metrics": {k.replace("Test ", ""): round(float(v), 5)
                         for k, v in tuned_res.loc[best_name].items() if str(k).startswith("Test ")},
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "data_summary": data_log,
    }
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # also keep a JSON of everything the report needs
    summary = {"data": data_log, "best_model": best_name, "best_params": best_params,
               "learning_curve_final": lc_summary, "url_only_best": best_u,
               "url_only_best_params": best_params_u, "top_features": imp.head(10).round(4).to_dict(),
               "seconds": round(time.time() - t0, 1)}
    with open(OUT_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---------------- console summary ---------------------------------------
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\n=== Comparison table ===")
    print(results.round(4).to_string(index=False))
    print("\n=== Cross-validation ===")
    print(cv_df.round(4).to_string(index=False))
    print("\n=== S1: without", STRONGEST_SINGLE, "===")
    print(s1.round(4).to_string(index=False))
    print("\n=== S2: URL-only features ===")
    print(s2.round(4).to_string(index=False)); print(cv_u.round(4).to_string(index=False))
    print("S2 best:", best_u, best_params_u[best_u])
    print("\nTop features:\n", imp.head(10).round(4))
    print(f"\nDone in {time.time() - t0:.0f}s. Selected: {best_name}")


if __name__ == "__main__":
    main()
