"""
Multi-Disease Prediction System
Diseases: Diabetes, Heart Disease, Breast Cancer
Models: Logistic Regression, Random Forest, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score,
    confusion_matrix, classification_report, accuracy_score
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERATE SYNTHETIC DATASETS
# ══════════════════════════════════════════════════════════════════════════════
def generate_diabetes(n=768, seed=42):
    np.random.seed(seed)
    n_pos = int(n * 0.35)
    n_neg = n - n_pos

    neg = {
        "Pregnancies":      np.random.poisson(3, n_neg),
        "Glucose":          np.random.normal(109, 26, n_neg).clip(0),
        "BloodPressure":    np.random.normal(70, 12, n_neg).clip(0),
        "SkinThickness":    np.random.normal(20, 15, n_neg).clip(0),
        "Insulin":          np.random.normal(79, 115, n_neg).clip(0),
        "BMI":              np.random.normal(30, 7, n_neg).clip(0),
        "DiabetesPedigree": np.random.exponential(0.4, n_neg).clip(0.08, 2.4),
        "Age":              np.random.normal(31, 11, n_neg).clip(18, 80),
    }
    pos = {
        "Pregnancies":      np.random.poisson(5, n_pos),
        "Glucose":          np.random.normal(141, 31, n_pos).clip(0),
        "BloodPressure":    np.random.normal(75, 14, n_pos).clip(0),
        "SkinThickness":    np.random.normal(33, 18, n_pos).clip(0),
        "Insulin":          np.random.normal(145, 138, n_pos).clip(0),
        "BMI":              np.random.normal(35, 7, n_pos).clip(0),
        "DiabetesPedigree": np.random.exponential(0.6, n_pos).clip(0.08, 2.4),
        "Age":              np.random.normal(37, 10, n_pos).clip(18, 80),
    }
    df_neg = pd.DataFrame(neg); df_neg["Outcome"] = 0
    df_pos = pd.DataFrame(pos); df_pos["Outcome"] = 1
    df = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Diabetes dataset: {df.shape} | Positive: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
    return df

def generate_heart(n=303, seed=42):
    np.random.seed(seed)
    n_pos = int(n * 0.54)
    n_neg = n - n_pos

    neg = {
        "age":      np.random.normal(52, 9, n_neg).clip(29, 77),
        "sex":      np.random.binomial(1, 0.7, n_neg),
        "cp":       np.random.choice([0,1,2,3], n_neg, p=[0.5,0.2,0.2,0.1]),
        "trestbps": np.random.normal(130, 17, n_neg).clip(90, 200),
        "chol":     np.random.normal(243, 52, n_neg).clip(140, 400),
        "fbs":      np.random.binomial(1, 0.15, n_neg),
        "restecg":  np.random.choice([0,1,2], n_neg, p=[0.5,0.4,0.1]),
        "thalach":  np.random.normal(158, 20, n_neg).clip(90, 202),
        "exang":    np.random.binomial(1, 0.14, n_neg),
        "oldpeak":  np.random.exponential(0.6, n_neg).clip(0, 6),
        "slope":    np.random.choice([0,1,2], n_neg, p=[0.2,0.5,0.3]),
        "ca":       np.random.choice([0,1,2,3], n_neg, p=[0.6,0.2,0.1,0.1]),
        "thal":     np.random.choice([0,1,2,3], n_neg, p=[0.1,0.1,0.7,0.1]),
    }
    pos = {
        "age":      np.random.normal(56, 8, n_pos).clip(29, 77),
        "sex":      np.random.binomial(1, 0.55, n_pos),
        "cp":       np.random.choice([0,1,2,3], n_pos, p=[0.2,0.3,0.3,0.2]),
        "trestbps": np.random.normal(134, 18, n_pos).clip(90, 200),
        "chol":     np.random.normal(251, 49, n_pos).clip(140, 400),
        "fbs":      np.random.binomial(1, 0.2, n_pos),
        "restecg":  np.random.choice([0,1,2], n_pos, p=[0.3,0.5,0.2]),
        "thalach":  np.random.normal(139, 22, n_pos).clip(90, 202),
        "exang":    np.random.binomial(1, 0.55, n_pos),
        "oldpeak":  np.random.exponential(1.5, n_pos).clip(0, 6),
        "slope":    np.random.choice([0,1,2], n_pos, p=[0.4,0.4,0.2]),
        "ca":       np.random.choice([0,1,2,3], n_pos, p=[0.3,0.3,0.2,0.2]),
        "thal":     np.random.choice([0,1,2,3], n_pos, p=[0.1,0.2,0.4,0.3]),
    }
    df_neg = pd.DataFrame(neg); df_neg["target"] = 0
    df_pos = pd.DataFrame(pos); df_pos["target"] = 1
    df = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.astype({"age":"int","trestbps":"int","chol":"int","thalach":"int"})
    print(f"Heart dataset: {df.shape} | Positive: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    return df

def generate_cancer(n=569, seed=42):
    np.random.seed(seed)
    n_ben = int(n * 0.63)
    n_mal = n - n_ben

    features = ["radius","texture","perimeter","area","smoothness",
                "compactness","concavity","concave_points","symmetry","fractal_dim"]
    ben_means = [12.1, 17.9, 78.0, 462, 0.092, 0.080, 0.046, 0.025, 0.174, 0.062]
    mal_means = [17.5, 21.6, 115.0, 978, 0.103, 0.145, 0.161, 0.088, 0.192, 0.063]
    stds      = [2.0,  4.0,  14.0,  350, 0.013, 0.053, 0.080, 0.040, 0.027, 0.007]

    ben_data = {f: np.random.normal(m, s, n_ben).clip(0)
                for f, m, s in zip(features, ben_means, stds)}
    mal_data = {f: np.random.normal(m, s, n_mal).clip(0)
                for f, m, s in zip(features, mal_means, stds)}

    df_ben = pd.DataFrame(ben_data); df_ben["diagnosis"] = 0
    df_mal = pd.DataFrame(mal_data); df_mal["diagnosis"] = 1
    df = pd.concat([df_ben, df_mal]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Cancer dataset: {df.shape} | Malignant: {df['diagnosis'].sum()} ({df['diagnosis'].mean()*100:.1f}%)")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def train_disease(name, df, target_col):
    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    features = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }

    results = []
    best_model, best_auc, best_name = None, 0, ""

    for mname, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc  = roc_auc_score(y_test, y_proba)
        f1   = f1_score(y_test, y_pred)
        acc  = accuracy_score(y_test, y_pred)
        pr   = average_precision_score(y_test, y_proba)
        rep  = classification_report(y_test, y_pred, output_dict=True)
        print(f"  {mname}: AUC={auc:.4f} | F1={f1:.4f} | Acc={acc:.4f}")
        results.append({"Model": mname, "AUC-ROC": round(auc,4),
                        "PR-AUC": round(pr,4), "F1": round(f1,4),
                        "Accuracy": round(acc,4),
                        "Precision": round(rep["1"]["precision"],4),
                        "Recall": round(rep["1"]["recall"],4)})
        if auc > best_auc:
            best_auc, best_model, best_name = auc, model, mname

    # Save best model + scaler
    tag = name.lower().replace(" ", "_")
    joblib.dump(best_model, f"models/{tag}_model.pkl")
    joblib.dump(scaler,     f"models/{tag}_scaler.pkl")
    joblib.dump(features,   f"models/{tag}_features.pkl")
    print(f"  ✅ Best: {best_name} (AUC={best_auc:.4f}) saved")

    # Save results
    df_r = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
    df_r.to_csv(f"outputs/{tag}_results.csv", index=False)

    # Plots
    plot_disease(name, tag, models, X_test, y_test, results, features)
    return df_r

# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_disease(name, tag, models, X_test, y_test, results, features):
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{name} — Model Evaluation", fontsize=15, fontweight="bold")

    # ROC curves
    for (mname, model), c in zip(models.items(), colors):
        yp = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        axes[0].plot(fpr, tpr, label=f"{mname} ({auc:.3f})", color=c, lw=2)
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Confusion matrix (best model = first after sort)
    best_model = list(models.values())[0]
    best_auc = 0
    for m in models.values():
        a = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
        if a > best_auc:
            best_auc = a; best_model = m
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                annot_kws={"size":14}, linewidths=0.5)
    axes[1].set_title("Confusion Matrix (Best Model)", fontweight="bold")
    axes[1].set_ylabel("Actual"); axes[1].set_xlabel("Predicted")

    # Feature importance
    rf = models["Random Forest"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    axes[2].barh([features[i] for i in idx][::-1], imp[idx][::-1],
                 color=plt.cm.RdYlGn(np.linspace(0.3,0.9,10)))
    axes[2].set_title("Feature Importance (RF)", fontweight="bold")
    axes[2].set_xlabel("Importance"); axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"outputs/{tag}_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Plot saved → outputs/{tag}_evaluation.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🏥 Multi-Disease Prediction System")
    print("="*50)

    diseases = [
        ("Diabetes",       generate_diabetes(), "Outcome"),
        ("Heart Disease",  generate_heart(),    "target"),
        ("Breast Cancer",  generate_cancer(),   "diagnosis"),
    ]

    all_results = {}
    for name, df, target in diseases:
        all_results[name] = train_disease(name, df, target)

    print("\n\n" + "="*50)
    print("  FINAL SUMMARY")
    print("="*50)
    for name, df_r in all_results.items():
        best = df_r.iloc[0]
        print(f"\n{name}:")
        print(f"  Best Model : {best['Model']}")
        print(f"  AUC-ROC    : {best['AUC-ROC']}")
        print(f"  F1 Score   : {best['F1']}")
        print(f"  Accuracy   : {best['Accuracy']}")

    print("\n🎉 All models trained! Check outputs/ and models/ folders.")
