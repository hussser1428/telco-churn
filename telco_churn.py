"""Reproducible churn analysis: python telco_churn.py."""
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, roc_auc_score, PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SEED = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def prepare_features(frame):
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    output = frame[FEATURES].copy()
    for column in NUMERIC:
        raw = output[column].astype("string").str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & raw.ne("") & numeric.isna()
        if invalid.any():
            raise ValueError(f"Invalid numeric values in {column}")
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite numeric values in {column}")
        output[column] = numeric.astype(float)
    for column in CATEGORICAL:
        # Explicit object dtype / np.nan also works across pandas string versions.
        values = output[column].astype("string").str.strip().replace("", pd.NA)
        output[column] = values.astype(object).where(values.notna(), np.nan)
    return output


def load_data(path=DATA_PATH):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["customerID"].isna().any() or frame["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique")
    if frame["Churn"].isna().any() or set(frame["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Expected both Yes and No churn labels")
    features = prepare_features(frame)
    target = frame["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    audit = {
        "rows": len(frame), "raw_columns": len(frame.columns), "features": len(FEATURES),
        "duplicate_customer_ids": int(frame["customerID"].duplicated().sum()),
        "missing_numeric": {c: int(features[c].isna().sum()) for c in NUMERIC},
        "churn_counts": {str(k): int(v) for k, v in target.value_counts().items()},
        "churn_rate": float(target.mean()),
    }
    return frame, features, target, audit


def make_pipeline(classifier):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def metrics_for(target, probability, threshold):
    predictions = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
    }


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_eda(frame, target, figures, reports):
    analysis = frame.assign(churn_flag=target)
    contract = analysis.groupby("Contract")["churn_flag"].agg(customers="size", churn_rate="mean")
    contract.to_csv(reports / "churn_by_contract.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    target.map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=axes[0], color=["#2563eb", "#f97316"], rot=0)
    axes[0].set(title="Customer outcome", ylabel="Customers")
    (contract["churn_rate"] * 100).plot.bar(ax=axes[1], color="#2563eb", rot=15)
    axes[1].set(title="Churn rate by contract", ylabel="Churn (%)")
    save_plot(fig, figures / "01_churn_and_contract.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, name, color in [(0, "Stayed", "#2563eb"), (1, "Churned", "#f97316")]:
        subset = frame.loc[target == label]
        axes[0].hist(subset["tenure"], bins=np.arange(0, 79, 6), density=True, alpha=.55, label=name, color=color)
        axes[1].hist(subset["MonthlyCharges"], bins=np.arange(15, 126, 10), density=True, alpha=.55, label=name, color=color)
    axes[0].set(title="Tenure distribution", xlabel="Months", ylabel="Density")
    axes[1].set(title="Monthly charge distribution", xlabel="Monthly charges (dataset units)", ylabel="Density")
    for ax in axes:
        ax.legend()
    save_plot(fig, figures / "02_tenure_and_charges.png")
    return contract


def write_report(audit, sizes, validation, test, selected, matrix, contract, source, reports):
    best = test.loc[test.model == selected].iloc[0]
    table = "| Mô hình | Ngưỡng | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for row in test.itertuples(index=False):
        table += f"| {row.model} | {row.threshold:.2f} | {row.accuracy:.3f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.roc_auc:.3f} | {row.average_precision:.3f} |\n"
    val_lines = "\n".join(f"- {r.model}: F1 = {r.f1:.3f}; ngưỡng = {r.threshold:.2f}." for r in validation.itertuples(index=False))
    contract_lines = "\n".join(f"- {name}: {int(row.customers):,} khách hàng, tỷ lệ churn {row.churn_rate:.1%}." for name, row in contract.iterrows())
    tn, fp, fn, tp = matrix.ravel()
    text = f"""# Báo cáo Telco Customer Churn

## Dữ liệu và mục tiêu

Phân loại nhãn `Churn = Yes/No` của khách hàng viễn thông. Nhãn cho biết khách hàng đã rời dịch vụ trong tháng gần nhất; đây là bài thực hành phân loại trên dữ liệu mẫu, chưa chứng minh khả năng dự báo tương lai.

- Nguồn thực tế: {source['publisher']} — {source['download_url']}
- Trang Kaggle: {source['dataset_page']}
- SHA-256: `{source['sha256']}`
- {audit['rows']:,} dòng, {audit['raw_columns']} cột gốc; dùng {audit['features']} đặc trưng sau khi bỏ `customerID` và `Churn`.
- Tỷ lệ churn: {audit['churn_rate']:.2%}; mã khách hàng trùng: {audit['duplicate_customer_ids']}.
- `TotalCharges`: {audit['missing_numeric']['TotalCharges']} giá trị thiếu sau chuyển số; điền bằng trung vị học từ train.

## Phân tích mô tả

{contract_lines}

![Churn và hợp đồng](figures/01_churn_and_contract.png)

![Thời gian sử dụng và chi phí](figures/02_tenure_and_charges.png)

Các biểu đồ dùng toàn bộ dữ liệu cho mô tả. Chúng thể hiện liên hệ, không xác lập nguyên nhân khách hàng rời đi.

## Phương pháp

Chia stratified với seed 42: train {sizes['train']:,}, validation {sizes['validation']:,}, test {sizes['test']:,}. Các tập không trùng khách hàng. Điền thiếu, chuẩn hóa số và one-hot encoding được fit chỉ trên train, giữ nguyên khi xử lý validation/test.

Baseline luôn dự đoán không churn. Logistic Regression (`max_iter=2000`) và Random Forest (300 cây, `min_samples_leaf=3`, `class_weight=balanced_subsample`) dùng cùng dữ liệu train. Tìm ngưỡng từ 0,10 đến 0,80, bước 0,01, bằng F1 validation; nếu hòa, ưu tiên gần 0,50. Chọn mô hình chỉ bằng F1 validation, không thay đổi lựa chọn theo test. Không refit bằng validation để giữ đúng mô hình và ngưỡng đã đánh giá.

### Kết quả validation

{val_lines}

Mô hình được chọn: **{selected}**.

### Kết quả test

{table}

Accuracy có thể cao dù mô hình bỏ sót mọi ca churn, vì đa số khách hàng không churn. Precision là tỷ lệ churn thật trong nhóm bị cảnh báo; recall là tỷ lệ ca churn mô hình phát hiện được; F1 cân bằng hai giá trị này. AP tóm tắt đường precision–recall, baseline xấp xỉ tỷ lệ churn.

Mô hình được chọn: recall {best['recall']:.1%}, precision {best['precision']:.1%}, F1 {best['f1']:.3f}. Ma trận nhầm lẫn: TN={tn}, FP={fp}, FN={fn}, TP={tp}; còn bỏ sót {fn} khách hàng churn và cảnh báo nhầm {fp} khách hàng ở lại.

![Ma trận nhầm lẫn](figures/03_confusion_matrix.png)

![Precision recall](figures/04_precision_recall.png)

## Cách sử dụng và giới hạn

`models/churn_pipeline.joblib` chứa preprocessing và classifier; `models/model_metadata.json` chứa ngưỡng và schema. Chạy `python predict.py --input du_lieu_moi.csv --output du_doan.csv` với CSV có đủ 19 cột đặc trưng. Kết quả gồm xác suất và nhãn suy luận; xác suất chưa được hiệu chỉnh calibration.

Đây là dữ liệu mẫu cho học tập, không phải đánh giá trên khách hàng của doanh nghiệp đang hoạt động. Dữ liệu không có mốc thời gian để chia theo thời gian hoặc biết chính xác đặc trưng được ghi trước churn. Chỉ cân nhắc vận hành sau khi có dữ liệu thực tế, định nghĩa cửa sổ dự báo, kiểm tra theo thời gian và chi phí cảnh báo/bỏ sót. Chưa có thử nghiệm cho thấy việc giữ chân dựa trên mô hình làm giảm churn.
"""
    (reports / "REPORT.md").write_text(text, encoding="utf-8")


def run_project(root=ROOT):
    root = Path(root)
    reports, figures, models = root / "reports", root / "reports" / "figures", root / "models"
    for folder in [reports, figures, models]:
        folder.mkdir(parents=True, exist_ok=True)
    frame, features, target, audit = load_data(root / "data" / DATA_PATH.name)
    source = json.loads((root / "data" / "source.json").read_text(encoding="utf-8"))
    contract = create_eda(frame, target, figures, reports)
    train_val, test_ids = train_test_split(frame.index, test_size=.2, stratify=target, random_state=SEED)
    train_ids, val_ids = train_test_split(train_val, test_size=.25, stratify=target.loc[train_val], random_state=SEED)
    splits = {"train": train_ids, "validation": val_ids, "test": test_ids}
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        raise AssertionError("Split overlap")
    if set(train_ids) | set(val_ids) | set(test_ids) != set(frame.index):
        raise AssertionError("Split coverage mismatch")
    split_frame = pd.concat([pd.DataFrame({"customerID": frame.loc[ids, "customerID"].values, "split": name}) for name, ids in splits.items()], ignore_index=True)
    split_frame.to_csv(reports / "splits.csv", index=False)
    sizes = {name: len(ids) for name, ids in splits.items()}
    classifiers = {
        "Baseline_NoChurn": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }
    fitted, thresholds, validation_rows, scans = {}, {}, [], []
    for name, classifier in classifiers.items():
        print(f"Training {name} ...", flush=True)
        pipeline = make_pipeline(classifier)
        pipeline.fit(features.loc[train_ids], target.loc[train_ids])
        probability = pipeline.predict_proba(features.loc[val_ids])[:, 1]
        if name == "Baseline_NoChurn":
            threshold = .5
        else:
            candidates = []
            for threshold in np.round(np.arange(.10, .801, .01), 2):
                score = f1_score(target.loc[val_ids], probability >= threshold, zero_division=0)
                candidates.append((float(score), float(threshold)))
                scans.append({"model": name, "threshold": float(threshold), "validation_f1": float(score)})
            _, threshold = max(candidates, key=lambda item: (item[0], -abs(item[1] - .5)))
        fitted[name], thresholds[name] = pipeline, threshold
        validation_rows.append({"model": name, **metrics_for(target.loc[val_ids], probability, threshold)})
    validation = pd.DataFrame(validation_rows)
    selected = validation.sort_values("f1", ascending=False, kind="stable").iloc[0]["model"]
    print(f"Locked selection from validation: {selected}", flush=True)
    # Only after model + thresholds are locked do we use test labels for evaluation.
    test_rows, test_probabilities = [], {}
    for name, pipeline in fitted.items():
        probability = pipeline.predict_proba(features.loc[test_ids])[:, 1]
        test_probabilities[name] = probability
        test_rows.append({"model": name, **metrics_for(target.loc[test_ids], probability, thresholds[name])})
    test = pd.DataFrame(test_rows)
    validation.to_csv(reports / "validation_metrics.csv", index=False)
    test.to_csv(reports / "test_metrics.csv", index=False)
    pd.DataFrame(scans).to_csv(reports / "threshold_selection.csv", index=False)
    chosen_probability = test_probabilities[selected]
    chosen_prediction = (chosen_probability >= thresholds[selected]).astype(int)
    matrix = confusion_matrix(target.loc[test_ids], chosen_prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=["Stayed", "Churned"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{selected}: test set")
    save_plot(fig, figures / "03_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["LogisticRegression", "RandomForest"]:
        PrecisionRecallDisplay.from_predictions(target.loc[test_ids], test_probabilities[name], name=name, ax=ax)
    ax.axhline(target.loc[test_ids].mean(), linestyle="--", color="gray", label="Churn prevalence")
    ax.set_title("Precision-recall on held-out test")
    ax.legend()
    save_plot(fig, figures / "04_precision_recall.png")
    predictions = pd.DataFrame({
        "customerID": frame.loc[test_ids, "customerID"].values,
        "actual_churn": target.loc[test_ids].values,
        "churn_probability": chosen_probability, "predicted_churn": chosen_prediction,
    })
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    joblib.dump(fitted[selected], models / "churn_pipeline.joblib")
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    metadata = {"selected_model": selected, "threshold": thresholds[selected], "feature_columns": FEATURES,
                "seed": SEED, "source_sha256": source["sha256"], "versions": versions,
                "selection_rule": "Highest validation F1 with threshold selected on validation only"}
    (models / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"audit": audit, "split_sizes": sizes, "selected_model": selected, "threshold": thresholds[selected],
               "validation": validation_rows, "test": test_rows, "confusion_matrix": matrix.tolist(), "versions": versions}
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(audit, sizes, validation, test, selected, matrix, contract, source, reports)
    print(test.round(4).to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    run_project()
