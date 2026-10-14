"""Predict a CSV with the saved pipeline and locked validation threshold."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from telco_churn import ROOT, prepare_features


def predict(input_path, output_path, model_dir=ROOT / "models"):
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / "churn_pipeline.joblib")
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError("Input CSV is empty")
    features = prepare_features(frame)
    probability = pipeline.predict_proba(features)[:, 1]
    output = pd.DataFrame({
        "customerID": frame["customerID"] if "customerID" in frame else np.arange(len(frame)),
        "churn_probability": probability,
        "predicted_churn": (probability >= metadata["threshold"]).astype(int),
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = predict(args.input, args.output)
    print(f"Saved predictions for {len(result):,} customers to {args.output}")
