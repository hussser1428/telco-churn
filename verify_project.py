"""Verify data provenance, split integrity, preprocessing and saved inference."""
from hashlib import sha256
import json
from pathlib import Path
import joblib
import nbformat
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from predict import predict
from telco_churn import ROOT, DATA_PATH, NUMERIC, load_data, metrics_for, prepare_features


def verify():
    raw, features, target, audit = load_data()
    source = json.loads((ROOT / 'data' / 'source.json').read_text(encoding='utf-8'))
    assert sha256(DATA_PATH.read_bytes()).hexdigest() == source['sha256']
    assert (audit['rows'], audit['raw_columns']) == (7043, 21)
    splits = pd.read_csv(ROOT / 'reports' / 'splits.csv')
    assert len(splits) == len(raw) and splits.customerID.is_unique
    assert set(splits.customerID) == set(raw.customerID)
    assert set(splits['split']) == {'train', 'validation', 'test'}
    indexed = raw.set_index('customerID')
    train_ids = splits.loc[splits['split'] == 'train', 'customerID']
    train_features = prepare_features(indexed.loc[train_ids])
    metadata = json.loads((ROOT / 'models' / 'model_metadata.json').read_text(encoding='utf-8'))
    model = joblib.load(ROOT / 'models' / 'churn_pipeline.joblib')
    numeric_pipeline = model.named_steps['preprocess'].named_transformers_['numeric']
    assert_allclose(numeric_pipeline.named_steps['imputer'].statistics_, train_features[NUMERIC].median().values)
    imputed = numeric_pipeline.named_steps['imputer'].transform(train_features[NUMERIC])
    assert_allclose(numeric_pipeline.named_steps['scaler'].mean_, imputed.mean(axis=0))
    val = pd.read_csv(ROOT / 'reports' / 'validation_metrics.csv')
    chosen = val.sort_values('f1', ascending=False, kind='stable').iloc[0]
    assert chosen['model'] == metadata['selected_model']
    assert_allclose(chosen['threshold'], metadata['threshold'])
    stored = pd.read_csv(ROOT / 'reports' / 'test_predictions.csv')
    test_ids = splits.loc[splits['split'] == 'test', 'customerID']
    assert stored.customerID.tolist() == test_ids.tolist()
    test_features = prepare_features(indexed.loc[test_ids])
    probability = model.predict_proba(test_features)[:, 1]
    assert_allclose(stored.churn_probability, probability, rtol=1e-12, atol=1e-12)
    assert np.array_equal(stored.predicted_churn, probability >= metadata['threshold'])
    assert np.array_equal(stored.actual_churn, indexed.loc[test_ids, 'Churn'].eq('Yes').astype(int))
    metrics = pd.read_csv(ROOT / 'reports' / 'test_metrics.csv').set_index('model').loc[metadata['selected_model']]
    recomputed = metrics_for(stored.actual_churn, probability, metadata['threshold'])
    for key, value in recomputed.items():
        assert_allclose(metrics[key], value, rtol=1e-12, atol=1e-12)
    example = pd.read_csv(ROOT / 'data' / 'example_customers.csv')
    output = predict(ROOT / 'data' / 'example_customers.csv', ROOT / 'reports' / 'example_predictions.csv')
    expected = stored.set_index('customerID').loc[example.customerID]
    assert_allclose(output.churn_probability, expected.churn_probability.values, rtol=1e-12, atol=1e-12)
    # Guardrails must reject missing columns and corrupted numeric values.
    for bad in [example.drop(columns='tenure'), example.assign(tenure='not-a-number')]:
        try:
            prepare_features(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input was accepted')
    notebook = nbformat.read(ROOT / 'telco_churn.ipynb', as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
    for path in (ROOT / 'reports' / 'figures').glob('*.png'):
        assert path.stat().st_size > 1000
    checks = {
        'source_checksum': 'passed', 'complete_disjoint_customer_splits': 'passed',
        'preprocessing_fit_on_train_only': 'passed', 'selection_uses_validation': 'passed',
        'saved_pipeline_matches_test_predictions': 'passed', 'test_metrics_recomputed': 'passed',
        'csv_inference_matches_saved_pipeline': 'passed', 'invalid_inputs_rejected': 'passed',
        'notebook_all_cells_executed_without_errors': 'passed', 'figures_exist': 'passed',
    }
    (ROOT / 'reports' / 'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    verify()
