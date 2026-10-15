"""
Unit and integration tests for Crop Yield model serialization and reusability.
"""

import os
import pytest
import numpy as np
import pandas as pd
from crop_yield_predictor import (
    load_model,
    predict_yield,
    prepare_input_data,
    CropYieldPredictor,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

MODEL_PATH = "model/crop_yield_model.joblib"
METADATA_PATH = "model/model_metadata.joblib"


def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}"
    assert os.path.getsize(MODEL_PATH) > 0, "Model artifact is empty"


def test_metadata_artifact_exists():
    assert os.path.exists(METADATA_PATH), f"Metadata artifact not found at {METADATA_PATH}"


def test_model_loading():
    pipeline = load_model(MODEL_PATH)
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_single_prediction():
    sample = {
        'crop': 'Rice',
        'year': 2021,
        'season': 'Kharif',
        'state': 'Tamil Nadu',
        'area': 1500.0,
        'fertilizer': 180000.0,
        'pesticide': 450.0
    }
    predictor = CropYieldPredictor(MODEL_PATH, METADATA_PATH)
    res = predictor.predict(sample)
    assert isinstance(res, pd.DataFrame)
    assert 'predicted_yield' in res.columns
    assert len(res) == 1
    assert res['predicted_yield'].iloc[0] > 0


def test_batch_prediction():
    samples = [
        {'crop': 'Wheat', 'year': 2022, 'season': 'Rabi', 'state': 'Punjab', 'area': 2500.0, 'fertilizer': 320000.0, 'pesticide': 600.0},
        {'crop': 'Cotton(lint)', 'year': 2022, 'season': 'Kharif', 'state': 'Gujarat', 'area': 3400.0, 'fertilizer': 410000.0, 'pesticide': 850.0}
    ]
    predictor = CropYieldPredictor(MODEL_PATH, METADATA_PATH)
    res = predictor.predict(samples)
    assert len(res) == 2
    assert all(res['predicted_yield'] > 0)


def test_unseen_category_handling():
    # Robustness check: unseen crop and state should not throw exceptions due to handle_unknown='ignore'
    sample = {
        'crop': 'NonExistentExoticCrop',
        'year': 2025,
        'season': 'Kharif',
        'state': 'NonExistentState',
        'area': 1000.0,
        'fertilizer': 100000.0,
        'pesticide': 200.0
    }
    predictor = CropYieldPredictor(MODEL_PATH, METADATA_PATH)
    res = predictor.predict(sample)
    assert len(res) == 1
    assert not np.isnan(res['predicted_yield'].iloc[0])


def test_target_leakage_removal():
    sample = {
        'crop': 'Rice',
        'year': 2021,
        'season': 'Kharif',
        'state': 'Tamil Nadu',
        'area': 1500.0,
        'production': 999999,  # Target leakage column
        'fertilizer': 180000.0,
        'pesticide': 450.0
    }
    cleaned_df = prepare_input_data(sample)
    assert 'production' not in cleaned_df.columns
    assert list(cleaned_df.columns) == FEATURE_COLUMNS
