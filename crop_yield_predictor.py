"""
Crop Yield Prediction: Reusable ML Pipeline & Inference Module
--------------------------------------------------------------
This module provides reusable components for building, training, serializing,
loading, and serving crop yield prediction models.

Features:
- Encapsulates preprocessing and modeling in an end-to-end scikit-learn Pipeline.
- Seamless model persistence (serialization & deserialization) via joblib.
- Flexible inference accepting single dictionaries, lists of dictionaries, or DataFrames.
- Robust handling of missing values, string whitespace, and unseen categories.
"""

import os
import json
import logging
from datetime import datetime
from typing import Union, List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature definitions
CATEGORICAL_FEATURES = ['crop', 'season', 'state']
NUMERICAL_FEATURES = ['year', 'area', 'fertilizer', 'pesticide']
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'yield'
LEAKAGE_COLUMNS = ['production']


def create_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for preprocessing agricultural data.
    - Numerical features: Median imputation + Standard scaling.
    - Categorical features: Most frequent imputation + One-Hot Encoding (ignoring unknown categories).
    """
    num_cols = numerical_features or NUMERICAL_FEATURES
    cat_cols = categorical_features or CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    return preprocessor


def build_crop_yield_pipeline(model: Optional[Any] = None) -> Pipeline:
    """
    Builds an end-to-end scikit-learn pipeline combining preprocessing and regression model.
    Defaults to RandomForestRegressor with 100 estimators.
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    preprocessor = create_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def save_model(
    pipeline: Pipeline,
    filepath: str = "model/crop_yield_model.joblib",
    metadata: Optional[Dict[str, Any]] = None,
    metadata_filepath: Optional[str] = "model/model_metadata.joblib"
) -> None:
    """
    Serializes the complete trained pipeline and optional metadata to disk using joblib.
    """
    target_dir = os.path.dirname(filepath)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    joblib.dump(pipeline, filepath, compress=3)
    logger.info(f"Model pipeline successfully serialized to: {filepath}")

    if metadata is not None and metadata_filepath:
        meta_dir = os.path.dirname(metadata_filepath)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)
        joblib.dump(metadata, metadata_filepath)
        # Also save as human-readable JSON if serializable
        json_path = metadata_filepath.replace('.joblib', '.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Could not dump json metadata: {e}")
        logger.info(f"Model metadata successfully serialized to: {metadata_filepath}")


def load_model(filepath: str = "model/crop_yield_model.joblib") -> Pipeline:
    """
    Deserializes and returns the trained pipeline from disk using joblib.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")
    pipeline = joblib.load(filepath)
    logger.info(f"Model pipeline loaded successfully from: {filepath}")
    return pipeline


def prepare_input_data(data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
    """
    Validates and standardizes raw input data into a clean DataFrame ready for inference.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input data must be a dictionary, list of dictionaries, or pandas DataFrame.")

    # Remove target leakage if accidentally included
    for col in LEAKAGE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure required columns are present
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}. Required: {FEATURE_COLUMNS}")

    # Strip whitespace from string/categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ensure correct feature ordering
    return df[FEATURE_COLUMNS]


def predict_yield(
    pipeline: Pipeline,
    data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
) -> np.ndarray:
    """
    Generates crop yield predictions for raw input data using the trained pipeline.
    """
    clean_df = prepare_input_data(data)
    predictions = pipeline.predict(clean_df)
    return predictions


class CropYieldPredictor:
    """
    High-level, reusable inference class for production crop yield prediction.
    """
    def __init__(
        self,
        model_path: str = "model/crop_yield_model.joblib",
        metadata_path: Optional[str] = "model/model_metadata.joblib"
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline: Pipeline = load_model(model_path)
        self.metadata: Optional[Dict[str, Any]] = None
        if metadata_path and os.path.exists(metadata_path):
            try:
                self.metadata = joblib.load(metadata_path)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

    def predict(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        include_input: bool = True
    ) -> pd.DataFrame:
        """
        Runs inference on the provided data and returns a DataFrame with inputs and predicted yield.
        """
        clean_df = prepare_input_data(data)
        preds = self.pipeline.predict(clean_df)

        if include_input:
            result = clean_df.copy()
            result['predicted_yield'] = preds
            return result
        return pd.DataFrame({'predicted_yield': preds})

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts top feature importances from the underlying Random Forest model.
        """
        model = self.pipeline.named_steps.get('model')
        preprocessor = self.pipeline.named_steps.get('preprocessor')
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError("The underlying model does not provide feature importances.")

        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features_encoded = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        all_feature_names = NUMERICAL_FEATURES + list(cat_features_encoded)

        importances = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        return df_imp.sort_values(by='Importance', ascending=False).head(top_n).reset_index(drop=True)
