"""
DataEncoder module for encoding categorical variables and scaling numerical features.

This module provides the DataEncoder class for encoding categorical variables
(transaction types, customer names) and scaling numerical features for machine learning models.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings

logger = logging.getLogger(__name__)


class DataEncoder:
    """
    DataEncoder class for encoding categorical variables and scaling numerical features.
    
    This class provides comprehensive data encoding and scaling functionality including:
    - Label encoding for categorical variables
    - One-hot encoding for categorical variables
    - Standard scaling for numerical features
    - Min-max scaling for numerical features
    - Stratified train-test splitting
    - Transformer persistence and loading
    """
    
    def __init__(self, 
                 categorical_encoding: str = 'label',
                 numerical_scaling: str = 'standard',
                 handle_unknown: str = 'ignore'):
        """
        Initialize DataEncoder with encoding and scaling options.
        
        Args:
            categorical_encoding: Type of categorical encoding ('label', 'onehot')
            numerical_scaling: Type of numerical scaling ('standard', 'minmax', 'none')
            handle_unknown: How to handle unknown categories ('ignore', 'error')
        """
        self.categorical_encoding = categorical_encoding
        self.numerical_scaling = numerical_scaling
        self.handle_unknown = handle_unknown
        
        # Store encoders and scalers for consistent transformation
        self._label_encoders = {}
        self._onehot_encoders = {}
        self._scalers = {}
        self._fitted = False
        
        # Track encoding statistics
        self.encoding_stats = {
            'categorical_features': [],
            'numerical_features': [],
            'encoded_features': [],
            'scaled_features': [],
            'original_shape': None,
            'encoded_shape': None
        }
    
    def fit_transform(self, 
                     df: pd.DataFrame, 
                     categorical_features: Optional[List[str]] = None,
                     numerical_features: Optional[List[str]] = None,
                     target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit encoders and scalers on the data and transform it.
        
        Args:
            df: DataFrame to encode and scale
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            target_column: Name of target column to exclude from encoding
            
        Returns:
            pd.DataFrame: Encoded and scaled DataFrame
        """
        logger.info("Fitting and transforming data with encoders and scalers")
        
        # Store original shape
        self.encoding_stats['original_shape'] = df.shape
        
        # Auto-detect feature types if not provided
        if categorical_features is None or numerical_features is None:
            categorical_features, numerical_features = self._auto_detect_feature_types(
                df, target_column
            )
        
        # Store feature lists
        self.encoding_stats['categorical_features'] = categorical_features
        self.encoding_stats['numerical_features'] = numerical_features
        
        # Create a copy to avoid modifying original
        df_encoded = df.copy()
        
        # Encode categorical features
        if categorical_features:
            df_encoded = self._fit_transform_categorical(df_encoded, categorical_features)
        
        # Scale numerical features
        if numerical_features and self.numerical_scaling != 'none':
            df_encoded = self._fit_transform_numerical(df_encoded, numerical_features)
        
        # Store final shape
        self.encoding_stats['encoded_shape'] = df_encoded.shape
        self._fitted = True
        
        logger.info(f"Data transformation completed: {self.encoding_stats['original_shape']} -> {self.encoding_stats['encoded_shape']}")
        return df_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted encoders and scalers.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            pd.DataFrame: Transformed DataFrame
            
        Raises:
            ValueError: If encoders haven't been fitted yet
        """
        if not self._fitted:
            raise ValueError("Encoders must be fitted before transforming data. Use fit_transform() first.")
        
        logger.info("Transforming data with fitted encoders and scalers")
        
        # Create a copy to avoid modifying original
        df_transformed = df.copy()
        
        # Transform categorical features
        if self.encoding_stats['categorical_features']:
            df_transformed = self._transform_categorical(
                df_transformed, 
                self.encoding_stats['categorical_features']
            )
        
        # Transform numerical features
        if self.encoding_stats['numerical_features'] and self.numerical_scaling != 'none':
            df_transformed = self._transform_numerical(
                df_transformed, 
                self.encoding_stats['numerical_features']
            )
        
        logger.info("Data transformation completed")
        return df_transformed
    
    def stratified_train_test_split(self, 
                                  df: pd.DataFrame, 
                                  target_column: str,
                                  test_size: float = 0.2,
                                  random_state: int = 42,
                                  encode_before_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform stratified train-test split on the data.
        
        Args:
            df: DataFrame to split
            target_column: Name of target column for stratification
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            encode_before_split: Whether to encode data before splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Performing stratified train-test split (test_size={test_size})")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode data if requested
        if encode_before_split:
            if not self._fitted:
                X = self.fit_transform(X, target_column=target_column)
            else:
                X = self.transform(X)
        
        # Perform stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Split completed: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        logger.info(f"Train fraud ratio: {y_train.mean():.4f}")
        logger.info(f"Test fraud ratio: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def encode_categorical_features(self, 
                                  df: pd.DataFrame, 
                                  categorical_features: List[str],
                                  fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using the specified encoding method.
        
        Args:
            df: DataFrame with categorical features
            categorical_features: List of categorical feature names
            fit: Whether to fit new encoders or use existing ones
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        logger.info(f"Encoding categorical features: {categorical_features}")
        
        if fit:
            return self._fit_transform_categorical(df, categorical_features)
        else:
            return self._transform_categorical(df, categorical_features)
    
    def scale_numerical_features(self, 
                               df: pd.DataFrame, 
                               numerical_features: List[str],
                               fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using the specified scaling method.
        
        Args:
            df: DataFrame with numerical features
            numerical_features: List of numerical feature names
            fit: Whether to fit new scalers or use existing ones
            
        Returns:
            pd.DataFrame: DataFrame with scaled numerical features
        """
        logger.info(f"Scaling numerical features: {numerical_features}")
        
        if self.numerical_scaling == 'none':
            logger.info("Numerical scaling disabled, returning original data")
            return df
        
        if fit:
            return self._fit_transform_numerical(df, numerical_features)
        else:
            return self._transform_numerical(df, numerical_features)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features after encoding.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Add categorical feature names
        if self.categorical_encoding == 'onehot':
            for feature in self.encoding_stats['categorical_features']:
                if feature in self._onehot_encoders:
                    encoder = self._onehot_encoders[feature]
                    if hasattr(encoder, 'get_feature_names_out'):
                        feature_names.extend(encoder.get_feature_names_out([feature]))
                    else:
                        # Fallback for older sklearn versions
                        categories = encoder.categories_[0]
                        feature_names.extend([f"{feature}_{cat}" for cat in categories])
        else:
            feature_names.extend(self.encoding_stats['categorical_features'])
        
        # Add numerical feature names
        feature_names.extend(self.encoding_stats['numerical_features'])
        
        return feature_names
    
    def save_transformers(self, filepath: str) -> None:
        """
        Save fitted transformers to disk.
        
        Args:
            filepath: Path to save transformers
        """
        if not self._fitted:
            raise ValueError("No fitted transformers to save. Fit transformers first.")
        
        transformers_data = {
            'label_encoders': self._label_encoders,
            'onehot_encoders': self._onehot_encoders,
            'scalers': self._scalers,
            'encoding_stats': self.encoding_stats,
            'categorical_encoding': self.categorical_encoding,
            'numerical_scaling': self.numerical_scaling,
            'handle_unknown': self.handle_unknown
        }
        
        joblib.dump(transformers_data, filepath)
        logger.info(f"Transformers saved to {filepath}")
    
    def load_transformers(self, filepath: str) -> None:
        """
        Load fitted transformers from disk.
        
        Args:
            filepath: Path to load transformers from
        """
        transformers_data = joblib.load(filepath)
        
        self._label_encoders = transformers_data['label_encoders']
        self._onehot_encoders = transformers_data['onehot_encoders']
        self._scalers = transformers_data['scalers']
        self.encoding_stats = transformers_data['encoding_stats']
        self.categorical_encoding = transformers_data['categorical_encoding']
        self.numerical_scaling = transformers_data['numerical_scaling']
        self.handle_unknown = transformers_data['handle_unknown']
        self._fitted = True
        
        logger.info(f"Transformers loaded from {filepath}")
    
    def get_encoding_summary(self) -> Dict[str, Any]:
        """
        Get summary of encoding operations.
        
        Returns:
            Dict containing encoding statistics
        """
        return {
            'categorical_encoding': self.categorical_encoding,
            'numerical_scaling': self.numerical_scaling,
            'categorical_features': self.encoding_stats['categorical_features'],
            'numerical_features': self.encoding_stats['numerical_features'],
            'original_shape': self.encoding_stats['original_shape'],
            'encoded_shape': self.encoding_stats['encoded_shape'],
            'fitted': self._fitted
        }
    
    # Private helper methods
    
    def _auto_detect_feature_types(self, 
                                 df: pd.DataFrame, 
                                 target_column: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Automatically detect categorical and numerical features.
        
        Args:
            df: DataFrame to analyze
            target_column: Target column to exclude
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        # Exclude target column
        columns = [col for col in df.columns if col != target_column]
        
        categorical_features = []
        numerical_features = []
        
        for col in columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                categorical_features.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if it's actually categorical (few unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
        
        logger.info(f"Auto-detected {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        return categorical_features, numerical_features
    
    def _fit_transform_categorical(self, df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """Fit and transform categorical features."""
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature not in df_encoded.columns:
                logger.warning(f"Feature {feature} not found in DataFrame, skipping")
                continue
            
            if self.categorical_encoding == 'label':
                encoder = LabelEncoder()
                df_encoded[feature] = encoder.fit_transform(df_encoded[feature].astype(str))
                self._label_encoders[feature] = encoder
                self.encoding_stats['encoded_features'].append(feature)
                
            elif self.categorical_encoding == 'onehot':
                encoder = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
                encoded_data = encoder.fit_transform(df_encoded[[feature]])
                
                # Get feature names
                if hasattr(encoder, 'get_feature_names_out'):
                    feature_names = encoder.get_feature_names_out([feature])
                else:
                    feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
                
                # Drop original feature and add encoded features
                df_encoded = df_encoded.drop(columns=[feature])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                
                self._onehot_encoders[feature] = encoder
                self.encoding_stats['encoded_features'].extend(feature_names)
        
        logger.info(f"Categorical encoding completed using {self.categorical_encoding} encoding")
        return df_encoded
    
    def _transform_categorical(self, df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature not in df_encoded.columns:
                logger.warning(f"Feature {feature} not found in DataFrame, skipping")
                continue
            
            if self.categorical_encoding == 'label' and feature in self._label_encoders:
                encoder = self._label_encoders[feature]
                # Handle unknown categories
                try:
                    df_encoded[feature] = encoder.transform(df_encoded[feature].astype(str))
                except ValueError as e:
                    if self.handle_unknown == 'ignore':
                        # Map unknown categories to a default value
                        known_classes = set(encoder.classes_)
                        df_encoded[feature] = df_encoded[feature].astype(str).apply(
                            lambda x: x if x in known_classes else encoder.classes_[0]
                        )
                        df_encoded[feature] = encoder.transform(df_encoded[feature])
                    else:
                        raise e
                        
            elif self.categorical_encoding == 'onehot' and feature in self._onehot_encoders:
                encoder = self._onehot_encoders[feature]
                encoded_data = encoder.transform(df_encoded[[feature]])
                
                # Get feature names
                if hasattr(encoder, 'get_feature_names_out'):
                    feature_names = encoder.get_feature_names_out([feature])
                else:
                    feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
                
                # Drop original feature and add encoded features
                df_encoded = df_encoded.drop(columns=[feature])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        return df_encoded
    
    def _fit_transform_numerical(self, df: pd.DataFrame, numerical_features: List[str]) -> pd.DataFrame:
        """Fit and transform numerical features."""
        df_scaled = df.copy()
        
        for feature in numerical_features:
            if feature not in df_scaled.columns:
                logger.warning(f"Feature {feature} not found in DataFrame, skipping")
                continue
            
            if self.numerical_scaling == 'standard':
                scaler = StandardScaler()
            elif self.numerical_scaling == 'minmax':
                scaler = MinMaxScaler()
            else:
                continue
            
            # Handle missing values
            feature_data = df_scaled[feature].values.reshape(-1, 1)
            if np.isnan(feature_data).any():
                logger.warning(f"Feature {feature} contains NaN values, filling with median")
                median_value = np.nanmedian(feature_data)
                feature_data = np.nan_to_num(feature_data, nan=median_value)
            
            df_scaled[feature] = scaler.fit_transform(feature_data).flatten()
            self._scalers[feature] = scaler
            self.encoding_stats['scaled_features'].append(feature)
        
        logger.info(f"Numerical scaling completed using {self.numerical_scaling} scaling")
        return df_scaled
    
    def _transform_numerical(self, df: pd.DataFrame, numerical_features: List[str]) -> pd.DataFrame:
        """Transform numerical features using fitted scalers."""
        df_scaled = df.copy()
        
        for feature in numerical_features:
            if feature not in df_scaled.columns:
                logger.warning(f"Feature {feature} not found in DataFrame, skipping")
                continue
            
            if feature in self._scalers:
                scaler = self._scalers[feature]
                
                # Handle missing values
                feature_data = df_scaled[feature].values.reshape(-1, 1)
                if np.isnan(feature_data).any():
                    logger.warning(f"Feature {feature} contains NaN values, filling with median")
                    median_value = np.nanmedian(feature_data)
                    feature_data = np.nan_to_num(feature_data, nan=median_value)
                
                df_scaled[feature] = scaler.transform(feature_data).flatten()
        
        return df_scaled


class StandardScalerWrapper:
    """
    Wrapper class for StandardScaler with additional functionality.
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Initialize StandardScaler wrapper.
        
        Args:
            with_mean: Whether to center data to mean
            with_std: Whether to scale data to unit variance
        """
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self.feature_names = None
        self.fitted = False
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'StandardScalerWrapper':
        """
        Fit the scaler to the data.
        
        Args:
            X: Data to fit scaler on
            
        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.scaler.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the data using fitted scaler.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                self.scaler.transform(X.values),
                columns=X.columns,
                index=X.index
            )
        else:
            return self.scaler.transform(X)
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit scaler and transform data in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform the scaled data.
        
        Args:
            X: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transforming data")
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                self.scaler.inverse_transform(X.values),
                columns=X.columns,
                index=X.index
            )
        else:
            return self.scaler.inverse_transform(X)


class MinMaxScalerWrapper:
    """
    Wrapper class for MinMaxScaler with additional functionality.
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize MinMaxScaler wrapper.
        
        Args:
            feature_range: Desired range of transformed data
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.feature_names = None
        self.fitted = False
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'MinMaxScalerWrapper':
        """
        Fit the scaler to the data.
        
        Args:
            X: Data to fit scaler on
            
        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.scaler.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the data using fitted scaler.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming data")
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                self.scaler.transform(X.values),
                columns=X.columns,
                index=X.index
            )
        else:
            return self.scaler.transform(X)
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit scaler and transform data in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform the scaled data.
        
        Args:
            X: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transforming data")
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                self.scaler.inverse_transform(X.values),
                columns=X.columns,
                index=X.index
            )
        else:
            return self.scaler.inverse_transform(X)