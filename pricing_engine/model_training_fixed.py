"""
AI Model Training Module for Pricing Engine System
Handles model training, evaluation, and prediction for pricing recommendations.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PricingModelTrainer:
    """AI Model Trainer for pricing predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metadata = {}
        self.confidence_interval = 0.0
        
    def load_processed_data(self, data_path: str) -> pd.DataFrame:
        """
        Load processed data from ETL pipeline
        
        Args:
            data_path (str): Path to processed data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            logger.info(f"Loading processed data from {data_path}")
            data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(data)} records for training")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable for training
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        try:
            logger.info("Preparing features for training")
            
            # Check available columns
            logger.info(f"Available columns: {list(data.columns)}")
            
            # Define target variable - use internal price as primary target
            target_column = None
            if 'price_per_m2_internal' in data.columns:
                target_column = 'price_per_m2_internal'
            elif 'price_per_m2' in data.columns:
                target_column = 'price_per_m2'
            else:
                raise ValueError("No suitable target column found in data")
            
            logger.info(f"Using target column: {target_column}")
            
            # Define feature columns - adapt to merged data structure
            potential_features = []
            
            # Size features
            if 'size_internal' in data.columns:
                potential_features.append('size_internal')
            elif 'size' in data.columns:
                potential_features.append('size')
            
            # Floor features
            if 'floor_internal' in data.columns:
                potential_features.append('floor_internal')
            elif 'floor' in data.columns:
                potential_features.append('floor')
            
            # Categorical features
            if 'city' in data.columns:
                potential_features.append('city')
            if 'property_type' in data.columns:
                potential_features.append('property_type')
            
            # Data source features
            if 'data_source_internal' in data.columns:
                potential_features.append('data_source_internal')
            elif 'data_source' in data.columns:
                potential_features.append('data_source')
            
            # Competitor price as feature
            if 'price_per_m2_competitor' in data.columns:
                potential_features.append('price_per_m2_competitor')
            
            # Select available features
            available_features = [col for col in potential_features if col in data.columns]
            logger.info(f"Available features: {available_features}")
            
            if not available_features:
                raise ValueError("No valid features found in data")
            
            # Create feature dataframe
            features_df = data[available_features].copy()
            target = data[target_column].copy()
            
            # Handle categorical variables
            categorical_features = ['city', 'property_type', 'data_source_internal', 'data_source']
            for feature in categorical_features:
                if feature in features_df.columns:
                    # Create label encoder for this feature
                    le = LabelEncoder()
                    features_df[feature] = le.fit_transform(features_df[feature].astype(str))
                    self.label_encoders[feature] = le
                    logger.info(f"Encoded categorical feature: {feature}")
            
            # Handle missing values
            features_df = features_df.fillna(features_df.mean())
            
            # Store feature columns for later use
            self.feature_columns = features_df.columns.tolist()
            
            logger.info(f"Prepared {len(features_df)} samples with {len(self.feature_columns)} features")
            return features_df, target
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_model(self, features: pd.DataFrame, target: pd.Series, model_type: str = 'xgboost') -> Dict[str, Any]:
        """
        Train the pricing prediction model
        
        Args:
            features (pd.DataFrame): Feature matrix
            target (pd.Series): Target variable
            model_type (str): Type of model to train
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            logger.info(f"Training {model_type} model")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Select and configure model
            if model_type.lower() == 'xgboost':
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
                
            elif model_type.lower() == 'randomforest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
                
            elif model_type.lower() == 'gradientboosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
                
            else:  # Default to Linear Regression
                self.model = LinearRegression()
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train_final, y_train)
            
            # Make predictions
            y_pred_train = self.model.predict(X_train_final)
            y_pred_test = self.model.predict(X_test_final)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train, "Training")
            test_metrics = self._calculate_metrics(y_test, y_pred_test, "Testing")
            
            # Calculate confidence interval
            residuals = y_test - y_pred_test
            self.confidence_interval = np.std(residuals) * 1.96  # 95% confidence interval
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_final, y_train, cv=5, scoring='r2')
            
            # Store model metadata
            self.model_metadata = {
                'model_type': model_type,
                'training_date': datetime.now().isoformat(),
                'feature_columns': self.feature_columns,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'confidence_interval': float(self.confidence_interval),
                'cv_mean_score': float(cv_scores.mean()),
                'cv_std_score': float(cv_scores.std())
            }
            
            results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'confidence_interval': self.confidence_interval,
                'model_metadata': self.model_metadata
            }
            
            logger.info(f"Model training completed. Test R²: {test_metrics['r2']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
            
            logger.info(f"{dataset_name} Metrics - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def predict_price(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make price prediction for new data
        
        Args:
            input_data (Dict[str, Any]): Input features
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            logger.info(f"Making prediction for: {input_data}")
            
            # Map input to feature columns used during training
            feature_mapping = {
                'size': 'size_internal',
                'floor': 'floor_internal',
                'data_source': 'data_source_internal'
            }
            
            # Create feature vector
            feature_vector = []
            for feature in self.feature_columns:
                value = 0  # Default value
                
                # Check direct match first
                if feature in input_data:
                    value = input_data[feature]
                else:
                    # Check mapped features
                    for input_key, mapped_feature in feature_mapping.items():
                        if feature == mapped_feature and input_key in input_data:
                            value = input_data[input_key]
                            break
                    
                    # Special handling for common features
                    if feature == 'city' and 'city' in input_data:
                        value = input_data['city']
                    elif feature == 'property_type' and 'property_type' in input_data:
                        value = input_data['property_type']
                    elif feature == 'price_per_m2_competitor':
                        # Use average competitor price as default
                        value = 38000  # Default competitor price
                
                # Handle categorical features
                if feature in self.label_encoders:
                    try:
                        value = self.label_encoders[feature].transform([str(value)])[0]
                    except ValueError:
                        # Handle unseen categories
                        logger.warning(f"Unseen category '{value}' for feature '{feature}', using default")
                        value = 0
                
                feature_vector.append(value)
            
            # Scale features
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            predicted_price = self.model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence bounds
            lower_bound = predicted_price - self.confidence_interval
            upper_bound = predicted_price + self.confidence_interval
            
            results = {
                'predicted_price': float(predicted_price),
                'confidence_interval': float(self.confidence_interval),
                'lower_bound': float(max(0, lower_bound)),  # Ensure non-negative
                'upper_bound': float(upper_bound),
                'model_version': self.model_metadata.get('training_date', 'unknown'),
                'input_features': input_data
            }
            
            logger.info(f"Prediction: ₪{predicted_price:.2f}/m² (±₪{self.confidence_interval:.2f})")
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def save_model(self, model_path: str = "pricing_engine/models/pricing_model.pkl"):
        """Save trained model and metadata"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'metadata': self.model_metadata,
                'confidence_interval': self.confidence_interval
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = "pricing_engine/models/pricing_model.pkl"):
        """Load trained model and metadata"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.model_metadata = model_data['metadata']
            self.confidence_interval = model_data['confidence_interval']
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
                # Sort by importance
                importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                logger.info(f"Feature importance: {importance_dict}")
                return importance_dict
            else:
                logger.warning("Model does not support feature importance")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

def main():
    """Main function to demonstrate model training"""
    try:
        # Initialize trainer
        trainer = PricingModelTrainer()
        
        # Check if processed data exists
        data_path = "pricing_engine/data/processed_data.csv"
        if not os.path.exists(data_path):
            logger.error(f"Processed data not found at {data_path}")
            logger.info("Please run etl_pipeline.py first to generate processed data")
            return
        
        # Load and prepare data
        data = trainer.load_processed_data(data_path)
        features, target = trainer.prepare_features(data)
        
        # Train model
        results = trainer.train_model(features, target, model_type='xgboost')
        
        # Save model
        trainer.save_model()
        
        # Get feature importance
        importance = trainer.get_feature_importance()
        
        # Test prediction
        sample_input = {
            'size': 100,
            'floor': 3,
            'city': 'Tel Aviv',
            'property_type': 'Apartment',
            'data_source': 'internal'
        }
        
        prediction = trainer.predict_price(sample_input)
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Test R² Score: {results['test_metrics']['r2']:.4f}")
        print(f"Test RMSE: {results['test_metrics']['rmse']:.2f}")
        print(f"Confidence Interval: ±₪{results['confidence_interval']:.2f}")
        print(f"\nSample Prediction: ₪{prediction['predicted_price']:.2f}/m²")
        print(f"Confidence Range: ₪{prediction['lower_bound']:.2f} - ₪{prediction['upper_bound']:.2f}")
        
        if importance:
            print(f"\nTop Features:")
            for feature, imp in list(importance.items())[:3]:
                print(f"  {feature}: {imp:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()
