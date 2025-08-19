"""
Signal Generation Service
Generates trading signals from market data using trained ML models
"""

import json
import logging
import pickle
from datetime import datetime
from typing import Dict, Optional, List
from queue import Queue
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalService:
    """Generate trading signals from market data"""
    
    def __init__(self, model_path: str = "models/xgboost_model.pkl", 
                 message_bus: Optional[Queue] = None):
        self.model_path = Path(model_path)
        self.message_bus = message_bus or Queue()
        self.model = None
        self.model_version = None
        self.feature_list = []
        
        # Signal generation parameters
        self.confidence_threshold = 0.7
        self.min_prediction_delta = 0.0005  # 0.05% minimum price change
        
        # Feature buffer for stateful features
        self.feature_buffer = []
        self.buffer_size = 60  # 1 hour of features
        
        self._load_model()
        self._load_feature_config()
        
    def _load_model(self):
        """Load the trained model"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.model = model_data.get('model')
                self.model_version = model_data.get('version', 'unknown')
                
                logger.info(f"Loaded model version: {self.model_version}")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                # Create dummy model for testing
                self.model = None
                self.model_version = 'dummy'
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            
    def _load_feature_config(self):
        """Load feature configuration"""
        config_path = Path("configs/selected_features.txt")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.feature_list = [line.strip() for line in f.readlines()]
                    
                logger.info(f"Loaded {len(self.feature_list)} features")
            else:
                # Use default feature set
                self.feature_list = [
                    'sma_10', 'sma_30', 'rsi_14', 'volatility',
                    'volume_ma', 'vwap', 'spread'
                ]
                
        except Exception as e:
            logger.error(f"Failed to load feature config: {e}")
            
    def process_market_data(self, market_data: Dict) -> Optional[Dict]:
        """Process market data and generate signals"""
        # Extract features from market data
        features = market_data.get('features', {})
        
        if not features:
            logger.debug("No features in market data")
            return None
            
        # Add to buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer.pop(0)
            
        # Need minimum data for prediction
        if len(self.feature_buffer) < 30:
            return None
            
        # Prepare features for model
        X = self._prepare_features(features)
        
        if X is None:
            return None
            
        # Generate prediction
        signal = self._generate_signal(X, market_data)
        
        if signal:
            self._emit_signal(signal)
            
        return signal
        
    def _prepare_features(self, current_features: Dict) -> Optional[np.ndarray]:
        """Prepare features for model prediction"""
        try:
            # Create feature vector in correct order
            feature_vector = []
            
            for feature_name in self.feature_list:
                if feature_name in current_features:
                    feature_vector.append(current_features[feature_name])
                else:
                    # Try to calculate from buffer
                    calculated = self._calculate_feature(feature_name)
                    if calculated is not None:
                        feature_vector.append(calculated)
                    else:
                        # Use default value for missing features
                        feature_vector.append(0.0)
                        
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
            
    def _calculate_feature(self, feature_name: str) -> Optional[float]:
        """Calculate feature from buffer if not directly available"""
        # Implement derived feature calculations
        if feature_name.startswith('lag_'):
            # Lagged features
            try:
                lag = int(feature_name.split('_')[1])
                base_feature = '_'.join(feature_name.split('_')[2:])
                
                if lag < len(self.feature_buffer):
                    return self.feature_buffer[-lag].get(base_feature)
            except:
                pass
                
        elif feature_name.startswith('diff_'):
            # Difference features
            try:
                parts = feature_name.split('_')
                feature1 = parts[1]
                feature2 = parts[2]
                
                val1 = self.feature_buffer[-1].get(feature1)
                val2 = self.feature_buffer[-1].get(feature2)
                
                if val1 and val2:
                    return val1 - val2
            except:
                pass
                
        return None
        
    def _generate_signal(self, X: np.ndarray, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal from features"""
        try:
            # If no model loaded, generate random signal for testing
            if self.model is None:
                # Generate dummy signal for testing
                import random
                prediction_proba = [random.random() for _ in range(3)]
                total = sum(prediction_proba)
                prediction_proba = [p/total for p in prediction_proba]
            else:
                # Get model prediction
                prediction_proba = self.model.predict_proba(X)[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class]
            
            # Map to signal
            class_map = {0: 'DOWN', 1: 'SIDEWAYS', 2: 'UP'}
            signal_type = class_map.get(predicted_class, 'SIDEWAYS')
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                signal_type = 'SIDEWAYS'  # No trade
                
            # Create signal object
            signal = {
                'timestamp': datetime.now(),
                'symbol': market_data.get('symbol', 'BTCUSDT'),
                'signal': signal_type,
                'confidence': float(confidence),
                'predicted_direction': signal_type,
                'current_price': market_data.get('close'),
                'features_snapshot': dict(list(zip(self.feature_list, X[0]))),
                'model_version': self.model_version
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
            
    def _emit_signal(self, signal: Dict):
        """Emit trading signal to message bus"""
        event = {
            'type': 'TRADING_SIGNAL',
            'timestamp': signal['timestamp'].isoformat(),
            'data': signal
        }
        
        self.message_bus.put(event)
        logger.info(f"Signal emitted: {signal['signal']} @ {signal['confidence']:.2%} confidence")
        
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'version': self.model_version,
            'features': self.feature_list,
            'confidence_threshold': self.confidence_threshold,
            'buffer_size': len(self.feature_buffer)
        }
        
    def update_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for signal generation"""
        if 0 < threshold <= 1:
            self.confidence_threshold = threshold
            logger.info(f"Updated confidence threshold to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}, keeping {self.confidence_threshold}")
