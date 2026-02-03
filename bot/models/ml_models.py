"""
Machine Learning models for price prediction.

Academic basis:
- XGBoost: 97% accuracy (Springer 2025), outperforms deep learning on tabular data
- Bi-LSTM: MAE 0.633 (Tripathy et al. 2024), best for magnitude prediction
- Recommendation: XGBoost for direction, LSTM for magnitude

IMPORTANT: Performance varies by prediction horizon, task type, cryptocurrency,
and market regime. No single model consistently dominates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import pickle
import json


@dataclass
class Prediction:
    """Model prediction output."""
    timestamp: datetime
    direction: int  # -1, 0, 1
    direction_probability: float  # Confidence in direction
    magnitude: float  # Predicted price change (normalized)
    magnitude_confidence: float
    model_name: str
    features_used: List[str]


class FeatureEngineering:
    """
    Feature engineering for ML models.

    Creates features from raw market data that have been shown
    to be predictive in academic literature.
    """

    @staticmethod
    def calculate_returns(prices: List[float], periods: List[int] = None) -> Dict[str, float]:
        """Calculate returns over various periods."""
        if periods is None:
            periods = [1, 5, 10, 20, 60]

        features = {}
        prices = np.array(prices)

        for period in periods:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period-1]) / prices[-period-1]
                features[f'return_{period}'] = ret

        return features

    @staticmethod
    def calculate_volatility(prices: List[float], windows: List[int] = None) -> Dict[str, float]:
        """Calculate realized volatility over various windows."""
        if windows is None:
            windows = [5, 10, 20, 60]

        features = {}
        prices = np.array(prices)

        if len(prices) < 2:
            return features

        returns = np.diff(prices) / prices[:-1]

        for window in windows:
            if len(returns) >= window:
                vol = np.std(returns[-window:])
                features[f'volatility_{window}'] = vol

        return features

    @staticmethod
    def calculate_momentum(prices: List[float]) -> Dict[str, float]:
        """Calculate momentum indicators."""
        features = {}
        prices = np.array(prices)

        if len(prices) < 20:
            return features

        # RSI-like momentum
        gains = np.maximum(np.diff(prices), 0)
        losses = np.maximum(-np.diff(prices), 0)

        if len(gains) >= 14:
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100

        # Moving average crossovers
        if len(prices) >= 20:
            ma5 = np.mean(prices[-5:])
            ma20 = np.mean(prices[-20:])
            features['ma_crossover'] = (ma5 - ma20) / ma20

        return features

    @staticmethod
    def calculate_obi_features(
        imbalances: List[float],
        windows: List[int] = None
    ) -> Dict[str, float]:
        """Calculate OBI-derived features."""
        if windows is None:
            windows = [5, 10, 20]

        features = {}
        imbalances = np.array(imbalances)

        if len(imbalances) == 0:
            return features

        features['obi_current'] = imbalances[-1]

        for window in windows:
            if len(imbalances) >= window:
                features[f'obi_ma_{window}'] = np.mean(imbalances[-window:])
                features[f'obi_std_{window}'] = np.std(imbalances[-window:])

        return features

    @staticmethod
    def prepare_features(
        prices: List[float],
        volumes: List[float] = None,
        imbalances: List[float] = None,
        funding_rates: List[float] = None,
        sentiment_scores: List[float] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare full feature vector for ML models.

        Returns: (feature_array, feature_names)
        """
        all_features = {}
        feature_names = []

        # Price-based features
        all_features.update(FeatureEngineering.calculate_returns(prices))
        all_features.update(FeatureEngineering.calculate_volatility(prices))
        all_features.update(FeatureEngineering.calculate_momentum(prices))

        # Volume features
        if volumes and len(volumes) > 0:
            volumes = np.array(volumes)
            if len(volumes) >= 20:
                all_features['volume_ma_ratio'] = volumes[-1] / np.mean(volumes[-20:])
                all_features['volume_std'] = np.std(volumes[-20:])

        # OBI features
        if imbalances:
            all_features.update(FeatureEngineering.calculate_obi_features(imbalances))

        # Funding rate features
        if funding_rates and len(funding_rates) > 0:
            rates = np.array(funding_rates)
            all_features['funding_current'] = rates[-1]
            if len(rates) >= 10:
                all_features['funding_ma'] = np.mean(rates[-10:])
                all_features['funding_extreme'] = 1 if abs(rates[-1]) > 0.001 else 0

        # Sentiment features
        if sentiment_scores and len(sentiment_scores) > 0:
            sentiment = np.array(sentiment_scores)
            all_features['sentiment_current'] = sentiment[-1]
            if len(sentiment) >= 10:
                all_features['sentiment_ma'] = np.mean(sentiment[-10:])

        # Convert to arrays
        feature_names = sorted(all_features.keys())
        feature_array = np.array([all_features[name] for name in feature_names])

        return feature_array, feature_names


class MLModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model. Returns training metrics."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        """Make prediction."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from file."""
        pass


class SimpleXGBoostDirectional(MLModel):
    """
    XGBoost-style model for directional prediction.

    This is a simplified implementation that mimics XGBoost behavior
    without requiring the actual library (for demo purposes).

    Academic basis:
    - Springer 2025: 97% accuracy for Bitcoin prediction
    - MDPI 2025: R² 0.9694-0.9827

    For production, use actual xgboost library.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._trees: List[Dict] = []
        self._feature_importance: Dict[str, float] = {}
        self._feature_names: List[str] = []
        self._is_trained: bool = False

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Train the model using gradient boosting.

        For demo, uses a simplified decision stump ensemble.
        """
        self._feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]

        # Initialize
        n_samples, n_features = X.shape
        predictions = np.zeros(n_samples)
        self._trees = []

        # Convert y to binary
        y_binary = (y > 0).astype(float)

        for _ in range(self.n_estimators):
            # Calculate residuals (gradient)
            probs = 1 / (1 + np.exp(-predictions))
            residuals = y_binary - probs

            # Find best split (simplified: single feature, single threshold)
            best_gain = -np.inf
            best_tree = None

            for feature_idx in range(n_features):
                thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])

                for threshold in thresholds:
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask

                    if left_mask.sum() < 5 or right_mask.sum() < 5:
                        continue

                    left_value = residuals[left_mask].mean()
                    right_value = residuals[right_mask].mean()

                    # Calculate gain
                    gain = (left_mask.sum() * left_value**2 +
                            right_mask.sum() * right_value**2)

                    if gain > best_gain:
                        best_gain = gain
                        best_tree = {
                            'feature': feature_idx,
                            'threshold': threshold,
                            'left_value': left_value,
                            'right_value': right_value,
                        }

            if best_tree is None:
                break

            # Update predictions
            left_mask = X[:, best_tree['feature']] <= best_tree['threshold']
            predictions[left_mask] += self.learning_rate * best_tree['left_value']
            predictions[~left_mask] += self.learning_rate * best_tree['right_value']

            self._trees.append(best_tree)

            # Update feature importance
            feature_name = self._feature_names[best_tree['feature']]
            self._feature_importance[feature_name] = self._feature_importance.get(feature_name, 0) + best_gain

        # Normalize feature importance
        total_importance = sum(self._feature_importance.values())
        if total_importance > 0:
            self._feature_importance = {
                k: v / total_importance for k, v in self._feature_importance.items()
            }

        self._is_trained = True

        # Calculate training accuracy
        final_probs = 1 / (1 + np.exp(-predictions))
        final_preds = (final_probs > 0.5).astype(int)
        accuracy = (final_preds == y_binary).mean()

        return {
            'accuracy': accuracy,
            'n_trees': len(self._trees),
            'feature_importance': self._feature_importance,
        }

    def predict(self, X: np.ndarray) -> Prediction:
        """Make directional prediction."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Handle feature dimension mismatch
        n_expected = len(self._feature_names)
        if X.shape[1] != n_expected:
            if X.shape[1] > n_expected:
                X = X[:, :n_expected]  # Truncate
            else:
                X = np.pad(X, ((0, 0), (0, n_expected - X.shape[1])))  # Pad

        # Sum tree predictions
        prediction = 0.0
        for tree in self._trees:
            feature_idx = tree['feature']
            if feature_idx < X.shape[1]:
                if X[0, feature_idx] <= tree['threshold']:
                    prediction += self.learning_rate * tree['left_value']
                else:
                    prediction += self.learning_rate * tree['right_value']

        # Convert to probability
        probability = 1 / (1 + np.exp(-prediction))

        # Direction
        if probability > 0.6:
            direction = 1
        elif probability < 0.4:
            direction = -1
        else:
            direction = 0

        return Prediction(
            timestamp=datetime.utcnow(),
            direction=direction,
            direction_probability=probability if direction >= 0 else 1 - probability,
            magnitude=0.0,  # XGBoost is for direction only
            magnitude_confidence=0.0,
            model_name='SimpleXGBoost',
            features_used=self._feature_names,
        )

    def save(self, path: str):
        """Save model."""
        state = {
            'trees': self._trees,
            'feature_importance': self._feature_importance,
            'feature_names': self._feature_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
            }
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        """Load model."""
        with open(path, 'r') as f:
            state = json.load(f)
        self._trees = state['trees']
        self._feature_importance = state['feature_importance']
        self._feature_names = state['feature_names']
        self.n_estimators = state['params']['n_estimators']
        self.max_depth = state['params']['max_depth']
        self.learning_rate = state['params']['learning_rate']
        self._is_trained = True


class SimpleLSTMMagnitude(MLModel):
    """
    LSTM-style model for magnitude prediction.

    This is a simplified implementation that mimics LSTM behavior
    without requiring deep learning frameworks (for demo purposes).

    Academic basis:
    - Tripathy et al. (2024): Bi-LSTM MAE 0.633
    - GRU sometimes outperforms LSTM

    For production, use TensorFlow/PyTorch LSTM.
    """

    def __init__(self, hidden_size: int = 64, lookback: int = 60):
        self.hidden_size = hidden_size
        self.lookback = lookback
        self._weights: Dict[str, np.ndarray] = {}
        self._is_trained: bool = False
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0
        self._n_features: int = 0  # Track expected feature dimension

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train the model.

        For demo, uses simple exponential smoothing with learned parameters.
        """
        # Normalize targets
        self._scaler_mean = y.mean()
        self._scaler_std = y.std() + 1e-8
        y_normalized = (y - self._scaler_mean) / self._scaler_std

        # Simple learned parameters (mock LSTM)
        # In reality, LSTM learns complex temporal patterns
        n_features = X.shape[1] if X.ndim > 1 else 1
        self._n_features = n_features  # Store for prediction validation

        # Initialize weights
        self._weights['input'] = np.random.randn(n_features) * 0.1
        self._weights['recurrent'] = np.random.randn(self.hidden_size) * 0.1
        self._weights['output'] = np.random.randn(self.hidden_size) * 0.1

        # Simple gradient descent
        learning_rate = 0.01
        hidden = np.zeros(self.hidden_size)

        for epoch in range(100):
            total_loss = 0.0

            for i in range(len(X)):
                # Forward pass (simplified)
                if X.ndim > 1:
                    input_val = np.dot(X[i], self._weights['input'])
                else:
                    input_val = X[i] * self._weights['input'][0]

                # Simple recurrent update
                hidden = 0.9 * hidden + 0.1 * np.tanh(input_val * self._weights['recurrent'])

                # Output
                pred = np.dot(hidden, self._weights['output'])

                # Loss
                loss = (pred - y_normalized[i]) ** 2
                total_loss += loss

                # Simple gradient update
                grad = 2 * (pred - y_normalized[i])
                self._weights['output'] -= learning_rate * grad * hidden

            if epoch % 20 == 0:
                avg_loss = total_loss / len(X)

        self._is_trained = True

        # Calculate metrics
        predictions = self._predict_batch(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
        }

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Internal batch prediction."""
        predictions = []
        hidden = np.zeros(self.hidden_size)

        for i in range(len(X)):
            if X.ndim > 1:
                x_i = X[i]
                # Handle feature dimension mismatch
                if len(x_i) != self._n_features:
                    if len(x_i) > self._n_features:
                        x_i = x_i[:self._n_features]  # Truncate
                    else:
                        x_i = np.pad(x_i, (0, self._n_features - len(x_i)))  # Pad
                input_val = np.dot(x_i, self._weights['input'])
            else:
                input_val = X[i] * self._weights['input'][0]

            hidden = 0.9 * hidden + 0.1 * np.tanh(input_val * self._weights['recurrent'])
            pred = np.dot(hidden, self._weights['output'])
            predictions.append(pred * self._scaler_std + self._scaler_mean)

        return np.array(predictions)

    def predict(self, X: np.ndarray) -> Prediction:
        """Make magnitude prediction."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = self._predict_batch(X)
        magnitude = predictions[-1]

        # Confidence based on prediction magnitude relative to historical std
        confidence = min(0.9, 0.5 + abs(magnitude) / (2 * self._scaler_std))

        return Prediction(
            timestamp=datetime.utcnow(),
            direction=1 if magnitude > 0 else -1 if magnitude < 0 else 0,
            direction_probability=0.5,  # LSTM is for magnitude, not direction
            magnitude=magnitude,
            magnitude_confidence=confidence,
            model_name='SimpleLSTM',
            features_used=[],
        )

    def save(self, path: str):
        """Save model."""
        state = {
            'weights': {k: v.tolist() for k, v in self._weights.items()},
            'scaler_mean': self._scaler_mean,
            'scaler_std': self._scaler_std,
            'hidden_size': self.hidden_size,
            'lookback': self.lookback,
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        """Load model."""
        with open(path, 'r') as f:
            state = json.load(f)
        self._weights = {k: np.array(v) for k, v in state['weights'].items()}
        self._scaler_mean = state['scaler_mean']
        self._scaler_std = state['scaler_std']
        self.hidden_size = state['hidden_size']
        self.lookback = state['lookback']
        self._is_trained = True


class EnsemblePredictor:
    """
    Ensemble predictor combining XGBoost (direction) and LSTM (magnitude).

    Academic recommendation: Use ensemble for improved robustness.
    """

    def __init__(self, config=None):
        self.direction_model = SimpleXGBoostDirectional()
        self.magnitude_model = SimpleLSTMMagnitude()
        self._is_trained = False

    def train(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict[str, any]:
        """Train both models."""
        direction_metrics = self.direction_model.train(X, y_direction, feature_names)
        magnitude_metrics = self.magnitude_model.train(X, y_magnitude)

        self._is_trained = True

        return {
            'direction': direction_metrics,
            'magnitude': magnitude_metrics,
        }

    def predict(self, X: np.ndarray) -> Prediction:
        """
        Combined prediction using both models.

        Direction from XGBoost, magnitude from LSTM.
        """
        if not self._is_trained:
            raise ValueError("Models not trained")

        direction_pred = self.direction_model.predict(X)
        magnitude_pred = self.magnitude_model.predict(X)

        # Combine predictions
        # Use direction from XGBoost but magnitude from LSTM
        combined_direction = direction_pred.direction
        combined_magnitude = magnitude_pred.magnitude

        # Align magnitude sign with direction
        if combined_direction != 0:
            combined_magnitude = abs(combined_magnitude) * combined_direction

        # Combined confidence
        combined_confidence = (
            direction_pred.direction_probability * 0.6 +
            magnitude_pred.magnitude_confidence * 0.4
        )

        return Prediction(
            timestamp=datetime.utcnow(),
            direction=combined_direction,
            direction_probability=direction_pred.direction_probability,
            magnitude=combined_magnitude,
            magnitude_confidence=magnitude_pred.magnitude_confidence,
            model_name='Ensemble(XGBoost+LSTM)',
            features_used=direction_pred.features_used,
        )
