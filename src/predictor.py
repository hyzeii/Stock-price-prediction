import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import json
import os
from typing import Dict, Any, Optional
from .technical_indicators import TechnicalIndicators
from .config import config_manager

class StockLSTMPredictor:
    def __init__(self, ticker: str, config: Optional[Dict[str, Any]] = None):
        self.ticker = ticker
        
        # Get configuration
        if config is None:
            config = config_manager.get_model_config(ticker)
        self.config = config
        
        # Extract key parameters from config
        self.lookback = self.config.get('lookback', 60)
        self.train_split = self.config.get('train_split', 0.8)
        
        # Initialize components
        self.model = None
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.tech_indicators = TechnicalIndicators(self.config)
        
        # Setup directories
        self.model_dir = config_manager.settings.model_dir
        self.data_dir = config_manager.settings.data_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, period: Optional[str] = None):
        """Fetch stock data from Yahoo Finance"""
        if period is None:
            period = config_manager.get('data.default_period', '2y')
            
        if start_date is None and period:
            print(f"Fetching data for {self.ticker} (period: {period})")
            self.data = yf.download(self.ticker, period=period, auto_adjust=True, progress=False)
        else:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            print(f"Fetching data for {self.ticker} from {start_date} to {end_date}")
            self.data = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Flatten multi-level columns if present
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        print(f"Fetched {len(self.data)} data points")
        return self.data
    
    def calculate_technical_indicators(self):
        """Calculate technical analysis indicators using configuration"""
        self.data_with_indicators = self.tech_indicators.calculate_all(self.data)
        print(f"Calculated technical indicators. Data points after cleaning: {len(self.data_with_indicators)}")
        return self.data_with_indicators
    
    def prepare_data(self, train_split: Optional[float] = None):
        """Prepare data for LSTM model"""
        if train_split is None:
            train_split = self.train_split
            
        df = self.data_with_indicators.copy()
        
        # Get features list from config
        features = config_manager.get_features_list(self.ticker)
        
        # Filter only available features
        available_features = [f for f in features if f in df.columns]
        print(f"Using {len(available_features)} features: {available_features}")
        
        data = df[available_features].values
        
        # Scale the data
        scaled_data = self.feature_scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price (index 0)
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * train_split)
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        print(f"Input shape: {self.X_train.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self):
        """Build LSTM model based on configuration"""
        arch_config = self.config.get('architecture', {})
        train_config = self.config.get('training', {})
        
        lstm_units = arch_config.get('lstm_units', [128, 64])
        dropout = arch_config.get('dropout', 0.2)
        dense_units = arch_config.get('dense_units', [32])
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        self.model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            return_seq = i < len(lstm_units) - 1
            self.model.add(LSTM(units=lstm_units[i], return_sequences=return_seq))
            self.model.add(Dropout(dropout))
        
        # Dense layers
        for units in dense_units:
            self.model.add(Dense(units=units, activation='relu'))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        optimizer = train_config.get('optimizer', 'adam')
        loss = train_config.get('loss', 'mean_squared_error')
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        
        print(self.model.summary())
        return self.model
    
    def train_model(self, epochs: Optional[int] = None, batch_size: Optional[int] = None, validation_split: Optional[float] = None):
        """Train the LSTM model"""
        train_config = self.config.get('training', {})
        
        if epochs is None:
            epochs = train_config.get('epochs', 50)
        if batch_size is None:
            batch_size = train_config.get('batch_size', 32)
        if validation_split is None:
            validation_split = self.config.get('validation_split', 0.1)
        
        # Early stopping configuration
        es_config = train_config.get('early_stopping', {})
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=es_config.get('patience', 10),
            min_delta=es_config.get('min_delta', 0.0001),
            restore_best_weights=True
        )
        
        # Model checkpoint
        model_path = os.path.join(self.model_dir, f'{self.ticker}_best_model.keras')
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        print(f"Training with epochs={epochs}, batch_size={batch_size}")
        
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        self.history = history
        print(f"\nBest model saved to: {model_path}")
        return history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        train_pred_inv = self.inverse_transform_predictions(train_pred)
        test_pred_inv = self.inverse_transform_predictions(test_pred)
        y_train_inv = self.inverse_transform_predictions(self.y_train.reshape(-1, 1))
        y_test_inv = self.inverse_transform_predictions(self.y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
        train_mae = mean_absolute_error(y_train_inv, train_pred_inv)
        test_mae = mean_absolute_error(y_test_inv, test_pred_inv)
        
        print(f"\nModel Performance:")
        print(f"Train RMSE: ${train_rmse:.2f}, MAE: ${train_mae:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}, MAE: ${test_mae:.2f}")
        
        return {
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_pred': train_pred_inv, 'test_pred': test_pred_inv,
            'y_train': y_train_inv, 'y_test': y_test_inv
        }
    
    def inverse_transform_predictions(self, predictions):
        """Inverse transform scaled predictions back to original price scale"""
        dummy = np.zeros((len(predictions), self.feature_scaler.n_features_in_))
        dummy[:, 0] = predictions.flatten()
        inv_transformed = self.feature_scaler.inverse_transform(dummy)
        return inv_transformed[:, 0]
    
    def predict_next_days(self, n_days: int = 5):
        """Predict next n days"""
        last_sequence = self.X_test[-1].reshape(1, self.lookback, -1)
        predictions = []
        
        for _ in range(n_days):
            pred = self.model.predict(last_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence with prediction
            new_row = last_sequence[0, -1, :].copy()
            new_row[0] = pred[0, 0]
            last_sequence = np.append(last_sequence[:, 1:, :], 
                                     new_row.reshape(1, 1, -1), axis=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_inv = self.inverse_transform_predictions(predictions)
        
        return predictions_inv
    
    def save_model(self, model_name: Optional[str] = None):
        """Save model, scalers, and metadata"""
        if model_name is None:
            model_name = f"{self.ticker}_model"
        
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save Keras model
        keras_model_path = os.path.join(model_path, 'model.keras')
        self.model.save(keras_model_path)
        
        # Save scalers
        scaler_path = os.path.join(model_path, 'feature_scaler.pkl')
        joblib.dump(self.feature_scaler, scaler_path)
        
        # Save metadata including config
        metadata = {
            'ticker': self.ticker,
            'lookback': self.lookback,
            'features': config_manager.get_features_list(self.ticker),
            'n_features': self.feature_scaler.n_features_in_,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }
        
        metadata_path = os.path.join(model_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Model package saved to: {model_path}")
        return model_path
    
    def load_model(self, model_name: Optional[str] = None):
        """Load model, scalers, and metadata"""
        if model_name is None:
            model_name = f"{self.ticker}_model"
        
        model_path = os.path.join(self.model_dir, model_name)
        
        # Load Keras model
        keras_model_path = os.path.join(model_path, 'model.keras')
        self.model = load_model(keras_model_path)
        
        # Load scalers
        scaler_path = os.path.join(model_path, 'feature_scaler.pkl')
        self.feature_scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.ticker = metadata['ticker']
        self.lookback = metadata['lookback']
        
        print(f"✓ Model loaded: {metadata['ticker']} (trained: {metadata['trained_date']})")
        return metadata