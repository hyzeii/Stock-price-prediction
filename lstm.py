import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import json
import os

class StockLSTMPredictor:
    def __init__(self, ticker='AAPL', lookback=60, model_dir='models'):
        self.ticker = ticker
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def fetch_data(self, start_date=None, end_date=None):
        """Fetch stock data from Yahoo Finance"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching data for {self.ticker} from {start_date} to {end_date}")
        self.data = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True)
        
        # Flatten multi-level columns if present
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)
        
        return self.data
    
    def calculate_technical_indicators(self):
        """Calculate technical analysis indicators"""
        df = self.data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Price Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        self.data_with_indicators = df
        return df
    
    def prepare_data(self, train_split=0.8):
        """Prepare data for LSTM model"""
        df = self.data_with_indicators.copy()
        
        # Select features
        features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                   'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                   'ATR', 'Volume_SMA', 'ROC']
        
        data = df[features].values
        
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
    
    def build_model(self, lstm_units=[128, 64], dropout=0.2):
        """Build LSTM model"""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=lstm_units[0], return_sequences=True, 
                           input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(dropout))
        
        # Second LSTM layer
        self.model.add(LSTM(units=lstm_units[1], return_sequences=False))
        self.model.add(Dropout(dropout))
        
        # Dense layers
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        print(self.model.summary())
        return self.model
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1):
        """Train the LSTM model"""
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Save best model during training
        model_path = os.path.join(self.model_dir, f'{self.ticker}_best_model.keras')
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
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
        # Create dummy array with all features
        dummy = np.zeros((len(predictions), self.feature_scaler.n_features_in_))
        dummy[:, 0] = predictions.flatten()  # Close price is first feature
        
        # Inverse transform
        inv_transformed = self.feature_scaler.inverse_transform(dummy)
        return inv_transformed[:, 0]
    
    def plot_results(self, metrics):
        """Plot training history and predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training history - Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training history - MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training predictions
        axes[1, 0].plot(metrics['y_train'], label='Actual', alpha=0.7)
        axes[1, 0].plot(metrics['train_pred'], label='Predicted', alpha=0.7)
        axes[1, 0].set_title(f'Training Set Predictions (RMSE: ${metrics["train_rmse"]:.2f})')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Test predictions
        axes[1, 1].plot(metrics['y_test'], label='Actual', alpha=0.7)
        axes[1, 1].plot(metrics['test_pred'], label='Predicted', alpha=0.7)
        axes[1, 1].set_title(f'Test Set Predictions (RMSE: ${metrics["test_rmse"]:.2f})')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_next_days(self, n_days=5):
        """Predict next n days"""
        last_sequence = self.X_test[-1].reshape(1, self.lookback, -1)
        predictions = []
        
        for _ in range(n_days):
            pred = self.model.predict(last_sequence, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence with prediction
            new_row = last_sequence[0, -1, :].copy()
            new_row[0] = pred[0, 0]  # Update close price
            last_sequence = np.append(last_sequence[:, 1:, :], 
                                     new_row.reshape(1, 1, -1), axis=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_inv = self.inverse_transform_predictions(predictions)
        
        return predictions_inv
    
    def save_model(self, model_name=None):
        """Save model, scalers, and metadata"""
        if model_name is None:
            model_name = f"{self.ticker}_model"
        
        # Create subdirectory for this model
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Save Keras model
        keras_model_path = os.path.join(model_path, 'model.keras')
        self.model.save(keras_model_path)
        print(f"Model saved to: {keras_model_path}")
        
        # Save scalers
        scaler_path = os.path.join(model_path, 'feature_scaler.pkl')
        joblib.dump(self.feature_scaler, scaler_path)
        print(f"Feature scaler saved to: {scaler_path}")
        
        # Save metadata
        metadata = {
            'ticker': self.ticker,
            'lookback': self.lookback,
            'features': ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                        'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
                        'ATR', 'Volume_SMA', 'ROC'],
            'n_features': self.feature_scaler.n_features_in_,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_architecture': {
                'input_shape': [self.lookback, self.X_train.shape[2]],
                'layers': [str(layer) for layer in self.model.layers]
            }
        }
        
        metadata_path = os.path.join(model_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
        
        print(f"\n✓ Complete model package saved to: {model_path}")
        return model_path
    
    def load_model(self, model_name=None):
        """Load model, scalers, and metadata"""
        if model_name is None:
            model_name = f"{self.ticker}_model"
        
        model_path = os.path.join(self.model_dir, model_name)
        
        # Load Keras model
        keras_model_path = os.path.join(model_path, 'model.keras')
        self.model = load_model(keras_model_path)
        print(f"Model loaded from: {keras_model_path}")
        
        # Load scalers
        scaler_path = os.path.join(model_path, 'feature_scaler.pkl')
        self.feature_scaler = joblib.load(scaler_path)
        print(f"Feature scaler loaded from: {scaler_path}")
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.ticker = metadata['ticker']
        self.lookback = metadata['lookback']
        
        print(f"Metadata loaded from: {metadata_path}")
        print(f"Model trained on: {metadata['trained_date']}")
        print(f"✓ Model package loaded successfully")
        
        return metadata
    
    @staticmethod
    def list_available_models(model_dir='models'):
        """List all available saved models"""
        if not os.path.exists(model_dir):
            print(f"Model directory '{model_dir}' does not exist")
            return []
        
        models = []
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append({
                        'name': item,
                        'ticker': metadata.get('ticker'),
                        'trained_date': metadata.get('trained_date'),
                        'lookback': metadata.get('lookback')
                    })
        
        return models

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockLSTMPredictor(ticker='AAPL', lookback=60)
    
    # Fetch and prepare data
    predictor.fetch_data()
    predictor.calculate_technical_indicators()
    predictor.prepare_data(train_split=0.8)
    
    # Build and train model
    predictor.build_model(lstm_units=[128, 64], dropout=0.2)
    predictor.train_model(epochs=50, batch_size=32)
    
    # Evaluate and visualize
    metrics = predictor.evaluate_model()
    predictor.plot_results(metrics)
    
    # Save the trained model
    model_path = predictor.save_model(model_name='AAPL_model')
    
    # Predict next 5 days
    future_predictions = predictor.predict_next_days(n_days=5)
    print(f"\nNext 5 days predictions:")
    for i, pred in enumerate(future_predictions, 1):
        print(f"Day {i}: ${pred:.2f}")
    
    # Example: Load model later for inference
    print("\n" + "="*50)
    print("Example: Loading saved model")
    print("="*50)
    
    new_predictor = StockLSTMPredictor(ticker='AAPL', lookback=60)
    metadata = new_predictor.load_model(model_name='AAPL_model')
    
    # List all available models
    print("\n" + "="*50)
    print("Available models:")
    print("="*50)
    models = StockLSTMPredictor.list_available_models()
    for model in models:
        print(f"- {model['name']}: {model['ticker']} (trained: {model['trained_date']})")