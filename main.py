# main.py
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import os
from pathlib import Path as PathlibPath

app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction with technical analysis",
    version="1.4.0"
)

# Configuration
MODEL_DIR = "models"
CACHE_DIR = "cache"

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Global model cache
model_cache = {}

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionResponse(BaseModel):
    ticker: str
    prediction: float
    confidence: Optional[str] = "medium"
    timestamp: str
    
class MultidayPredictionResponse(BaseModel):
    ticker: str
    predictions: List[dict]
    timestamp: str
    
class ModelInfo(BaseModel):
    name: str
    ticker: str
    trained_date: str
    lookback: int
    
class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]
    count: int

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    timestamp: str

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_technical_indicators(df):
    """Calculate technical analysis indicators"""
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
    
    return df

def fetch_stock_data(ticker: str, period: str = "2y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data for {ticker}: {str(e)}")

def load_model_package(model_name: str):
    """Load model, scaler, and metadata"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        # Load Keras model
        keras_model_path = os.path.join(model_path, 'model.keras')
        model = load_model(keras_model_path)
        
        # Load scaler
        scaler_path = os.path.join(model_path, 'feature_scaler.pkl')
        scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cache the model
        model_package = {
            'model': model,
            'scaler': scaler,
            'metadata': metadata
        }
        model_cache[model_name] = model_package
        
        return model_package
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def prepare_sequence(data, lookback, scaler):
    """Prepare sequence for prediction"""
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
               'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
               'ATR', 'Volume_SMA', 'ROC']
    
    feature_data = data[features].values
    scaled_data = scaler.transform(feature_data)
    
    # Get last lookback days
    sequence = scaled_data[-lookback:]
    return sequence.reshape(1, lookback, -1)

def inverse_transform_prediction(prediction, scaler):
    """Inverse transform scaled prediction"""
    dummy = np.zeros((len(prediction), scaler.n_features_in_))
    dummy[:, 0] = prediction.flatten()
    inv_transformed = scaler.inverse_transform(dummy)
    return inv_transformed[:, 0]

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Price Prediction API",
        "version": "1.4.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/{ticker}",
            "predict_multiday": "/predict/{ticker}/multiday",
            "models": "/models"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_cache),
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=AvailableModelsResponse, tags=["Models"])
async def list_models():
    """List all available trained models"""
    models = []
    
    if not os.path.exists(MODEL_DIR):
        return AvailableModelsResponse(models=[], count=0)
    
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(item_path):
            metadata_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                models.append(ModelInfo(
                    name=item,
                    ticker=metadata.get('ticker', 'unknown'),
                    trained_date=metadata.get('trained_date', 'unknown'),
                    lookback=metadata.get('lookback', 60)
                ))
    
    return AvailableModelsResponse(models=models, count=len(models))

@app.get("/predict/{ticker}", response_model=PredictionResponse, tags=["Prediction"])
async def predict_next_day(
    ticker: str = Path(..., description="Stock ticker symbol (e.g., AAPL, TSLA)"),
    model_name: Optional[str] = Query(None, description="Model name to use (defaults to {ticker}_model)")
):
    """
    Predict next day's closing price for a stock
    
    - **ticker**: Stock ticker symbol
    - **model_name**: Optional custom model name
    """
    if model_name is None:
        model_name = f"{ticker.upper()}_model"
    
    # Load model
    model_package = load_model_package(model_name)
    model = model_package['model']
    scaler = model_package['scaler']
    metadata = model_package['metadata']
    lookback = metadata['lookback']
    
    # Fetch and prepare data
    data = fetch_stock_data(ticker, period="2y")
    data = calculate_technical_indicators(data)
    
    if len(data) < lookback:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient data. Need at least {lookback} days, got {len(data)}"
        )
    
    # Prepare sequence and predict
    sequence = prepare_sequence(data, lookback, scaler)
    prediction = model.predict(sequence, verbose=0)
    
    # Inverse transform
    predicted_price = inverse_transform_prediction(prediction, scaler)[0]
    
    # Calculate confidence based on recent volatility
    recent_std = data['Close'].tail(20).std()
    recent_mean = data['Close'].tail(20).mean()
    volatility = (recent_std / recent_mean) * 100
    
    if volatility < 2:
        confidence = "high"
    elif volatility < 5:
        confidence = "medium"
    else:
        confidence = "low"
    
    return PredictionResponse(
        ticker=ticker.upper(),
        prediction=round(float(predicted_price), 2),
        confidence=confidence,
        timestamp=datetime.now().isoformat()
    )

@app.get("/predict/{ticker}/multiday", response_model=MultidayPredictionResponse, tags=["Prediction"])
async def predict_multiple_days(
    ticker: str = Path(..., description="Stock ticker symbol (e.g., AAPL, TSLA)"),
    days: int = Query(5, ge=1, le=30, description="Number of days to predict (1-30)"),
    model_name: Optional[str] = Query(None, description="Model name to use")
):
    """
    Predict multiple days ahead for a stock
    
    - **ticker**: Stock ticker symbol
    - **days**: Number of days to predict (1-30)
    - **model_name**: Optional custom model name
    """
    if model_name is None:
        model_name = f"{ticker.upper()}_model"
    
    # Load model
    model_package = load_model_package(model_name)
    model = model_package['model']
    scaler = model_package['scaler']
    metadata = model_package['metadata']
    lookback = metadata['lookback']
    
    # Fetch and prepare data
    data = fetch_stock_data(ticker, period="2y")
    data = calculate_technical_indicators(data)
    
    if len(data) < lookback:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient data. Need at least {lookback} days, got {len(data)}"
        )
    
    # Prepare initial sequence
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
               'MACD', 'Signal_Line', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
               'ATR', 'Volume_SMA', 'ROC']
    
    feature_data = data[features].values
    scaled_data = scaler.transform(feature_data)
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, -1)
    
    # Predict multiple days
    predictions = []
    current_date = data.index[-1]
    
    for i in range(days):
        pred = model.predict(last_sequence, verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence
        new_row = last_sequence[0, -1, :].copy()
        new_row[0] = pred[0, 0]
        last_sequence = np.append(last_sequence[:, 1:, :], 
                                  new_row.reshape(1, 1, -1), axis=1)
        
        current_date = current_date + timedelta(days=1)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predicted_prices = inverse_transform_prediction(predictions, scaler)
    
    # Format response
    prediction_list = []
    base_date = data.index[-1]
    for i, price in enumerate(predicted_prices, 1):
        pred_date = base_date + timedelta(days=i)
        prediction_list.append({
            "day": i,
            "date": pred_date.strftime('%Y-%m-%d'),
            "predicted_price": round(float(price), 2)
        })
    
    return MultidayPredictionResponse(
        ticker=ticker.upper(),
        predictions=prediction_list,
        timestamp=datetime.now().isoformat()
    )

@app.delete("/models/{model_name}/cache", tags=["Models"])
async def clear_model_cache(model_name: str):
    """Clear cached model from memory"""
    if model_name in model_cache:
        del model_cache[model_name]
        return {"message": f"Cache cleared for model '{model_name}'"}
    return {"message": f"Model '{model_name}' not in cache"}

@app.delete("/models/cache/all", tags=["Models"])
async def clear_all_cache():
    """Clear all cached models from memory"""
    count = len(model_cache)
    model_cache.clear()
    return {"message": f"Cleared {count} models from cache"}

# ============================================================================
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================