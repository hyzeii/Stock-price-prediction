from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Handle imports for both direct run and module import
try:
    from .config import config_manager
    from .predictor import StockLSTMPredictor
    from .technical_indicators import TechnicalIndicators
except ImportError:
    from config import config_manager
    from predictor import StockLSTMPredictor
    from technical_indicators import TechnicalIndicators

# Initialize FastAPI app
app = FastAPI(
    title=config_manager.get('api.title', 'Stock Price Prediction API'),
    version=config_manager.get('api.version', '1.0.0'),
    debug=config_manager.get('api.debug', False)
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
model_cache = {}

# Pydantic models
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
    config: Optional[dict] = None

class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]
    count: int

class ConfigResponse(BaseModel):
    ticker: Optional[str] = None
    config: dict

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": config_manager.get('api.title'),
        "version": config_manager.get('api.version'),
        "endpoints": {
            "health": "/health",
            "predict": "/predict/{ticker}",
            "config": "/config/{ticker}",
            "models": "/models",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_cache),
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=AvailableModelsResponse)
async def list_models():
    """List all available trained models"""
    models = []
    model_dir = config_manager.settings.model_dir
    
    if not os.path.exists(model_dir):
        return AvailableModelsResponse(models=[], count=0)
    
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            metadata_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                models.append(ModelInfo(
                    name=item,
                    ticker=metadata.get('ticker', 'unknown'),
                    trained_date=metadata.get('trained_date', 'unknown'),
                    lookback=metadata.get('lookback', 60),
                    config=metadata.get('config')
                ))
    
    return AvailableModelsResponse(models=models, count=len(models))

@app.get("/config/{ticker}", response_model=ConfigResponse)
async def get_config(ticker: str):
    """Get configuration for a specific ticker"""
    config = config_manager.get_model_config(ticker)
    return ConfigResponse(ticker=ticker, config=config)

@app.get("/features/{ticker}")
async def get_features(ticker: str):
    """Get list of features for a ticker"""
    features = config_manager.get_features_list(ticker)
    return {"ticker": ticker, "features": features, "count": len(features)}

@app.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_next_day(
    ticker: str = Path(..., description="Stock ticker symbol"),
    model_name: Optional[str] = Query(None, description="Model name")
):
    """Predict next day's closing price"""
    if model_name is None:
        model_name = f"{ticker.upper()}_model"
    
    try:
        if model_name not in model_cache:
            predictor = StockLSTMPredictor(ticker=ticker.upper())
            predictor.load_model(model_name)
            model_cache[model_name] = predictor
        else:
            predictor = model_cache[model_name]
        
        # Fetch latest data
        predictor.fetch_data()
        predictor.calculate_technical_indicators()
        
        # Prepare data for prediction
        features = config_manager.get_features_list(ticker)
        available_features = [f for f in features if f in predictor.data_with_indicators.columns]
        
        feature_data = predictor.data_with_indicators[available_features].values
        scaled_data = predictor.feature_scaler.transform(feature_data)
        sequence = scaled_data[-predictor.lookback:].reshape(1, predictor.lookback, -1)
        
        # Predict
        prediction = predictor.model.predict(sequence, verbose=0)
        predicted_price = predictor.inverse_transform_predictions(prediction)[0]
        
        # Calculate confidence
        recent_std = predictor.data_with_indicators['Close'].tail(20).std()
        recent_mean = predictor.data_with_indicators['Close'].tail(20).mean()
        volatility = (recent_std / recent_mean) * 100
        
        thresholds = config_manager.get('prediction.confidence_thresholds', {})
        if volatility < thresholds.get('high', 2.0):
            confidence = "high"
        elif volatility < thresholds.get('medium', 5.0):
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            ticker=ticker.upper(),
            prediction=round(float(predicted_price), 2),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{ticker}/multiday", response_model=MultidayPredictionResponse)
async def predict_multiple_days(
    ticker: str = Path(..., description="Stock ticker symbol"),
    days: int = Query(5, ge=1, le=30, description="Number of days"),
    model_name: Optional[str] = Query(None, description="Model name")
):
    """Predict multiple days ahead"""
    if model_name is None:
        model_name = f"{ticker.upper()}_model"
    
    try:
        if model_name not in model_cache:
            predictor = StockLSTMPredictor(ticker=ticker.upper())
            predictor.load_model(model_name)
            model_cache[model_name] = predictor
        else:
            predictor = model_cache[model_name]
        
        # Fetch latest data
        predictor.fetch_data()
        predictor.calculate_technical_indicators()
        
        # Get predictions
        predictions = predictor.predict_next_days(n_days=days)
        
        # Format response
        prediction_list = []
        base_date = predictor.data_with_indicators.index[-1]
        for i, price in enumerate(predictions, 1):
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
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config_manager.settings.api_host,
        port=config_manager.settings.api_port
    )

