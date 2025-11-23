#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


from src.predictor import StockLSTMPredictor
from src.config import config_manager

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for stock prediction')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--lookback', type=int, help='Lookback period (overrides config)')
    parser.add_argument('--epochs', type=int, help='Training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--model-name', type=str, help='Custom model name')
    
    args = parser.parse_args()
    
    # Get model config
    model_config = config_manager.get_model_config(args.ticker)
    
    # Override with command line arguments
    if args.lookback:
        model_config['lookback'] = args.lookback
    if args.epochs:
        model_config['training']['epochs'] = args.epochs
    if args.batch_size:
        model_config['training']['batch_size'] = args.batch_size
    
    # Initialize predictor
    predictor = StockLSTMPredictor(
        ticker=args.ticker,
        config=model_config
    )
    
    print(f"Training model for {args.ticker}")
    print(f"Configuration: {model_config}")
    
    # Fetch and prepare data
    predictor.fetch_data()
    predictor.calculate_technical_indicators()
    predictor.prepare_data()
    
    # Build and train
    predictor.build_model()
    predictor.train_model()
    
    # Evaluate
    metrics = predictor.evaluate_model()
    
    # Save model
    model_name = args.model_name or f"{args.ticker}_model"
    predictor.save_model(model_name=model_name)
    
    print(f"Training complete!")
    print(f"Test RMSE: ${metrics['test_rmse']:.2f}")
    print(f"Test MAE: ${metrics['test_mae']:.2f}")
    print(f"Test MAPE: {metrics['test_mape']:.2f}%")

if __name__ == "__main__":
    main()