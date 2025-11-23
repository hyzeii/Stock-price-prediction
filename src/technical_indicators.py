import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators based on config"""
        df = df.copy()
        tech_config = self.config.get('features', {}).get('technical_indicators', {})
        
        # Simple Moving Averages
        for period in tech_config.get('sma', []):
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in tech_config.get('ema', []):
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        if 'macd' in tech_config:
            macd_config = tech_config['macd']
            fast = macd_config.get('fast', 12)
            slow = macd_config.get('slow', 26)
            signal = macd_config.get('signal', 9)
            
            df[f'EMA_{fast}'] = df['Close'].ewm(span=fast, adjust=False).mean()
            df[f'EMA_{slow}'] = df['Close'].ewm(span=slow, adjust=False).mean()
            df['MACD'] = df[f'EMA_{fast}'] - df[f'EMA_{slow}']
            df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # RSI
        if 'rsi' in tech_config:
            period = tech_config['rsi'].get('period', 14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'bollinger_bands' in tech_config:
            bb_config = tech_config['bollinger_bands']
            period = bb_config.get('period', 20)
            std_dev = bb_config.get('std_dev', 2)
            
            df['BB_Middle'] = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
        
        # ATR
        if 'atr' in tech_config:
            period = tech_config['atr'].get('period', 14)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(period).mean()
        
        # Volume SMA
        if 'volume_sma' in tech_config:
            period = tech_config['volume_sma'].get('period', 20)
            df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()
        
        # Rate of Change
        if 'roc' in tech_config:
            period = tech_config['roc'].get('period', 10)
            df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df