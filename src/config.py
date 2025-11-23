import os
import yaml
from typing import Dict, Any, Optional

class Settings:
    def __init__(self):
        self.model_dir = os.getenv('MODEL_DIR', './models')
        self.data_dir = os.getenv('DATA_DIR', './data')
        self.cache_dir = os.getenv('CACHE_DIR', './cache')
        self.log_dir = os.getenv('LOG_DIR', './logs')
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))

class ConfigManager:
    def __init__(self):
        self.settings = Settings()
        self.config = self._load_yaml('./config/default_config.yaml')
        self.model_config = self._load_yaml('./config/model_config.yaml')
        
    def _load_yaml(self, path):
        try:
            if os.path.exists(path):
                with open(path) as f:
                    return yaml.safe_load(f) or {}
        except: pass
        return {}
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def get_model_config(self, ticker=None):
        default = self.model_config.get('default', {})
        if ticker and ticker.upper() in self.model_config:
            return {**default, **self.model_config[ticker.upper()]}
        return default
    
    def get_features_list(self, ticker=None):
        config = self.get_model_config(ticker)
        features = config.get('features', {}).get('base', ['Close', 'Volume'])
        tech = config.get('features', {}).get('technical_indicators', {})
        for p in tech.get('sma', []): features.append(f'SMA_{p}')
        for p in tech.get('ema', []): features.append(f'EMA_{p}')
        if 'macd' in tech: features.extend(['MACD', 'Signal_Line'])
        if 'rsi' in tech: features.append('RSI')
        if 'bollinger_bands' in tech: features.extend(['BB_Upper', 'BB_Middle', 'BB_Lower'])
        if 'atr' in tech: features.append('ATR')
        if 'volume_sma' in tech: features.append('Volume_SMA')
        if 'roc' in tech: features.append('ROC')
        return features

config_manager = ConfigManager()