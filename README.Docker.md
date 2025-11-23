# Stock Price Predictor - Dockerized

## Quick Start

1. Copy environment file:
```bash
cp .env.example .env
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

3. API available at: http://localhost:8000

## Training Models

Train inside container:
```bash
docker-compose exec stock-predictor python scripts/train_model.py AAPL
```

Train with custom parameters:
```bash
docker-compose exec stock-predictor python scripts/train_model.py TSLA --epochs 100 --lookback 90
```

## Configuration

### Environment Variables (.env)
- Modify `.env` for runtime settings
- Changes take effect on container restart

### Model Configuration (config/model_config.yaml)
- Define per-stock configurations
- Set default hyperparameters
- Add new technical indicators

### API Configuration (config/default_config.yaml)
- API settings
- Caching configuration
- Logging preferences

## Volumes

- `./models` - Trained models (persisted)
- `./data` - Downloaded stock data
- `./logs` - Application logs
- `./config` - Configuration files

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /models` - List models
- `GET /predict/{ticker}` - Single day prediction
- `GET /predict/{ticker}/multiday?days=5` - Multi-day prediction
- `GET /docs` - Interactive API documentation

## Adding New Stocks

1. Add configuration in `config/model_config.yaml`:
```yaml
GOOGL:
  lookback: 60
  training:
    epochs: 75
  architecture:
    lstm_units: [256, 128]
```

2. Train the model:
```bash
docker-compose exec stock-predictor python scripts/train_model.py GOOGL
```

3. Use the API:
```bash
curl http://localhost:8000/predict/GOOGL
```