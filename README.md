# Two Sigma Financial Market Prediction

A machine learning pipeline for predicting 10-day stock market returns using LightGBM.

## Project Overview

This project implements a financial market prediction system that:
- Predicts 10-day stock returns using market and news data
- Processes financial market and news data with lag-based and sentiment features
- Implements Bayesian hyperparameter optimization for model tuning

### Basic Usage

```python
from src.pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline()

# Prepare training data
X, y = pipeline.prepare_training_data(market_data, news_data)

# Train model
model = pipeline.train_model(X, y)

# Make predictions
predictions = pipeline.predict_batch(market_obs, news_obs, template)
```

### Run Complete Pipeline

```bash
# Run main prediction pipeline
python src/main.py

# Run with validation
python src/main.py --validate

# Test pipeline functionality
python src/main.py --test
```

## Features

### Data Processing
- **Market Data**: Price outlier removal, trading day mapping
- **News Data**: Sentiment analysis, trading day alignment
- **Memory Optimization**: Automatic dtype optimization for large datasets

### Feature Engineering
- **Lag Features**: Rolling window statistics (3, 7, 14 days)
- **Market Features**: Price-volume ratios, daily returns
- **News Features**: Sentiment coverage, word count analysis
- **Parallel Processing**: Multiprocessing for efficient feature creation

### Model Training
- **LightGBM Classifier**: Optimized gradient boosting
- **Bayesian Optimization**: Automated hyperparameter tuning
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Binary Classification**: Converts returns to up/down predictions

## Configuration

Customize the pipeline via `src/config.py`:

```python
# Model parameters (Bayesian optimized)
LGBM_PARAMS = {
    'learning_rate': 0.10192437737356348,
    'num_leaves': 1011,
    'min_data_in_leaf': 399,
    'num_iteration': 500,
    'max_bin': 242
}

# Feature engineering
LAG_WINDOWS = [3, 7, 14]
RETURN_FEATURES = ['returnsClosePrevRaw1', 'open', 'close']
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
two-sigma-financial-prediction/
├── src/
│   ├── pipeline.py          # Main SOLID refactored classes
│   ├── main.py             # Execution script
│   └── config.py           # Configuration
├── tests/                  # Unit tests
├── notebooks/              # Original Jupyter notebook
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- lightgbm
- kaggle (for competition data)s