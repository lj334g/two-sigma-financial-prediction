"""
Unit tests for Two Sigma Financial Prediction Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import (
    DateTimeConverter,
    MemoryOptimizer,
    MarketDataProcessor,
    NewsDataProcessor,
    LagFeatureEngineer,
    LGBMModelTrainer,
    ModelPredictor,
    DataMerger,
    TwoSigmaPredictionPipeline,
    create_pipeline
)


class TestDateTimeConverter:
    """Test datetime conversion utilities"""
    
    def test_date_to_int(self):
        date = datetime(2024, 3, 15)
        result = DateTimeConverter.date_to_int(date)
        assert result == 20240315
    
    def test_int_to_date(self):
        date_int = 20240315
        result = DateTimeConverter.int_to_date(date_int)
        assert result == datetime(2024, 3, 15)
    
    def test_round_trip_conversion(self):
        original_date = datetime(2023, 12, 31)
        date_int = DateTimeConverter.date_to_int(original_date)
        converted_back = DateTimeConverter.int_to_date(date_int)
        assert original_date == converted_back


class TestMemoryOptimizer:
    """Test memory optimization functionality"""
    
    def test_reduce_memory_usage(self):
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'large_int': [1000000, 2000000, 3000000, 4000000, 5000000]
        })
        
        original_memory = df.memory_usage().sum()
        optimized_df = MemoryOptimizer.reduce_memory_usage(df, verbose=False)
        optimized_memory = optimized_df.memory_usage().sum()
        
        assert optimized_memory <= original_memory
        pd.testing.assert_frame_equal(df, optimized_df)


class TestMarketDataProcessor:
    """Test market data processing"""
    
    @pytest.fixture
    def sample_market_data(self):
        return pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'assetName': ['AAPL', 'GOOGL', 'MSFT'],
            'open': [100.0, 200.0, 150.0],
            'close': [105.0, 195.0, 155.0],
            'volume': [1000000, 2000000, 1500000]
        })
    
    def test_process_normal_data(self, sample_market_data):
        processor = MarketDataProcessor(train_cutoff=20240101)
        result = processor.process(sample_market_data)
        
        assert all(isinstance(t, int) for t in result['time'])
        assert 'pricevolume' in result.columns
        assert len(result) > 0
    
    def test_price_outlier_removal(self):
        df = pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'open': [100.0, 100.0],
            'close': [20.0, 300.0],
            'volume': [1000000, 1000000]
        })
        
        processor = MarketDataProcessor(train_cutoff=20240101)
        result = processor.process(df)
        
        assert result.iloc[0]['open'] == result.iloc[0]['close']
        assert result.iloc[1]['close'] == result.iloc[1]['open']


class TestNewsDataProcessor:
    """Test news data processing"""
    
    @pytest.fixture
    def sample_news_data(self):
        return pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'assetName': ['AAPL', 'GOOGL'],
            'sentimentWordCount': [10, 15],
            'wordCount': [100, 200]
        })
    
    @pytest.fixture  
    def trading_days(self):
        return np.array([20240101, 20240102, 20240103])
    
    def test_process_with_trading_days(self, sample_news_data, trading_days):
        processor = NewsDataProcessor(train_cutoff=20240101)
        processor.set_trading_days(trading_days)
        
        result = processor.process(sample_news_data)
        
        assert 'coverage' in result.columns
        expected_coverage = sample_news_data['sentimentWordCount'] / sample_news_data['wordCount']
        assert np.allclose(result['coverage'], expected_coverage)
    
    def test_trading_day_mapping(self, trading_days):
        processor = NewsDataProcessor()
        processor.set_trading_days(trading_days)
        
        assert processor._map_trading_day(20240101) == 20240101
        assert processor._map_trading_day(20231231) == 20240101


class TestLagFeatureEngineer:
    """Test lag feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'time': [20240101, 20240102, 20240103, 20240104, 20240105] * 2,
            'assetCode': ['AAPL'] * 5 + ['GOOGL'] * 5,
            'returnsClosePrevRaw1': [0.01, 0.02, -0.01, 0.03, -0.02] * 2,
            'open': [100, 101, 102, 103, 104] * 2,
            'close': [101, 103, 101, 106, 102] * 2,
            'volume': [1000000] * 10,
            'returnsOpenPrevRaw1': [0.01] * 10,
            'returnsClosePrevMktres1': [0.01] * 10,
            'returnsOpenPrevMktres1': [0.01] * 10,
            'returnsClosePrevRaw10': [0.01] * 10,
            'returnsOpenPrevRaw10': [0.01] * 10,
            'returnsClosePrevMktres10': [0.01] * 10,
            'returnsOpenPrevMktres10': [0.01] * 10
        })
    
    def test_create_features(self, sample_data):
        engineer = LagFeatureEngineer(lag_windows=[3], return_features=['open', 'close'])
        result = engineer.create_features(sample_data)
        
        lag_columns = [col for col in result.columns if 'lag' in col]
        assert len(lag_columns) > 0
        
        assert any('open_lag_3_mean' in col for col in result.columns)
        assert any('close_lag_3_mean' in col for col in result.columns)


class TestLGBMModelTrainer:
    """Test LGBM model training"""
    
    @pytest.fixture
    def sample_training_data(self):
        X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_train_model(self, sample_training_data):
        X, y = sample_training_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        trainer = LGBMModelTrainer(model_params={
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'num_iteration': 10,
            'max_bin': 255
        })
        
        model = trainer.train(X_train, y_train, X_val, y_val)
        
        assert model is not None
        predictions = model.predict_proba(X_val)
        assert predictions.shape == (len(X_val), 2)


class TestTwoSigmaPredictionPipeline:
    """Test main pipeline orchestration"""
    
    def test_create_pipeline(self):
        pipeline = create_pipeline()
        assert isinstance(pipeline, TwoSigmaPredictionPipeline)
        assert pipeline.market_processor is not None
        assert pipeline.news_processor is not None
        assert pipeline.feature_engineer is not None
        assert pipeline.model_trainer is not None


class TestIntegration:
    """Integration tests for full pipeline"""
    
    @pytest.fixture
    def mock_market_data(self):
        return pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01'] * 3),
            'assetCode': ['AAPL', 'GOOGL', 'MSFT'],
            'assetName': ['AAPL', 'GOOGL', 'MSFT'],
            'open': [100.0, 200.0, 150.0],
            'close': [105.0, 195.0, 155.0],
            'volume': [1000000, 2000000, 1500000],
            'returnsClosePrevRaw1': [0.01, 0.02, 0.03],
            'returnsOpenPrevRaw1': [0.01, 0.02, 0.03],
            'returnsClosePrevMktres1': [0.01, 0.02, 0.03],
            'returnsOpenPrevMktres1': [0.01, 0.02, 0.03],
            'returnsClosePrevRaw10': [0.01, 0.02, 0.03],
            'returnsOpenPrevRaw10': [0.01, 0.02, 0.03],
            'returnsClosePrevMktres10': [0.01, 0.02, 0.03],
            'returnsOpenPrevMktres10': [0.01, 0.02, 0.03],
            'returnsOpenNextMktres10': [0.01, -0.02, 0.03]
        })
    
    @pytest.fixture
    def mock_news_data(self):
        return pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01'] * 3),
            'assetName': ['AAPL', 'GOOGL', 'MSFT'],
            'sentimentWordCount': [10, 15, 5],
            'wordCount': [100, 200, 50]
        })
    
    def test_pipeline_end_to_end(self, mock_market_data, mock_news_data):
        """Test the entire pipeline works end-to-end"""
        pipeline = create_pipeline()
        
        X, y = pipeline.prepare_training_data(mock_market_data, mock_news_data)
        
        assert len(X) == len(y)
        assert len(X) > 0
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
