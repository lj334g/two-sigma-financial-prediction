"""
Two Sigma Financial Market Prediction
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import gc
from multiprocessing import Pool


class IDataProcessor(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IFeatureEngineer(ABC):
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IModelTrainer(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: np.ndarray) -> object:
        pass


class IPredictor(ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass


class DateTimeConverter:
    @staticmethod
    def date_to_int(date: datetime) -> int:
        return 10000 * date.year + 100 * date.month + date.day
    
    @staticmethod
    def int_to_date(date_int: int) -> datetime:
        return datetime.strptime(str(date_int), '%Y%m%d')


class MemoryOptimizer:
    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        start_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            print(f'Memory usage: {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                c_min, c_max = df[col].min(), df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            print(f'Optimized memory: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        
        return df


class MarketDataProcessor(IDataProcessor):
    def __init__(self, train_cutoff: int = 20101231):
        self.train_cutoff = train_cutoff
        self.datetime_converter = DateTimeConverter()
    
    def process(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Process market data with outlier removal and feature creation"""
        processed_data = market_data.copy()
        processed_data = self._replace_price_outliers(processed_data)
        processed_data['time'] = processed_data['time'].dt.strftime("%Y%m%d").astype(int)
        processed_data = processed_data[processed_data.time >= self.train_cutoff]
        processed_data['pricevolume'] = processed_data['volume'] / processed_data['close']
        
        return processed_data
    
    def _replace_price_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove unrealistic price movements"""
        data['dailychange'] = data['close'] / data['open']
        data.loc[data['dailychange'] < 0.33, 'open'] = data.loc[data['dailychange'] < 0.33, 'close']
        data.loc[data['dailychange'] > 2, 'close'] = data.loc[data['dailychange'] > 2, 'open']
        return data


class NewsDataProcessor(IDataProcessor):
    def __init__(self, train_cutoff: int = 20101231):
        self.train_cutoff = train_cutoff
        self.trading_days: Optional[np.ndarray] = None
    
    def set_trading_days(self, trading_days: np.ndarray):
        """Set available trading days for mapping news to trading days"""
        self.trading_days = trading_days
    
    def process(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Process news data and map to trading days"""
        if self.trading_days is None:
            raise ValueError("Trading days must be set before processing news data")
        
        processed_data = news_data.copy()
        processed_data['time'] = processed_data['time'].dt.strftime("%Y%m%d").astype(int)
        
        # Filter by cutoff date
        processed_data = processed_data[processed_data.time >= self.train_cutoff]
        
        # Map to trading days
        processed_data['time'] = processed_data['time'].apply(self._map_trading_day)
        
        # Create news features
        processed_data['coverage'] = (
            processed_data['sentimentWordCount'] / processed_data['wordCount']
        )
        
        return processed_data
    
    def _map_trading_day(self, news_date: int) -> int:
        """Map news date to nearest future trading day"""
        if news_date in self.trading_days:
            return news_date
        
        values = self.trading_days - news_date
        mask = values >= 0
        try:
            return self.trading_days[mask][0]
        except IndexError:
            return 0


class DataMerger:
    @staticmethod
    def merge_market_news(market_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Merge market and news data"""
        # Aggregate news data by time and asset
        news_agg = news_df.groupby(['time', 'assetName'], sort=False).agg(np.mean).reset_index()
        
        # Merge with market data
        merged = pd.merge(market_df, news_agg, how='left', on=['time', 'assetName'], copy=False)
        merged.fillna(value=0, inplace=True)
        
        return merged


class LagFeatureEngineer(IFeatureEngineer):
    def __init__(self, 
                 asset_id: str = 'assetCode',
                 lag_windows: List[int] = None,
                 shift_size: int = 1,
                 return_features: List[str] = None):
        
        self.asset_id = asset_id
        self.lag_windows = lag_windows or [3, 7, 14]
        self.shift_size = shift_size
        self.return_features = return_features or [
            'returnsClosePrevMktres10', 'returnsClosePrevRaw10',
            'returnsOpenPrevMktres1', 'returnsOpenPrevRaw1',
            'open', 'close'
        ]
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for the dataset"""
        # Select base features
        base_features = ['time', self.asset_id, 'volume', 'close', 'open',
                        'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
                        'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                        'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
                        'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
        
        feature_df = df.loc[:, base_features].copy()
        
        # Generate lag features by asset
        lag_features_df = self._generate_lag_features_parallel(feature_df)
        result = pd.merge(df, lag_features_df, how='left', on=['time', self.asset_id])
        
        return self._impute_missing_values(result)
    
    def _generate_lag_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lag features using multiprocessing"""
        # Group by asset
        grouped_data = [group[1][['time', self.asset_id] + self.return_features] 
                       for group in df.groupby(self.asset_id)]
        
        # Process in parallel
        with Pool(4) as pool:
            processed_groups = pool.map(self._create_lag_for_asset, grouped_data)
        
        lag_features_df = pd.concat(processed_groups)
        lag_features_df.drop(self.return_features, axis=1, inplace=True)
        
        return lag_features_df
    
    def _create_lag_for_asset(self, asset_df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for a single asset"""
        for feature in self.return_features:
            for window in self.lag_windows:
                rolled = asset_df[feature].shift(self.shift_size).rolling(window=window)
                
                asset_df[f'{feature}_lag_{window}_mean'] = rolled.mean()
                asset_df[f'{feature}_lag_{window}_max'] = rolled.max()
                asset_df[f'{feature}_lag_{window}_min'] = rolled.min()
        
        return asset_df.fillna(-1)
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values based on data type"""
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("other")
            elif df[col].dtype in ["int32", "float32", "int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
        
        return df


class LGBMModelTrainer(IModelTrainer):
    def __init__(self, model_params: Dict = None, boosting_type: str = 'gbdt'):
        self.model_params = model_params or {
            'learning_rate': 0.10192437737356348,
            'num_leaves': 1011,
            'min_data_in_leaf': 399,
            'num_iteration': 500,
            'max_bin': 242
        }
        self.boosting_type = boosting_type
        self.model = None
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray, 
              X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> LGBMClassifier:
        """Train the LightGBM model"""
        self.model = LGBMClassifier(
            boosting_type=self.boosting_type,
            **self.model_params,
            verbose=1,
            n_jobs=-1
        )
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['binary_logloss'],
                verbose=True,
                early_stopping_rounds=10
            )
        else:
            self.model.fit(X_train, y_train)
        
        return self.model


class ModelPredictor(IPredictor):    
    def __init__(self, model: object, scaler: MinMaxScaler = None):
        self.model = model
        self.scaler = scaler or MinMaxScaler()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make scaled predictions"""
        raw_predictions = self.model.predict_proba(X)
        scaled_predictions = self.scaler.fit_transform(raw_predictions)[:, 1]
        return np.clip(scaled_predictions * 2 - 1, -0.99, 0.99)


class TwoSigmaPredictionPipeline:    
    def __init__(self,
                 market_processor: IDataProcessor,
                 news_processor: IDataProcessor,
                 feature_engineer: IFeatureEngineer,
                 model_trainer: IModelTrainer,
                 predictor: IPredictor = None):
        
        self.market_processor = market_processor
        self.news_processor = news_processor
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.predictor = predictor
        
        self.data_merger = DataMerger()
        self.memory_optimizer = MemoryOptimizer()
        self.model = None
        self.feature_columns = None
    
    def prepare_training_data(self, market_data: pd.DataFrame, 
                            news_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        
        # Process market data
        processed_market = self.market_processor.process(market_data)
        if hasattr(self.news_processor, 'set_trading_days'):
            self.news_processor.set_trading_days(processed_market['time'].unique())
        
        processed_news = self.news_processor.process(news_data)
        merged_data = self.data_merger.merge_market_news(processed_market, processed_news)
        target = merged_data.pop('returnsOpenNextMktres10').values
        featured_data = self.feature_engineer.create_features(merged_data)
        featured_data = self.memory_optimizer.reduce_memory_usage(featured_data)
        
        # Select feature columns
        drop_cols = ['assetCode', 'assetName', 'marketCommentary', 'time']
        self.feature_columns = [c for c in featured_data.columns if c not in drop_cols]
        
        return featured_data[self.feature_columns], target
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray, 
                   test_size: float = 0.25, random_state: int = 0) -> object:
        """Train the model with train/validation split"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Convert to binary classification
        y_train_binary = np.where(y_train > 0, 1, 0).astype(int)
        y_val_binary = np.where(y_val > 0, 1, 0).astype(int)
        self.model = self.model_trainer.train(X_train, y_train_binary, X_val, y_val_binary)
        
        # Setup predictor
        if self.predictor is None:
            self.predictor = ModelPredictor(self.model)
        
        return self.model
    
    def predict_batch(self, market_obs: pd.DataFrame, 
                     news_obs: pd.DataFrame, 
                     predictions_template: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for a batch of observations"""
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Process new data
        processed_market = self.market_processor.process(market_obs)
        processed_news = self.news_processor.process(news_obs)
        
        # Merge and engineer features
        merged = self.data_merger.merge_market_news(processed_market, processed_news)
        featured = self.feature_engineer.create_features(merged)
        
        # Align with template
        ordered_df = pd.merge(predictions_template, featured, 
                            how='left', on='assetCode')
        
        # Make predictions
        X_pred = ordered_df[self.feature_columns]
        predictions = self.predictor.predict(X_pred)
        
        # Update template
        predictions_template = predictions_template.copy()
        predictions_template['confidenceValue'] = predictions
        
        return predictions_template


def create_pipeline() -> TwoSigmaPredictionPipeline:
    # Create components
    market_processor = MarketDataProcessor(train_cutoff=20101231)
    news_processor = NewsDataProcessor(train_cutoff=20101231)
    feature_engineer = LagFeatureEngineer()
    model_trainer = LGBMModelTrainer()
    
    # Create pipeline
    pipeline = TwoSigmaPredictionPipeline(
        market_processor=market_processor,
        news_processor=news_processor,
        feature_engineer=feature_engineer,
        model_trainer=model_trainer
    )
    
    return pipeline
