__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .pipeline import (
    TwoSigmaPredictionPipeline,
    create_pipeline,
    MarketDataProcessor,
    NewsDataProcessor,
    LagFeatureEngineer,
    LGBMModelTrainer,
    ModelPredictor
)

from .config import Config, AlternativeConfigs

__all__ = [
    "TwoSigmaPredictionPipeline",
    "create_pipeline",
    "MarketDataProcessor", 
    "NewsDataProcessor",
    "LagFeatureEngineer",
    "LGBMModelTrainer",
    "ModelPredictor",
    "Config",
    "AlternativeConfigs"
]
