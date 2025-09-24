from typing import List, Dict, Any

class Config:
    """Main configuration class"""
    
    # Data Processing Settings
    TRAIN_CUTOFF_DATE = 20101231
    MEMORY_OPTIMIZE = True
    
    # Feature Engineering Settings
    LAG_WINDOWS = [3, 7, 14]
    SHIFT_SIZE = 1
    RETURN_FEATURES = [
        'returnsClosePrevMktres10', 'returnsClosePrevRaw10',
        'returnsOpenPrevMktres1', 'returnsOpenPrevRaw1',
        'open', 'close'
    ]
    
    # Model Parameters (optimized via Bayesian optimization)
    LGBM_PARAMS = {
        'learning_rate': 0.10192437737356348,
        'num_leaves': 1011,
        'min_data_in_leaf': 399,
        'num_iteration': 500,
        'max_bin': 242
    }
    
    # Training Settings
    BOOSTING_TYPE = 'gbdt'
    TEST_SIZE = 0.25
    RANDOM_STATE = 0
    EARLY_STOPPING_ROUNDS = 10
    
    # Feature Selection
    DROP_COLUMNS = ['assetCode', 'assetName', 'marketCommentary', 'time']
    
    # Prediction Settings
    CONFIDENCE_CLIP_MIN = -0.99
    CONFIDENCE_CLIP_MAX = 0.99
    
    # Parallel Processing
    N_JOBS = 4
    
    # Validation Settings
    VALIDATION_VERBOSE = True


class AlternativeConfigs:
    """Alternative configurations for experimentation"""
    
    FAST_TRAINING = {
        'LGBM_PARAMS': {
            'learning_rate': 0.2,
            'num_leaves': 31,
            'min_data_in_leaf': 100,
            'num_iteration': 100,
            'max_bin': 255
        },
        'LAG_WINDOWS': [3, 7],
        'EARLY_STOPPING_ROUNDS': 5
    }
    
    CONSERVATIVE_MODEL = {
        'LGBM_PARAMS': {
            'learning_rate': 0.05,
            'num_leaves': 2000,
            'min_data_in_leaf': 500,
            'num_iteration': 1000,
            'max_bin': 300
        },
        'CONFIDENCE_CLIP_MIN': -0.5,
        'CONFIDENCE_CLIP_MAX': 0.5
    }


# Default configuration instance
config = Config()
