import gc
import sys
import argparse
import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

from pipeline import create_pipeline


def main(validate_first: bool = False):
    """Main execution function"""
    
    print("Initializing Two Sigma environment...")
    env = twosigmanews.make_env()
    
    print("Loading training data...")
    (env_market_train, env_news_train) = env.get_training_data()
    print(f"Market data: {env_market_train.shape}")
    print(f"News data: {env_news_train.shape}")
    
    print("Creating prediction pipeline...")
    pipeline = create_pipeline()
    
    if validate_first:
        print("Running validation...")
        run_validation(pipeline, env_market_train.copy(), env_news_train.copy())
    
    print("Processing and engineering features...")
    X, y = pipeline.prepare_training_data(env_market_train, env_news_train)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    del env_market_train, env_news_train
    gc.collect()
    
    print("Training LightGBM model...")
    model = pipeline.train_model(X, y)
    print("Model training completed")
    
    del X, y
    gc.collect()
    
    print("Starting prediction loop...")
    days = env.get_prediction_days()
    
    day_count = 0
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        day_count += 1
        
        try:
            print(f"Processing day {day_count}...")
            
            predictions = pipeline.predict_batch(
                market_obs_df, 
                news_obs_df, 
                predictions_template_df
            )
            
            print(f"Generated {len(predictions)} predictions")
            print(f"Confidence range: [{predictions['confidenceValue'].min():.3f}, {predictions['confidenceValue'].max():.3f}]")
            
            env.predict(predictions)
            print("Predictions submitted successfully")
            
            gc.collect()
            
        except Exception as e:
            print(f"Error on day {day_count}: {str(e)}")
            predictions_template_df = predictions_template_df.copy()
            predictions_template_df['confidenceValue'] = 0.0
            env.predict(predictions_template_df)
            print("Submitted zero predictions as fallback")
    
    print(f"Completed predictions for {day_count} days")
    
    print("Writing submission file...")
    env.write_submission_file()
    print("Submission file created successfully")


def run_validation(pipeline, market_data, news_data):
    """Run model validation before main prediction loop"""
    
    print("Running model validation...")
    
    X, y = pipeline.prepare_training_data(market_data, news_data)
    model = pipeline.train_model(X, y, test_size=0.2)
    
    print("Validation completed")
    
    del X, y, market_data, news_data
    gc.collect()


def run_quick_test():
    """Quick test to verify pipeline works"""
    
    print("Running quick pipeline test...")
    
    try:
        pipeline = create_pipeline()
        print("Pipeline created successfully")
        
        mock_market = pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01']),
            'assetCode': ['TEST'],
            'assetName': ['TEST'],
            'open': [100.0],
            'close': [105.0],
            'volume': [1000000],
            'returnsClosePrevRaw1': [0.01],
            'returnsOpenPrevRaw1': [0.01],
            'returnsClosePrevMktres1': [0.01],
            'returnsOpenPrevMktres1': [0.01],
            'returnsClosePrevRaw10': [0.01],
            'returnsOpenPrevRaw10': [0.01],
            'returnsClosePrevMktres10': [0.01],
            'returnsOpenPrevMktres10': [0.01],
            'returnsOpenNextMktres10': [0.05]
        })
        
        mock_news = pd.DataFrame({
            'time': pd.to_datetime(['2024-01-01']),
            'assetName': ['TEST'],
            'sentimentWordCount': [10],
            'wordCount': [100]
        })
        
        X, y = pipeline.prepare_training_data(mock_market, mock_news)
        print(f"Data processing successful - Features: {X.shape}, Target: {y.shape}")
        
        print("Quick test passed - Pipeline is ready to run")
        
    except Exception as e:
        print(f"Quick test failed: {str(e)}")
        print("Please check your setup and dependencies")
        return False
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Two Sigma Financial Market Prediction')
    parser.add_argument('--validate', action='store_true', 
                       help='Run validation before main prediction loop')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test of pipeline functionality')
    
    args = parser.parse_args()
    
    if args.test:
        if run_quick_test():
            print("\nReady to run main pipeline!")
            print("Run: python main.py")
    else:
        try:
            main(validate_first=args.validate)
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user")
        except Exception as e:
            print(f"\nPipeline failed with error: {str(e)}")
            print("Try running 'python main.py --test' first")
            sys.exit(1)
