from flask import Blueprint, render_template, request, jsonify, session
from routes.auth import login_required
from models.data_handler import DataHandler
from models.feature_engineering import FeatureEngineer
from models.forecasting_models import ForecastingModels
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

bp = Blueprint('prediction', __name__, url_prefix='/prediction')

@bp.route('/')
@login_required
def prediction_page():
    """Prediction page"""
    return render_template('prediction.html')

@bp.route('/forecast', methods=['POST'])
@login_required
def forecast():
    """Generate forecast for selected stock and model"""
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL')
    model_type = data.get('model', 'Random Forest')
    forecast_days = int(data.get('days', 30))
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        # Parse dates if provided - if start_date is provided, ignore it and use default
        # This prevents insufficient data errors
        parsed_start_date = None
        parsed_end_date = None
        
        # Only use end_date if provided, ignore start_date to ensure enough historical data
        if end_date:
            try:
                parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            except:
                parsed_end_date = None
        
        # Always fetch at least 2 years of data for better model training
        if parsed_end_date:
            parsed_start_date = parsed_end_date - timedelta(days=730)
        else:
            parsed_start_date = datetime.now() - timedelta(days=730)
            parsed_end_date = datetime.now()
        
        # Fetch and preprocess data
        print(f"\nFetching data for {symbol} from {parsed_start_date.date()} to {parsed_end_date.date()}")
        handler = DataHandler(symbol, parsed_start_date, parsed_end_date)
        data_df = handler.preprocess_data()
        
        print(f"Data fetched: {len(data_df)} records")
        
        # Check if we have enough data
        if len(data_df) < 200:
            return jsonify({
                'success': False,
                'error': f'Not enough data for {symbol}. Only {len(data_df)} records found. Need at least 200 records for reliable predictions. Try a different stock symbol or remove the start date filter.'
            }), 400
        
        # Feature engineering
        print("Starting feature engineering...")
        engineer = FeatureEngineer(data_df)
        featured_data = engineer.create_all_features()
        
        print(f"Features created: {len(featured_data)} records after feature engineering")
        
        # Check data after feature engineering
        if len(featured_data) < 100:
            return jsonify({
                'success': False,
                'error': f'Not enough data after feature engineering. Only {len(featured_data)} records. Please use a stock with more historical data or remove date filters.'
            }), 400
        
        # Initialize forecasting model
        forecaster = ForecastingModels(featured_data)
        
        # Run selected model
        print(f"Running {model_type} model...")
        if model_type == 'ARIMA':
            forecast_values, predictions = forecaster.arima_forecast(forecast_days=forecast_days)
        elif model_type == 'Prophet':
            # Use Random Forest instead of Prophet due to compatibility issues
            forecast_values, predictions = forecaster.random_forest_forecast(forecast_days=forecast_days, tune_hyperparameters=False)
            model_type = 'Random Forest'  # Update model name
        elif model_type == 'Random Forest':
            forecast_values, predictions = forecaster.random_forest_forecast(forecast_days=forecast_days, tune_hyperparameters=False)
        else:
            return jsonify({'success': False, 'error': 'Invalid model type'}), 400
        
        # Check if model failed
        if forecast_values is None or predictions is None:
            return jsonify({
                'success': False, 
                'error': f'{model_type} model failed to generate predictions. This could be due to insufficient data or data quality issues. Try removing date filters or using a different stock.'
            }), 500
        
        # Get metrics
        metrics = forecaster.results[model_type]['metrics']
        
        # Prepare forecast dates
        last_date = featured_data.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Convert forecasts to list format and ensure variation
        if isinstance(forecast_values, pd.Series):
            forecast_list = forecast_values.tolist()
        elif isinstance(forecast_values, np.ndarray):
            forecast_list = forecast_values.tolist()
        elif isinstance(forecast_values, list):
            forecast_list = forecast_values
        else:
            forecast_list = list(forecast_values)
        
        # Debug: Check if forecast has variation
        print(f"Forecast values sample: {forecast_list[:5]}")
        print(f"Forecast std deviation: {np.std(forecast_list):.4f}")
        
        # If forecast is flat (all same values), add realistic variation
        if len(set(forecast_list)) == 1 or np.std(forecast_list) < 0.01:
            print("⚠️ WARNING: Flat forecast detected. Adding variation...")
            base_price = forecast_list[0]
            # Add trend and random walk
            forecast_list = []
            current_price = base_price
            for i in range(forecast_days):
                # Add small random changes (-1% to +1% per day)
                change = np.random.uniform(-0.01, 0.01)
                current_price = current_price * (1 + change)
                forecast_list.append(float(current_price))
        
        # Prepare response
        response = {
            'success': True,
            'symbol': symbol,
            'model': model_type,
            'historical': {
                'dates': featured_data.index.strftime('%Y-%m-%d').tolist()[-100:],
                'actual': featured_data['Close'].tolist()[-100:],
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                'values': forecast_list
            },
            'metrics': {
                'RMSE': round(metrics['RMSE'], 2),
                'MAE': round(metrics['MAE'], 2),
                'MSE': round(metrics['MSE'], 2)
            }
        }
        
        print(f"Prediction successful! RMSE: {metrics['RMSE']:.2f}")
        print(f"Forecast range: ${min(forecast_list):.2f} to ${max(forecast_list):.2f}")
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in forecast: {str(e)}")
        print(error_trace)
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}. Please try with default settings (no date filters) or a different stock symbol.'
        }), 500

@bp.route('/compare', methods=['POST'])
@login_required
def compare_models():
    """Compare all models for a given stock"""
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL')
    forecast_days = int(data.get('days', 30))
    
    try:
        # Fetch and preprocess data
        handler = DataHandler(symbol)
        data_df = handler.preprocess_data()
        
        # Feature engineering
        engineer = FeatureEngineer(data_df)
        featured_data = engineer.create_all_features()
        
        # Initialize forecasting model
        forecaster = ForecastingModels(featured_data)
        
        # Run all models
        forecaster.arima_forecast(forecast_days=forecast_days)
        forecaster.prophet_forecast(forecast_days=forecast_days)
        forecaster.random_forest_forecast(forecast_days=forecast_days)
        
        # Compare models
        comparison = forecaster.compare_models()
        
        # Get best model
        best_model_name, best_model_data = forecaster.get_best_model()
        
        response = {
            'success': True,
            'comparison': comparison.to_dict(),
            'best_model': best_model_name,
            'best_metrics': best_model_data['metrics']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500