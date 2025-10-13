from flask import Blueprint, render_template, request, jsonify, session
from routes.auth import login_required
from models.data_handler import DataHandler
from models.feature_engineering import FeatureEngineer
from models.forecasting_models import ForecastingModels
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay # Importing Business Day offset for last_date check

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
        # --- Date Handling (Ensuring enough historical data) ---
        parsed_start_date = None
        parsed_end_date = None
        
        # Only use end_date if provided, ignore start_date to ensure enough historical data
        if end_date:
            try:
                # Handle M/D/YYYY format
                parsed_end_date = datetime.strptime(end_date, '%m/%d/%Y')
            except:
                try:
                    # Handle YYYY-M-D format
                    parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except:
                    parsed_end_date = datetime.now()
        else:
            parsed_end_date = datetime.now()
        
        # Always fetch at least 2 years (730 days) of data for better model training
        parsed_start_date = parsed_end_date - timedelta(days=730)

        
        # Fetch and preprocess data
        print(f"\nFetching data for {symbol} from {parsed_start_date.date()} to {parsed_end_date.date()}")
        handler = DataHandler(symbol, parsed_start_date, parsed_end_date)
        data_df = handler.preprocess_data()
        
        print(f"Data fetched: {len(data_df)} records")
        
        # Check if we have enough data
        if len(data_df) < 200:
            return jsonify({
                'success': False,
                'error': f'Not enough data for {symbol}. Only {len(data_df)} records found. Need at least 200 records for reliable predictions.'
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
            metrics = forecaster.results['ARIMA']['metrics']
        elif model_type == 'Prophet':
             # Use Random Forest instead of Prophet due to compatibility issues/if Prophet is missing
            forecast_values, predictions = forecaster.random_forest_forecast(forecast_days=forecast_days, tune_hyperparameters=False)
            model_type = 'Random Forest'  # Update model name
            metrics = forecaster.results['Random Forest']['metrics']
        elif model_type == 'Random Forest':
            forecast_values, predictions = forecaster.random_forest_forecast(forecast_days=forecast_days, tune_hyperparameters=False)
            metrics = forecaster.results['Random Forest']['metrics']
        else:
            return jsonify({'success': False, 'error': 'Invalid model type'}), 400
        
        # Check if model failed
        if forecast_values is None or predictions is None:
            return jsonify({
                'success': False, 
                'error': f'{model_type} model failed to generate predictions. This could be due to insufficient data or data quality issues.'
            }), 500
        
        # --- FIX: Prepare Business Day Forecast Dates ---
        last_date = featured_data.index[-1].to_pydatetime()
        
        # Determine the next business day start point
        # pandas BDay automatically jumps to Monday if the last_date is Friday/Weekend.
        start_date_for_range = last_date + BDay(1)
        
        # Use pandas bdate_range (Business Date Range) to get only Mon-Fri
        # 'B' frequency automatically excludes weekends
        forecast_dates_index = pd.bdate_range(
            start=start_date_for_range, 
            periods=forecast_days, 
            freq='B' 
        )
        forecast_dates = forecast_dates_index.to_list()
        
        # --- End FIX ---
        
        # Convert forecasts to list format
        if not isinstance(forecast_values, list):
            forecast_list = list(forecast_values)
        else:
            forecast_list = forecast_values
        
        # Ensure the lengths match after generating business days
        if len(forecast_list) > len(forecast_dates):
            # If the model generated more days than business days, truncate the forecast data
            forecast_list = forecast_list[:len(forecast_dates)]
        elif len(forecast_list) < len(forecast_dates):
            # If the model generated fewer days (shouldn't happen with fixed code), truncate dates
            forecast_dates = forecast_dates[:len(forecast_list)]


        print(f"Prediction successful! Model: {model_type}, RMSE: {metrics['RMSE']:.2f}")
        print(f"Forecast range: ${min(forecast_list):.2f} to ${max(forecast_list):.2f}")
        
        # Prepare response (METRICS REMOVED from the client response)
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
            # Metrics intentionally omitted
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in forecast: {str(e)}")
        print(error_trace)
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}. Please try with default settings or a different stock symbol.'
        }), 500

@bp.route('/compare', methods=['POST'])
@login_required
def compare_models():
    """Compare all models for a given stock"""
    # ... (No changes needed for this route, using default logic)
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL')
    forecast_days = int(data.get('days', 30))
    
    try:
        # Fetch and preprocess data (use simple 2 year fetch for comparison)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        handler = DataHandler(symbol, start_date, end_date)
        data_df = handler.preprocess_data()
        
        # Feature engineering
        engineer = FeatureEngineer(data_df)
        featured_data = engineer.create_all_features()
        
        # Initialize forecasting model
        forecaster = ForecastingModels(featured_data)
        
        # Run all models
        forecaster.arima_forecast(forecast_days=forecast_days)
        # Assuming Prophet might still fail, but we keep the call
        forecaster.prophet_forecast(forecast_days=forecast_days)
        forecaster.random_forest_forecast(forecast_days=forecast_days)
        
        # Compare models
        comparison = forecaster.compare_models()
        
        # Get best model
        best_model_name, best_model_data = forecaster.get_best_model()
        
        # The comparison route is often used for a different display, so we keep the metrics here
        # for comparison purposes, but you may want to remove them if not used.
        response = {
            'success': True,
            'comparison': comparison.to_dict(),
            'best_model': best_model_name,
            'best_metrics': best_model_data['metrics'] if best_model_data else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500