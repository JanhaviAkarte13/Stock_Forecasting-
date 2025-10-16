from flask import Blueprint, render_template, request, jsonify, session
from routes.auth import login_required
from models.data_handler import DataHandler
from models.feature_engineering import FeatureEngineer
from models.forecasting_models import ForecastingModels
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

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
    
    # Clean and validate ticker symbol
    original_symbol = symbol.upper().strip()
    cleaned_symbol = original_symbol
    
    # Fix common ticker format issues
    if cleaned_symbol.endswith('.NSE'):
        cleaned_symbol = cleaned_symbol.replace('.NSE', '.NS')
    
    symbol = cleaned_symbol

    try:
        # Date Handling - fetch at least 2 years of data
        parsed_end_date = None
        
        if end_date:
            try:
                parsed_end_date = datetime.strptime(end_date, '%m/%d/%Y')
            except:
                try:
                    parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except:
                    parsed_end_date = datetime.now()
        else:
            parsed_end_date = datetime.now()
        
        # Fetch 2 years of data minimum
        parsed_start_date = parsed_end_date - timedelta(days=730)
        
        # Fetch and preprocess data
        print(f"\n{'='*70}")
        print(f"FETCHING DATA FOR {original_symbol} (using ticker: {symbol})") 
        print(f"Date Range: {parsed_start_date.date()} to {parsed_end_date.date()}")
        print(f"{'='*70}")
        
        handler = DataHandler(symbol, parsed_start_date, parsed_end_date)
        
        # TRY to fetch data - catch specific yfinance errors
        try:
            data_df = handler.preprocess_data()
        except Exception as fetch_error:
            error_msg = str(fetch_error).lower()
            
            # Check for common data fetching errors
            if 'no data' in error_msg or 'empty' in error_msg:
                return jsonify({
                    'success': False,
                    'error': f'No data found for ticker "{symbol}". Please verify the ticker symbol is correct. For Indian stocks, use format: SYMBOL.NS (e.g., RELIANCE.NS)'
                }), 400
            elif 'invalid' in error_msg or 'not found' in error_msg:
                return jsonify({
                    'success': False,
                    'error': f'Invalid ticker symbol "{symbol}". Please check the symbol and try again.'
                }), 400
            else:
                # Re-raise other errors
                raise
        
        print(f"✓ Data fetched: {len(data_df)} records")
        
        # Check if we have enough data
        if len(data_df) < 200:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for {symbol}. Found {len(data_df)} records, but need at least 200 for reliable predictions. This ticker may be newly listed or have limited trading history.'
            }), 400
        
        # Feature engineering
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70)
        engineer = FeatureEngineer(data_df)
        featured_data = engineer.create_all_features()
        
        # Initialize forecasting model
        forecaster = ForecastingModels(featured_data)
        
        # Run selected model
        print(f"\n{'='*70}")
        print(f"RUNNING {model_type.upper()} MODEL")
        print(f"{'='*70}")
        
        forecast_values = None
        predictions = None
        metrics = None
        
        if model_type == 'ARIMA':
            forecast_values, predictions = forecaster.arima_forecast(forecast_days=forecast_days)
            if forecast_values is not None:
                metrics = forecaster.results['ARIMA']['metrics']
        
        elif model_type == 'Prophet':
            return jsonify({
                'success': False,
                'error': 'Prophet model is currently not supported. Please use Random Forest or ARIMA instead.'
            }), 400
        
        elif model_type == 'Random Forest':
            forecast_values, predictions = forecaster.random_forest_forecast(
                forecast_days=forecast_days, 
                tune_hyperparameters=False
            )
            if forecast_values is not None:
                metrics = forecaster.results['Random Forest']['metrics']
        
        else:
            return jsonify({
                'success': False, 
                'error': f'Invalid model type: {model_type}. Choose ARIMA or Random Forest.'
            }), 400
        
        # Check if model failed
        if forecast_values is None or predictions is None:
            return jsonify({
                'success': False, 
                'error': f'{model_type} model failed to generate predictions. This could be due to insufficient data quality or volatility. Try using a different stock or model.'
            }), 500
        
        # Validate forecast values
        current_price = featured_data['Close'].iloc[-1]
        
        # Check for unrealistic predictions
        forecast_array = np.array(forecast_values)
        if np.any(forecast_array <= 0):
            print("⚠️ Warning: Negative or zero prices detected in forecast!")
            forecast_values = [max(0.01, v) for v in forecast_values]
        
        # Check for extreme jumps
        first_forecast = forecast_values[0]
        price_change_pct = abs(first_forecast - current_price) / current_price * 100
        
        if price_change_pct > 50:
            print(f"⚠️ Warning: Extreme price jump detected: {price_change_pct:.1f}% change")
            print(f"   Current: ${current_price:.2f}, Forecast: ${first_forecast:.2f}")
        
        # Prepare Business Day Forecast Dates
        last_date = featured_data.index[-1].to_pydatetime()
        start_date_for_range = last_date + BDay(1)
        
        forecast_dates_index = pd.bdate_range(
            start=start_date_for_range, 
            periods=forecast_days, 
            freq='B' 
        )
        forecast_dates = forecast_dates_index.to_list()
        
        # Convert forecasts to list format
        if not isinstance(forecast_values, list):
            forecast_list = list(forecast_values)
        else:
            forecast_list = forecast_values
        
        # Ensure the lengths match
        if len(forecast_list) > len(forecast_dates):
            forecast_list = forecast_list[:len(forecast_dates)]
        elif len(forecast_list) < len(forecast_dates):
            forecast_dates = forecast_dates[:len(forecast_list)]
        
        # Send last 180 days of historical data
        historical_window = min(180, len(featured_data))
        
        print(f"\n{'='*70}")
        print("PREDICTION SUMMARY")
        print(f"{'='*70}")
        print(f"✓ Model: {model_type}")
        print(f"✓ RMSE: ${metrics['RMSE']:.2f}")
        if 'R2' in metrics:
            print(f"✓ R² Score: {metrics['R2']:.3f}")
        print(f"✓ Current Price: ${current_price:.2f}")
        print(f"✓ First Forecast: ${forecast_list[0]:.2f} ({((forecast_list[0]/current_price - 1) * 100):.2f}% change)")
        print(f"✓ Last Forecast: ${forecast_list[-1]:.2f}")
        print(f"✓ Forecast Range: ${min(forecast_list):.2f} to ${max(forecast_list):.2f}")
        print(f"✓ Forecast Std Dev: ${np.std(forecast_list):.2f}")
        print(f"✓ Historical data sent: {historical_window} days")
        print(f"{'='*70}\n")
        
        # Prepare response
        response = {
            'success': True,
            'symbol': symbol,
            'model': model_type,
            'current_price': float(current_price),
            'historical': {
                'dates': featured_data.index.strftime('%Y-%m-%d').tolist()[-historical_window:],
                'actual': featured_data['Close'].tolist()[-historical_window:],
            },
            'forecast': {
                'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                'values': forecast_list
            },
            'summary': {
                'first_forecast': float(forecast_list[0]),
                'last_forecast': float(forecast_list[-1]),
                'min_forecast': float(min(forecast_list)),
                'max_forecast': float(max(forecast_list)),
                'avg_forecast': float(np.mean(forecast_list)),
                'forecast_trend': 'upward' if forecast_list[-1] > forecast_list[0] else 'downward'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n{'='*70}")
        print("❌ ERROR IN FORECAST")
        print(f"{'='*70}")
        print(f"Error: {str(e)}")
        print(f"\nFull traceback:")
        print(error_trace)
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}. Check the ticker symbol and try again.',
            'details': str(e)
        }), 500

@bp.route('/compare', methods=['POST'])
@login_required
def compare_models():
    """Compare all models for a given stock"""
    data = request.get_json()
    symbol = data.get('symbol', 'AAPL').upper().strip()
    forecast_days = int(data.get('days', 30))
    
    # Clean ticker symbol
    if symbol.endswith('.NSE'):
        symbol = symbol.replace('.NSE', '.NS')
    
    try:
        # Fetch and preprocess data (use 2 year fetch for comparison)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        print(f"\n{'='*70}")
        print(f"COMPARING MODELS FOR {symbol}")
        print(f"{'='*70}")
        
        handler = DataHandler(symbol, start_date, end_date)
        data_df = handler.preprocess_data()
        
        if len(data_df) < 200:
            return jsonify({
                'success': False,
                'error': f'Insufficient data for comparison. Found {len(data_df)} records.'
            }), 400
        
        # Feature engineering
        engineer = FeatureEngineer(data_df)
        featured_data = engineer.create_all_features()
        
        # Initialize forecasting model
        forecaster = ForecastingModels(featured_data)
        
        # Run all models
        print("\nRunning ARIMA...")
        forecaster.arima_forecast(forecast_days=forecast_days)
        
        print("\nRunning Random Forest...")
        forecaster.random_forest_forecast(forecast_days=forecast_days)
        
        # Compare models
        comparison = forecaster.compare_models()
        
        # Get best model
        best_model_name, best_model_data = forecaster.get_best_model()
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"✓ Best RMSE: ${best_model_data['metrics']['RMSE']:.2f}")
        print(f"{'='*70}\n")
        
        response = {
            'success': True,
            'comparison': comparison.to_dict(),
            'best_model': best_model_name,
            'best_metrics': best_model_data['metrics'] if best_model_data else None
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in compare_models: {str(e)}")
        print(error_trace)
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500