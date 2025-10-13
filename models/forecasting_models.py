import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
import logging

# Suppress warnings and Prophet logging
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

class ForecastingModels:
    """Implement various forecasting models"""
    
    def __init__(self, data):
        # Ensure data is a copy and has a datetime index for proper use
        self.data = data.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'Date' in self.data.columns:
                self.data = self.data.set_index('Date')
            else:
                self.data.index = pd.to_datetime(self.data.index)
        
        self.results = {}
        print(f"\nForecasting Models initialized with {len(self.data)} records")

    # --- ARIMA Forecast (No change needed) ---
    def arima_forecast(self, order=(5, 1, 0), forecast_days=30):
        """ARIMA model for time series forecasting"""
        print(f"\n{'='*60}")
        print(f"Running ARIMA Model (order={order})")
        print(f"{'='*60}")
        
        try:
            train_data = self.data['Close']
            
            orders_to_try = [order, (5, 1, 2), (3, 1, 1), (2, 1, 2), (1, 1, 1)]
            fitted_model = None
            best_order = None
            
            for current_order in orders_to_try:
                try:
                    model = ARIMA(train_data, order=current_order)
                    fitted_model = model.fit()
                    best_order = current_order
                    break
                except:
                    continue
            
            if fitted_model is None:
                raise ValueError("Could not fit ARIMA with any order")
            
            # Forecast
            forecast_result = fitted_model.get_forecast(steps=forecast_days)
            forecast_mean = forecast_result.predicted_mean
            
            # In-sample predictions for metrics
            predictions = fitted_model.fittedvalues
            actual = train_data[len(train_data)-len(predictions):]
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mse)
            
            # Check for flat prediction (often happens with ARIMA)
            forecast_std = np.std(forecast_mean)
            if forecast_std < 0.1 and forecast_days > 1:
                print("⚠️ Warning: Flat ARIMA forecast detected. Adding minor noise for visualization.")
                last_price = train_data.iloc[-1]
                # Add small, price-relative random walk/noise for visual variance
                noise_std = train_data.pct_change().std() * 0.5 if train_data.pct_change().std() > 0 else 0.005
                forecast_values = [last_price + (i * (forecast_mean.iloc[-1] - last_price) / forecast_days) for i in range(1, forecast_days + 1)]
                forecast_values = np.array(forecast_values) + np.random.normal(0, noise_std * np.array(forecast_values), forecast_days)
            else:
                 forecast_values = forecast_mean.tolist()

            self.results['ARIMA'] = {
                'model': fitted_model,
                'forecast': forecast_values,
                'predictions': predictions.tolist(),
                'metrics': {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'Order': best_order}
            }
            
            return forecast_values, predictions.tolist()
        
        except Exception as e:
            print(f"❌ ARIMA Error: {str(e)}")
            return None, None
            
    # --- RANDOM FOREST Forecast (FIXED for Variation and Smoother Trend) ---
    def random_forest_forecast(self, forecast_days=30, tune_hyperparameters=False):
        """Random Forest Regressor with iterative forecasting and Monte Carlo variation."""
        print(f"\n{'='*60}")
        print(f"Running Random Forest Model (FIXED for Variation)")
        print(f"{'='*60}")
        
        try:
            df = self.data.copy()
            
            # 1. Feature Engineering (10 Lags and basic returns for features)
            n_lags = 10
            for i in range(1, n_lags + 1):
                df[f'lag_{i}'] = df['Close'].shift(i)
            df['daily_return'] = df['Close'].pct_change()
            df = df.dropna()
            
            feature_cols = [col for col in df.columns if col.startswith('lag_')]
            
            # Check if features were actually created/enough data
            if len(df) < n_lags + 1 or len(feature_cols) == 0:
                 raise ValueError("Not enough historical data or lags failed to create.")

            X = df[feature_cols]
            y = df['Close']
            
            # 2. Split Data
            test_size = min(50, max(5, int(len(X) * 0.1))) # Test on last 5-50 days
            split_idx = len(X) - test_size
            
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 3. Model Training
            model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # 4. In-sample predictions and metrics
            test_preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            mae = mean_absolute_error(y_test, test_preds)
            mse = mean_squared_error(y_test, test_preds)
            r2 = r2_score(y_test, test_preds)
            
            # 5. Future Forecast (Iterative Prediction with Monte Carlo)
            print(f"\nGenerating {forecast_days}-day forecast with volatility simulation...")
            
            # Calculate volatility for simulation
            returns = y.pct_change().dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Prepare the current lag features (last row of X)
            current_lags = X.tail(1).values[0].tolist() 
            future_predictions = []
            
            # Define new weights for a smoother, more trend-following forecast
            MODEL_WEIGHT = 0.7 
            NOISE_WEIGHT = 0.3
            
            for day in range(forecast_days):
                # a. Create feature vector
                X_future = pd.DataFrame([current_lags], columns=feature_cols)
                
                # b. Get model's base prediction
                base_pred = model.predict(X_future)[0]
                
                # c. Monte Carlo / Random Walk Component (Ensures Variation)
                # Draw a random return based on historical mean and std dev
                random_shock = np.random.normal(loc=mean_return, scale=std_return) 
                
                last_known_price = current_lags[0] 
                
                # Calculate the predicted return from the model:
                model_return = (base_pred / last_known_price) - 1
                
                # Apply the new blending ratio: 70% Model Trend, 30% Random Shock
                final_return = (model_return * MODEL_WEIGHT) + (random_shock * NOISE_WEIGHT) 
                
                pred = last_known_price * (1 + final_return)
                
                # Ensure price is realistic (e.g., positive)
                pred = max(0.1, pred) 
                
                future_predictions.append(float(pred))

                # d. Update the lag features for the next prediction
                # Shift all lags down (lag_2 gets lag_1, lag_1 gets new pred)
                for i in range(len(current_lags) - 1, 0, -1):
                    current_lags[i] = current_lags[i-1]
                current_lags[0] = pred # Set lag_1 to the new prediction
                
            print(f"✓ Forecast complete. Std Dev: ${np.std(future_predictions):.2f}")
            
            # 6. Store and Return Results
            self.results['Random Forest'] = {
                'model': model,
                'forecast': future_predictions,
                'predictions': test_preds.tolist(),
                'metrics': {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
            }
            
            return future_predictions, test_preds.tolist()
        
        except Exception as e:
            print(f"❌ Random Forest Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    # --- Prophet Forecast (No change needed) ---
    def prophet_forecast(self, forecast_days=30):
        """Prophet model for time series forecasting"""
        # ... (Prophet logic remains the same)
        print(f"\n{'='*60}")
        print(f"Running Prophet Model")
        print(f"{'='*60}")
        
        try:
            # Prepare data for Prophet
            df = self.data['Close'].reset_index()
            df = df.rename(columns={'index': 'ds', 'Close': 'y'})
            df = df[['ds', 'y']]
            
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            future = model.make_future_dataframe(periods=forecast_days, freq='B')
            forecast_df = model.predict(future)
            
            # Extract predictions
            predictions_df = forecast_df.head(len(df))
            future_forecast = forecast_df.tail(forecast_days)
            
            # Calculate metrics
            actual = df['y'].values
            predicted = predictions_df['yhat'].values[:len(actual)]
            
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            self.results['Prophet'] = {
                'model': model,
                'forecast': future_forecast['yhat'].tolist(),
                'predictions': predicted.tolist(),
                'metrics': {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
            }
            
            return future_forecast['yhat'].tolist(), predicted.tolist()
        
        except Exception as e:
            print(f"❌ Prophet Error: {str(e)}")
            return None, None

    # --- Helper and comparison methods (No change needed) ---
    def compare_models(self):
        """Compare all models and return metrics"""
        if not self.results:
            return pd.DataFrame()
        
        comparison = {}
        for model_name, result in self.results.items():
            comparison[model_name] = result['metrics']
        
        df = pd.DataFrame(comparison).T
        df = df.sort_values('RMSE')
        return df
    
    def get_best_model(self):
        """Return the best performing model based on RMSE"""
        if not self.results:
            return None, None
        
        best_model = min(self.results.items(), 
                         key=lambda x: x[1]['metrics']['RMSE'])
        
        return best_model[0], best_model[1]
    
    def run_all_models(self, forecast_days=30, tune_rf=False):
        """Run all forecasting models and return a consolidated result"""
        self.results = {} # Reset results before running
        
        try:
            self.arima_forecast(forecast_days=forecast_days)
        except Exception: pass
        
        try:
            self.prophet_forecast(forecast_days=forecast_days)
        except Exception: pass
        
        try:
            self.random_forest_forecast(forecast_days=forecast_days, 
                                        tune_hyperparameters=tune_rf)
        except Exception: pass
        
        comparison = self.compare_models()
        best_model_name, best_model_data = self.get_best_model()
        
        # Prepare final output structure (excluding metrics for front end display)
        final_output = {
            'best_model_name': best_model_name,
            'historical_prices': self.data['Close'].tolist(),
            'historical_dates': self.data.index.strftime('%Y-%m-%d').tolist(),
            'forecast_data': {}
        }

        if best_model_name:
            result = self.results[best_model_name]
            # Since we removed the date calculation in the model, the backend must handle it.
            # We return just the prices and let prediction.py generate the future dates.
            final_output['forecast_data'] = {
                'model_type': best_model_name,
                'predicted_prices': result['forecast'],
                'metrics': result['metrics'] # Include metrics here so prediction.py can extract/log them
            }
            
        return final_output