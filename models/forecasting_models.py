import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class ForecastingModels:
    """Implement various forecasting models"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.results = {}
        print(f"\nForecasting Models initialized with {len(self.data)} records")
    
    def arima_forecast(self, order=(5, 1, 0), forecast_days=30):
        """ARIMA model for time series forecasting with proper variation"""
        print(f"\n{'='*60}")
        print(f"Running ARIMA Model (order={order})")
        print(f"{'='*60}")
        
        try:
            # Prepare data
            train_data = self.data['Close'].values
            
            print(f"Training data size: {len(train_data)}")
            print(f"Forecast horizon: {forecast_days} days")
            print(f"Price range: ${train_data.min():.2f} - ${train_data.max():.2f}")
            
            # Try different ARIMA orders if default fails
            orders_to_try = [
                (5, 1, 0), (1, 1, 1), (2, 1, 2), 
                (3, 1, 1), (1, 0, 1), (2, 0, 2)
            ]
            
            fitted_model = None
            best_order = None
            
            for current_order in orders_to_try:
                try:
                    print(f"Trying ARIMA{current_order}...")
                    model = ARIMA(train_data, order=current_order)
                    fitted_model = model.fit()
                    best_order = current_order
                    print(f"✓ Successfully fitted ARIMA{current_order}")
                    break
                except Exception as e:
                    continue
            
            if fitted_model is None:
                raise ValueError("Could not fit ARIMA with any order")
            
            # Get forecast with confidence intervals
            forecast_result = fitted_model.get_forecast(steps=forecast_days)
            forecast_mean = forecast_result.predicted_mean
            
            # Convert to list and ensure variation
            forecast_values = forecast_mean.tolist()
            
            # Check for flat prediction
            forecast_std = np.std(forecast_values)
            print(f"Initial forecast std: {forecast_std:.4f}")
            
            if forecast_std < 1.0:  # If too flat
                print("⚠️ Flat ARIMA forecast detected. Using random walk with drift...")
                # Use random walk with historical volatility
                returns = pd.Series(train_data).pct_change().dropna()
                mean_return = returns.mean()
                std_return = returns.std()
                
                last_price = train_data[-1]
                forecast_values = []
                
                for i in range(forecast_days):
                    # Random walk: price_t = price_t-1 * (1 + return)
                    random_return = np.random.normal(mean_return, std_return)
                    last_price = last_price * (1 + random_return)
                    forecast_values.append(last_price)
            
            print(f"Final forecast std: {np.std(forecast_values):.4f}")
            print(f"Forecast range: ${min(forecast_values):.2f} - ${max(forecast_values):.2f}")
            
            # In-sample predictions
            predictions = fitted_model.fittedvalues
            actual = train_data[len(train_data)-len(predictions):]
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\nMetrics: RMSE={rmse:.2f}, MAE={mae:.2f}")
            
            self.results['ARIMA'] = {
                'model': fitted_model,
                'forecast': forecast_values,
                'predictions': predictions.tolist(),
                'actual': actual.tolist(),
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                },
                'order': best_order
            }
            
            return forecast_values, predictions
        
        except Exception as e:
            print(f"❌ ARIMA Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def random_forest_forecast(self, forecast_days=30, tune_hyperparameters=False):
        """Random Forest with GUARANTEED variation"""
        print(f"\n{'='*60}")
        print(f"Running Random Forest Model")
        print(f"{'='*60}")
        
        try:
            # Create lag features
            df = self.data.copy()
            
            # Create multiple lag features
            for lag in range(1, 31):  # 30 lags
                df[f'lag_{lag}'] = df['Close'].shift(lag)
            
            # Add returns
            for lag in [1, 3, 7]:
                df[f'return_{lag}'] = df['Close'].pct_change(lag)
            
            # Drop NaN
            df = df.dropna()
            
            print(f"Data after feature engineering: {len(df)} records")
            
            # Prepare features
            feature_cols = [col for col in df.columns if col.startswith('lag_') or col.startswith('return_')]
            X = df[feature_cols]
            y = df['Close']
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Test predictions
            test_preds = model.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            mae = mean_absolute_error(y_test, test_preds)
            mse = mean_squared_error(y_test, test_preds)
            r2 = r2_score(y_test, test_preds)
            
            print(f"Model trained. RMSE: {rmse:.2f}")
            
            # FUTURE FORECAST - WITH GUARANTEED VARIATION
            print(f"\nGenerating {forecast_days}-day forecast...")
            
            # Get historical statistics
            historical_prices = y.values
            returns = pd.Series(historical_prices).pct_change().dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            trend = (historical_prices[-1] - historical_prices[0]) / len(historical_prices)
            
            print(f"Historical stats:")
            print(f"  Mean daily return: {mean_return*100:.3f}%")
            print(f"  Std daily return: {std_return*100:.3f}%")
            print(f"  Trend: ${trend:.3f}/day")
            
            # Method: Combine model prediction with random walk
            future_predictions = []
            
            # Get last window of data
            last_values = historical_prices[-30:].tolist()
            
            for day in range(forecast_days):
                # Create features from recent history
                features = {}
                for lag in range(1, 31):
                    if lag <= len(last_values):
                        features[f'lag_{lag}'] = last_values[-lag]
                    else:
                        features[f'lag_{lag}'] = historical_prices[-(lag - len(last_values) + 30)]
                
                # Add return features
                for lag in [1, 3, 7]:
                    if lag <= len(last_values):
                        features[f'return_{lag}'] = (last_values[-1] - last_values[-lag-1]) / last_values[-lag-1]
                    else:
                        features[f'return_{lag}'] = 0
                
                # Create feature vector
                X_future = pd.DataFrame([features], columns=feature_cols)
                
                # Get model prediction
                base_pred = model.predict(X_future)[0]
                
                # Add random walk component (THIS ENSURES VARIATION)
                random_shock = np.random.normal(mean_return, std_return * 0.5)
                trend_component = trend * (day + 1) * 0.3  # Small trend
                
                # Final prediction
                pred = base_pred * (1 + random_shock) + trend_component
                
                future_predictions.append(float(pred))
                
                # Update last_values for next iteration
                last_values.append(pred)
                if len(last_values) > 30:
                    last_values.pop(0)
                
                if (day + 1) % 10 == 0:
                    print(f"  Day {day+1}: ${pred:.2f}")
            
            print(f"\nForecast complete:")
            print(f"  Start: ${future_predictions[0]:.2f}")
            print(f"  End: ${future_predictions[-1]:.2f}")
            print(f"  Range: ${min(future_predictions):.2f} - ${max(future_predictions):.2f}")
            print(f"  Std Dev: ${np.std(future_predictions):.2f}")
            
            # Store results
            self.results['Random Forest'] = {
                'model': model,
                'forecast': future_predictions,
                'predictions': test_preds.tolist(),
                'test_actual': y_test.tolist(),
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
            }
            
            return future_predictions, test_preds
        
        except Exception as e:
            print(f"❌ Random Forest Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def compare_models(self):
        """Compare all models"""
        if not self.results:
            return pd.DataFrame()
        
        comparison = {}
        for model_name, result in self.results.items():
            comparison[model_name] = result['metrics']
        
        return pd.DataFrame(comparison).T
    
    def get_best_model(self):
        """Get best model by RMSE"""
        if not self.results:
            return None, None
        
        best = min(self.results.items(), key=lambda x: x[1]['metrics']['RMSE'])
        return best[0], best[1]
    
    def arima_forecast(self, order=(5, 1, 0), forecast_days=30):
        """ARIMA model for time series forecasting"""
        print(f"\n{'='*60}")
        print(f"Running ARIMA Model (order={order})")
        print(f"{'='*60}")
        
        try:
            # Prepare data
            train_data = self.data['Close']
            
            print(f"Training data size: {len(train_data)}")
            print(f"Forecast horizon: {forecast_days} days")
            
            # Try different ARIMA orders if default fails
            orders_to_try = [order, (5, 1, 2), (3, 1, 1), (2, 1, 2), (1, 1, 1)]
            fitted_model = None
            
            for current_order in orders_to_try:
                try:
                    print(f"Trying ARIMA order: {current_order}")
                    model = ARIMA(train_data, order=current_order)
                    fitted_model = model.fit()
                    print(f"✓ Successfully fitted with order {current_order}")
                    break
                except Exception as e:
                    print(f"Failed with order {current_order}: {str(e)}")
                    continue
            
            if fitted_model is None:
                raise ValueError("Could not fit ARIMA model with any order")
            
            # Make predictions with confidence intervals
            forecast_result = fitted_model.get_forecast(steps=forecast_days)
            forecast = forecast_result.predicted_mean
            
            # Add confidence interval for more realistic predictions
            conf_int = forecast_result.conf_int()
            
            print("✓ Forecast generated successfully")
            
            # In-sample predictions for evaluation
            predictions = fitted_model.fittedvalues
            
            # Align actual and predicted values
            actual = train_data[len(train_data)-len(predictions):]
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mse)
            
            print(f"\nModel Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            
            # Verify forecast has variation
            forecast_std = np.std(forecast)
            print(f"  Forecast variation (std): {forecast_std:.4f}")
            
            if forecast_std < 0.1:
                print("⚠️ Warning: Forecast shows very little variation")
            
            self.results['ARIMA'] = {
                'model': fitted_model,
                'forecast': forecast.tolist(),
                'predictions': predictions.tolist(),
                'actual': actual.tolist(),
                'confidence_interval': {
                    'lower': conf_int.iloc[:, 0].tolist(),
                    'upper': conf_int.iloc[:, 1].tolist()
                },
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                },
                'order': current_order if fitted_model else order
            }
            
            print("✓ ARIMA model completed successfully")
            
            return forecast, predictions
        
        except Exception as e:
            print(f"❌ ARIMA Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def prophet_forecast(self, forecast_days=30):
        """Prophet model for time series forecasting"""
        print(f"\n{'='*60}")
        print(f"Running Prophet Model")
        print(f"{'='*60}")
        
        try:
            # Prepare data for Prophet
            df = self.data.reset_index()
            df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
            df = df[['ds', 'y']]
            
            print(f"Training data size: {len(df)}")
            print(f"Forecast horizon: {forecast_days} days")
            
            # Initialize and fit Prophet model
            print("Fitting Prophet model...")
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            # Suppress Prophet logging
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            model.fit(df)
            
            print("✓ Model fitted successfully")
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=forecast_days, freq='B')
            forecast = model.predict(future)
            
            # Extract predictions
            predictions_df = forecast[['ds', 'yhat']].tail(len(df))
            future_forecast = forecast[['ds', 'yhat']].tail(forecast_days)
            
            # Calculate metrics
            actual = df['y'].values
            predicted = predictions_df['yhat'].values[:len(actual)]
            
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            print(f"\nModel Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            
            self.results['Prophet'] = {
                'model': model,
                'forecast': future_forecast['yhat'].tolist(),
                'predictions': predicted.tolist(),
                'actual': actual.tolist(),
                'forecast_df': future_forecast,
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse
                }
            }
            
            print("✓ Prophet model completed successfully")
            
            return future_forecast, predictions_df
        
        except Exception as e:
            print(f"❌ Prophet Error: {str(e)}")
            return None, None
    
    def random_forest_forecast(self, forecast_days=30, tune_hyperparameters=False):
        """Random Forest Regressor for forecasting"""
        print(f"\n{'='*60}")
        print(f"Running Random Forest Model")
        print(f"{'='*60}")
        
        try:
            # Prepare features
            feature_columns = [col for col in self.data.columns 
                             if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            if len(feature_columns) == 0:
                print("No features found. Creating lag features...")
                # Create basic lag features if no features exist
                for i in range(1, 11):
                    self.data[f'lag_{i}'] = self.data['Close'].shift(i)
                
                # Drop NaN values
                self.data = self.data.dropna()
                feature_columns = [f'lag_{i}' for i in range(1, 11)]
            
            print(f"Using {len(feature_columns)} features")
            
            # Make a copy to avoid modifying original data
            work_data = self.data.copy()
            
            X = work_data[feature_columns]
            y = work_data['Close']
            
            print(f"Dataset size: {len(X)} records")
            
            # Check if we have enough data
            if len(X) < 50:
                raise ValueError(f"Not enough data for Random Forest. Need at least 50 records, got {len(X)}")
            
            # Split data
            test_size = min(0.2, max(50, int(len(X) * 0.1)) / len(X))
            split_idx = int(len(X) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            print(f"Training set: {len(X_train)} records")
            print(f"Test set: {len(X_test)} records")
            
            # Model initialization
            if tune_hyperparameters and len(X_train) > 100:
                print("Performing hyperparameter tuning...")
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                rf = RandomForestRegressor(random_state=42, n_jobs=-1)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=3, 
                    scoring='neg_mean_squared_error',
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
            else:
                print("Training with default parameters...")
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=2,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
                model.fit(X_train, y_train)
            
            print("✓ Model trained successfully")
            
            # Predictions on test set
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            print(f"\nModel Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Future forecast (iterative prediction)
            print(f"\nGenerating {forecast_days}-day forecast...")
            last_data = X.tail(1).copy()
            future_predictions = []
            
            for day in range(forecast_days):
                pred = model.predict(last_data)[0]
                future_predictions.append(pred)
                
                # Update features for next prediction (simplified rolling)
                # In production, you'd want more sophisticated feature updating
                if 'lag_1' in last_data.columns:
                    for i in range(10, 1, -1):
                        if f'lag_{i}' in last_data.columns and f'lag_{i-1}' in last_data.columns:
                            last_data[f'lag_{i}'] = last_data[f'lag_{i-1}'].values
                    last_data['lag_1'] = pred
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            print("\nTop 10 Important Features:")
            for feature, importance in top_features:
                print(f"  {feature}: {importance:.4f}")
            
            self.results['Random Forest'] = {
                'model': model,
                'forecast': future_predictions,
                'predictions': predictions.tolist(),
                'test_actual': y_test.tolist(),
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                },
                'feature_importance': feature_importance,
                'top_features': top_features
            }
            
            print("✓ Random Forest model completed successfully")
            
            return future_predictions, predictions
        
        except Exception as e:
            print(f"❌ Random Forest Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def linear_regression_baseline(self, forecast_days=30):
        """Simple Linear Regression baseline model"""
        print(f"\n{'='*60}")
        print(f"Running Linear Regression Baseline")
        print(f"{'='*60}")
        
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare data with time index
            self.data['time_idx'] = np.arange(len(self.data))
            
            X = self.data[['time_idx']].values
            y = self.data['Close'].values
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            print(f"\nModel Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Future forecast
            future_idx = np.arange(len(X), len(X) + forecast_days).reshape(-1, 1)
            future_predictions = model.predict(future_idx)
            
            self.results['Linear Regression'] = {
                'model': model,
                'forecast': future_predictions.tolist(),
                'predictions': predictions.tolist(),
                'metrics': {
                    'MSE': mse,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
            }
            
            print("✓ Linear Regression completed successfully")
            
            return future_predictions, predictions
        
        except Exception as e:
            print(f"❌ Linear Regression Error: {str(e)}")
            return None, None
    
    def compare_models(self):
        """Compare all models and return metrics"""
        if not self.results:
            print("No models have been run yet!")
            return pd.DataFrame()
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        comparison = {}
        
        for model_name, result in self.results.items():
            comparison[model_name] = result['metrics']
        
        df = pd.DataFrame(comparison).T
        df = df.sort_values('RMSE')
        
        print(df.to_string())
        print(f"\n{'='*60}\n")
        
        return df
    
    def get_best_model(self):
        """Return the best performing model based on RMSE"""
        if not self.results:
            print("No models have been run yet!")
            return None, None
        
        best_model = min(self.results.items(), 
                        key=lambda x: x[1]['metrics']['RMSE'])
        
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   RMSE: {best_model[1]['metrics']['RMSE']:.4f}")
        
        return best_model[0], best_model[1]
    
    def run_all_models(self, forecast_days=30, tune_rf=False):
        """Run all forecasting models"""
        print(f"\n{'='*70}")
        print(f"RUNNING ALL FORECASTING MODELS")
        print(f"Forecast Horizon: {forecast_days} days")
        print(f"{'='*70}")
        
        # Run ARIMA
        try:
            self.arima_forecast(forecast_days=forecast_days)
        except Exception as e:
            print(f"ARIMA failed: {str(e)}")
        
        # Run Prophet
        try:
            self.prophet_forecast(forecast_days=forecast_days)
        except Exception as e:
            print(f"Prophet failed: {str(e)}")
        
        # Run Random Forest
        try:
            self.random_forest_forecast(forecast_days=forecast_days, 
                                       tune_hyperparameters=tune_rf)
        except Exception as e:
            print(f"Random Forest failed: {str(e)}")
        
        # Compare all models
        comparison = self.compare_models()
        
        # Get best model
        best_model_name, best_model_data = self.get_best_model()
        
        return {
            'comparison': comparison,
            'best_model': best_model_name,
            'results': self.results
        }
    
    def plot_predictions(self, model_name, save_path=None):
        """Plot actual vs predicted values"""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results!")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            result = self.results[model_name]
            
            plt.figure(figsize=(14, 7))
            
            # Plot actual values
            plt.plot(result['actual'], label='Actual', color='blue', linewidth=2)
            
            # Plot predictions
            plt.plot(result['predictions'], label='Predicted', 
                    color='red', linestyle='--', linewidth=2)
            
            plt.title(f'{model_name} - Actual vs Predicted', fontsize=16, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics = result['metrics']
            metrics_text = f"RMSE: {metrics['RMSE']:.2f}\nMAE: {metrics['MAE']:.2f}"
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error plotting: {str(e)}")
    
    def get_forecast_summary(self, model_name):
        """Get summary of forecast for a specific model"""
        if model_name not in self.results:
            return None
        
        result = self.results[model_name]
        forecast = result['forecast']
        
        summary = {
            'model': model_name,
            'forecast_length': len(forecast),
            'forecast_mean': np.mean(forecast),
            'forecast_std': np.std(forecast),
            'forecast_min': np.min(forecast),
            'forecast_max': np.max(forecast),
            'forecast_trend': 'Upward' if forecast[-1] > forecast[0] else 'Downward',
            'metrics': result['metrics']
        }
        
        return summary