import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

class ForecastingModels:
    """Implement ARIMA and Random Forest forecasting models"""
    
    def __init__(self, data):
        self.data = data.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'Date' in self.data.columns:
                self.data = self.data.set_index('Date')
            else:
                self.data.index = pd.to_datetime(self.data.index)
        
        self.results = {}
        print(f"\nForecasting Models initialized with {len(self.data)} records")

    def arima_forecast(self, order=(2, 1, 2), forecast_days=30):
        """
        ARIMA model - Produces smooth, linear trend forecasts.
        Good for capturing overall market direction.
        NOW TRULY LINEAR - removed artificial noise
        """
        print(f"\n{'='*60}")
        print(f"Running ARIMA Model (Should produce SMOOTH/LINEAR forecast)")
        print(f"{'='*60}")
        
        try:
            train_data = self.data['Close'].copy()
            current_price = train_data.iloc[-1]
            
            print(f"Current price: ₹{current_price:.2f}")
            print(f"Training on {len(train_data)} data points")
            
            # Try multiple ARIMA orders
            orders_to_try = [
                (1, 1, 1),
                (2, 1, 1), 
                (1, 1, 2),
                (2, 1, 2),
                (3, 1, 2),
            ]
            
            fitted_model = None
            best_order = None
            best_aic = float('inf')
            
            for current_order in orders_to_try:
                try:
                    model = ARIMA(train_data, order=current_order)
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        fitted_model = fitted
                        best_order = current_order
                except:
                    continue
            
            if fitted_model is None:
                raise ValueError("Could not fit ARIMA")
            
            print(f"✓ Using ARIMA{best_order} (AIC: {best_aic:.2f})")
            
            # ===== FIX: Get pure ARIMA forecast WITHOUT artificial noise =====
            forecast_obj = fitted_model.get_forecast(steps=forecast_days)
            forecast_mean = forecast_obj.predicted_mean
            
            # Convert to list and ensure positive values ONLY
            # NO noise addition - this is what makes ARIMA truly linear
            forecast_values = []
            for val in forecast_mean:
                val = float(val)
                val = max(0.01, val)  # Only ensure positive
                forecast_values.append(val)
            
            # ===== REMOVED: The noise that was creating zig-zag patterns =====
            # OLD CODE (was adding random noise):
            # for i in range(len(forecast_values)):
            #     small_noise = np.random.normal(0, forecast_values[i] * 0.005)
            #     forecast_values[i] = max(0.01, forecast_values[i] + small_noise)
            # ===== NOW: Pure ARIMA output (smooth/linear) =====
            
            print(f"\n✓ ARIMA Forecast Generated:")
            print(f"   First day: ₹{forecast_values[0]:.2f}")
            print(f"   Last day: ₹{forecast_values[-1]:.2f}")
            print(f"   Range: ₹{min(forecast_values):.2f} to ₹{max(forecast_values):.2f}")
            print(f"   Total change: {((forecast_values[-1]-current_price)/current_price*100):+.2f}%")
            print(f"   Pattern: SMOOTH/LINEAR (pure ARIMA, no artificial noise)")
            
            # Calculate direction changes to verify linearity
            changes = np.diff(forecast_values)
            direction_changes = sum(1 for i in range(1, len(changes)) if changes[i] * changes[i-1] < 0)
            print(f"   Direction reversals: {direction_changes} (should be 0-2 for linear)")
            
            # Calculate metrics on fitted values
            predictions = fitted_model.fittedvalues
            actual = train_data[len(train_data)-len(predictions):]
            min_len = min(len(actual), len(predictions))
            actual = actual[-min_len:]
            predictions = predictions[-min_len:]
            
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            print(f"   RMSE: ₹{rmse:.2f}, MAE: ₹{mae:.2f}, MAPE: {mape:.2f}%")
            
            self.results['ARIMA'] = {
                'model': fitted_model,
                'forecast': forecast_values,
                'predictions': predictions.tolist(),
                'metrics': {
                    'MSE': float(mean_squared_error(actual, predictions)),
                    'MAE': float(mae),
                    'RMSE': float(rmse),
                    'MAPE': float(mape),
                    'Order': best_order,
                    'AIC': float(best_aic)
                }
            }
            
            return forecast_values, predictions.tolist()
        
        except Exception as e:
            print(f"❌ ARIMA Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def random_forest_forecast(self, forecast_days=30, tune_hyperparameters=False):
        """
        Random Forest model - Produces non-linear forecasts with curves and reversals.
        Uses technical indicators to capture complex patterns.
        """
        print(f"\n{'='*60}")
        print(f"Running Random Forest Model (Should produce NON-LINEAR forecast)")
        print(f"{'='*60}")
        
        try:
            df = self.data.copy()
            current_price = df['Close'].iloc[-1]
            
            # Get feature columns (exclude OHLCV)
            exclude_cols = ['Open', 'High', 'Low', 'Volume', 'Adj Close']
            feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'Close']
            
            if len(feature_cols) == 0:
                raise ValueError("No features found! Run feature engineering first.")
            
            print(f"Using {len(feature_cols)} features")
            print(f"Current price: ₹{current_price:.2f}")
            
            # Prepare data
            X = df[feature_cols].copy()
            y = df['Close'].copy()
            
            # Clean
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Split data
            split_idx = int(len(X) * 0.75)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            print("✓ Model trained")
            
            # Evaluate
            test_preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            mae = mean_absolute_error(y_test, test_preds)
            r2 = r2_score(y_test, test_preds)
            mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
            
            print(f"✓ Test RMSE: ₹{rmse:.2f}, MAE: ₹{mae:.2f}, R²: {r2:.3f}")
            
            # ===== MULTI-STEP FORECASTING WITH REALISTIC DYNAMICS =====
            print(f"\nGenerating {forecast_days}-day NON-LINEAR forecast...")
            
            # Calculate market characteristics
            returns = y.pct_change().dropna()
            daily_volatility = returns.tail(30).std()
            avg_return = returns.tail(30).mean()
            
            print(f"   Historical volatility: {daily_volatility*100:.2f}%/day")
            print(f"   Recent avg return: {avg_return*100:.2f}%/day")
            
            forecast_values = []
            
            # Start with last known feature values
            last_row = X.iloc[-1:].copy()
            current_pred_price = current_price
            
            for day in range(forecast_days):
                # Align features
                aligned_features = last_row[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                
                # Get base prediction
                base_prediction = model.predict(aligned_features)[0]
                
                # Apply multi-component dynamics for non-linearity
                
                # 1. Base model component (60%)
                pred = base_prediction * 0.6 + current_pred_price * 0.4
                
                # 2. Trend momentum (helps create slopes)
                if len(forecast_values) >= 3:
                    recent_change = forecast_values[-1] - forecast_values[-3]
                    momentum = recent_change * 0.4  # Continue recent direction
                    pred += momentum
                
                # 3. Mean reversion (creates curves/reversals)
                if len(forecast_values) >= 10:
                    recent_avg = np.mean(forecast_values[-10:])
                    reversion = (recent_avg - current_pred_price) * 0.15
                    pred += reversion
                
                # 4. Random walk component (adds realistic noise)
                noise = np.random.normal(0, daily_volatility * current_pred_price * 0.5)
                pred += noise
                
                # 5. Occasional shocks (every 7-10 days for non-linearity)
                if day > 0 and day % np.random.randint(7, 11) == 0:
                    shock = np.random.choice([-1, 1]) * current_pred_price * np.random.uniform(0.02, 0.04)
                    pred += shock
                    print(f"   Day {day+1}: Market shock applied ({shock:+.2f})")
                
                # Safety constraints
                max_daily_change = 0.08  # 8% max daily change
                if abs(pred - current_pred_price) / current_pred_price > max_daily_change:
                    direction = 1 if pred > current_pred_price else -1
                    pred = current_pred_price * (1 + direction * max_daily_change)
                
                pred = max(current_price * 0.5, pred)  # Don't drop below 50% of current
                pred = min(current_price * 1.5, pred)  # Don't exceed 150% of current
                pred = max(0.01, pred)  # Always positive
                
                forecast_values.append(pred)
                current_pred_price = pred
                
                # Update features for next iteration
                # Shift lag features
                for lag in [1, 2, 3, 5, 10]:
                    col_name = f'Close_Lag_{lag}'
                    if col_name in last_row.columns:
                        if lag == 1:
                            last_row[col_name] = pred
                        else:
                            prev_col = f'Close_Lag_{lag-1}'
                            if prev_col in last_row.columns:
                                last_row[col_name] = last_row[prev_col].values[0]
                
                # Update moving averages (approximate)
                if 'SMA_20' in last_row.columns:
                    last_row['SMA_20'] = last_row['SMA_20'].values[0] * 0.95 + pred * 0.05
                if 'SMA_50' in last_row.columns:
                    last_row['SMA_50'] = last_row['SMA_50'].values[0] * 0.98 + pred * 0.02
                
                if (day + 1) % 10 == 0:
                    pct_change = ((pred - current_price) / current_price) * 100
                    print(f"   Day {day+1}: ₹{pred:.2f} ({pct_change:+.2f}% from start)")
            
            # Analyze forecast
            forecast_std = np.std(forecast_values)
            total_return = ((forecast_values[-1] - current_price) / current_price) * 100
            
            # Count direction changes (more = more non-linear)
            changes = np.diff(forecast_values)
            direction_changes = sum(1 for i in range(1, len(changes)) if changes[i] * changes[i-1] < 0)
            
            print(f"\n✓ Random Forest Forecast Generated:")
            print(f"   First day: ₹{forecast_values[0]:.2f}")
            print(f"   Last day: ₹{forecast_values[-1]:.2f}")
            print(f"   Range: ₹{min(forecast_values):.2f} to ₹{max(forecast_values):.2f}")
            print(f"   Total return: {total_return:+.2f}%")
            print(f"   Std deviation: ₹{forecast_std:.2f}")
            print(f"   Direction reversals: {direction_changes}")
            print(f"   Pattern: {'NON-LINEAR with curves' if direction_changes >= 5 else 'RELATIVELY SMOOTH'}")
            
            self.results['Random Forest'] = {
                'model': model,
                'forecast': forecast_values,
                'predictions': test_preds.tolist(),
                'metrics': {
                    'MSE': float(mean_squared_error(y_test, test_preds)),
                    'MAE': float(mae),
                    'RMSE': float(rmse),
                    'R2': float(r2),
                    'MAPE': float(mape)
                }
            }
            
            return forecast_values, test_preds.tolist()
        
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
        
        df = pd.DataFrame(comparison).T
        if 'RMSE' in df.columns:
            df = df.sort_values('RMSE')
        return df
    
    def get_best_model(self):
        """Get best model by RMSE"""
        if not self.results:
            return None, None
        
        best_model = min(self.results.items(), 
                          key=lambda x: x[1]['metrics'].get('RMSE', float('inf')))
        
        return best_model[0], best_model[1]