import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

class FeatureEngineer:
    """Create technical indicators and derived features"""
    
    def __init__(self, data):
        self.data = data.copy()
        print(f"\nFeature Engineering initialized with {len(self.data)} records")
    
    def add_moving_averages(self, windows=[5, 10, 20, 50, 200]):
        """Add Simple Moving Averages"""
        print(f"Adding Simple Moving Averages: {windows}")
        
        for window in windows:
            if len(self.data) >= window:
                self.data[f'SMA_{window}'] = self.data['Close'].rolling(window=window).mean()
            else:
                print(f"⚠ Warning: Not enough data for SMA_{window}")
        
        return self.data
    
    def add_exponential_moving_averages(self, windows=[12, 26]):
        """Add Exponential Moving Averages"""
        print(f"Adding Exponential Moving Averages: {windows}")
        
        for window in windows:
            if len(self.data) >= window:
                self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
            else:
                print(f"⚠ Warning: Not enough data for EMA_{window}")
        
        return self.data
    
    def add_rsi(self, window=14):
        """Add Relative Strength Index"""
        print(f"Adding RSI (window={window})")
        
        try:
            rsi_indicator = RSIIndicator(close=self.data['Close'], window=window)
            self.data['RSI'] = rsi_indicator.rsi()
        except Exception as e:
            print(f"Error adding RSI: {str(e)}")
        
        return self.data
    
    def add_macd(self, fast=12, slow=26, signal=9):
        """Add MACD indicator"""
        print(f"Adding MACD (fast={fast}, slow={slow}, signal={signal})")
        
        try:
            macd_indicator = MACD(close=self.data['Close'], 
                                 window_fast=fast, 
                                 window_slow=slow, 
                                 window_sign=signal)
            
            self.data['MACD'] = macd_indicator.macd()
            self.data['MACD_Signal'] = macd_indicator.macd_signal()
            self.data['MACD_Diff'] = macd_indicator.macd_diff()
        except Exception as e:
            print(f"Error adding MACD: {str(e)}")
        
        return self.data
    
    def add_bollinger_bands(self, window=20, num_std=2):
        """Add Bollinger Bands"""
        print(f"Adding Bollinger Bands (window={window}, std={num_std})")
        
        try:
            bb_indicator = BollingerBands(close=self.data['Close'], 
                                         window=window, 
                                         window_dev=num_std)
            
            self.data['BB_Middle'] = bb_indicator.bollinger_mavg()
            self.data['BB_Upper'] = bb_indicator.bollinger_hband()
            self.data['BB_Lower'] = bb_indicator.bollinger_lband()
            self.data['BB_Width'] = bb_indicator.bollinger_wband()
        except Exception as e:
            print(f"Error adding Bollinger Bands: {str(e)}")
        
        return self.data
    
    def add_volatility(self, window=20):
        """Add volatility indicator"""
        print(f"Adding Volatility (window={window})")
        
        try:
            self.data['Volatility'] = self.data['Close'].pct_change().rolling(window=window).std()
            
            # Add Average True Range
            atr_indicator = AverageTrueRange(high=self.data['High'], 
                                            low=self.data['Low'], 
                                            close=self.data['Close'], 
                                            window=window)
            self.data['ATR'] = atr_indicator.average_true_range()
        except Exception as e:
            print(f"Error adding volatility: {str(e)}")
        
        return self.data
    
    def add_volume_features(self):
        """Add volume-based features"""
        print("Adding Volume Features")
        
        try:
            # Volume moving average
            self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
            
            # Volume ratio
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA_20']
            
            # On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(self.data)):
                if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + self.data['Volume'].iloc[i])
                elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - self.data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            self.data['OBV'] = obv
        except Exception as e:
            print(f"Error adding volume features: {str(e)}")
        
        return self.data
    
    def add_price_changes(self):
        """Add price change features"""
        print("Adding Price Change Features")
        
        try:
            # Daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change()
            
            # Price change
            self.data['Price_Change'] = self.data['Close'].diff()
            
            # High-Low range
            self.data['HL_Range'] = self.data['High'] - self.data['Low']
            
            # Close-Open range
            self.data['CO_Range'] = self.data['Close'] - self.data['Open']
            
            # Percentage change from open
            self.data['Open_Close_Pct'] = ((self.data['Close'] - self.data['Open']) / 
                                           self.data['Open'] * 100)
        except Exception as e:
            print(f"Error adding price changes: {str(e)}")
        
        return self.data
    
    def add_lag_features(self, lags=[1, 2, 3, 5, 10, 20, 50]):
        """Add lagged features"""
        print(f"Adding Lag Features: {lags}")
        
        try:
            for lag in lags:
                if lag < len(self.data):
                    self.data[f'Close_Lag_{lag}'] = self.data['Close'].shift(lag)
                    self.data[f'Volume_Lag_{lag}'] = self.data['Volume'].shift(lag)
        except Exception as e:
            print(f"Error adding lag features: {str(e)}")
        
        return self.data
    
    def add_stochastic_oscillator(self, window=14):
        """Add Stochastic Oscillator"""
        print(f"Adding Stochastic Oscillator (window={window})")
        
        try:
            stoch_indicator = StochasticOscillator(high=self.data['High'],
                                                   low=self.data['Low'],
                                                   close=self.data['Close'],
                                                   window=window)
            
            self.data['Stoch_K'] = stoch_indicator.stoch()
            self.data['Stoch_D'] = stoch_indicator.stoch_signal()
        except Exception as e:
            print(f"Error adding Stochastic Oscillator: {str(e)}")
        
        return self.data
    
    def add_momentum_indicators(self):
        """Add momentum indicators"""
        print("Adding Momentum Indicators")
        
        try:
            # Rate of Change (ROC)
            self.data['ROC'] = self.data['Close'].pct_change(periods=10) * 100
            
            # Momentum
            self.data['Momentum'] = self.data['Close'] - self.data['Close'].shift(4)
        except Exception as e:
            print(f"Error adding momentum indicators: {str(e)}")
        
        return self.data
    
    def create_all_features(self):
        """Create all technical indicators"""
        print(f"\n{'='*60}")
        print("Creating All Features")
        print(f"{'='*60}\n")
        
        # Add all features
        self.add_moving_averages()
        self.add_exponential_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_volatility()
        self.add_volume_features()
        self.add_price_changes()
        self.add_lag_features()
        self.add_stochastic_oscillator()
        self.add_momentum_indicators()
        
        # Drop NaN values created by indicators
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        final_rows = len(self.data)
        
        print(f"\n{'='*60}")
        print(f"Feature Engineering Complete!")
        print(f"Initial rows: {initial_rows}")
        print(f"Final rows: {final_rows}")
        print(f"Rows dropped: {initial_rows - final_rows}")
        print(f"Total features: {len(self.data.columns)}")
        print(f"{'='*60}\n")
        
        return self.data
    
    def get_feature_importance_data(self):
        """Prepare data for feature importance analysis"""
        # Exclude original OHLCV columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        return self.data[feature_columns], self.data['Close']
    
    def get_feature_summary(self):
        """Get summary of created features"""
        feature_groups = {
            'Moving Averages': [col for col in self.data.columns if 'MA' in col],
            'RSI': [col for col in self.data.columns if 'RSI' in col],
            'MACD': [col for col in self.data.columns if 'MACD' in col],
            'Bollinger Bands': [col for col in self.data.columns if 'BB' in col],
            'Volatility': [col for col in self.data.columns if 'Volatility' in col or 'ATR' in col],
            'Volume': [col for col in self.data.columns if 'Volume' in col or 'OBV' in col],
            'Price Changes': [col for col in self.data.columns if any(x in col for x in ['Return', 'Change', 'Range'])],
            'Lag Features': [col for col in self.data.columns if 'Lag' in col],
            'Stochastic': [col for col in self.data.columns if 'Stoch' in col],
            'Momentum': [col for col in self.data.columns if 'ROC' in col or 'Momentum' in col]
        }
        
        print("\nFeature Summary:")
        print("=" * 50)
        for group, features in feature_groups.items():
            if features:
                print(f"\n{group}: {len(features)} features")
                for feature in features:
                    print(f"  - {feature}")
        
        return feature_groups