import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataHandler:
    """Handle stock data fetching and preprocessing"""
    
    def __init__(self, symbol, start_date=None, end_date=None):
        self.symbol = symbol.upper()
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=730))
        self.data = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Fetch data with auto_adjust=True for split/dividend adjusted prices
            # This ensures historical accuracy
            self.data = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Remove timezone if present
            if self.data.index.tz is not None:
                self.data.index = self.data.index.tz_localize(None)
            
            # Normalize to midnight (remove time component)
            self.data.index = pd.to_datetime(self.data.index).normalize()
            
            # Remove any duplicate dates (keep last)
            self.data = self.data[~self.data.index.duplicated(keep='last')]
            
            # Handle infinite values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            
            # Only drop rows where ALL OHLC values are NaN (keep partial data)
            self.data = self.data.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all')
            
            if self.data.empty:
                raise ValueError(f"No usable data found for {self.symbol} after cleaning.")
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {self.symbol}: {str(e)}")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset - CONSERVATIVE approach"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        # Check for missing values
        missing_count = self.data.isnull().sum()
        if missing_count.sum() > 0:
            print(f"Missing values found:\n{missing_count[missing_count > 0]}")
        
        # Only forward fill small gaps (max 3 days) to avoid creating fake data
        # This prevents filling large gaps with stale prices
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(method='ffill', limit=3)
        
        # For any remaining NaN at the start, use backward fill (limit 3)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(method='bfill', limit=3)
        
        # Drop any rows that still have NaN (these are large gaps we shouldn't fill)
        self.data = self.data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        return self.data
    
    def handle_outliers(self, columns=['Close'], method='iqr', threshold=3.0):
        """
        Detect outliers but DON'T modify them - just log them.
        Real stock prices can have legitimate spikes/drops (earnings, news, etc.)
        Only remove truly impossible values (like negative prices).
        """
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        # Remove impossible values only
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.data.columns:
                # Remove negative prices (impossible)
                invalid_count = (self.data[col] <= 0).sum()
                if invalid_count > 0:
                    print(f"Removing {invalid_count} invalid (<=0) values from {col}")
                    self.data = self.data[self.data[col] > 0]
        
        # Remove negative volume
        if 'Volume' in self.data.columns:
            invalid_volume = (self.data['Volume'] < 0).sum()
            if invalid_volume > 0:
                print(f"Removing {invalid_volume} negative volume values")
                self.data = self.data[self.data['Volume'] >= 0]
        
        # Log outliers but don't modify (for diagnostic purposes)
        for col in columns:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"Note: {outliers} statistical outliers detected in {col} (not modified)")
        
        return self.data
    
    def handle_non_trading_days(self):
        """
        REMOVED: Do NOT fill non-trading days with artificial data.
        This was creating fake prices for weekends/holidays.
        Keep only actual trading days from the exchange.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        print(f"Data contains {len(self.data)} actual trading days (no artificial fills)")
        
        return self.data
    
    def preprocess_data(self):
        """Complete preprocessing pipeline - CONSERVATIVE approach"""
        print(f"\n{'='*50}")
        print(f"Preprocessing data for {self.symbol}")
        print(f"{'='*50}\n")
        
        # Step 1: Fetch data
        print("Step 1: Fetching data...")
        self.fetch_data()
        print(f"✓ Fetched {len(self.data)} records")
        
        # Step 2: Handle missing values (conservative)
        print("\nStep 2: Handling missing values (conservative)...")
        self.handle_missing_values()
        print(f"✓ Missing values handled - {len(self.data)} records remaining")
        
        # Step 3: Validate data (don't modify legitimate outliers)
        print("\nStep 3: Validating data...")
        self.handle_outliers(columns=['Open', 'High', 'Low', 'Close'])
        print("✓ Data validated")
        
        # Step 4: Keep only actual trading days (removed artificial filling)
        print("\nStep 4: Using actual trading days...")
        self.handle_non_trading_days()
        print("✓ Data ready")
        
        print(f"\n{'='*50}")
        print("Preprocessing complete!")
        print(f"Final dataset: {len(self.data)} records")
        print(f"{'='*50}\n")
        
        return self.data
    
    def get_stock_info(self):
        """Get additional stock information"""
        try:
            ticker = yf.Ticker(self.symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', self.symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'currency': info.get('currency', 'USD'),
                'website': info.get('website', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')
            }
        except Exception as e:
            print(f"Error fetching stock info: {str(e)}")
            return {
                'name': self.symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'pe_ratio': 0,
                'dividend_yield': 0,
                'fifty_two_week_high': 0,
                'fifty_two_week_low': 0,
                'currency': 'USD',
                'website': 'N/A',
                'description': 'N/A'
            }
    
    def get_summary_statistics(self):
        """Get summary statistics of the data"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        stats = {
            'mean_close': self.data['Close'].mean(),
            'median_close': self.data['Close'].median(),
            'std_close': self.data['Close'].std(),
            'min_close': self.data['Close'].min(),
            'max_close': self.data['Close'].max(),
            'mean_volume': self.data['Volume'].mean(),
            'total_records': len(self.data),
            'date_range': f"{self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}"
        }
        
        return stats
    
    def export_to_csv(self, filename=None):
        """Export data to CSV file"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        if filename is None:
            filename = f"{self.symbol}_stock_data.csv"
        
        self.data.to_csv(filename)
        print(f"Data exported to {filename}")
        
        return filename