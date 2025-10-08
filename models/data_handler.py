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
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Reset index to make Date a column
            self.data = self.data.reset_index()
            self.data.set_index('Date', inplace=True)
            
            return self.data
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        # Check for missing values
        missing_count = self.data.isnull().sum()
        print(f"Missing values before handling:\n{missing_count}")
        
        # Forward fill for missing values
        self.data = self.data.fillna(method='ffill')
        # Backward fill for any remaining NaN at the start
        self.data = self.data.fillna(method='bfill')
        
        # Check again
        missing_count_after = self.data.isnull().sum()
        print(f"Missing values after handling:\n{missing_count_after}")
        
        return self.data
    
    def handle_outliers(self, columns=['Close'], method='iqr', threshold=1.5):
        """Remove or cap outliers using IQR method"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        for col in columns:
            if col in self.data.columns:
                # Calculate IQR
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define bounds
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Count outliers
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                print(f"Outliers found in {col}: {outliers}")
                
                # Cap outliers instead of removing (preserves data points)
                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return self.data
    
    def handle_non_trading_days(self):
        """Fill gaps for non-trading days"""
        if self.data is None:
            raise ValueError("No data loaded. Call fetch_data() first.")
        
        # Get date range
        date_range = pd.date_range(start=self.data.index.min(), 
                                   end=self.data.index.max(), 
                                   freq='B')  # Business days
        
        # Reindex to business days and forward fill
        self.data = self.data.reindex(date_range)
        self.data.fillna(method='ffill', inplace=True)
        
        print(f"Data resampled to business days. Total rows: {len(self.data)}")
        
        return self.data
    
    def preprocess_data(self):
        """Complete preprocessing pipeline"""
        print(f"\n{'='*50}")
        print(f"Preprocessing data for {self.symbol}")
        print(f"{'='*50}\n")
        
        # Step 1: Fetch data
        print("Step 1: Fetching data...")
        self.fetch_data()
        print(f"✓ Fetched {len(self.data)} records")
        
        # Step 2: Handle missing values
        print("\nStep 2: Handling missing values...")
        self.handle_missing_values()
        print("✓ Missing values handled")
        
        # Step 3: Handle outliers
        print("\nStep 3: Handling outliers...")
        self.handle_outliers(columns=['Open', 'High', 'Low', 'Close'])
        print("✓ Outliers handled")
        
        # Step 4: Handle non-trading days
        print("\nStep 4: Handling non-trading days...")
        self.handle_non_trading_days()
        print("✓ Non-trading days handled")
        
        print(f"\n{'='*50}")
        print("Preprocessing complete!")
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
            'date_range': f"{self.data.index.min().date()} to {self.data.index.max().date()}"
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