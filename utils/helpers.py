from datetime import datetime, timedelta
import pandas as pd

def format_currency(value, currency='USD'):
    """Format value as currency"""
    if currency == 'USD':
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def format_large_number(value):
    """Format large numbers (e.g., market cap)"""
    if value >= 1_000_000_000_000:
        return f"${value/1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    return f"${value:,.0f}"

def calculate_date_range(days=30):
    """Calculate start and end date for given number of days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

def validate_stock_symbol(symbol):
    """Validate stock symbol format"""
    if not symbol:
        return False
    if len(symbol) > 5:
        return False
    if not symbol.isalpha():
        return False
    return True

def get_color_for_change(change):
    """Get color code for price change"""
    if change > 0:
        return 'green'
    elif change < 0:
        return 'red'
    return 'gray'

def calculate_moving_average(data, window):
    """Calculate simple moving average"""
    return data.rolling(window=window).mean()

def calculate_returns(prices):
    """Calculate returns from price series"""
    return prices.pct_change()

def get_trading_days(start_date, end_date):
    """Get number of trading days between dates"""
    date_range = pd.bdate_range(start=start_date, end=end_date)
    return len(date_range)