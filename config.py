import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///stock_app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Stock settings
    DEFAULT_STOCK = 'AAPL'
    PREDICTION_DAYS = 30
    HISTORICAL_DAYS = 730  # 2 years
    
    # Model settings
    MODELS = ['ARIMA', 'Prophet', 'Random Forest']
    
    # Popular stock symbols
    POPULAR_STOCKS = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
        'META', 'NVDA', 'JPM', 'V', 'WMT'
    ]