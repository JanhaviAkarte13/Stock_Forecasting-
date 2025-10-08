from flask import Blueprint, render_template, jsonify, session
from routes.auth import login_required
from models.data_handler import DataHandler
from models.feature_engineering import FeatureEngineer
import yfinance as yf
from datetime import datetime, timedelta

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@bp.route('/')
@login_required
def dashboard_page():
    """Main dashboard page"""
    return render_template('dashboard.html')

@bp.route('/portfolio')
@login_required
def get_portfolio():
    """Get portfolio overview"""
    try:
        # Get popular stocks data
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        portfolio_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[0]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    portfolio_data.append({
                        'symbol': symbol,
                        'name': ticker.info.get('longName', symbol),
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'volume': int(hist['Volume'].iloc[-1])
                    })
            except:
                continue
        
        return jsonify({
            'success': True,
            'portfolio': portfolio_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/market-overview')
@login_required
def market_overview():
    """Get market overview with indices"""
    try:
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^FTSE': 'FTSE 100'
        }
        
        market_data = []
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    market_data.append({
                        'name': name,
                        'value': round(current, 2),
                        'change': round(change, 2)
                    })
            except:
                continue
        
        return jsonify({
            'success': True,
            'market': market_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/recent-activity')
@login_required
def recent_activity():
    """Get recent activity for the user"""
    user_id = session.get('user_id')
    
    # Mock data - in production, fetch from database
    activity = [
        {
            'action': '📈 Predicted',
            'symbol': 'AAPL',
            'model': 'Prophet',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'accuracy': '95%'
        },
        {
            'action': '🔍 Viewed',
            'symbol': 'GOOGL',
            'model': 'Random Forest',
            'date': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
            'accuracy': '92%'
        },
        {
            'action': '📊 Analyzed',
            'symbol': 'TSLA',
            'model': 'ARIMA',
            'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
            'accuracy': '88%'
        }
    ]
    
    return jsonify({
        'success': True,
        'activity': activity
    })

@bp.route('/stats/<symbol>')
@login_required
def get_stock_stats(symbol):
    """Get detailed statistics for a stock"""
    try:
        handler = DataHandler(symbol)
        data = handler.preprocess_data()
        
        # Feature engineering for advanced stats
        engineer = FeatureEngineer(data)
        featured_data = engineer.create_all_features()
        
        # Calculate statistics
        stats = {
            'current_price': round(data['Close'].iloc[-1], 2),
            'high_52week': round(data['High'].tail(252).max(), 2),
            'low_52week': round(data['Low'].tail(252).min(), 2),
            'avg_volume': int(data['Volume'].mean()),
            'volatility': round(data['Close'].pct_change().std() * 100, 2),
            'rsi': round(featured_data['RSI'].iloc[-1], 2) if 'RSI' in featured_data else 0,
            'sma_20': round(featured_data['SMA_20'].iloc[-1], 2) if 'SMA_20' in featured_data else 0,
            'sma_50': round(featured_data['SMA_50'].iloc[-1], 2) if 'SMA_50' in featured_data else 0
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500