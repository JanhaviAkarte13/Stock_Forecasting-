from flask import Blueprint, render_template, jsonify, request
from routes.auth import login_required
from models.data_handler import DataHandler
from models.feature_engineering import FeatureEngineer
import plotly.graph_objs as go
import plotly.utils
import json

bp = Blueprint('visuals', __name__, url_prefix='/visuals')

@bp.route('/')
@login_required
def visuals_page():
    """Visualizations page"""
    return render_template('visuals.html')

@bp.route('/candlestick/<symbol>')
@login_required
def candlestick_chart(symbol):
    """Generate candlestick chart data"""
    try:
        handler = DataHandler(symbol)
        data = handler.preprocess_data()
        
        # Take last 90 days
        data = data.tail(90)
        
        chart_data = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'volume': data['Volume'].tolist()
        }
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/technical/<symbol>')
@login_required
def technical_indicators(symbol):
    """Generate technical indicators chart data"""
    try:
        handler = DataHandler(symbol)
        data = handler.preprocess_data()
        
        # Feature engineering
        engineer = FeatureEngineer(data)
        featured_data = engineer.create_all_features()
        
        # Take last 90 days
        featured_data = featured_data.tail(90)
        
        chart_data = {
            'dates': featured_data.index.strftime('%Y-%m-%d').tolist(),
            'close': featured_data['Close'].tolist(),
            'sma_20': featured_data['SMA_20'].tolist() if 'SMA_20' in featured_data else [],
            'sma_50': featured_data['SMA_50'].tolist() if 'SMA_50' in featured_data else [],
            'rsi': featured_data['RSI'].tolist() if 'RSI' in featured_data else [],
            'macd': featured_data['MACD'].tolist() if 'MACD' in featured_data else [],
            'macd_signal': featured_data['MACD_Signal'].tolist() if 'MACD_Signal' in featured_data else [],
            'bb_upper': featured_data['BB_Upper'].tolist() if 'BB_Upper' in featured_data else [],
            'bb_middle': featured_data['BB_Middle'].tolist() if 'BB_Middle' in featured_data else [],
            'bb_lower': featured_data['BB_Lower'].tolist() if 'BB_Lower' in featured_data else []
        }
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/volume/<symbol>')
@login_required
def volume_chart(symbol):
    """Generate volume chart data"""
    try:
        handler = DataHandler(symbol)
        data = handler.preprocess_data()
        
        # Take last 90 days
        data = data.tail(90)
        
        chart_data = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'volume': data['Volume'].tolist(),
            'close': data['Close'].tolist()
        }
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/correlation', methods=['POST'])
@login_required
def correlation_matrix():
    """Generate correlation matrix for multiple stocks"""
    data = request.get_json()
    symbols = data.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
    
    try:
        import pandas as pd
        
        # Fetch data for all symbols
        close_prices = pd.DataFrame()
        
        for symbol in symbols:
            handler = DataHandler(symbol)
            stock_data = handler.preprocess_data()
            close_prices[symbol] = stock_data['Close']
        
        # Calculate correlation
        correlation = close_prices.corr()
        
        return jsonify({
            'success': True,
            'correlation': correlation.to_dict(),
            'symbols': symbols
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/returns/<symbol>')
@login_required
def returns_distribution(symbol):
    """Generate returns distribution data"""
    try:
        handler = DataHandler(symbol)
        data = handler.preprocess_data()
        
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        chart_data = {
            'returns': returns.tolist(),
            'mean': round(returns.mean() * 100, 2),
            'std': round(returns.std() * 100, 2),
            'min': round(returns.min() * 100, 2),
            'max': round(returns.max() * 100, 2)
        }
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500