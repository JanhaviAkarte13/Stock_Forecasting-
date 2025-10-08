from flask import Blueprint, render_template, request, jsonify, session
from routes.auth import login_required
from models.data_handler import DataHandler
from config import Config
import yfinance as yf

bp = Blueprint('stock', __name__, url_prefix='/stock')

@bp.route('/')
@login_required
def stock_page():
    """Stock selection page"""
    return render_template('stock.html', popular_stocks=Config.POPULAR_STOCKS)

@bp.route('/search', methods=['POST'])
@login_required
def search_stock():
    """Search for stock by symbol"""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    
    try:
        handler = DataHandler(symbol)
        stock_info = handler.get_stock_info()
        
        # Get current price
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'name': stock_info['name'],
            'sector': stock_info['sector'],
            'industry': stock_info['industry'],
            'current_price': round(current_price, 2),
            'currency': stock_info['currency']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@bp.route('/data/<symbol>')
@login_required
def get_stock_data(symbol):
    """Get historical stock data"""
    try:
        handler = DataHandler(symbol)
        data = handler.preprocess_data()
        
        # Convert to JSON-friendly format
        result = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'volume': data['Volume'].tolist()
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@bp.route('/info/<symbol>')
@login_required
def get_stock_info(symbol):
    """Get detailed stock information"""
    try:
        handler = DataHandler(symbol)
        info = handler.get_stock_info()
        
        return jsonify({
            'success': True,
            'info': info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400