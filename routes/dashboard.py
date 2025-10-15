from flask import Blueprint, render_template, jsonify, session
from routes.auth import login_required
import yfinance as yf
from datetime import datetime, timedelta
import requests
import concurrent.futures

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@bp.route('/')
@login_required
def dashboard_page():
    """Dashboard home page"""
    username = session.get('username', 'User')
    return render_template('dashboard.html', username=username)

def get_usd_to_inr():
    """Get USD to INR exchange rate with fallback sources"""
    # Try primary source
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=3)
        rate = response.json()['rates']['INR']
        if rate and rate > 0:
            return rate
    except:
        pass
    
    # Fallback to alternative source
    try:
        response = requests.get('https://api.exchangerate.host/latest?base=USD&symbols=INR', timeout=3)
        rate = response.json()['rates']['INR']
        if rate and rate > 0:
            return rate
    except:
        pass
    
    # Final fallback to approximate current rate
    return 88.75

@bp.route('/api/portfolio-summary')
@login_required
def portfolio_summary():
    """Get portfolio summary with key stocks"""
    try:
        portfolio_stocks = {
            'Indian': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        }
        
        usd_to_inr = get_usd_to_inr()
        portfolio_value_inr = 0
        portfolio_value_usd = 0
        total_change = 0
        stocks_data = []
        
        def fetch_stock_quick(symbol):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get REAL-TIME intraday 1-minute data
                hist_1m = ticker.history(period='1d', interval='1m')
                hist_2d = ticker.history(period='2d', interval='1d')
                
                if hist_2d.empty or len(hist_2d) < 2:
                    return None
                
                # Use intraday for current price if available
                if not hist_1m.empty and len(hist_1m) > 0:
                    current = float(hist_1m['Close'].iloc[-1])
                else:
                    current = float(hist_2d['Close'].iloc[-1])
                
                prev = float(hist_2d['Close'].iloc[-2])
                change_pct = ((current - prev) / prev) * 100
                
                is_indian = '.NS' in symbol
                
                return {
                    'symbol': symbol.replace('.NS', ''),
                    'price': current,
                    'change': round(change_pct, 2),
                    'is_indian': is_indian
                }
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                return None
        
        all_symbols = portfolio_stocks['Indian'] + portfolio_stocks['US']
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_stock_quick, all_symbols))
        
        for stock in results:
            if stock:
                stocks_data.append(stock)
                if stock['is_indian']:
                    portfolio_value_inr += stock['price']
                    portfolio_value_usd += stock['price'] / usd_to_inr
                else:
                    portfolio_value_usd += stock['price']
                    portfolio_value_inr += stock['price'] * usd_to_inr
                
                total_change += stock['change']
        
        avg_change = total_change / len(stocks_data) if stocks_data else 0
        
        return jsonify({
            'success': True,
            'total_value_inr': round(portfolio_value_inr, 2),
            'total_value_usd': round(portfolio_value_usd, 2),
            'total_change': round(avg_change, 2),
            'stocks_count': len(stocks_data),
            'exchange_rate': round(usd_to_inr, 2)
        })
    
    except Exception as e:
        print(f"Portfolio error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/market-indices')
@login_required
def market_indices():
    """Get major market indices with REAL-TIME data"""
    try:
        indices = {
            '^NSEI': 'NIFTY 50',
            '^BSESN': 'SENSEX',
            '^DJI': 'Dow Jones',
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ'
        }
        
        def fetch_index(symbol_name):
            symbol, name = symbol_name
            try:
                ticker = yf.Ticker(symbol)
                
                # Get REAL-TIME intraday data
                hist_1m = ticker.history(period='1d', interval='1m')
                hist_2d = ticker.history(period='2d', interval='1d')
                
                if hist_2d.empty or len(hist_2d) < 2:
                    return None
                
                # Use intraday for current value if available
                if not hist_1m.empty and len(hist_1m) > 0:
                    current = float(hist_1m['Close'].iloc[-1])
                else:
                    current = float(hist_2d['Close'].iloc[-1])
                
                prev = float(hist_2d['Close'].iloc[-2])
                change = current - prev
                change_pct = (change / prev) * 100
                
                print(f"✓ {name}: {current:.2f} ({change_pct:+.2f}%)")
                
                return {
                    'name': name,
                    'value': round(current, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2)
                }
            except Exception as e:
                print(f"✗ {name}: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_index, indices.items()))
        
        indices_data = [r for r in results if r]
        
        return jsonify({
            'success': True,
            'indices': indices_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')
        })
    
    except Exception as e:
        print(f"Indices error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/top-movers')
@login_required
def top_movers():
    """Get REAL-TIME top gainers and losers using Yahoo Finance Screener API"""
    try:
        def get_yahoo_screener(category):
            api_map = {'gainers': 'day_gainers', 'losers': 'day_losers'}
            url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds={api_map[category]}&count=20"
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                response = requests.get(url, headers=headers, timeout=5)
                data = response.json()
                if 'finance' in data and 'result' in data['finance']:
                    quotes = data['finance']['result'][0].get('quotes', [])
                    return [q['symbol'] for q in quotes[:10] if 'symbol' in q]
            except:
                pass
            return []
        
        # Get symbols from Yahoo API
        gainer_symbols = get_yahoo_screener('gainers')
        loser_symbols = get_yahoo_screener('losers')
        
        # Add Indian stocks
        indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
                         'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'WIPRO.NS', 'TATAMOTORS.NS']
        
        all_symbols = list(set(gainer_symbols + loser_symbols + indian_stocks))
        
        print(f"📊 Fetching {len(all_symbols)} stocks for movers...")
        
        def fetch_mover(symbol):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get REAL-TIME intraday data
                hist_1m = ticker.history(period='1d', interval='1m')
                hist_2d = ticker.history(period='2d', interval='1d')
                
                if hist_2d.empty or len(hist_2d) < 2:
                    return None
                
                # Use intraday for current price
                if not hist_1m.empty and len(hist_1m) > 0:
                    current = float(hist_1m['Close'].iloc[-1])
                else:
                    current = float(hist_2d['Close'].iloc[-1])
                
                prev = float(hist_2d['Close'].iloc[-2])
                change_pct = ((current - prev) / prev) * 100
                
                try:
                    name = ticker.info.get('shortName', symbol)
                except:
                    name = symbol
                
                return {
                    'symbol': symbol.replace('.NS', ''),
                    'name': name,
                    'change': round(change_pct, 2)
                }
            except:
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(fetch_mover, all_symbols))
        
        valid = [r for r in results if r]
        gainers = sorted([s for s in valid if s['change'] > 0], key=lambda x: x['change'], reverse=True)[:5]
        losers = sorted([s for s in valid if s['change'] < 0], key=lambda x: x['change'])[:5]
        
        print(f"✅ Gainers: {len(gainers)}, Losers: {len(losers)}")
        
        return jsonify({
            'success': True,
            'gainers': gainers,
            'losers': losers
        })
    
    except Exception as e:
        print(f"Movers error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/sector-performance')
@login_required
def sector_performance():
    """Get REAL-TIME sector ETF performance"""
    try:
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLY': 'Consumer'
        }
        
        def fetch_sector(symbol_name):
            symbol, name = symbol_name
            try:
                ticker = yf.Ticker(symbol)
                
                # Get REAL-TIME intraday data
                hist_1d = ticker.history(period='1d', interval='5m')
                hist_5d = ticker.history(period='5d', interval='1d')
                
                if hist_5d.empty or len(hist_5d) < 2:
                    return None
                
                # Use intraday for current price
                if not hist_1d.empty and len(hist_1d) > 0:
                    current = float(hist_1d['Close'].iloc[-1])
                else:
                    current = float(hist_5d['Close'].iloc[-1])
                
                week_ago = float(hist_5d['Close'].iloc[0])
                change_pct = ((current - week_ago) / week_ago) * 100
                
                print(f"✓ {name}: {change_pct:+.2f}%")
                
                return {
                    'sector': name,
                    'change': round(change_pct, 2)
                }
            except Exception as e:
                print(f"✗ {name}: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(fetch_sector, sectors.items()))
        
        sector_data = [r for r in results if r]
        sector_data.sort(key=lambda x: x['change'], reverse=True)
        
        print(f"✅ Sectors fetched: {len(sector_data)}")
        
        return jsonify({
            'success': True,
            'sectors': sector_data
        })
    
    except Exception as e:
        print(f"Sector error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500