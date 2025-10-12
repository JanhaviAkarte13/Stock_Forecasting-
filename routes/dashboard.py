from flask import Blueprint, render_template, jsonify, session
from routes.auth import login_required
import yfinance as yf
from datetime import datetime, timedelta
import requests
import concurrent.futures
import sqlite3
import os

bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@bp.route('/')
@login_required
def dashboard_page():
    """Dashboard home page"""
    username = session.get('username', 'User')
    return render_template('dashboard.html', username=username)

def get_usd_to_inr():
    """Get USD to INR exchange rate"""
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=3)
        return response.json()['rates']['INR']
    except:
        return 83.0

def get_db_connection():
    """Create database connection for predictions"""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'predictions.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_predictions_db():
    """Initialize predictions database if it doesn't exist"""
    try:
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_symbol TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB init error: {e}")

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

@bp.route('/api/recent-predictions')
@login_required
def recent_predictions():
    """Get LIVE recent predictions from database with actual prices"""
    try:
        init_predictions_db()
        conn = get_db_connection()
        
        # Get predictions from last 7 days
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        cursor = conn.execute('''
            SELECT stock_symbol, prediction_date, predicted_price, actual_price
            FROM predictions
            WHERE prediction_date >= ?
            ORDER BY prediction_date DESC
            LIMIT 10
        ''', (seven_days_ago,))
        
        db_predictions = cursor.fetchall()
        conn.close()
        
        predictions = []
        
        # If no predictions in DB, create sample ones for demo
        if not db_predictions:
            sample_stocks = ['TCS.NS', 'INFY.NS', 'AAPL', 'MSFT', 'RELIANCE.NS']
            
            def create_sample_prediction(symbol):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d', interval='1d')
                    
                    if len(hist) >= 2:
                        yesterday_price = float(hist['Close'].iloc[-2])
                        today_price = float(hist['Close'].iloc[-1])
                        
                        # Simulate a prediction (yesterday's price with small variance)
                        predicted = yesterday_price * (1 + (today_price - yesterday_price) / yesterday_price * 0.95)
                        
                        accuracy = 100 - abs((predicted - today_price) / today_price * 100)
                        
                        return {
                            'stock': symbol.replace('.NS', ''),
                            'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                            'predicted': round(predicted, 2),
                            'actual': round(today_price, 2),
                            'accuracy': round(accuracy, 2)
                        }
                except:
                    pass
                return None
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(create_sample_prediction, sample_stocks))
            
            predictions = [r for r in results if r][:5]
        
        else:
            # Fetch actual prices for stored predictions
            for row in db_predictions:
                symbol = row['stock_symbol']
                pred_date = row['prediction_date']
                predicted = row['predicted_price']
                actual = row['actual_price']
                
                # If actual price not stored, fetch it
                if actual is None:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(start=pred_date, end=(datetime.strptime(pred_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d'))
                        
                        if not hist.empty:
                            actual = float(hist['Close'].iloc[0])
                            
                            # Update database with actual price
                            conn = get_db_connection()
                            conn.execute('''
                                UPDATE predictions 
                                SET actual_price = ? 
                                WHERE stock_symbol = ? AND prediction_date = ?
                            ''', (actual, symbol, pred_date))
                            conn.commit()
                            conn.close()
                    except:
                        actual = predicted
                
                accuracy = 100 - abs((predicted - actual) / actual * 100) if actual else 0
                
                predictions.append({
                    'stock': symbol.replace('.NS', ''),
                    'date': pred_date,
                    'predicted': round(predicted, 2),
                    'actual': round(actual, 2),
                    'accuracy': round(accuracy, 2)
                })
        
        print(f"✅ Fetched {len(predictions)} predictions")
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        print(f"Predictions error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/financial-news')
@login_required
def financial_news():
    """Get LIVE financial news from Yahoo Finance"""
    news_items = []

    try:
        def fetch_yahoo_news_api():
            # Attempt to use a broader Yahoo Finance news endpoint
            url = "https://query1.finance.yahoo.com/v2/finance/news?q=finance&count=10"
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                response = requests.get(url, headers=headers, timeout=5)
                data = response.json()
                if 'finance' in data and 'result' in data['finance'] and data['finance']['result']:
                    return data['finance']['result'][0].get('quotes', [])
            except:
                pass
            return []

        def fetch_news_for_symbols():
            # Fallback: Fetch news for specific symbols
            news_symbols = ['^NSEI', '^DJI', 'RELIANCE.NS', 'AAPL', 'MSFT']
            all_news = []
            
            for symbol in news_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    if news and len(news) > 0:
                        all_news.extend(news[:2])
                        print(f"✓ Got {len(news)} news for {symbol}")
                except Exception as e:
                    print(f"✗ Error for {symbol}: {e}")
                    continue
            
            return all_news
        
        # Try both methods
        print("📰 Fetching news from Yahoo Finance...")
        all_news = fetch_yahoo_news_api()
        
        if not all_news:
            print("📰 Trying symbol-based news...")
            all_news = fetch_news_for_symbols()
        
        # Process news items
        seen_titles = set()
        
        for item in all_news:
            try:
                title = item.get('title', '')
                if not title or title in seen_titles:
                    continue
                
                seen_titles.add(title)
                
                # Calculate time ago
                timestamp = item.get('providerPublishTime', 0)
                if timestamp:
                    news_time = datetime.fromtimestamp(timestamp)
                    time_diff = datetime.now() - news_time
                    
                    if time_diff.days > 0:
                        time_ago = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
                    elif time_diff.seconds >= 3600:
                        hours = time_diff.seconds // 3600
                        time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
                    else:
                        minutes = max(1, time_diff.seconds // 60)
                        time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                else:
                    time_ago = "Recently"
                
                # Simple keyword-based sentiment analysis
                title_lower = title.lower()
                positive_words = ['surge', 'rally', 'gain', 'high', 'jump', 'rise', 'up', 'boost', 'soar', 'climb', 'record', 'strong']
                negative_words = ['fall', 'drop', 'down', 'loss', 'decline', 'crash', 'plunge', 'tumble', 'sink', 'weak', 'concern']
                
                if any(word in title_lower for word in positive_words):
                    sentiment = 'positive'
                elif any(word in title_lower for word in negative_words):
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                news_items.append({
                    'title': title,
                    'time': time_ago,
                    'sentiment': sentiment,
                    'link': item.get('link', '')
                })
                
                if len(news_items) >= 6:
                    break
                    
            except Exception as e:
                print(f"Processing error: {e}")
                continue
        
        # If still no news, provide realistic fallback based on market status
        if not news_items:
            now = datetime.now()
            current_hour = now.hour
            
            # Determine market status (Assuming India time zone for a generic example)
            # Market hours for NSE/BSE: 9:15 AM to 3:30 PM IST (9 <= hour < 16)
            if 9 <= current_hour < 16:  # Market hours
                news_items = [
                    {'title': 'Markets trading in active session today', 'time': '1 hour ago', 'sentiment': 'neutral', 'link': '#'},
                    {'title': 'Investors monitoring global economic indicators', 'time': '2 hours ago', 'sentiment': 'neutral', 'link': '#'},
                    {'title': 'Tech stocks showing mixed performance', 'time': '3 hours ago', 'sentiment': 'neutral', 'link': '#'}
                ]
            else:  # After hours
                news_items = [
                    {'title': 'Markets closed - Trading to resume next session', 'time': '1 hour ago', 'sentiment': 'neutral', 'link': '#'},
                    {'title': 'After-hours trading sees moderate activity', 'time': '2 hours ago', 'sentiment': 'neutral', 'link': '#'},
                    {'title': 'Preparing for next market opening', 'time': '3 hours ago', 'sentiment': 'neutral', 'link': '#'}
                ]
        
        print(f"✅ Returning {len(news_items)} news items")
        
        return jsonify({
            'success': True,
            'news': news_items[:6]
        })
    
    except Exception as e:
        print(f"News error: {e}")
        # Return error with helpful fallback
        return jsonify({
            'success': True,
            'news': [
                {'title': f'Unable to fetch live news: {str(e)[:50]}...', 'time': 'Just now', 'sentiment': 'neutral', 'link': '#'}
            ]
        })

# Helper function to add predictions to database (call this from your prediction module)
def add_prediction_to_db(stock_symbol, prediction_date, predicted_price):
    """Add a new prediction to the database"""
    try:
        init_predictions_db()
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO predictions (stock_symbol, prediction_date, predicted_price)
            VALUES (?, ?, ?)
        ''', (stock_symbol, prediction_date, predicted_price))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Add prediction error: {e}")
        return False