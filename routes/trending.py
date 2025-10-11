from flask import Blueprint, render_template, jsonify
from routes.auth import login_required
import yfinance as yf
from datetime import datetime, timedelta
import requests
import traceback
import concurrent.futures
from threading import Lock
from pandas import DataFrame # Used by yfinance for dynamic lists
import pytz # Needed for market status timezones

bp = Blueprint('trending', __name__, url_prefix='/trending')

# --- Global Caches ---
_exchange_rate_cache = {'rate': 83.0, 'timestamp': None}
_cache_lock = Lock()

# Fallback lists (used if dynamic API calls fail)
US_STOCKS_FALLBACK = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'WMT']
INDIAN_STOCKS_FALLBACK = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'MARUTI.NS', 'HCLTECH.NS'
]
US_STOCKS_FALLBACK = list(set(US_STOCKS_FALLBACK))
INDIAN_STOCKS_FALLBACK = list(set(INDIAN_STOCKS_FALLBACK))


def get_usd_to_inr():
    """Get USD to INR with caching (5-minute cache)"""
    global _exchange_rate_cache
    now = datetime.now()
    
    with _cache_lock:
        if _exchange_rate_cache['timestamp']:
            age = (now - _exchange_rate_cache['timestamp']).seconds
            if age < 300:
                return _exchange_rate_cache['rate']
    
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=3)
        rate = response.json()['rates']['INR']
        with _cache_lock:
            _exchange_rate_cache = {'rate': rate, 'timestamp': now}
        return rate
    except:
        return _exchange_rate_cache['rate']

@bp.route('/')
@login_required
def trending_page():
    return render_template('trending.html')

# --- DYNAMIC SYMBOL FETCHERS ---

def get_nse_dynamic_symbols(category):
    """Fetch dynamic Indian symbols from the unofficial NSE live API."""
    
    api_map = {'gainers': 'gainers', 'losers': 'losers', 'mostactive': 'mostactive'}
    if category not in api_map: return []
    
    nse_url = f"https://www.nseindia.com/api/live-analysis-variations?index={api_map[category]}"
    nse_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.nseindia.com/'
    }
    
    indian_symbols = []
    try:
        nse_response = requests.get(nse_url, headers=nse_headers, timeout=5)
        if nse_response.status_code == 200:
            nse_data = nse_response.json()
            data_list = nse_data.get('data', [])
            # Take the top 50 symbols for a robust buffer
            indian_symbols = [item['symbol'] + '.NS' for item in data_list[:50] if 'symbol' in item]
            return indian_symbols
        else:
            print(f"⚠️ NSE API responded with status {nse_response.status_code}. Falling back.")
            
    except Exception as e:
        print(f"❌ NSE API fetch failed for {category}: {e}")

    return INDIAN_STOCKS_FALLBACK

def get_yahoo_screener_api(category):
    """Fetch US gainers/losers/active from Yahoo Finance Screener API (US region only)"""
    
    api_map = {
        'gainers': 'day_gainers',
        'losers': 'day_losers',
        'mostactive': 'most_actives'
    }
    
    if category not in api_map: return []
        
    # Requesting 50 stocks from Yahoo Screener for a robust buffer
    url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds={api_map[category]}&count=50"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    symbols = []
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if 'finance' in data and 'result' in data['finance']:
            quotes = data['finance']['result'][0].get('quotes', [])
            # Take the top 50 symbols from the raw API response
            symbols = [q['symbol'] for q in quotes[:50] if 'symbol' in q] 
    except Exception as e:
        print(f"⚠️ Yahoo US {category} API error: {e}. Falling back.")
        symbols = US_STOCKS_FALLBACK
        
    return symbols

# --- STOCK DETAIL FETCH FUNCTIONS ---

def fetch_single_stock_data(symbol, usd_to_inr):
    """Fetch real-time data for a single stock using yfinance"""
    try:
        # Use period='2d' interval='1d' for stable, current day data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2d', interval='1d') 
        
        if hist.empty or len(hist) < 2: return None
        
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2])
        volume = int(hist['Volume'].iloc[-1])
        
        if prev_close == 0: return None
        
        change_percent = ((current_price - prev_close) / prev_close) * 100
        
        try:
            info = ticker.info
            company_name = info.get('longName') or info.get('shortName') or symbol
        except:
            company_name = symbol.replace('.NS', '').replace('.BO', '')
        
        is_indian = '.NS' in symbol or '.BO' in symbol
        
        if is_indian:
            price_inr = current_price
            price_usd = current_price / usd_to_inr
            currency = '₹'
            market = '🇮🇳 NSE'
            display_price = price_inr
        else:
            price_usd = current_price
            price_inr = current_price * usd_to_inr
            currency = '$'
            market = '🇺🇸 US'
            display_price = price_usd
        
        if len(company_name) > 40: company_name = company_name[:37] + '...'
        
        return {
            'symbol': symbol.replace('.NS', '').replace('.BO', ''),
            'name': company_name,
            'price': round(display_price, 2),
            'currency': currency,
            'price_inr': round(price_inr, 2),
            'price_usd': round(price_usd, 2),
            'change': round(change_percent, 2),
            'volume': volume,
            'market': market,
            'is_indian': is_indian
        }
        
    except Exception as e:
        return None

def fetch_stocks_parallel(symbols, usd_to_inr, max_workers=20):
    """Fetch stocks in parallel"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_stock_data, symbol, usd_to_inr): symbol 
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result: results.append(result)
    
    return results

# --- MAIN ROUTES (Implementing Top 10 US + Top 10 India Logic) ---

@bp.route('/gainers')
@login_required
def top_gainers():
    try:
        usd_to_inr = get_usd_to_inr()
        
        # Step 1: Get dynamic gainers lists (buffers of 50 each)
        us_symbols = get_yahoo_screener_api('gainers')
        indian_symbols = get_nse_dynamic_symbols('gainers')
        
        # Step 2: Combine all symbols to fetch details in parallel
        all_symbols = list(set(us_symbols + indian_symbols)) 
        all_stocks = fetch_stocks_parallel(all_symbols, usd_to_inr)
        
        # Step 3: Filter and sort all valid gainers
        gainers = [s for s in all_stocks if s['change'] > 0]
        gainers.sort(key=lambda x: x['change'], reverse=True)
        
        # Step 4: Separate and slice to guarantee Top 10 of each (if available)
        us_gainers = [s for s in gainers if not s['is_indian']][:10]
        indian_gainers = [s for s in gainers if s['is_indian']][:10]
        
        # Step 5: Merge results
        top_20 = us_gainers + indian_gainers
        
        return jsonify({
            'success': True,
            'gainers': top_20,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2),
            'total_gainers': len(gainers),
            'source': 'Yahoo US Screener + NSE Live Scrape'
        })
    
    except Exception as e:
        print(f"❌ ERROR in /gainers: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/losers')
@login_required
def top_losers():
    try:
        usd_to_inr = get_usd_to_inr()
        
        # Step 1: Get dynamic losers lists (buffers of 50 each)
        us_symbols = get_yahoo_screener_api('losers')
        indian_symbols = get_nse_dynamic_symbols('losers')
        
        # Step 2: Combine all symbols to fetch details in parallel
        all_symbols = list(set(us_symbols + indian_symbols))
        all_stocks = fetch_stocks_parallel(all_symbols, usd_to_inr)
        
        # Step 3: Filter and sort all valid losers
        losers = [s for s in all_stocks if s['change'] < 0]
        losers.sort(key=lambda x: x['change']) # Sort ascending: most negative first (biggest loser)
        
        # Step 4: Separate and slice to guarantee Top 10 of each (if available)
        us_losers = [s for s in losers if not s['is_indian']][:10]
        indian_losers = [s for s in losers if s['is_indian']][:10]
        
        # Step 5: Merge results
        top_20 = us_losers + indian_losers
        
        return jsonify({
            'success': True,
            'losers': top_20,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2),
            'total_losers': len(losers),
            'source': 'Yahoo US Screener + NSE Live Scrape'
        })
    
    except Exception as e:
        print(f"❌ ERROR in /losers: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/most-active')
@login_required
def most_active():
    try:
        usd_to_inr = get_usd_to_inr()
        
        # Step 1: Get dynamic active lists (buffers of 50 each)
        us_symbols = get_yahoo_screener_api('mostactive')
        indian_symbols = get_nse_dynamic_symbols('mostactive')
        
        # Step 2: Combine all symbols to fetch details in parallel
        all_symbols = list(set(us_symbols + indian_symbols))
        all_stocks = fetch_stocks_parallel(all_symbols, usd_to_inr)
        
        # Step 3: Sort all stocks by volume
        all_stocks.sort(key=lambda x: x['volume'], reverse=True)
        
        # Step 4: Separate Top 10 US and Top 10 India by volume
        us_active = [s for s in all_stocks if not s['is_indian']][:10]
        indian_active = [s for s in all_stocks if s['is_indian']][:10]
        
        # Step 5: Merge results
        top_20 = us_active + indian_active
        
        return jsonify({
            'success': True,
            'active': top_20,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2),
            'source': 'Yahoo US Screener + NSE Live Scrape'
        })
    
    except Exception as e:
        print(f"❌ ERROR in /most-active: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# --- SECTOR PERFORMANCE (Fixed ETFs, No Dynamic Stock Lists Needed) ---

def fetch_sector_data(symbol_name_tuple):
    """Fetch sector ETF data"""
    symbol, name = symbol_name_tuple
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='7d', interval='1d') 
        
        if hist.empty or len(hist) < 2:
            return {
                'sector': name, 'symbol': symbol, 'today_change': 0.0,
                'change_5d': None, 'price': float(hist['Close'].iloc[-1]) if not hist.empty else 0.0
            }
        
        current_price = float(hist['Close'].iloc[-1])
        yesterday_close = float(hist['Close'].iloc[-2])
        today_change = ((current_price - yesterday_close) / yesterday_close) * 100 if yesterday_close != 0 else 0

        change_5d = None
        if len(hist) >= 5:
            five_days_ago_close = float(hist['Close'].iloc[-5])
            change_5d = ((current_price - five_days_ago_close) / five_days_ago_close) * 100 if five_days_ago_close != 0 else 0
        
        return {
            'sector': name, 'symbol': symbol, 'today_change': round(today_change, 2),
            'change_5d': round(change_5d, 2) if change_5d is not None else None, 
            'price': round(current_price, 2)
        }
        
    except Exception as e:
        return {'sector': name, 'symbol': symbol, 'today_change': 0.0, 'change_5d': None, 'price': 0.0}

@bp.route('/sector-performance')
@login_required
def sector_performance():
    try:
        sectors = [
            ('XLK', 'Technology'), ('XLF', 'Financial'), ('XLV', 'Healthcare'),
            ('XLE', 'Energy'), ('XLI', 'Industrial'), ('XLY', 'Consumer Discretionary'),
            ('XLP', 'Consumer Staples'), ('XLB', 'Materials'), ('XLRE', 'Real Estate'),
            ('XLU', 'Utilities'), ('XLC', 'Communication')
        ]
        
        sector_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
            futures = [executor.submit(fetch_sector_data, s) for s in sectors]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result: sector_data.append(result)
        
        sector_data.sort(key=lambda x: x['today_change'], reverse=True)
        usd_to_inr = get_usd_to_inr()
        
        return jsonify({
            'success': True,
            'sectors': sector_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# --- MARKET STATUS ---

@bp.route('/market-status')
@login_required
def market_status():
    try:
        now_utc = datetime.now(pytz.UTC)
        
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = now_utc.astimezone(ist)
        # NSE open hours: 9:15 AM to 3:30 PM, Monday to Friday
        nse_open = (now_ist.weekday() < 5 and 
                    ((now_ist.hour == 9 and now_ist.minute >= 15) or 
                     (9 < now_ist.hour < 15) or 
                     (now_ist.hour == 15 and now_ist.minute <= 30)))
        
        est = pytz.timezone('America/New_York')
        now_est = now_utc.astimezone(est)
        # NYSE/NASDAQ open hours: 9:30 AM to 4:00 PM EST, Monday to Friday
        nyse_open = (now_est.weekday() < 5 and 
                     ((now_est.hour == 9 and now_est.minute >= 30) or 
                      (9 < now_est.hour < 16)))
        
        return jsonify({
            'success': True,
            'nse': {'is_open': nse_open, 'status': '🟢 OPEN' if nse_open else '🔴 CLOSED', 'time': now_ist.strftime('%I:%M %p IST')},
            'nyse': {'is_open': nyse_open, 'status': '🟢 OPEN' if nyse_open else '🔴 CLOSED', 'time': now_est.strftime('%I:%M %p EST')}
        })
    except:

        return jsonify({'success': True, 'nse': {'is_open': False, 'status': '❓ UNKNOWN'}, 'nyse': {'is_open': False, 'status': '❓ UNKNOWN'}})
