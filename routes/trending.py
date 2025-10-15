from flask import Blueprint, render_template, jsonify
from routes.auth import login_required
import yfinance as yf
from datetime import datetime, timedelta
import requests
import traceback
import concurrent.futures
from threading import Lock
import pytz
import time

bp = Blueprint('trending', __name__, url_prefix='/trending')

_exchange_rate_cache = {'rate': 88.79, 'timestamp': None}
_cache_lock = Lock()

def get_usd_to_inr():
    """Get LIVE USD to INR exchange rate with caching"""
    global _exchange_rate_cache
    now = datetime.now()
    
    with _cache_lock:
        if _exchange_rate_cache['timestamp']:
            age = (now - _exchange_rate_cache['timestamp']).seconds
            if age < 300:  # Cache for 5 minutes
                return _exchange_rate_cache['rate']
    
    # Try multiple forex APIs
    sources = [
        ('https://api.exchangerate-api.com/v4/latest/USD', lambda r: r.json()['rates']['INR']),
        ('https://open.er-api.com/v6/latest/USD', lambda r: r.json()['rates']['INR']),
        ('https://api.fxratesapi.com/latest?base=USD&currencies=INR', lambda r: r.json()['rates']['INR']),
    ]
    
    for url, parser in sources:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                rate = float(parser(response))
                if 80 <= rate <= 95:  # Validate range
                    with _cache_lock:
                        _exchange_rate_cache = {'rate': rate, 'timestamp': now}
                    print(f"✓ Exchange Rate: $1 = ₹{rate:.2f}")
                    return rate
        except:
            continue
    
    print(f"⚠ Using cached rate: ₹{_exchange_rate_cache['rate']:.2f}")
    return _exchange_rate_cache['rate']

@bp.route('/')
@login_required
def trending_page():
    return render_template('trending.html')

def get_top_indian_stocks():
    """Get top liquid Indian stocks to scan for real-time data"""
    return [
        # NIFTY 50 - Most liquid
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
        'HCLTECH.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS',
        'WIPRO.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'M&M.NS', 'TECHM.NS',
        'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS', 'COALINDIA.NS', 'ADANIPORTS.NS',
        'BAJAJFINSV.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HEROMOTOCO.NS',
        'INDUSINDBK.NS', 'JSWSTEEL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'DIVISLAB.NS',
        'SHREECEM.NS', 'HINDALCO.NS', 'BPCL.NS', 'SBILIFE.NS', 'VEDL.NS',
        'TATACONSUM.NS', 'APOLLOHOSP.NS', 'HDFCLIFE.NS', 'ADANIENT.NS', 'ICICIGI.NS',
        
        # High liquidity stocks
        'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'PIDILITIND.NS', 'HAVELLS.NS',
        'LUPIN.NS', 'TORNTPHARM.NS', 'BIOCON.NS', 'AUROPHARMA.NS', 'GODREJCP.NS',
        'MARICO.NS', 'DABUR.NS', 'COLPAL.NS', 'AMBUJACEM.NS', 'ACC.NS',
        'DLF.NS', 'GODREJPROP.NS', 'SIEMENS.NS', 'ABB.NS', 'BOSCHLTD.NS',
        'INDIGO.NS', 'PFC.NS', 'RECLTD.NS', 'BANKBARODA.NS', 'PNB.NS',
        
        # Popular stocks
        'ZOMATO.NS', 'NYKAA.NS', 'PAYTM.NS', 'DMART.NS', 'IRCTC.NS',
        'HAL.NS', 'BEL.NS', 'LTIM.NS', 'PERSISTENT.NS', 'COFORGE.NS',
        'DIXON.NS', 'CUMMINSIND.NS', 'ESCORTS.NS', 'TVSMOTOR.NS', 'BAJAJ-AUTO.NS',
        'TATAPOWER.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'SUZLON.NS', 'IRFC.NS'
    ]

def fetch_indian_stock_realtime(symbol, usd_to_inr):
    """Fetch real-time data for Indian stocks using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get info for company name
        try:
            info = ticker.info
            company_name = info.get('longName') or info.get('shortName') or symbol.replace('.NS', '')
        except:
            company_name = symbol.replace('.NS', '')
        
        # Get real-time price using fast_info (fastest method)
        try:
            fast_info = ticker.fast_info
            current_price = float(fast_info.get('lastPrice', 0))
            prev_close = float(fast_info.get('previousClose', 0))
            
            if current_price <= 0 or prev_close <= 0:
                raise ValueError("Invalid fast_info")
        except:
            # Fallback to history
            hist = ticker.history(period='2d', interval='1d')
            if hist.empty or len(hist) < 2:
                return None
            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2])
        
        if prev_close <= 0 or current_price <= 0:
            return None
        
        # Get volume
        try:
            hist = ticker.history(period='1d', interval='1d')
            current_volume = int(hist['Volume'].iloc[-1]) if not hist.empty else 0
        except:
            current_volume = 0
        
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        # Filter out unrealistic changes (likely data errors)
        if abs(change_pct) > 50:  # Skip stocks with >50% change (likely errors)
            return None
        
        # Format volume
        if current_volume >= 1_000_000:
            volume_display = f"{current_volume / 1_000_000:.2f}M"
        elif current_volume >= 1000:
            volume_display = f"{current_volume / 1000:.2f}K"
        else:
            volume_display = str(current_volume)
        
        # Truncate long names
        if len(company_name) > 40:
            company_name = company_name[:37] + '...'
        
        price_inr = current_price
        price_usd = current_price / usd_to_inr
        
        return {
            'symbol': symbol.replace('.NS', ''),
            'name': company_name,
            'price': round(price_inr, 2),
            'currency': '₹',
            'price_inr': round(price_inr, 2),
            'price_usd': round(price_usd, 2),
            'change': round(change_pct, 2),
            'change_abs': round(current_price - prev_close, 2),
            'volume': current_volume,
            'volume_display': volume_display,
            'market': 'NSE',
            'is_indian': True,
            'last_updated': datetime.now().strftime('%I:%M:%S %p')
        }
    except:
        return None

def fetch_indian_stocks_parallel(symbols, usd_to_inr, max_workers=50):
    """Fetch Indian stocks in parallel"""
    stocks = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_indian_stock_realtime, symbol, usd_to_inr): symbol 
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                stocks.append(result)
    
    return stocks

def fetch_yahoo_screener_direct(category, region='US'):
    """
    Fetch real-time stock data directly from Yahoo Finance screener API
    This gets LIVE data that matches finance.yahoo.com exactly
    """
    
    if region == 'US':
        # US Screeners - Same as finance.yahoo.com
        screener_map = {
            'gainers': 'day_gainers',
            'losers': 'day_losers',
            'active': 'most_actives'
        }
        
        if category not in screener_map:
            return []
        
        url = f"https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            'formatted': 'false',
            'scrIds': screener_map[category],
            'count': 25,
            'start': 0
        }
        
    else:  # India
        # Indian Screeners
        screener_map = {
            'gainers': 'day_gainers_in',
            'losers': 'day_losers_in', 
            'active': 'most_actives_in'
        }
        
        if category not in screener_map:
            return []
            
        url = f"https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            'formatted': 'false',
            'scrIds': screener_map[category],
            'count': 25,
            'start': 0,
            'region': 'IN'
        }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finance.yahoo.com/',
        'Origin': 'https://finance.yahoo.com'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"✗ Yahoo API error: {response.status_code}")
            return []
        
        data = response.json()
        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
        
        if not quotes:
            print(f"✗ No quotes returned for {region} {category}")
            return []
        
        stocks = []
        usd_to_inr = get_usd_to_inr()
        
        for quote in quotes:
            try:
                symbol = quote.get('symbol', '')
                if not symbol:
                    continue
                
                # Get real-time data from the screener response
                regular_market_price = quote.get('regularMarketPrice', 0)
                regular_market_change = quote.get('regularMarketChange', 0)
                regular_market_change_percent = quote.get('regularMarketChangePercent', 0)
                regular_market_volume = quote.get('regularMarketVolume', 0)
                
                # Get company name
                short_name = quote.get('shortName', '')
                long_name = quote.get('longName', '')
                display_name = long_name or short_name or symbol
                
                if len(display_name) > 45:
                    display_name = display_name[:42] + '...'
                
                # Determine if Indian stock
                is_indian = '.NS' in symbol or '.BO' in symbol or region == 'IN'
                
                if regular_market_price <= 0:
                    continue
                
                # Calculate prices
                if is_indian:
                    price_inr = regular_market_price
                    price_usd = regular_market_price / usd_to_inr
                    currency = '₹'
                    market = 'NSE'
                    display_price = price_inr
                else:
                    price_usd = regular_market_price
                    price_inr = regular_market_price * usd_to_inr
                    currency = '$'
                    market = 'US'
                    display_price = price_usd
                
                # Format volume
                if regular_market_volume >= 1_000_000:
                    volume_display = f"{regular_market_volume / 1_000_000:.2f}M"
                elif regular_market_volume >= 1000:
                    volume_display = f"{regular_market_volume / 1000:.2f}K"
                else:
                    volume_display = str(regular_market_volume)
                
                stocks.append({
                    'symbol': symbol.replace('.NS', '').replace('.BO', ''),
                    'name': display_name,
                    'price': round(display_price, 2),
                    'currency': currency,
                    'price_inr': round(price_inr, 2),
                    'price_usd': round(price_usd, 2),
                    'change': round(regular_market_change_percent, 2),
                    'change_abs': round(regular_market_change, 2),
                    'volume': regular_market_volume,
                    'volume_display': volume_display,
                    'market': market,
                    'is_indian': is_indian,
                    'last_updated': datetime.now().strftime('%I:%M:%S %p')
                })
                
            except Exception as e:
                continue
        
        print(f"✓ Yahoo {region} {category}: {len(stocks)} stocks fetched with LIVE data")
        return stocks
        
    except Exception as e:
        print(f"✗ Yahoo {region} {category} error: {e}")
        traceback.print_exc()
        return []

@bp.route('/gainers')
@login_required
def top_gainers():
    """Top Gainers: 10 US + 10 India (Real-time from Yahoo Finance API)"""
    try:
        print("\n" + "="*60)
        print("FETCHING TOP GAINERS (REAL-TIME FROM YAHOO FINANCE)")
        print("="*60)
        
        usd_to_inr = get_usd_to_inr()
        
        # Fetch US stocks from Yahoo screener (real-time API)
        us_gainers = fetch_yahoo_screener_direct('gainers', 'US')
        us_top10 = us_gainers[:10]
        
        # Fetch Indian stocks using yfinance (more reliable for India)
        print("✓ Scanning top Indian stocks for real-time gainers...")
        indian_symbols = get_top_indian_stocks()
        all_indian = fetch_indian_stocks_parallel(indian_symbols, usd_to_inr, max_workers=60)
        
        # Filter gainers and sort by change %
        india_gainers = [s for s in all_indian if s['change'] > 0]
        india_gainers.sort(key=lambda x: x['change'], reverse=True)
        india_top10 = india_gainers[:10]
        
        # Combine: First 10 US, then 10 India
        top_20 = us_top10 + india_top10
        
        print(f"✓ Results: US: {len(us_top10)} gainers | India: {len(india_top10)} gainers")
        if us_top10:
            print(f"✓ #1 US Gainer: {us_top10[0]['name']} (+{us_top10[0]['change']}%) @ ${us_top10[0]['price']}")
        if india_top10:
            print(f"✓ #1 India Gainer: {india_top10[0]['name']} (+{india_top10[0]['change']}%) @ ₹{india_top10[0]['price']}")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'gainers': top_20,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'),
            'exchange_rate': round(usd_to_inr, 2),
            'stats': {
                'us_count': len(us_top10),
                'india_count': len(india_top10),
                'total_count': len(us_top10) + len(india_top10)
            }
        })
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/losers')
@login_required
def top_losers():
    """Top Losers: 10 US + 10 India (Real-time from Yahoo Finance API)"""
    try:
        print("\n" + "="*60)
        print("FETCHING TOP LOSERS (REAL-TIME FROM YAHOO FINANCE)")
        print("="*60)
        
        usd_to_inr = get_usd_to_inr()
        
        # Fetch US stocks from Yahoo screener (real-time API)
        us_losers = fetch_yahoo_screener_direct('losers', 'US')
        us_top10 = us_losers[:10]
        
        # Fetch Indian stocks using yfinance (more reliable for India)
        print("✓ Scanning top Indian stocks for real-time losers...")
        indian_symbols = get_top_indian_stocks()
        all_indian = fetch_indian_stocks_parallel(indian_symbols, usd_to_inr, max_workers=60)
        
        # Filter losers and sort by change %
        india_losers = [s for s in all_indian if s['change'] < 0]
        india_losers.sort(key=lambda x: x['change'])
        india_top10 = india_losers[:10]
        
        # Combine: First 10 US, then 10 India
        top_20 = us_top10 + india_top10
        
        print(f"✓ Results: US: {len(us_top10)} losers | India: {len(india_top10)} losers")
        if us_top10:
            print(f"✓ #1 US Loser: {us_top10[0]['name']} ({us_top10[0]['change']}%) @ ${us_top10[0]['price']}")
        if india_top10:
            print(f"✓ #1 India Loser: {india_top10[0]['name']} ({india_top10[0]['change']}%) @ ₹{india_top10[0]['price']}")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'losers': top_20,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'),
            'exchange_rate': round(usd_to_inr, 2),
            'stats': {
                'us_count': len(us_top10),
                'india_count': len(india_top10),
                'total_count': len(us_top10) + len(india_top10)
            }
        })
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/most-active')
@login_required
def most_active():
    """Most Active: 10 US + 10 India (Real-time from Yahoo Finance API)"""
    try:
        print("\n" + "="*60)
        print("FETCHING MOST ACTIVE (REAL-TIME FROM YAHOO FINANCE)")
        print("="*60)
        
        usd_to_inr = get_usd_to_inr()
        
        # Fetch US stocks from Yahoo screener (real-time API)
        us_active = fetch_yahoo_screener_direct('active', 'US')
        us_top10 = us_active[:10]
        
        # Fetch Indian stocks using yfinance (more reliable for India)
        print("✓ Scanning top Indian stocks for real-time volume...")
        indian_symbols = get_top_indian_stocks()
        all_indian = fetch_indian_stocks_parallel(indian_symbols, usd_to_inr, max_workers=60)
        
        # Sort by volume
        all_indian.sort(key=lambda x: x['volume'], reverse=True)
        india_top10 = all_indian[:10]
        
        # Combine: First 10 US, then 10 India
        top_20 = us_top10 + india_top10
        
        print(f"✓ Results: US: {len(us_top10)} active | India: {len(india_top10)} active")
        if us_top10:
            print(f"✓ #1 US Active: {us_top10[0]['name']} ({us_top10[0]['volume_display']}) @ ${us_top10[0]['price']}")
        if india_top10:
            print(f"✓ #1 India Active: {india_top10[0]['name']} ({india_top10[0]['volume_display']}) @ ₹{india_top10[0]['price']}")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'active': top_20,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'),
            'exchange_rate': round(usd_to_inr, 2),
            'stats': {
                'us_count': len(us_top10),
                'india_count': len(india_top10),
                'total_count': len(us_top10) + len(india_top10)
            }
        })
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/sector-performance')
@login_required
def sector_performance():
    """US & Indian Sector Performance (Real-time)"""
    try:
        print("\n" + "="*50)
        print("FETCHING SECTOR PERFORMANCE")
        print("="*50)
        
        # US Sector ETFs
        us_sectors = [
            ('XLK', 'Technology'), ('XLF', 'Financial'), ('XLV', 'Healthcare'),
            ('XLE', 'Energy'), ('XLI', 'Industrial'), ('XLY', 'Consumer Discretionary'),
            ('XLP', 'Consumer Staples'), ('XLB', 'Materials'), ('XLRE', 'Real Estate'),
            ('XLU', 'Utilities'), ('XLC', 'Communication')
        ]
        
        # Indian Sector Indices
        indian_sectors = [
            ('^NSEI', 'NIFTY 50'), ('^NSEBANK', 'Bank Nifty'),
            ('NIFTYMETAL.NS', 'Metal'), ('NIFTYIT.NS', 'IT'),
            ('NIFTYPHARMA.NS', 'Pharma'), ('NIFTYAUTO.NS', 'Auto'),
            ('NIFTYFMCG.NS', 'FMCG'), ('NIFTYREALTY.NS', 'Realty'),
            ('NIFTYENERGY.NS', 'Energy'), ('NIFTYPSE.NS', 'PSU')
        ]
        
        def fetch_sector(symbol_name, market):
            symbol, name = symbol_name
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='7d', interval='1d')
                
                if hist.empty or len(hist) < 2:
                    return None
                
                current = float(hist['Close'].iloc[-1])
                yesterday = float(hist['Close'].iloc[-2])
                today_change = ((current - yesterday) / yesterday) * 100 if yesterday > 0 else 0
                
                change_5d = None
                if len(hist) >= 5:
                    five_days_ago = float(hist['Close'].iloc[-5])
                    change_5d = ((current - five_days_ago) / five_days_ago) * 100 if five_days_ago > 0 else 0
                
                return {
                    'sector': name,
                    'symbol': symbol,
                    'market': market,
                    'today_change': round(today_change, 2),
                    'change_5d': round(change_5d, 2) if change_5d else None,
                    'price': round(current, 2)
                }
            except:
                return None
        
        sector_data = []
        
        # Fetch both US and Indian sectors in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            us_futures = [executor.submit(fetch_sector, s, 'US') for s in us_sectors]
            india_futures = [executor.submit(fetch_sector, s, 'India') for s in indian_sectors]
            
            for future in concurrent.futures.as_completed(us_futures + india_futures):
                result = future.result()
                if result:
                    sector_data.append(result)
        
        # Sort by today's change
        sector_data.sort(key=lambda x: x['today_change'], reverse=True)
        
        print(f"✓ Fetched {len(sector_data)} sectors (US + India)")
        print("="*50 + "\n")
        
        return jsonify({
            'success': True,
            'sectors': sector_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p'),
            'exchange_rate': round(get_usd_to_inr(), 2)
        })
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/market-status')
@login_required
def market_status():
    """Real-time Market Status for NSE and NYSE"""
    try:
        now_utc = datetime.now(pytz.UTC)
        
        # NSE timing (9:15 AM - 3:30 PM IST)
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = now_utc.astimezone(ist)
        nse_open = (now_ist.weekday() < 5 and
                    ((now_ist.hour == 9 and now_ist.minute >= 15) or
                     (9 < now_ist.hour < 15) or
                     (now_ist.hour == 15 and now_ist.minute <= 30)))
        
        # NYSE timing (9:30 AM - 4:00 PM EST)
        est = pytz.timezone('America/New_York')
        now_est = now_utc.astimezone(est)
        nyse_open = (now_est.weekday() < 5 and
                     ((now_est.hour == 9 and now_est.minute >= 30) or
                      (9 < now_est.hour < 16)))
        
        return jsonify({
            'success': True,
            'nse': {
                'is_open': nse_open,
                'status': 'OPEN' if nse_open else 'CLOSED',
                'time': now_ist.strftime('%I:%M %p IST'),
                'trading_hours': '9:15 AM - 3:30 PM IST'
            },
            'nyse': {
                'is_open': nyse_open,
                'status': 'OPEN' if nyse_open else 'CLOSED',
                'time': now_est.strftime('%I:%M %p EST'),
                'trading_hours': '9:30 AM - 4:00 PM EST'
            },
            'timestamp': now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        })
    
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500