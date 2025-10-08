from flask import Blueprint, render_template, jsonify
from routes.auth import login_required
import yfinance as yf
from datetime import datetime
import requests
import traceback
import sys

# Set encoding for print statements, useful for symbols like '₹'
# sys.stdout.reconfigure(encoding='utf-8')

bp = Blueprint('trending', __name__, url_prefix='/trending')

def get_usd_to_inr():
    """Get REAL-TIME USD to INR conversion rate"""
    try:
        # Using a reliable free exchange rate API
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        inr_rate = data['rates']['INR']
        print(f"💱 Exchange Rate: $1 = ₹{inr_rate:.2f}")
        return inr_rate
    except requests.exceptions.RequestException as e:
        print(f"❌ Exchange Rate API Error: {e}. Falling back to default rate.")
        return 83.0
    except Exception as e:
        print(f"❌ Exchange Rate JSON/Other Error: {e}. Falling back to default rate.")
        return 83.0

@bp.route('/')
@login_required
def trending_page():
    """Trending stocks page"""
    return render_template('trending.html')

def fetch_stock_data(symbols, usd_to_inr, is_gainer=True):
    """
    Helper function to fetch stock data, calculate change, and filter gainers/losers.
    """
    results = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get REAL-TIME intraday data (1-minute intervals) for live change
            hist = ticker.history(period='1d', interval='1m')
            
            # Fallback to 2-day daily data if no live/intraday data is available
            if hist.empty or len(hist) < 2:
                hist = ticker.history(period='2d')

            if len(hist) >= 2:
                current_price = float(hist['Close'].iloc[-1])
                volume = int(hist['Volume'].iloc[-1])
                
                # Determine previous price for change calculation
                if len(hist) > 10:  # Enough intraday data, compare with today's open
                    previous_price = float(hist['Open'].iloc[0])
                else: # Daily data, compare with previous close
                    previous_price = float(hist['Close'].iloc[-2])
                
                if previous_price == 0: # Avoid division by zero
                    continue 

                change_percent = ((current_price - previous_price) / previous_price) * 100
                
                # Filter based on whether we are looking for gainers or losers
                if (is_gainer and change_percent > 0) or (not is_gainer and change_percent < 0):
                    info = ticker.info
                    company_name = info.get('longName', symbol.replace('.NS', ''))
                    
                    is_indian = '.NS' in symbol
                    
                    if is_indian:
                        price_inr = current_price
                        price_usd = current_price / usd_to_inr
                        currency_symbol = '₹'
                        market = '🇮🇳 NSE'
                        price_display = round(price_inr, 2)
                    else:
                        price_usd = current_price
                        price_inr = current_price * usd_to_inr
                        currency_symbol = '$'
                        market = '🇺🇸 NYSE'
                        price_display = round(price_usd, 2)
                    
                    # Truncate long names
                    if len(company_name) > 40:
                        company_name = company_name[:37] + '...'
                    
                    results.append({
                        'symbol': symbol.replace('.NS', ''),
                        'name': company_name,
                        'price': price_display, # Price in its native currency
                        'currency': currency_symbol,
                        'price_inr': round(price_inr, 2),
                        'price_usd': round(price_usd, 2),
                        'change': round(change_percent, 2),
                        'volume': volume,
                        'market': market,
                        'is_indian': is_indian
                    })
                    
                    sign = '+' if change_percent > 0 else ''
                    print(f"✅ {symbol:10s} {currency_symbol}{current_price:8.2f} {sign}{change_percent:5.2f}% {market}")
            
        except Exception as e:
            print(f"❌ {symbol:10s} Error: {str(e)[:30]}")
            continue
            
    return results


# ---

@bp.route('/gainers')
@login_required
def top_gainers():
    """Get REAL-TIME top gainers - INDIAN STOCKS (NSE) + US STOCKS"""
    try:
        print(f"\n{'='*70}")
        print(f"🚀 FETCHING REAL-TIME TOP GAINERS (INDIA + US)")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        print(f"{'='*70}")
        
        # Define stock lists
        indian_stocks = [
            'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS',
            'AXISBANK.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'MARUTI.NS', 'SUNPHARMA.NS'
        ]
        us_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'V', 'MA', 'WMT'
        ]
        all_symbols = indian_stocks + us_stocks
        
        usd_to_inr = get_usd_to_inr()
        
        gainers = fetch_stock_data(all_symbols, usd_to_inr, is_gainer=True)
        
        # Sort by change percentage (highest first)
        gainers.sort(key=lambda x: x['change'], reverse=True)
        top_20_gainers = gainers[:20]
        
        print(f"\n✅ SUCCESS: Found {len(gainers)} gainers, returning top 20")
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': True,
            'gainers': top_20_gainers,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2),
            'total_processed': len(all_symbols)
        })
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR in top_gainers: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ---

@bp.route('/losers')
@login_required
def top_losers():
    """Get REAL-TIME top losers - INDIAN + US STOCKS"""
    try:
        print(f"\n{'='*70}")
        print(f"📉 FETCHING REAL-TIME TOP LOSERS (INDIA + US)")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        print(f"{'='*70}")
        
        # Define stock lists
        indian_stocks = [
            'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS',
            'AXISBANK.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'MARUTI.NS', 'SUNPHARMA.NS'
        ]
        us_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'V', 'MA', 'WMT'
        ]
        all_symbols = indian_stocks + us_stocks
        
        usd_to_inr = get_usd_to_inr()
        
        losers = fetch_stock_data(all_symbols, usd_to_inr, is_gainer=False)
        
        # Sort by change percentage (lowest first, which is most negative)
        losers.sort(key=lambda x: x['change'])
        top_20_losers = losers[:20]
        
        print(f"\n✅ SUCCESS: Found {len(losers)} losers, returning top 20")
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': True,
            'losers': top_20_losers,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2),
            'total_processed': len(all_symbols)
        })
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR in top_losers: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ---

@bp.route('/most-active')
@login_required
def most_active():
    """Get REAL-TIME most active - INDIAN + US STOCKS"""
    try:
        print(f"\n{'='*70}")
        print(f"💹 FETCHING MOST ACTIVE (INDIA + US)")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        print(f"{'='*70}")
        
        # Define stock lists
        indian_stocks = [
            'TCS.NS', 'INFY.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS',
            'AXISBANK.NS', 'KOTAKBANK.NS', 'WIPRO.NS', 'MARUTI.NS', 'LT.NS'
        ]
        us_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'V', 'MA', 'WMT'
        ]
        all_symbols = indian_stocks + us_stocks
        
        active_stocks = []
        usd_to_inr = get_usd_to_inr()
        
        for symbol in all_symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Fetch recent data for volume and price
                # Use 1d 1m for live volume, fall back to 1d 1d if empty
                hist = ticker.history(period='1d', interval='1m')
                if hist.empty:
                     hist = ticker.history(period='1d', interval='1d')
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1])
                    
                    # Calculate change from previous close (using 2 days for stability)
                    hist_2d = ticker.history(period='2d')
                    if len(hist_2d) >= 2:
                        prev = float(hist_2d['Close'].iloc[-2])
                        change = ((current_price - prev) / prev) * 100
                    else:
                        change = 0
                    
                    info = ticker.info
                    is_indian = '.NS' in symbol
                    
                    if is_indian:
                        price_inr = current_price
                        price_usd = current_price / usd_to_inr
                        market = '🇮🇳 NSE'
                    else:
                        price_usd = current_price
                        price_inr = current_price * usd_to_inr
                        market = '🇺🇸 NYSE'
                    
                    company_name = info.get('longName', symbol.replace('.NS', ''))
                    if len(company_name) > 40:
                        company_name = company_name[:37] + '...'

                    active_stocks.append({
                        'symbol': symbol.replace('.NS', ''),
                        'name': company_name,
                        'volume': volume,
                        'price_inr': round(price_inr, 2),
                        'price_usd': round(price_usd, 2),
                        'change': round(change, 2),
                        'market': market
                    })
                    
                    print(f"✅ {symbol:10s} Vol: {volume:15,d}")
            
            except Exception as e:
                print(f"❌ {symbol:10s} Error: {str(e)[:30]}")
                continue
        
        # Sort by volume (highest first)
        active_stocks.sort(key=lambda x: x['volume'], reverse=True)
        top_active = active_stocks[:20]
        
        print(f"\n✅ SUCCESS: Found {len(active_stocks)} active stocks, returning top 20")
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': True,
            'active': top_active,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2),
            'total_processed': len(all_symbols)
        })
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR in most_active: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# ---

@bp.route('/sector-performance')
@login_required
def sector_performance():
    """Get REAL-TIME sector performance using sector ETFs"""
    try:
        print(f"\n{'='*70}")
        print(f"🏢 FETCHING REAL-TIME SECTOR PERFORMANCE")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')}")
        print(f"{'='*70}")
        
        sectors = {
            'XLK': 'Technology', 'XLF': 'Financial Services', 'XLV': 'Healthcare', 
            'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Discretionary', 
            'XLP': 'Consumer Staples', 'XLB': 'Materials', 'XLRE': 'Real Estate', 
            'XLU': 'Utilities', 'XLC': 'Communication Services'
        }
        
        sector_data = []
        usd_to_inr = get_usd_to_inr()
        
        for symbol, name in sectors.items():
            try:
                ticker = yf.Ticker(symbol)
                
                # Get 5-day history for trend
                hist_5d = ticker.history(period='5d', interval='1d')
                # Get today's intraday data for current price and open
                hist_1d_intraday = ticker.history(period='1d', interval='5m')
                
                if hist_1d_intraday.empty:
                    # Fallback to 1-day daily data if no intraday available
                    hist_1d_intraday = ticker.history(period='1d', interval='1d')
                
                if len(hist_5d) >= 2 and not hist_1d_intraday.empty:
                    current_price = float(hist_1d_intraday['Close'].iloc[-1])
                    
                    # 5-day change (current price vs. close 5 days ago)
                    week_ago_price = float(hist_5d['Close'].iloc[0])
                    if week_ago_price != 0:
                        change_5d = ((current_price - week_ago_price) / week_ago_price) * 100
                    else:
                        change_5d = 0
                    
                    # Today's change (current price vs. today's open or previous day's close)
                    if len(hist_1d_intraday) > 1 and 'Open' in hist_1d_intraday:
                        today_open = float(hist_1d_intraday['Open'].iloc[0])
                        today_change = ((current_price - today_open) / today_open) * 100 if today_open != 0 else 0
                    elif len(hist_5d) >= 2:
                        # Use previous day's close from 5-day history
                        prev_close = float(hist_5d['Close'].iloc[-2])
                        today_change = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                    else:
                        today_change = 0
                    
                    sector_data.append({
                        'sector': name,
                        'symbol': symbol,
                        'change_5d': round(change_5d, 2), # Changed key for clarity
                        'today_change': round(today_change, 2),
                        'price': round(current_price, 2),
                        'price_inr': round(current_price * usd_to_inr, 2)
                    })
                    
                    print(f"✅ {name:25s} 5D: {change_5d:+6.2f}% | Today: {today_change:+6.2f}%")
                
            except Exception as e:
                print(f"❌ {name:25s} Error: {str(e)[:30]}")
                continue
        
        # Sort by 5-day performance
        sector_data.sort(key=lambda x: x['change_5d'], reverse=True)
        
        print(f"\n✅ SUCCESS: Fetched {len(sector_data)} sectors")
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': True,
            'sectors': sector_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST'),
            'exchange_rate': round(usd_to_inr, 2)
        })
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR in sector_performance: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
