import os
import time
import re
import traceback
import requests
import yfinance as yf
import google.generativeai as genai
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, session
from routes.auth import login_required
from config import Config
from dateutil.parser import parse
import concurrent.futures
from threading import Lock

try:
    from google.generativeai.errors import APIError as GeminiAPIError
except ImportError:
    class GeminiAPIError(Exception): pass

bp = Blueprint('chatbot', __name__, url_prefix='/chatbot')

# Global Configuration
GEMINI_KEY_STATUS = "PENDING"
MIN_KEY_LENGTH = 35

if Config.GEMINI_API_KEY:
    if len(Config.GEMINI_API_KEY) < MIN_KEY_LENGTH or not Config.GEMINI_API_KEY.startswith('AIzaSy'):
        GEMINI_KEY_STATUS = "ERROR: Key is too short or malformed."
        print(f"CRITICAL ERROR: {GEMINI_KEY_STATUS}")
    else:
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            print(f"SUCCESS: Gemini configured. Key length: {len(Config.GEMINI_API_KEY)}")
            GEMINI_KEY_STATUS = "CONFIGURED"
        except Exception as e:
            GEMINI_KEY_STATUS = f"CRITICAL ERROR: {e}"
            print(f"CRITICAL ERROR: {GEMINI_KEY_STATUS}")

@bp.route('/')
@login_required
def chatbot_page():
    return render_template('chatbot.html')

# Utility Functions
def get_usd_to_inr():
    """Get real-time USD to INR rate"""
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=3)
        return response.json()['rates']['INR']
    except:
        return 83.0

def extract_stock_symbol(question):
    """Extract single stock symbol from question"""
    q_lower = question.lower()
    
    mappings = {
        'TCS': ['tcs', 'tata consultancy'], 'INFY': ['infosys'], 
        'RELIANCE': ['reliance', 'ril'], 'HDFCBANK': ['hdfc bank', 'hdfc'],
        'ICICIBANK': ['icici bank', 'icici'], 'SBIN': ['sbi', 'state bank'],
        'BHARTIARTL': ['airtel', 'bharti'], 'TATAMOTORS': ['tata motors'],
        'AAPL': ['apple'], 'MSFT': ['microsoft'], 'GOOGL': ['google', 'alphabet'],
        'TSLA': ['tesla'], 'META': ['meta', 'facebook'], 'AMZN': ['amazon']
    }
    
    for symbol, names in mappings.items():
        for name in names:
            if name in q_lower:
                if symbol in ['TCS', 'INFY', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'TATAMOTORS']:
                    return f"{symbol}.NS"
                return symbol
    
    symbols = re.findall(r'\b[A-Z]{2,10}(?:\.[A-Z]{2})?\b', question.upper())
    if symbols:
        return symbols[0]
    
    return None

def extract_all_symbols(question):
    """Extract multiple symbols for comparison"""
    q_upper = question.upper()
    q_lower = question.lower()
    symbols = set()
    
    # Check mappings first
    mappings = {
        'AAPL': ['apple'], 'MSFT': ['microsoft'], 'TCS.NS': ['tcs', 'tata consultancy'],
        'INFY.NS': ['infosys'], 'GOOGL': ['google'], 'TSLA': ['tesla'],
        'META': ['meta', 'facebook'], 'AMZN': ['amazon'], 'RELIANCE.NS': ['reliance'],
        'HDFCBANK.NS': ['hdfc'], 'ICICIBANK.NS': ['icici']
    }
    
    for ticker, names in mappings.items():
        if any(name in q_lower for name in names):
            symbols.add(ticker)
    
    # Find explicit tickers
    explicit = re.findall(r'\b[A-Z]{2,10}(?:\.[A-Z]{2})?\b', q_upper)
    for sym in explicit:
        if sym not in ['VS', 'OR', 'AND']:  # Exclude common words
            symbols.add(sym)
    
    return list(symbols)[:5]

def detect_question_type(question):
    """Detect query type"""
    q = question.lower()
    
    # Check for comparison first (highest priority)
    has_vs = ' vs ' in q or ' versus ' in q
    has_compare = any(w in q for w in ['compare', 'comparison', 'difference between', 'which is better'])
    
    # Check for historical
    has_historical = any(w in q for w in ['price on', 'on 20', 'yesterday', 'last week', 'last month', 'ago', 'was the price', 'historical'])
    
    return {
        'comparison': (has_vs or has_compare) and len(extract_all_symbols(question)) >= 2,
        'historical': has_historical and not (has_vs or has_compare),
        'gainers': any(w in q for w in ['gainer', 'best', 'rising', 'top performer']) and not (has_vs or has_compare),
        'losers': any(w in q for w in ['loser', 'worst', 'falling', 'declining']) and not (has_vs or has_compare),
        'active': any(w in q for w in ['active', 'volume', 'traded', 'most bought']) and not (has_vs or has_compare),
        'sector': 'sector' in q and not (has_vs or has_compare),
        'overview': 'overview' in q or 'market summary' in q,
        'price': any(w in q for w in ['price', 'current', 'quote', 'value', 'trading']) and not (has_vs or has_compare or has_historical)
    }

def parse_date_from_question(question):
    """
    Parse date from question, supporting relative dates and common date formats.
    """
    q = question.lower()
    
    # 1. Handle Relative Dates
    if 'yesterday' in q:
        return (datetime.now() - timedelta(days=1)).date()
    if 'last week' in q or 'week ago' in q:
        return (datetime.now() - timedelta(days=7)).date()
    if 'last month' in q or 'month ago' in q:
        return (datetime.now() - timedelta(days=30)).date()
    
    # 2. Handle Explicit Dates using dateutil.parser (IMPROVED FOR DD MONTH YYYY)
    date_patterns = [
        # 1. Specifically target '23 september 2025' or '23rd sept 2025'
        r'\b(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,}\s+\d{4})\b', 
        # 2. YYYY-M-D (e.g., 2025-09-23)
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b', 
        # 3. M-D-YYYY (e.g., 9-23-2025)
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
        # 4. on Month DD, YYYY or on Month YYYY
        r'\b(on|at|for)\s+([a-z]+\s+\d{1,2},?\s+\d{4})\b', 
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, q)
        if match:
            try:
                # Group 2 is for patterns with 'on|at|for', otherwise Group 1
                date_str = match.group(2) if len(match.groups()) > 1 and match.group(2) else match.group(1)
                
                # Use dateutil.parser for robust parsing
                parsed_date = parse(date_str, fuzzy=True).date()
                print(f"DEBUG: Successfully parsed date string: '{date_str}' to {parsed_date}")
                return parsed_date
            except Exception as e:
                print(f"DEBUG: Failed to parse '{date_str}' with current pattern: {e}")
                continue
    
    return None

# Data Fetching Functions
def fetch_single_stock_realtime(symbol):
    """Fetch REAL-TIME stock data"""
    try:
        ticker = yf.Ticker(symbol)
        hist_1m = ticker.history(period='1d', interval='1m')
        hist_2d = ticker.history(period='2d', interval='1d')
        
        if hist_2d.empty or len(hist_2d) < 2:
            return None
        
        if not hist_1m.empty:
            current_price = float(hist_1m['Close'].iloc[-1])
            volume = int(hist_1m['Volume'].sum())
        else:
            current_price = float(hist_2d['Close'].iloc[-1])
            volume = int(hist_2d['Volume'].iloc[-1])
        
        prev_close = float(hist_2d['Close'].iloc[-2])
        if prev_close == 0:
            return None
        
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        try:
            info = ticker.info
            company_name = info.get('longName') or info.get('shortName') or symbol
            sector = info.get('sector', 'N/A')
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 'N/A')
        except:
            company_name = symbol
            sector = 'N/A'
            market_cap = 0
            pe_ratio = 'N/A'
        
        is_indian = '.NS' in symbol or '.BO' in symbol
        usd_to_inr = get_usd_to_inr()
        
        if is_indian:
            price_inr = current_price
            price_usd = current_price / usd_to_inr
        else:
            price_usd = current_price
            price_inr = current_price * usd_to_inr
        
        mc = market_cap if is_indian else market_cap * usd_to_inr
        if mc >= 1e12:
            mc_str = f"₹{mc/1e12:.2f} Lakh Crore"
        elif mc >= 1e10:
            mc_str = f"₹{mc/1e10:.2f} Thousand Crore"
        elif mc >= 1e7:
            mc_str = f"₹{mc/1e7:.2f} Crore"
        else:
            mc_str = f"₹{mc:,.0f}"
        
        return {
            'symbol': symbol.replace('.NS', '').replace('.BO', ''),
            'company': company_name,
            'sector': sector,
            'price_inr': round(price_inr, 2),
            'price_usd': round(price_usd, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'prev_close': round(prev_close, 2),
            'volume': volume,
            'market_cap': mc_str,
            'pe_ratio': round(pe_ratio, 2) if isinstance(pe_ratio, (int, float)) else 'N/A',
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')
        }
    except Exception as e:
        print(f"Error fetching REAL-TIME {symbol}: {e}")
        return None

def fetch_historical_stock(symbol, target_date):
    """
    Fetch historical stock data for the closest trading day on or before target_date.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Define the search window
        # We search from a week before the target date up to one day after (to be safe).
        start_date = target_date - timedelta(days=7)
        end_date = target_date + timedelta(days=1)
        
        # Fetch history for the window
        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if hist.empty:
            print(f"Historical data empty for {symbol} in range {start_date} to {end_date.date()}")
            return None

        # 2. Find the closest historical data point
        # Convert index to datetime.date objects for comparison
        hist.index = hist.index.date
        
        # Filter for dates up to and including the target date
        valid_history = hist[hist.index <= target_date]
        
        if valid_history.empty:
            print(f"No trading data found on or before {target_date.strftime('%Y-%m-%d')}")
            return None

        # Get the latest data point from the valid range (i.e., the closest trading day)
        data_point = valid_history.iloc[-1]
        close_date = valid_history.index[-1]
        
        # 3. Extract and format data
        price = float(data_point['Close'])
        volume = int(data_point['Volume'])
        
        usd_to_inr = get_usd_to_inr()
        is_indian = '.NS' in symbol or '.BO' in symbol
        
        if is_indian:
            price_inr = price
            price_usd = price / usd_to_inr
        else:
            price_usd = price
            price_inr = price * usd_to_inr
        
        try:
            company_name = ticker.info.get('longName', symbol)
        except:
            company_name = symbol
        
        return {
            'symbol': symbol.replace('.NS', '').replace('.BO', ''),
            'company': company_name,
            'date': close_date.strftime('%Y-%m-%d'), # Use the actual trading date found
            'price_inr': round(price_inr, 2),
            'price_usd': round(price_usd, 2),
            'volume': volume
        }
    except Exception as e:
        print(f"Historical error for {symbol}: {e}")
        return None

def get_market_movers_fast():
    """Fetch market movers"""
    def get_yahoo_screener(category):
        api_map = {'gainers': 'day_gainers', 'losers': 'day_losers', 'active': 'most_actives'}
        url = f"https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds={api_map[category]}&count=30"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            if 'finance' in data and 'result' in data['finance']:
                quotes = data['finance']['result'][0].get('quotes', [])
                return [q['symbol'] for q in quotes[:15] if 'symbol' in q]
        except:
            pass
        return []
    
    indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                     'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'TATAMOTORS.NS', 'WIPRO.NS']
    
    us_gainers = get_yahoo_screener('gainers')
    us_losers = get_yahoo_screener('losers')
    us_active = get_yahoo_screener('active')
    all_symbols = list(set(us_gainers + us_losers + us_active + indian_stocks))
    
    def fetch_realtime(sym):
        try:
            ticker = yf.Ticker(sym)
            hist_1m = ticker.history(period='1d', interval='1m')
            hist_2d = ticker.history(period='2d', interval='1d')
            
            if hist_2d.empty or len(hist_2d) < 2:
                return None
            
            if not hist_1m.empty:
                current = float(hist_1m['Close'].iloc[-1])
                volume = int(hist_1m['Volume'].sum())
            else:
                current = float(hist_2d['Close'].iloc[-1])
                volume = int(hist_2d['Volume'].iloc[-1])
            
            prev = float(hist_2d['Close'].iloc[-2])
            if prev == 0:
                return None
            
            change = ((current - prev) / prev) * 100
            usd_to_inr = get_usd_to_inr()
            is_indian = '.NS' in sym
            
            return {
                'symbol': sym.replace('.NS', ''),
                'price_usd': round(current / usd_to_inr if is_indian else current, 2),
                'price_inr': round(current if is_indian else current * usd_to_inr, 2),
                'change': round(change, 2),
                'volume': volume
            }
        except:
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        results = list(executor.map(fetch_realtime, all_symbols))
    
    valid = [r for r in results if r]
    gainers = sorted([s for s in valid if s['change'] > 0], key=lambda x: x['change'], reverse=True)[:10]
    losers = sorted([s for s in valid if s['change'] < 0], key=lambda x: x['change'])[:10]
    active = sorted(valid, key=lambda x: x['volume'], reverse=True)[:10]
    
    return gainers, losers, active

# Response Formatting
def format_single_stock_response(data):
    """Format single stock data"""
    if not data:
        return ""
    
    emoji = "📈" if data['change_pct'] >= 0 else "📉"
    
    return f"""📊 Real-Time Stock Data: {data['symbol']} - {data['company']}

{data['symbol']} - {data['company']}
Sector: {data['sector']}

💰 CURRENT PRICE:
 • INR: ₹{data['price_inr']}
 • USD: ${data['price_usd']}

{emoji} TODAY'S PERFORMANCE:
 • Change: {'+' if data['change'] >= 0 else ''}{data['change']:.2f} ({data['change_pct']:+.2f}%)
 • Previous Close: {data['prev_close']}

📊 KEY METRICS:
 • Market Cap: {data['market_cap']}
 • P/E Ratio: {data['pe_ratio']}
 • Volume: {data['volume']:,}

⏰ Last Updated: {data['timestamp']}
✅ Real-time data from Yahoo Finance"""

def format_comparison_response(comparison_data, question):
    """Format and generate AI analysis for stock comparison."""
    comp_text = "📊 STOCK COMPARISON\n\n"
    
    for idx, stock in enumerate(comparison_data, 1):
        comp_text += f"STOCK {idx}: {stock['symbol']} - {stock['company']}\n"
        comp_text += f"Sector: {stock['sector']}\n\n"
        comp_text += f"💰 CURRENT PRICE:\n"
        comp_text += f"  • INR: ₹{stock['price_inr']}\n"
        comp_text += f"  • USD: ${stock['price_usd']}\n\n"
        emoji = "📈" if stock['change_pct'] >= 0 else "📉"
        comp_text += f"{emoji} TODAY'S PERFORMANCE:\n"
        comp_text += f"  • Change: {stock['change']:+.2f} ({stock['change_pct']:+.2f}%)\n"
        comp_text += f"  • Previous Close: {stock['prev_close']}\n\n"
        comp_text += f"📊 KEY METRICS:\n"
        comp_text += f"  • Market Cap: {stock['market_cap']}\n"
        comp_text += f"  • P/E Ratio: {stock['pe_ratio']}\n"
        comp_text += f"  • Volume: {stock['volume']:,}\n\n"
            
    comp_text += f"⏰ {datetime.now().strftime('%Y-%m-%d %I:%M %p IST')}\n"
    comp_text += f"✅ Real-time data from Yahoo Finance\n\n"
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        summary = "COMPARISON DATA:\n"
        for stock in comparison_data:
            summary += f"Stock: {stock['symbol']} - {stock['company']}, Price: ₹{stock['price_inr']}, Change: {stock['change_pct']:+.2f}%, Market Cap: {stock['market_cap']}, P/E: {stock['pe_ratio']}\n"
        
        prompt = f"""{summary}

User Question: {question}

Analyze these stocks. Provide:
1. Key differences
2. Which appears stronger
3. Brief recommendation

Keep it under 150 words."""
        
        response = model.generate_content(prompt, generation_config={'temperature': 0.7})
        comp_text += "📝 AI ANALYSIS:\n" + response.text
    except Exception as e:
        print(f"Gemini Analysis Error: {e}")
        # Gracefully handle AI failure, but return the raw data
        pass

    return comp_text

def format_market_movers_response(gainers, losers, active, q_type):
    """Format market movers"""
    usd_to_inr = get_usd_to_inr()
    response = "📊 REAL-TIME MARKET DATA:\n\n"
    
    if q_type['gainers']:
        response += "🚀 TOP 10 GAINERS:\n"
        for i, s in enumerate(gainers, 1):
            response += f"{i}. {s['symbol']}: ${s['price_usd']} (₹{s['price_inr']}) {s['change']:+.2f}%\n"
    elif q_type['losers']:
        response += "📉 TOP 10 LOSERS:\n"
        for i, s in enumerate(losers, 1):
            response += f"{i}. {s['symbol']}: ${s['price_usd']} (₹{s['price_inr']}) {s['change']:+.2f}%\n"
    elif q_type['active']:
        response += "💹 TOP 10 MOST ACTIVE:\n"
        for i, s in enumerate(active, 1):
            response += f"{i}. {s['symbol']}: ${s['price_usd']} (₹{s['price_inr']}) | Vol: {s['volume']:,}\n"
    
    response += f"\n💱 Exchange Rate: $1 = ₹{usd_to_inr:.2f}"
    response += f"\n⏰ Updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p IST')}"
    response += f"\n✅ Real-time data from Yahoo Finance"
    
    return response

# Main Route
@bp.route('/ask', methods=['POST'])
@login_required
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'}), 400
        
        global GEMINI_KEY_STATUS
        if GEMINI_KEY_STATUS != "CONFIGURED":
            return jsonify({'success': True, 'answer': f"❌ GEMINI API ERROR: {GEMINI_KEY_STATUS}"})
        
        q_type = detect_question_type(question)
        
        # --- KEY CHANGE: Extract symbol and date first for high-priority routing ---
        symbol = extract_stock_symbol(question)
        target_date = parse_date_from_question(question)
        
        print(f"Question: {question}")
        print(f"Detected Type: {q_type}")
        print(f"Extracted Symbol: {symbol}")
        print(f"Extracted Date: {target_date}")
        # --------------------------------------------------------------------------
        
        # 1. COMPARISON (Highest Priority)
        if q_type['comparison']:
            symbols = extract_all_symbols(question)
            print(f"Comparison detected. Symbols: {symbols}")
            
            if len(symbols) >= 2:
                comparison_data = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    results = executor.map(fetch_single_stock_realtime, symbols)
                    comparison_data = [r for r in results if r]
                
                if len(comparison_data) >= 2:
                    # NEW: Use the dedicated formatting function
                    comp_text = format_comparison_response(comparison_data, question)
                    return jsonify({'success': True, 'answer': comp_text})
                
                # Fallback if two stocks couldn't be fetched
                return jsonify({'success': True, 'answer': f"❌ I detected a comparison query but could not retrieve real-time data for at least two stocks among: {', '.join(symbols)}."})

        
        # 2. HISTORICAL DATA (FIXED PRIORITY: Check for symbol AND date presence)
        if symbol and target_date:
            print("Historical query execution started...")
            
            hist_data = fetch_historical_stock(symbol, target_date)
            if hist_data:
                # SUCCESS
                response = f"""📊 Historical Stock Data: {hist_data['symbol']} - {hist_data['company']}

Date: {hist_data['date']}
Closing Price (INR): ₹{hist_data['price_inr']}
Closing Price (USD): ${hist_data['price_usd']}
Volume: {hist_data['volume']:,}

✅ Historical data from Yahoo Finance"""
                return jsonify({'success': True, 'answer': response})
            else:
                # FAILURE in data fetch (e.g., non-trading day and no preceding data found)
                return jsonify({'success': True, 'answer': f"❌ No trading data found for {symbol} on or near {target_date.strftime('%Y-%m-%d')}. This date may be in the future or a non-trading day with no preceding data."})

        # 3. MARKET MOVERS
        if q_type['gainers'] or q_type['losers'] or q_type['active']:
            print("Market movers query detected")
            gainers, losers, active = get_market_movers_fast()
            return jsonify({'success': True, 'answer': format_market_movers_response(gainers, losers, active, q_type)})
        
        # 4. SINGLE STOCK PRICE (If historical failed, and a symbol is present)
        if symbol:
            print(f"Single stock price query: {symbol}")
            stock_data = fetch_single_stock_realtime(symbol)
            if stock_data:
                return jsonify({'success': True, 'answer': format_single_stock_response(stock_data)})
            
        # 5. GENERAL QUESTION (Default)
        print("General question detected")
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = f"""You are an expert stock market advisor. Answer clearly and concisely.

Question: {question}

Provide a helpful answer in under 150 words."""
            
            response = model.generate_content(prompt, generation_config={'temperature': 0.7})
            return jsonify({'success': True, 'answer': response.text})
        except Exception as e:
            print(f"Gemini error: {e}")
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                return jsonify({'success': True, 'answer': response.text})
            except:
                return jsonify({'success': True, 'answer': f"❌ AI Error: {str(e)}"})
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'success': True, 'answer': f"❌ Server error: {str(e)}"})

@bp.route('/suggestions')
@login_required
def get_suggestions():
    suggestions = [
        "💰 What is TCS stock price?",
        "📊 Compare TCS vs Infosys",
        "🚀 Show top 10 gainers today",
        "📉 What are top 10 losers?",
        "💹 Most actively traded stocks",
        "🔍 Apple price yesterday",
        "💡 What is P/E ratio?",
        "⚡ Compare Apple and Microsoft"
    ]
    return jsonify({'success': True, 'suggestions': suggestions})