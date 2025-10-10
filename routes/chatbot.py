import os
import time
import re
import traceback
import requests
import yfinance as yf
import google.generativeai as genai
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, session
from routes.auth import login_required
from config import Config
from dateutil.parser import parse
from datetime import datetime, timedelta 
import re
# Need to import APIError for resilient try/except blocks
try:
    from google.generativeai.errors import APIError as GeminiAPIError
except ImportError:
    # Fallback for different library versions
    class GeminiAPIError(Exception): pass

bp = Blueprint('chatbot', __name__, url_prefix='/chatbot')

# --- GLOBAL KEY CHECK AND CONFIGURATION ---
GEMINI_KEY_STATUS = "PENDING"
MIN_KEY_LENGTH = 35 

if Config.GEMINI_API_KEY:
    if len(Config.GEMINI_API_KEY) < MIN_KEY_LENGTH or not Config.GEMINI_API_KEY.startswith('AIzaSy'):
        GEMINI_KEY_STATUS = "ERROR: Key is too short or malformed. Please check .env file and restart."
        print(f"CRITICAL ERROR: {GEMINI_KEY_STATUS}")
    else:
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            print(f"SUCCESS: Gemini client configured. Key length: {len(Config.GEMINI_API_KEY)}.")
            GEMINI_KEY_STATUS = "CONFIGURED"
        except Exception as e:
            GEMINI_KEY_STATUS = f"CRITICAL ERROR: Failed to configure genai during startup: {e}"
            print(f"CRITICAL ERROR: {GEMINI_KEY_STATUS}")
            
@bp.route('/')
@login_required
def chatbot_page():
    """Chatbot page"""
    return render_template('chatbot.html')

def get_usd_to_inr():
    """Get current USD to INR conversion rate"""
    try:
        for i in range(3):
            try:
                response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
                response.raise_for_status()
                data = response.json()
                return data['rates']['INR']
            except (requests.exceptions.RequestException, KeyError):
                if i < 2:
                    time.sleep(2 ** i) 
        return 83.0 # Default fallback
    except Exception as e:
        print(f"Currency API error: {e}")
        return 83.0

def search_ticker_by_name(company_name):
    """Searches Yahoo Finance for the best matching ticker symbol."""
    search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}&quotes_count=5&news_count=0&lists_count=0"
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i in range(3):
        try:
            response = requests.get(search_url, timeout=5, headers=headers)
            response.raise_for_status() 
            data = response.json()
            
            if 'quotes' in data and data['quotes']:
                for quote in data['quotes']:
                    symbol = quote.get('symbol')
                    if not symbol or quote.get('exchDisp', '').upper() in ['INDEXT', 'FUND']:
                        continue
                        
                    try:
                        test_ticker = yf.Ticker(symbol)
                        hist = test_ticker.history(period='1d')
                        if not hist.empty:
                            print(f"✓ Ticker found and validated via search API: {symbol}")
                            return symbol
                    except Exception as e:
                        print(f"  Attempted symbol {symbol} failed validation.")
                        continue
                break 
        except requests.exceptions.RequestException as e:
            print(f"Yahoo Search API error on attempt {i+1}: {e}")
            if i < 2:
                time.sleep(2 ** i) 
                continue
        except Exception as e:
            print(f"Error processing Yahoo Search data: {e}")
            break
    return None

def extract_stock_symbol(question):
    """Extract stock symbol from question."""
    q_upper = question.upper()
    q_lower = question.lower()
    
    stock_mappings = {
        'TCS': ['tcs', 'tata consultancy'],
        'INFY': ['infosys'],
        'RELIANCE': ['reliance', 'ril', 'reliance industries'],
        'HDFCBANK': ['hdfc bank', 'hdfc'],
        'ICICIBANK': ['icici bank', 'icici'],
        'SBIN': ['sbi', 'state bank of india'],
        'IDFCFIRSTB': ['idfc', 'idfc limited', 'idfc first bank'],
        'ITC': ['itc'],
        'WIPRO': ['wipro'],
        'BHARTIARTL': ['airtel', 'bharti airtel'],
        'TATAMOTORS': ['tata motors'],
        'AAPL': ['apple'],
        'MSFT': ['microsoft'],
        'GOOGL': ['google', 'alphabet'],
        'TSLA': ['tesla']
    }
    
    for symbol, names in stock_mappings.items():
        for name in names:
            if name in q_lower:
                if symbol in ['TCS', 'INFY', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 
                              'WIPRO', 'BHARTIARTL', 'ITC', 'SBIN', 'HINDUNILVR',
                              'TATAMOTORS', 'ADANIENT', 'AXISBANK', 'BAJFINANCE', 'MARUTI', 'IDFCFIRSTB']:
                    return f"{symbol}.NS"
                return symbol
    
    symbols = re.findall(r'\b[A-Z]{2,10}(?:\.[A-Z]{2})?\b', q_upper)
    if symbols:
        symbol = symbols[0]
        if any(word in q_lower for word in ['indian', 'nse', 'bse', 'india', 'rupee', 'inr']):
            if '.' not in symbol:
                return f"{symbol}.NS"
        return symbol
    
    company_name_match = re.search(r'(?:about|for|of|price of|share price|current price of|stock price for)\s+([a-zA-Z\s]+?)(?:\s+stock|\s+share|\s+ltd|$|\?)', q_lower)
    if company_name_match:
        company_name = company_name_match.group(1).strip()
        print(f"Attempting robust API search for company name: {company_name}")
        found_symbol = search_ticker_by_name(company_name)
        if found_symbol:
            return found_symbol 
    
    return None

from datetime import datetime, timedelta

def get_any_stock_realtime(symbol, target_date=None):
    """Fetch data for a stock, either real-time or historical."""
    try:
        print(f"\n{'='*70}")
        
        ticker = yf.Ticker(symbol)
        
        if target_date:
            
            print(f"FETCHING HISTORICAL DATA FOR: {symbol} on {target_date}")
            start_date = target_date
            end_date = target_date + timedelta(days=1) 
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                print(f"❌ No historical data found for {symbol} on {target_date}")
                return None

            current_price = float(hist['Close'].iloc[0]) 
            volume = int(hist['Volume'].iloc[0])
            
            info = {}
            company_name = ticker.info.get('longName', symbol)
            sector = ticker.info.get('sector', 'N/A')
            prev_close = current_price
            change = 0.0 
            change_pct = 'N/A'

        else:
            
            print(f"FETCHING REAL-TIME DATA FOR: {symbol}")
            
            hist = ticker.history(period='1d', interval='5m') 
            
            if hist.empty or len(hist) < 2:
                hist = ticker.history(period='5d')
            
            if hist.empty:
                print(f"❌ No data found for {symbol}")
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])
            
            info = ticker.info
            company_name = info.get('longName', symbol)
            sector = info.get('sector', 'N/A')
            
            prev_close = info.get('previousClose', current_price) 
            
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0
        
        # --- Common logic for both paths ---
        
        currency = info.get('currency', 'USD')
        is_inr = currency == 'INR' or symbol.endswith('.NS') or symbol.endswith('.BO')
        
        usd_to_inr = get_usd_to_inr()
        
        if is_inr:
            price_inr = current_price
            price_usd = current_price / usd_to_inr
        else:
            price_usd = current_price
            price_inr = current_price * usd_to_inr
        
        market_cap = info.get('marketCap', 0)
        
        mc = market_cap if is_inr else market_cap * usd_to_inr
        
        if mc >= 1_00_000_00_00_000:
            mc_str = f"₹{mc/1_00_000_00_00_000:,.2f} Lakh Crore"
        elif mc >= 1_00_00_00_000:
            mc_str = f"₹{mc/1_00_00_00_000:,.2f} Thousand Crore"
        elif mc >= 1_00_00_000:
            mc_str = f"₹{mc/1_00_00_000:,.2f} Crore"
        else:
            mc_str = f"₹{mc:,.0f}"

        # --- Final Return ---
        timestamp = str(target_date) if target_date else datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')
        
        return {
            'symbol': symbol,
            'company': company_name,
            'sector': sector,
            'price_inr': round(price_inr, 2),
            'price_usd': round(price_usd, 2),
            'currency': currency,
            'change': round(change, 2) if not target_date else 'N/A',
            'change_pct': round(change_pct, 2) if not target_date else 'N/A',
            'prev_close': round(prev_close, 2) if not target_date else 'N/A',
            'volume': volume,
            'market_cap': mc_str if not target_date else 'N/A',
            'pe_ratio': info.get('trailingPE', 'N/A') if not target_date else 'N/A',
            'week_52_high': round(info.get('fiftyTwoWeekHigh', 0), 2) if not target_date else 'N/A',
            'week_52_low': round(info.get('fiftyTwoWeekLow', 0), 2) if not target_date else 'N/A',
            'exchange_rate': round(usd_to_inr, 2),
            'timestamp': timestamp
        }
        
    except Exception as e:
        print(f"❌ ERROR fetching stock data for {symbol}: {e}")
        traceback.print_exc()
        return None

def clean_text_for_encoding(text):
    """
    Clean text to remove problematic characters that cause UTF-8 encoding errors.
    This fixes the 'surrogates not allowed' error when sending to Gemini API.
    """
    if not text:
        return text
    
    # Remove surrogate characters and other problematic Unicode
    # Replace currency symbols that might cause issues
    text = text.replace('₹', 'INR ')
    text = text.replace('$', 'USD ')
    
    # Encode to ASCII, ignore errors, then decode back
    # This removes any non-ASCII characters that might cause issues
    try:
        # Try to encode as UTF-8 first, replacing errors
        cleaned = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        # Further sanitize by removing any remaining surrogate pairs
        cleaned = ''.join(char for char in cleaned if ord(char) < 0x110000 and not (0xD800 <= ord(char) <= 0xDFFF))
        return cleaned
    except Exception as e:
        print(f"Warning: Error cleaning text: {e}")
        # Fallback: return ASCII-only version
        return text.encode('ascii', errors='ignore').decode('ascii')

def format_stock_data_text(data):
    """Format single stock data into text - Fixed to use actual stock symbol"""
    if not data:
        return ""
    
    # FIX: Use actual stock symbol instead of hardcoded "INFY"
    change_emoji = "📈" if isinstance(data['change_pct'], (int, float)) and data['change_pct'] >= 0 else "📉"
    display_symbol = data['symbol'].replace('.NS', '').replace('.BO', '')

    # Format change values properly
    change_str = f"+{data['change']:.2f}" if isinstance(data['change'], (int, float)) and data['change'] >= 0 else str(data['change'])
    change_pct_str = f"{data['change_pct']:+.2f}%" if isinstance(data['change_pct'], (int, float)) else str(data['change_pct'])
    
    # Build the response text
    text = (
        f"{display_symbol} - {data['company']} "
        f"Sector: {data['sector']} "
        f"💰 CURRENT PRICE: • INR: Rs.{data['price_inr']} (Primary) • USD: ${data['price_usd']} "
        f"{change_emoji} TODAY'S PERFORMANCE: • Change: {change_str} ({change_pct_str}) "
        f"• Previous Close: {data['prev_close']} "
        f"📊 KEY METRICS: • Market Cap: {data['market_cap']} "
        f"• P/E Ratio: {data['pe_ratio']} "
        f"• 52-Week High: {data['week_52_high']} "
        f"• 52-Week Low: {data['week_52_low']} "
        f"• Volume: {data['volume']:,} "
        f"💱 Exchange Rate: $1 = Rs.{data['exchange_rate']} "
        f"⏰ Last Updated: {data['timestamp']} "
        f"✅ This is REAL-TIME data from Yahoo Finance."
    )
    
    # Adding the specific leading phrase
    final_output = f"📊 Real-Time Stock Data Found: 🟢 {display_symbol} - {data['company']}\n\n{text}"

    return final_output

def get_market_movers():
    """Get top gainers/losers/active by sampling a larger, combined list."""
    
    indian_symbols = [
        'RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'ICICIBANK.NS', 'INFY.NS', 'LICI.NS', 
        'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'LT.NS', 'KOTAKBANK.NS', 
        'HCLTECH.NS', 'AXISBANK.NS', 'SUNPHARMA.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS', 
        'TITAN.NS', 'WIPRO.NS', 'MARUTI.NS', 'POWERGRID.NS', 'ADANIENT.NS', 
        'ULTRACEMCO.NS', 'NTPC.NS', 'GRASIM.NS', 'PIDILITIND.NS', 'PNB.NS', 'GAIL.NS', 
        'INDUSINDBK.NS', 'DLF.NS', 'CANBK.NS', 'BOSCHLTD.NS', 'BEL.NS', 'SAIL.NS',
        'ZOMATO.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'TATASTEEL.NS', 'M&M.NS', 'NESTLEIND.NS',
        'BAJAJFINSV.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS', 'DRREDDY.NS', 'HINDALCO.NS',
        'JSWSTEEL.NS', 'EICHERMOT.NS', 'UPL.NS', 'APOLLOHOSP.NS', 'CIPLA.NS', 'TECHM.NS'
    ]
    
    us_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 
        'WMT', 'UNH', 'HD', 'DIS', 'BAC', 'NFLX', 'ADBE', 'CRM', 'PYPL', 
        'INTC', 'AMD', 'CSCO', 'ORCL', 'IBM', 'BA', 'NKE', 'MCD', 'PFE', 'KO', 'PEP'
    ]
    
    symbols_to_check = indian_symbols + us_symbols 
    
    gainers, losers, most_active = [], [], []
    usd_to_inr = get_usd_to_inr()

    try:
        data = yf.download(symbols_to_check, period='2d', interval='1d', progress=False)
    except Exception as e:
        print(f"Error in batch yfinance download: {e}")
        data = {} 

    
    for symbol in symbols_to_check:
        try:
            current, previous, volume = None, None, None
            
            if not data.empty and symbol in data['Close'].columns and len(data['Close'][symbol].dropna()) >= 2:
                current = float(data['Close'][symbol].iloc[-1])
                previous = float(data['Close'][symbol].iloc[-2])
                volume = int(data['Volume'][symbol].iloc[-1])
            else:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='2d')
                if len(hist) < 2: continue
                current = float(hist['Close'].iloc[-1])
                previous = float(hist['Close'].iloc[-2])
                volume = int(hist['Volume'].iloc[-1])
                
            change_pct = ((current - previous) / previous) * 100
            
            is_inr_stock = symbol.endswith('.NS') or symbol.endswith('.BO')
            
            price_usd = round(current / usd_to_inr, 2) if is_inr_stock else round(current, 2)
            price_inr = round(current, 2) if is_inr_stock else round(current * usd_to_inr, 2)
            
            stock_data = {
                'symbol': symbol.replace('.NS', '').replace('.BO', ''), 
                'price_usd': price_usd,
                'price_inr': price_inr,
                'change': round(change_pct, 2),
                'volume': volume
            }
            
            if change_pct > 0:
                gainers.append(stock_data)
            elif change_pct < 0:
                losers.append(stock_data)
            
            most_active.append(stock_data)
        except Exception as e:
            continue

    return (sorted(gainers, key=lambda x: x['change'], reverse=True)[:10],
            sorted(losers, key=lambda x: x['change'])[:10],
            sorted(most_active, key=lambda x: x['volume'], reverse=True)[:10])

def get_sector_performance():
    """Get sector data based on US sector ETFs (e.g., XLK for Tech)"""
    sectors = {
        'XLK': 'Technology', 'XLF': 'Financial', 'XLV': 'Healthcare',
        'XLE': 'Energy', 'XLI': 'Industrial', 'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples', 'XLB': 'Materials', 
        'XLU': 'Utilities', 'XLRE': 'Real Estate'
    }
    
    sector_data = []
    
    for symbol, name in sectors.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if len(hist) >= 2:
                current = float(hist['Close'].iloc[-1])
                week_ago = float(hist['Close'].iloc[0]) 
                change_pct = ((current - week_ago) / week_ago) * 100
                
                sector_data.append({
                    'sector': name,
                    'change': round(change_pct, 2)
                })
        except:
            continue
    
    return sorted(sector_data, key=lambda x: x['change'], reverse=True)

def detect_question_type(question):
    """Detect question type"""
    q = question.lower()
    return {
        'gainers': any(w in q for w in ['gainer', 'best', 'rising', 'top performer']),
        'losers': any(w in q for w in ['loser', 'worst', 'falling', 'bottom performer']),
        'active': any(w in q for w in ['active', 'volume', 'traded', 'most bought']),
        'sector': any(w in q for w in ['sector', 'industry', 'which sector']),
        'overview': any(w in q for w in ['market', 'overview', 'summary', 'today in stock'])
    }

def is_simple_price_query(question):
    """Determines if the question is ONLY asking for the price of a single stock."""
    q = question.lower()
    price_keywords = ['price', 'current', 'share value', 'stock value', 'quote']
    
    stock_symbol = extract_stock_symbol(question)
    
    if stock_symbol:
        complex_keywords = ['why', 'news', 'outlook', 'analysis', 'buy', 'sell', 'forecast', 'should i']
        
        is_complex = any(word in q for word in complex_keywords)
        is_price_focus = any(word in q for word in price_keywords)
        
        if is_price_focus and not is_complex:
            return True
            
    return False

def is_market_data_query(question):
    """Determines if the question is asking for market data (gainers/losers/active/sector/overview)"""
    q_type = detect_question_type(question)
    return any([q_type['gainers'], q_type['losers'], q_type['active'], q_type['sector'], q_type['overview']])

def format_market_data(question):
    """
    Attempts to fetch specific stock data or general market data based on the question.
    """
    stock_symbol = extract_stock_symbol(question)
    
    # 1. Specific Stock Data Lookup
    if stock_symbol:
        stock_data = get_any_stock_realtime(stock_symbol)
        if stock_data:
            return format_stock_data_text(stock_data)
    
    # 2. General Market Data Lookup (Gainers/Losers/Sectors)
    q_type = detect_question_type(question)
    
    if any([q_type['gainers'], q_type['losers'], q_type['active'], q_type['sector'], q_type['overview']]):
        
        gainers, losers, most_active = ([], [], [])
        if any([q_type['gainers'], q_type['losers'], q_type['active'], q_type['overview']]):
             gainers, losers, most_active = get_market_movers()
             
        sectors = []
        if q_type['sector'] or q_type['overview']:
             sectors = get_sector_performance()
             
        usd_to_inr = get_usd_to_inr()
        
        response = "\n📊 MARKET DATA (Sampled from Major Indices):\n\n"
        
        if q_type['gainers'] or q_type['overview']:
            response += "🚀 TOP 10 GAINERS (from sample):\n"
            for i, s in enumerate(gainers, 1):
                response += f"{i}. {s['symbol']}: ${s['price_usd']} (Rs.{s['price_inr']}) +{s['change']}%\n"
            response += "\n"
        
        if q_type['losers'] or q_type['overview']:
            response += "📉 TOP 10 LOSERS (from sample):\n"
            for i, s in enumerate(losers, 1):
                response += f"{i}. {s['symbol']}: ${s['price_usd']} (Rs.{s['price_inr']}) {s['change']}%\n"
            response += "\n"
        
        if q_type['active'] or q_type['overview']:
            response += "💹 TOP 10 MOST ACTIVE (Volume):\n"
            for i, s in enumerate(most_active, 1):
                response += f"{i}. {s['symbol']}: ${s['price_usd']} (Rs.{s['price_inr']}) | Vol: {s['volume']:,}\n"
            response += "\n"
        
        if q_type['sector'] or q_type['overview']:
            response += "🏢 SECTOR PERFORMANCE (US ETFs, Last 5 Days):\n"
            for sec in sectors:
                emoji = "📈" if sec['change'] > 0 else "📉"
                response += f"{emoji} {sec['sector']}: {sec['change']:+.2f}%\n"
            response += "\n"
        
        response += f"💱 Exchange Rate: $1 = Rs.{usd_to_inr:.2f}\n"
        response += f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p IST')}\n"
        
        return response
    
    return ""

@bp.route('/ask', methods=['POST'])
@login_required
def ask_question():
    """Handle chatbot questions"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'}), 400
        
        market_context = "" 
        
        # --- CRITICAL: Check API Key Status ---
        global GEMINI_KEY_STATUS
        if GEMINI_KEY_STATUS != "CONFIGURED":
            return jsonify({
                'success': True,
                'answer': f"❌ **GEMINI API KEY ERROR:** Could not configure AI client.\n\n**Reason:** {GEMINI_KEY_STATUS}\n\nPlease check your `.env` file and restart the Flask server."
            })
        
        # 1. Fetch real-time data or market context
        market_context = format_market_data(question)
        
        # 2. Check for SIMPLE Price Query OR Market Data Query and return raw data directly
        if market_context and (is_simple_price_query(question) or is_market_data_query(question)):
            print("INFO: Returning raw detailed output (Bypassing AI for simple/market query).")
            return jsonify({'success': True, 'answer': market_context})

        answer = None
        gemini_error_message = ""
        
        # 3. Structure the prompt for complex or general queries
        contents_list = []
        
        if market_context:
            # FIX: Clean the market context before sending to Gemini
            cleaned_market_context = clean_text_for_encoding(market_context)
            
            full_prompt_text = f"""
            You are a highly specialized Stock Market Analyst AI.
            Your response must **ALWAYS** be derived from the data in the 'DATA SOURCE' block.
            **NEVER** under any circumstance use the phrase "I do not have access to real-time market data" or any variation of it. The data provided **IS** the source of truth.

            **DATA SOURCE:**
            {cleaned_market_context}

            **USER QUESTION:** {question}

            **INSTRUCTIONS:**
            1. Use the data in the 'DATA SOURCE' block to answer the user's question directly.
            2. If the question is general but refers to the stocks in the list (e.g., "why is SAIL a loser?"), use your general knowledge *in addition* to the price data to give context.
            3. Keep the response professional, concise, and under 250 words.

            Answer now:
            """
            contents_list.append(full_prompt_text)

        else:
            full_prompt_text = f"""
            You are a highly knowledgeable and comprehensive Stock Market AI assistant. 
            You answer general questions about finance, investment concepts, company history, definitions, and market trends based on your intrinsic knowledge. 
            Do not provide any specific stock prices or real-time lists. Keep the response concise, accurate, and educational, and under 250 words.

            Question: {question}
            Provide a clear, detailed, and accurate answer to this question.
            """
            contents_list.append(full_prompt_text)
        
        # 4. Execute Gemini Call
        try:
            models_to_try = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro']
            
            for model_name in models_to_try:
                try:
                    print(f"Trying Gemini model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    
                    for i in range(3):
                        try:
                            response = model.generate_content(
                                contents=contents_list, 
                                generation_config={"temperature": 0.5}
                            )
                            answer = response.text
                            print(f"✓ Success with {model_name} on attempt {i+1}")
                            break 
                        except GeminiAPIError as api_error:
                            gemini_error_message = str(api_error)
                            print(f"✗ {model_name} failed on attempt {i+1} due to API Error: {gemini_error_message}")
                            if i < 2:
                                time.sleep(2 ** i)
                                continue
                            raise 
                        except Exception as model_error:
                            gemini_error_message = str(model_error)
                            raise model_error
                    
                    if answer:
                        break 
                except Exception as model_error:
                    print(f"✗ Model {model_name} failed entirely: {str(model_error)}")
                    continue
            
            if not answer:
                raise Exception(f"All Gemini models failed. Last API Error: {gemini_error_message}")
                
        except Exception as e:
            # Final failure reporting
            print(f"AI Error: {e}")
            final_error = str(e)
            
            if market_context:
                # FIX: Return ONLY the cleaned market data without the error warning
                # This ensures users see the data without the "AI Processing Failed" message
                answer = market_context
            else:
                # If no stock data was found AND AI failed (general question failed).
                answer = f"❌ **AI SERVICE ERROR.** The question could not be answered. \n\n**Raw Error:** {final_error}"
        
        return jsonify({'success': True, 'answer': answer})
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'success': True, 'answer': f"❌ Unexpected server error: {str(e)}"})

@bp.route('/suggestions')
@login_required
def get_suggestions():
    """Get suggestions"""
    suggestions = [
        "💰 What is TCS stock price?",
        "📊 Show me Apple stock price in rupees",
        "🚀 Show top 10 gainers today",
        "📉 What are top 10 losers?",
        "💹 Most actively traded stocks",
        "🏢 Show sector performance",
        "📈 Infosys stock price in INR",
        "🔍 Reliance Industries current price",
        "💡 Market overview today",
        "⚡ Tesla stock price"
    ]
    
    return jsonify({'success': True, 'suggestions': suggestions})