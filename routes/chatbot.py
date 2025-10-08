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

# Need to import APIError for resilient try/except blocks
try:
    from google.generativeai.errors import APIError as GeminiAPIError
except ImportError:
    # Fallback for different library versions
    class GeminiAPIError(Exception): pass


bp = Blueprint('chatbot', __name__, url_prefix='/chatbot')

# Configure Gemini API
if Config.GEMINI_API_KEY:
    try:
        # Initialize the AI client using the key from config.py
        genai.configure(api_key=Config.GEMINI_API_KEY)
        print("SUCCESS: Gemini client configured using Config.GEMINI_API_KEY.")
    except Exception as e:
        # This catches errors during the initial configuration (e.g., malformed key)
        print(f"CRITICAL ERROR: Failed to configure genai using Config.GEMINI_API_KEY: {e}")
        
@bp.route('/')
@login_required
def chatbot_page():
    """Chatbot page"""
    return render_template('chatbot.html')

def get_usd_to_inr():
    """Get current USD to INR conversion rate"""
    try:
        # Use exponential backoff for external API
        for i in range(3):
            try:
                response = requests.get('https://api.exchangerate-api.com/v4/latest/USD', timeout=5)
                response.raise_for_status()
                data = response.json()
                return data['rates']['INR']
            except (requests.exceptions.RequestException, KeyError):
                if i < 2:
                    time.sleep(2 ** i) # Exponential backoff
        return 83.0 # Default fallback
    except Exception as e:
        print(f"Currency API error: {e}")
        return 83.0

def search_ticker_by_name(company_name):
    """
    Searches Yahoo Finance for the best matching ticker symbol.
    This function checks multiple search results and validates the ticker 
    by fetching a small amount of history to ensure it's a valid, fetchable stock.
    """
    search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={company_name}&quotes_count=5&news_count=0&lists_count=0"
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i in range(3):
        try:
            response = requests.get(search_url, timeout=5, headers=headers)
            response.raise_for_status() 
            data = response.json()
            
            if 'quotes' in data and data['quotes']:
                
                # Check top 5 results to ensure a valid ticker is found
                for quote in data['quotes']:
                    symbol = quote.get('symbol')
                    if not symbol or quote.get('exchDisp', '').upper() in ['INDEXT', 'FUND']:
                        continue
                        
                    try:
                        # Validation: Check if yfinance can actually fetch data for this symbol
                        test_ticker = yf.Ticker(symbol)
                        hist = test_ticker.history(period='1d')
                        if not hist.empty:
                            print(f"✓ Ticker found and validated via search API: {symbol}")
                            return symbol
                    except Exception as e:
                        # Validation failed, try next result
                        print(f"  Attempted symbol {symbol} failed validation.")
                        continue
                
                break # If we checked all 5 and none worked, break retry loop

        except requests.exceptions.RequestException as e:
            print(f"Yahoo Search API error on attempt {i+1}: {e}")
            if i < 2:
                time.sleep(2 ** i) # Exponential backoff
                continue
        except Exception as e:
            print(f"Error processing Yahoo Search data: {e}")
            break

    return None

def extract_stock_symbol(question):
    """
    Extract stock symbol from question.
    1. Check hardcoded map (fastest for popular stocks).
    2. Check for explicit ticker pattern (e.g., 'AAPL', 'RELIANCE.NS').
    3. Use Yahoo Search API for generic company names (UNIVERSAL LOOKUP).
    """
    q_upper = question.upper()
    q_lower = question.lower()
    
    # 1. Hardcoded stock mappings (for common names)
    stock_mappings = {
        'TCS': ['tcs', 'tata consultancy'],
        'INFY': ['infosys'],
        'RELIANCE': ['reliance', 'ril'],
        # ... (rest of the Indian hardcoded stocks)
        'IDFCFIRSTB': ['idfc', 'idfc limited', 'idfc first bank'], # Mapped to current ticker
        # ... (US hardcoded stocks)
    }
    
    for symbol, names in stock_mappings.items():
        for name in names:
            if name in q_lower:
                if symbol in ['TCS', 'INFY', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 
                              'WIPRO', 'BHARTIARTL', 'ITC', 'SBIN', 'HINDUNILVR',
                              'TATAMOTORS', 'ADANIENT', 'AXISBANK', 'BAJFINANCE', 'MARUTI', 'IDFCFIRSTB']:
                    return f"{symbol}.NS"
                return symbol
    
    # 2. Look for explicit symbols: ABC or ABC.XX format
    symbols = re.findall(r'\b[A-Z]{2,10}(?:\.[A-Z]{2})?\b', q_upper)
    if symbols:
        symbol = symbols[0]
        if any(word in q_lower for word in ['indian', 'nse', 'bse', 'india', 'rupee', 'inr']):
            if '.' not in symbol:
                return f"{symbol}.NS"
        return symbol
    
    # 3. Fallback: Use Yahoo Search API for general company name lookup (ANY STOCK)
    company_name_match = re.search(r'(?:about|for|of|price of)\s+([a-zA-Z\s]+?)(?:\s+stock|\s+share|\s+ltd|$|\?)', q_lower)
    if company_name_match:
        company_name = company_name_match.group(1).strip()
        print(f"Attempting robust API search for company name: {company_name}")
        
        found_symbol = search_ticker_by_name(company_name)
        
        if found_symbol:
            return found_symbol 
    
    return None

def get_any_stock_realtime(symbol):
    """Fetch real-time data for ANY stock"""
    try:
        # ... (unchanged stock fetching logic)
        print(f"\n{'='*70}")
        print(f"FETCHING REAL-TIME DATA FOR: {symbol}")
        print(f"{'='*70}")
        
        ticker = yf.Ticker(symbol)
        
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
        
        currency = info.get('currency', 'USD')
        is_inr = currency == 'INR'
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
        
        print(f"✅ {company_name}: {currency}{current_price:.2f} ({change_pct:+.2f}%)")
        print(f"{'='*70}\n")
        
        return {
            'symbol': symbol,
            'company': company_name,
            'sector': sector,
            'price_inr': round(price_inr, 2),
            'price_usd': round(price_usd, 2),
            'currency': currency,
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'prev_close': round(prev_close, 2),
            'volume': volume,
            'market_cap': mc_str,
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'week_52_high': round(info.get('fiftyTwoWeekHigh', 0), 2),
            'week_52_low': round(info.get('fiftyTwoWeekLow', 0), 2),
            'exchange_rate': round(usd_to_inr, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M:%S %p IST')
        }
    except Exception as e:
        print(f"❌ ERROR fetching stock data for {symbol}: {e}")
        traceback.print_exc()
        return None

def format_stock_data_text(data):
    """Format stock data into text"""
    # ... (unchanged formatting logic)
    if not data:
        return ""
    
    change_emoji = "📈" if data['change'] >= 0 else "📉"
    
    text = f"""
🟢 REAL-TIME STOCK DATA

📊 {data['symbol'].replace('.NS', '').replace('.BO', '')} - {data['company']}
Sector: {data['sector']}

💰 CURRENT PRICE:
"""
    
    if data['currency'] == 'INR':
        text += f"• INR: ₹{data['price_inr']} (Primary)\n• USD: ${data['price_usd']}\n"
    else:
        text += f"• USD: ${data['price_usd']} (Primary)\n• INR: ₹{data['price_inr']}\n"
    
    text += f"""
{change_emoji} TODAY'S PERFORMANCE:
• Change: {'+' if data['change'] >= 0 else ''}{data['change']} ({data['change_pct']:+.2f}%)
• Previous Close: {data['prev_close']}

📊 KEY METRICS:
• Market Cap: {data['market_cap']}
• P/E Ratio: {data['pe_ratio']}
• 52-Week High: {data['week_52_high']}
• 52-Week Low: {data['week_52_low']}
• Volume: {data['volume']:,}

💱 Exchange Rate: $1 = ₹{data['exchange_rate']}
⏰ Last Updated: {data['timestamp']}

✅ This is REAL-TIME data from Yahoo Finance.
"""
    return text

# ... (get_market_movers, get_sector_performance, detect_question_type, format_market_data are unchanged) ...

def get_market_movers():
    """Get top gainers/losers/active"""
    symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 
        'JPM', 'V', 'WMT', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
        'NFLX', 'ADBE', 'CRM', 'PYPL', 'INTC', 'AMD', 'CSCO',
        'ORCL', 'IBM', 'BA', 'NKE', 'MCD', 'PFE', 'KO', 'PEP'
    ]
    
    gainers, losers, most_active = [], [], []
    usd_to_inr = get_usd_to_inr()
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='2d')
            
            if len(hist) >= 2:
                current = float(hist['Close'].iloc[-1])
                previous = float(hist['Close'].iloc[-2])
                change_pct = ((current - previous) / previous) * 100
                volume = int(hist['Volume'].iloc[-1])
                
                stock_data = {
                    'symbol': symbol,
                    'price_usd': round(current, 2),
                    'price_inr': round(current * usd_to_inr, 2),
                    'change': round(change_pct, 2),
                    'volume': volume
                }
                
                if change_pct > 0:
                    gainers.append(stock_data)
                elif change_pct < 0:
                    losers.append(stock_data)
                
                most_active.append(stock_data)
        except:
            continue
    
    return (sorted(gainers, key=lambda x: x['change'], reverse=True)[:10],
            sorted(losers, key=lambda x: x['change'])[:10],
            sorted(most_active, key=lambda x: x['volume'], reverse=True)[:10])

def get_sector_performance():
    """Get sector data"""
    sectors = {
        'XLK': 'Technology', 'XLF': 'Financial', 'XLV': 'Healthcare',
        'XLE': 'Energy', 'XLI': 'Industrial', 'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples', 'XLB': 'Materials', 
        'XLRE': 'Real Estate', 'XLU': 'Utilities'
    }
    
    sector_data = []
    usd_to_inr = get_usd_to_inr()
    
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
        'gainers': any(w in q for w in ['gainer', 'best', 'rising']),
        'losers': any(w in q for w in ['loser', 'worst', 'falling']),
        'active': any(w in q for w in ['active', 'volume', 'traded']),
        'sector': any(w in q for w in ['sector', 'industry']),
        'overview': any(w in q for w in ['market', 'overview', 'summary'])
    }

def format_market_data(question):
    """Format market data"""
    stock_symbol = extract_stock_symbol(question)
    
    if stock_symbol:
        stock_data = get_any_stock_realtime(stock_symbol)
        if stock_data:
            return format_stock_data_text(stock_data)
    
    q_type = detect_question_type(question)
    
    if any([q_type['gainers'], q_type['losers'], q_type['active'], q_type['overview']]):
        gainers, losers, most_active = get_market_movers()
        sectors = get_sector_performance()
        usd_to_inr = get_usd_to_inr()
        
        response = "\n📊 REAL-TIME MARKET DATA:\n\n"
        
        if q_type['gainers'] or q_type['overview']:
            response += "🚀 TOP 10 GAINERS:\n"
            for i, s in enumerate(gainers, 1):
                response += f"{i}. {s['symbol']}: ${s['price_usd']} (₹{s['price_inr']}) +{s['change']}%\n"
            response += "\n"
        
        if q_type['losers'] or q_type['overview']:
            response += "📉 TOP 10 LOSERS:\n"
            for i, s in enumerate(losers, 1):
                response += f"{i}. {s['symbol']}: ${s['price_usd']} (₹{s['price_inr']}) {s['change']}%\n"
            response += "\n"
        
        if q_type['active'] or q_type['overview']:
            response += "💹 TOP 10 MOST ACTIVE:\n"
            for i, s in enumerate(most_active, 1):
                response += f"{i}. {s['symbol']}: ${s['price_usd']} (₹{s['price_inr']}) | Vol: {s['volume']:,}\n"
            response += "\n"
        
        if q_type['sector'] or q_type['overview']:
            response += "🏢 SECTOR PERFORMANCE (5-Day):\n"
            for sec in sectors:
                emoji = "📈" if sec['change'] > 0 else "📉"
                response += f"{emoji} {sec['sector']}: {sec['change']:+.2f}%\n"
            response += "\n"
        
        response += f"💱 Exchange Rate: $1 = ₹{usd_to_inr:.2f}\n"
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
        
        # Check API Key configuration first
        if not Config.GEMINI_API_KEY:
            return jsonify({
                'success': True,
                'answer': "⚠️ Gemini API not configured. Please ensure GEMINI_API_KEY is set in your environment."
            })
        
        # 1. Fetch real-time data - This is the universal stock lookup!
        market_context = format_market_data(question)
        
        # Define the system instruction for the AI, which sets its persona and rules.
        system_prompt = """
        You are a highly specialized Stock Market Analyst AI.
        Your absolute and primary instruction is to **ALWAYS** use the data provided in the 'REAL-TIME DATA' block.
        **NEVER** decline to provide a price, and **NEVER** state that you cannot access real-time data or refer the user to external sites.
        Your response must be based *only* on the injected data.
        """
        
        if market_context:
            prompt = f"""
            User Question: {question}

            REAL-TIME DATA (USE THIS AS THE ONLY SOURCE OF TRUTH):
            {market_context}

            INSTRUCTIONS:
            1. Use the provided stock symbol, company name, and prices (both USD and INR) directly in your response.
            2. Keep the response concise, informative, and under 200 words.
            3. If the question is about a specific stock, provide its current price and change percentage.

            Answer now:
            """
        else:
            # If no real-time data is available (i.e., ticker not found), use general knowledge.
            prompt = f"""
            Question: {question}
            Provide a clear answer. Keep under 200 words."""
            # Set system prompt for general knowledge queries
            system_prompt = "You are a stock market AI assistant. Provide a clear answer."
        
        answer = None
        
        try:
            models_to_try = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-pro']
            
            for model_name in models_to_try:
                try:
                    print(f"Trying Gemini model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    
                    # === EXPONENTIAL BACKOFF AND RETRIES FOR API CALLS ===
                    for i in range(3):
                        try:
                            response = model.generate_content(
                                contents=prompt, 
                                generation_config={"temperature": 0.5},
                                system_instruction=system_prompt 
                            )
                            answer = response.text
                            print(f"✓ Success with {model_name} on attempt {i+1}")
                            break # Success, break out of retry loop
                        except GeminiAPIError as api_error:
                            print(f"✗ {model_name} failed on attempt {i+1} due to API Error: {str(api_error)}")
                            if i < 2:
                                time.sleep(2 ** i) # Exponential backoff
                                continue
                            raise # If retries failed, raise the final error
                        except Exception as model_error:
                            raise model_error
                    
                    if answer:
                        break # Success, break out of model loop
                except Exception as model_error:
                    print(f"✗ Model {model_name} failed entirely: {str(model_error)}")
                    continue
            
            if not answer:
                raise Exception("All Gemini models failed after all retries.")
                
        except Exception as e:
            # This is the AI failure handling
            print(f"AI Error: {e}")
            traceback.print_exc()
            if market_context:
                # 🏆 SUCCESS: Stock data was found, but the AI API failed. Show the data!
                answer = f"📊 Real-Time Data Found (AI currently unavailable):\n\n{market_context}"
            else:
                # ❌ FAILURE: Both AI failed AND no stock data was found (ticker not in market/typo).
                # We show the original error message now.
                answer = f"❌ AI service error. Please check your GEMINI_API_KEY and API console status, or check stock symbol/spelling. (No real-time data found)."
        
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











