import os
import json
import time
import yfinance as yf
import pandas as pd
import ta
from duckduckgo_search import DDGS
from ddgs import DDGS
from newspaper import Article
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configure Hugging Face Model (Qwen 2.5 - 1.5B Instruct)
print("Loading Qwen 2.5 - 1.5B Instruct model (CPU-optimized)...")
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map=None,
    low_cpu_mem_usage=True
)

model = model.to('cpu')
print("Model loaded successfully! Running on CPU.\n")

# Watchlist: Major IDX, US, and Crypto assets
WATCHLIST = [
    # IDX Stocks (Indonesian Stock Exchange)
    "BBCA.JK",  # Bank Central Asia
    "BBRI.JK",  # Bank Rakyat Indonesia
    "BMRI.JK",  # Bank Mandiri
    "TLKM.JK",  # Telkom Indonesia
    "ASII.JK",  # Astra International
    "UNVR.JK",  # Unilever Indonesia
    "ICBP.JK",  # Indofood CBP
    "INDF.JK",  # Indofood
    "KLBF.JK",  # Kalbe Farma
    "BBTN.JK",  # Bank Tabungan Negara
    "ACES.JK",  # Ace Hardware
    "ADRO.JK",  # Adaro Energy
    "ANTM.JK",  # Aneka Tambang
    "INCO.JK",  # Vale Indonesia
    
    # US Stocks
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Google
    "AMZN",     # Amazon
    "NVDA",     # Nvidia
    "TSLA",     # Tesla
    "META",     # Meta
    "JPM",      # JP Morgan
    "V",        # Visa
    "WMT",      # Walmart
    "JNJ",      # Johnson & Johnson
    "PG",       # Procter & Gamble
    "MA",       # Mastercard
    "HD",       # Home Depot
    "BAC",      # Bank of America
    "DIS",      # Disney
    "NFLX",     # Netflix
    "CSCO",     # Cisco
    "KO",       # Coca-Cola
    
    # Crypto (via Yahoo Finance)
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
]


def fetch_stock_data(ticker):
    """Fetch price, technical indicators, and dividend data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data (60 days for technical indicators)
        hist = stock.history(period="60d")
        
        if hist.empty:
            print(f"  No data available for {ticker}")
            return None
        
        # Calculate technical indicators
        hist['RSI'] = ta.momentum.RSIIndicator(hist['Close'], window=14).rsi()
        
        macd = ta.trend.MACD(hist['Close'])
        hist['MACD'] = macd.macd()
        hist['MACD_Signal'] = macd.macd_signal()
        
        hist['SMA_20'] = ta.trend.SMAIndicator(hist['Close'], window=20).sma_indicator()
        hist['SMA_50'] = ta.trend.SMAIndicator(hist['Close'], window=50).sma_indicator()
        
        hist['ATR'] = ta.volatility.AverageTrueRange(
            hist['High'], hist['Low'], hist['Close'], window=14
        ).average_true_range()
        
        # Get latest values
        latest = hist.iloc[-1]
        current_price = latest['Close']
        
        # Determine currency based on ticker
        currency = 'IDR' if ticker.endswith('.JK') else 'USD'
        
        # Get dividend info
        dividend_yield = 0
        try:
            info = stock.info
            dividend_yield = info.get('dividendYield', 0)
            # Already in decimal form (e.g., 0.05 = 5%) no need to convert
        except:
            pass
        
        # Prepare historical data for different timeframes
        hist_1m = stock.history(period="1mo")
        hist_6m = stock.history(period="6mo")
        hist_max = stock.history(period="max")
        
        chart_data = {
            '1M': [],
            '6M': [],
            'MAX': []
        }
        
        # Process each timeframe
        for period_key, hist_period in [('1M', hist_1m), ('6M', hist_6m), ('MAX', hist_max)]:
            if not hist_period.empty:
                for idx in range(len(hist_period)):
                    row = hist_period.iloc[idx]
                    # Calculate SMA 20 if enough data
                    sma_20_val = None
                    if idx >= 19:
                        sma_20_val = round(hist_period['Close'].iloc[max(0, idx-19):idx+1].mean(), 2)
                    
                    chart_data[period_key].append({
                        'date': hist_period.index[idx].strftime('%Y-%m-%d'),
                        'price': round(row['Close'], 2),
                        'sma_20': sma_20_val
                    })
        
        data = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'currency': currency,
            'rsi': round(latest['RSI'], 2) if pd.notna(latest['RSI']) else None,
            'macd': round(latest['MACD'], 2) if pd.notna(latest['MACD']) else None,
            'macd_signal': round(latest['MACD_Signal'], 2) if pd.notna(latest['MACD_Signal']) else None,
            'sma_20': round(latest['SMA_20'], 2) if pd.notna(latest['SMA_20']) else None,
            'sma_50': round(latest['SMA_50'], 2) if pd.notna(latest['SMA_50']) else None,
            'atr': round(latest['ATR'], 2) if pd.notna(latest['ATR']) else None,
            'dividend_yield': round(dividend_yield, 2) if dividend_yield else 0,
            'volume': int(latest['Volume']),
            'chart_data': chart_data
        }
        
        return data
        
    except Exception as e:
        print(f"  Error fetching data for {ticker}: {e}")
        return None


def fetch_news(ticker):
    """Fetch recent news headlines and snippets for a ticker."""
    try:
        # Clean ticker for search
        if ticker.endswith('.JK'):
            # Indonesian stocks - use company name mapping for better news
            stock_names = {
                'BBCA.JK': 'Bank BCA Indonesia',
                'BBRI.JK': 'Bank BRI Indonesia',
                'BMRI.JK': 'Bank Mandiri Indonesia',
                'TLKM.JK': 'Telkom Indonesia',
                'ASII.JK': 'Astra International',
                'UNVR.JK': 'Unilever Indonesia',
                'ICBP.JK': 'Indofood CBP',
                'INDF.JK': 'Indofood',
                'KLBF.JK': 'Kalbe Farma',
                'GGRM.JK': 'Gudang Garam',
                'BBTN.JK': 'BTN Bank',
                'ACES.JK': 'Ace Hardware Indonesia',
                'ADRO.JK': 'Adaro Energy',
                'ANTM.JK': 'Aneka Tambang',
                'INCO.JK': 'Vale Indonesia'
            }
            search_query = stock_names.get(ticker, ticker.replace('.JK', '')) + ' saham'
        else:
            search_query = f"{ticker} stock"
        
        # Search for news
        ddgs = DDGS()
        results = ddgs.news(search_query, max_results=5)
        
        news_items = []
        for result in results[:3]:  # Limit to 3 articles
            news_items.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('body', '')
            })
        
        return news_items
        
    except Exception as e:
        print(f"  Error fetching news for {ticker}: {e}")
        return []


def analyze_with_qwen(ticker, stock_data, news):
    """Use Qwen model to analyze stock and return structured recommendation."""
    try:
        # Get currency symbol
        currency = stock_data.get('currency', 'USD')
        currency_symbol = 'Rp' if currency == 'IDR' else '$'
        
        # Prepare prompt with strict JSON structure
        prompt = f"""Analyze {ticker} stock data and provide investment recommendation.

TECHNICAL INDICATORS:
- Price: {currency_symbol}{stock_data['current_price']}
- RSI(14): {stock_data['rsi']} (>70 overbought, <30 oversold)
- MACD: {stock_data['macd']} | Signal: {stock_data['macd_signal']}
- SMA(20): {currency_symbol}{stock_data['sma_20']} | SMA(50): {currency_symbol}{stock_data['sma_50']}
- ATR: {stock_data['atr']} (volatility)
- Dividend Yield: {stock_data['dividend_yield']}%
- Volume: {stock_data['volume']:,}

RECENT NEWS:
"""
        for idx, article in enumerate(news, 1):
            prompt += f"{idx}. {article['title']}\n"
        
        prompt += f"""\n
RESPOND WITH VALID JSON ONLY (no markdown, no explanation):
{{
  "interest_level": "Interesting" or "Not Interesting",
  "recommendation": "Strong Buy" or "Buy" or "Hold" or "Sell" or "Strong Sell",
  "confidence_score": 0-100,
  "time_horizon": "Scalp Trading (Minutes/Hours)" or "Day Trading (Days)" or "Investment (Long-Term)",
  "reasoning": "Brief analysis based on technical indicators and news",
  "entry_zone": "{currency_symbol}X.XX-{currency_symbol}Y.YY",
  "take_profit": "{currency_symbol}X.XX",
  "stop_loss": "{currency_symbol}X.XX"
}}

RULES:
1. interest_level "Not Interesting" = fundamentally weak, bad news, poor technicals, not worth monitoring
2. interest_level "Interesting" = has potential, worth watching or trading
3. time_horizon (choose ONE only):
   - "Scalp Trading (Minutes/Hours)": ONLY for extreme volatility with RSI >75 or <25, expected to move sharply within minutes/hours
   - "Day Trading (Days)": Short-term momentum, technical breakout/breakdown, hold for 1-5 days
   - "Investment (Long-Term)": Strong fundamentals, good dividend, stable company, hold for months/years
4. Differentiate clearly: Scalp = very short-term high volatility, Day Trading = short-term momentum, Investment = long-term value
5. entry_zone, take_profit, stop_loss must be NUMBERS with currency symbol (e.g., "$150.50" or "Rp5000")
6. No objects, no arrays in entry_zone/take_profit/stop_loss
7. Base recommendation on BOTH technicals AND news sentiment"""
        
        # Prepare messages for chat format
        messages = [
            {"role": "system", "content": "You are a professional stock analyst. Output ONLY valid JSON. No markdown formatting."},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response with more constrained parameters
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=400,
                temperature=0.3,  # Lower temperature for more consistent output
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Aggressive JSON extraction
        response_text = response_text.strip()
        # Remove markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        # Find JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            response_text = response_text[start_idx:end_idx]
        
        analysis = json.loads(response_text.strip())
        
        # Validate and fix common issues
        currency_symbol = 'Rp' if stock_data.get('currency') == 'IDR' else '$'
        
        # Fix entry_zone if it's an object or array
        if isinstance(analysis.get('entry_zone'), (dict, list)):
            analysis['entry_zone'] = f"{currency_symbol}{stock_data['current_price']}-{currency_symbol}{round(stock_data['current_price'] * 1.02, 2)}"
        
        # Fix take_profit if it's an object or array
        if isinstance(analysis.get('take_profit'), (dict, list)):
            analysis['take_profit'] = f"{currency_symbol}{round(stock_data['current_price'] * 1.1, 2)}"
        
        # Fix stop_loss if it's an object or array
        if isinstance(analysis.get('stop_loss'), (dict, list)):
            analysis['stop_loss'] = f"{currency_symbol}{round(stock_data['current_price'] * 0.95, 2)}"
        
        return analysis
        
    except Exception as e:
        print(f"  Error analyzing with Qwen for {ticker}: {e}")
        # Return a fallback analysis
        currency_symbol = 'Rp' if stock_data.get('currency') == 'IDR' else '$'
        return {
            "interest_level": "Interesting",
            "recommendation": "Hold",
            "confidence_score": 50,
            "time_horizon": "Investment (Long-Term)",
            "reasoning": f"Technical analysis pending. RSI: {stock_data['rsi']}, Price near SMA(20).",
            "entry_zone": f"{currency_symbol}{stock_data['current_price']}",
            "take_profit": f"{currency_symbol}{round(stock_data['current_price'] * 1.08, 2)}",
            "stop_loss": f"{currency_symbol}{round(stock_data['current_price'] * 0.95, 2)}"
        }


def main():
    """Main function to analyze all stocks in the watchlist."""
    print("StockSentinel Automated Analysis")
    print("=" * 50)
    print(f"Analyzing {len(WATCHLIST)} assets...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    for idx, ticker in enumerate(WATCHLIST, 1):
        print(f"[{idx}/{len(WATCHLIST)}] Processing {ticker}...")
        
        # Fetch stock data
        stock_data = fetch_stock_data(ticker)
        if not stock_data:
            continue
        
        # Fetch news
        news = fetch_news(ticker)
        
        # Analyze with Qwen
        analysis = analyze_with_qwen(ticker, stock_data, news)
        
        # Combine all data
        result = {
            **stock_data,
            **analysis,
            'news': news,
            'last_updated': datetime.now().isoformat()
        }
        
        results.append(result)
        
        print(f"  ✓ {ticker}: {analysis['recommendation']} ({analysis['time_horizon']})")
        
        # Small delay between analyses (not needed for local model, but helps with stability)
        if idx < len(WATCHLIST):
            time.sleep(0.5)  # Short delay for stability
    
    # Save to docs/data.json
    output_dir = 'docs'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'data.json')
    with open(output_file, 'w') as f:
        json.dump({
            'last_updated': datetime.now().isoformat(),
            'total_assets': len(results),
            'assets': results
        }, f, indent=2)
    
    print()
    print("=" * 50)
    print(f"✓ Analysis complete! {len(results)} assets analyzed.")
    print(f"✓ Results saved to: {output_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
