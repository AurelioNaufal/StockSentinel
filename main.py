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
import numpy as np

# Configure Hugging Face Model (Qwen 2.5 - 3B Instruct)
print("Loading Qwen 2.5 - 3B Instruct model (better accuracy)...")
model_name = "Qwen/Qwen2.5-3B-Instruct"

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
    "DIS",      # Disney
    "NFLX",     # Netflix
    
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
        results = ddgs.news(search_query, max_results=10)
        
        news_items = []
        for result in results[:3]:  # Limit to 3 articles for speed
            news_items.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('body', '')
            })
        
        return news_items
        
    except Exception as e:
        print(f"  Error fetching news for {ticker}: {e}")
        return []


def calculate_fundamental_score(ticker, stock_data):
    """Calculate fundamental score 0-100 based on financial metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        scores = []
        
        # 1. P/E Ratio (lower is better, typical range 10-30)
        pe = info.get('trailingPE', None) or info.get('forwardPE', None)
        if pe and pe > 0:
            if pe < 15:
                scores.append(90)  # Undervalued
            elif pe < 25:
                scores.append(70)  # Fair
            elif pe < 40:
                scores.append(50)  # Expensive
            else:
                scores.append(30)  # Overvalued
        
        # 2. ROE (Return on Equity) - higher is better
        roe = info.get('returnOnEquity', None)
        if roe:
            roe_pct = roe * 100
            if roe_pct > 20:
                scores.append(90)
            elif roe_pct > 15:
                scores.append(75)
            elif roe_pct > 10:
                scores.append(60)
            else:
                scores.append(40)
        
        # 3. Debt to Equity (lower is better)
        debt_to_equity = info.get('debtToEquity', None)
        if debt_to_equity is not None:
            if debt_to_equity < 50:
                scores.append(90)
            elif debt_to_equity < 100:
                scores.append(70)
            elif debt_to_equity < 200:
                scores.append(50)
            else:
                scores.append(30)
        
        # 4. Profit Margins (higher is better)
        profit_margin = info.get('profitMargins', None)
        if profit_margin:
            margin_pct = profit_margin * 100
            if margin_pct > 20:
                scores.append(90)
            elif margin_pct > 10:
                scores.append(70)
            elif margin_pct > 5:
                scores.append(50)
            else:
                scores.append(30)
        
        # 5. Revenue Growth (YoY)
        revenue_growth = info.get('revenueGrowth', None)
        if revenue_growth:
            growth_pct = revenue_growth * 100
            if growth_pct > 20:
                scores.append(90)
            elif growth_pct > 10:
                scores.append(75)
            elif growth_pct > 0:
                scores.append(60)
            else:
                scores.append(40)
        
        # 6. Dividend Yield (bonus points)
        if stock_data.get('dividend_yield', 0) > 3:
            scores.append(85)
        elif stock_data.get('dividend_yield', 0) > 1:
            scores.append(65)
        
        # Average all available scores
        if scores:
            return round(np.mean(scores), 2)
        else:
            return 50  # Neutral if no data
            
    except Exception as e:
        print(f"  Fundamental scoring error: {e}")
        return 50


def calculate_technical_score(stock_data):
    """Calculate technical score 0-100 based on indicators."""
    scores = []
    
    # 1. RSI Score (30-70 is ideal, extremes are warning signs)
    rsi = stock_data.get('rsi')
    if rsi:
        if 40 <= rsi <= 60:
            scores.append(80)  # Neutral zone - good
        elif 30 <= rsi < 40:
            scores.append(90)  # Oversold - buy opportunity
        elif 60 < rsi <= 70:
            scores.append(60)  # Mildly overbought
        elif rsi < 30:
            scores.append(95)  # Very oversold - strong buy signal
        elif rsi > 70:
            scores.append(40)  # Overbought - caution
        else:
            scores.append(50)
    
    # 2. MACD Signal (bullish crossover is positive)
    macd = stock_data.get('macd')
    macd_signal = stock_data.get('macd_signal')
    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd > 0:
            scores.append(85)  # Bullish momentum
        elif macd > macd_signal:
            scores.append(70)  # Gaining strength
        elif macd < macd_signal and macd < 0:
            scores.append(40)  # Bearish momentum
        else:
            scores.append(55)
    
    # 3. Price vs Moving Averages
    price = stock_data.get('current_price')
    sma_20 = stock_data.get('sma_20')
    sma_50 = stock_data.get('sma_50')
    
    if price and sma_20 and sma_50:
        if price > sma_20 > sma_50:
            scores.append(85)  # Strong uptrend
        elif price > sma_20:
            scores.append(70)  # Above short-term MA
        elif price < sma_20 < sma_50:
            scores.append(40)  # Downtrend
        elif price < sma_20:
            scores.append(90)  # Below MA - potential buy
        else:
            scores.append(60)
    
    # 4. Volume Analysis (higher volume = more conviction)
    volume = stock_data.get('volume', 0)
    if volume > 0:
        # Placeholder: ideally compare to average volume
        scores.append(65)
    
    # Average all scores
    if scores:
        return round(np.mean(scores), 2)
    else:
        return 50


def calculate_composite_score(sentiment_score, fundamental_score, technical_score):
    """Calculate weighted composite score (33% sentiment, 33% fundamental, 34% technical)."""
    composite = (
        sentiment_score * 0.33 +      # Sentiment: 33%
        fundamental_score * 0.33 +     # Fundamentals: 33%
        technical_score * 0.34         # Technicals: 34%
    )
    
    # Determine suggested recommendation based on composite score
    if composite >= 80:
        suggested_rec = "Strong Buy"
    elif composite >= 65:
        suggested_rec = "Buy"
    elif composite >= 45:
        suggested_rec = "Hold"
    elif composite >= 30:
        suggested_rec = "Sell"
    else:
        suggested_rec = "Strong Sell"
    
    return round(composite, 2), suggested_rec


def analyze_with_qwen(ticker, stock_data, news):
    """Use Qwen model with AI-powered composite scoring to analyze stock."""
    try:
        # Calculate AI scores (fundamental + technical, sentiment from Qwen)
        print(f"  Calculating AI scores...")
        fundamental_score = calculate_fundamental_score(ticker, stock_data)
        technical_score = calculate_technical_score(stock_data)
        
        print(f"  Scores - Fundamental: {fundamental_score}, Technical: {technical_score}")
        
        # Get currency symbol
        currency = stock_data.get('currency', 'USD')
        currency_symbol = 'Rp' if currency == 'IDR' else '$'
        
        # Prepare prompt with AI scoring insights
        prompt = f"""You are an expert stock analyst. Analyze {ticker} using the provided scoring system.

SCORING SYSTEM:
- Fundamental Score: {fundamental_score}/100 (PE, ROE, debt, margins, growth)
- Technical Score: {technical_score}/100 (RSI, MACD, moving averages)

TECHNICAL DETAILS:
- Price: {currency_symbol}{stock_data['current_price']}
- RSI: {stock_data['rsi']} | MACD: {stock_data['macd']} | Signal: {stock_data['macd_signal']}
- SMA(20): {currency_symbol}{stock_data['sma_20']} | SMA(50): {currency_symbol}{stock_data['sma_50']}
- Dividend Yield: {stock_data['dividend_yield']}%

RECENT NEWS HEADLINES:
"""
        for idx, article in enumerate(news, 1):
            prompt += f"{idx}. {article['title']}\n"
        
        prompt += f"""

YOUR TASK:
1. Read and analyze the news sentiment comprehensively (0-100 score)
   - 0-30: Very negative
   - 30-45: Negative
   - 45-55: Neutral  
   - 55-70: Positive
   - 70-100: Very positive
2. Calculate composite score: (sentiment*33% + fundamental*33% + technical*34%)
3. Provide recommendation based on composite score

RESPOND WITH VALID JSON ONLY (no markdown):
{{
  "sentiment_score": X (0-100, your analysis of news sentiment),
  "composite_score": Y (calculated: sentiment*0.33 + {fundamental_score}*0.33 + {technical_score}*0.34),
  "interest_level": "Interesting" or "Not Interesting",
  "recommendation": "Strong Buy" or "Buy" or "Hold" or "Sell" or "Strong Sell",
  "confidence_score": Z (same as composite_score),
  "time_horizon": "Scalp Trading (Minutes/Hours)" or "Day Trading (Days)" or "Investment (Long-Term)",
  "reasoning": "Brief explanation mentioning sentiment, fundamentals, technicals.",
  "entry_zone": "{currency_symbol}X.XX-{currency_symbol}Y.YY",
  "take_profit": "{currency_symbol}X.XX",
  "stop_loss": "{currency_symbol}X.XX"
}}

GUIDELINES:
1. Analyze news carefully for sentiment (not just keywords)
2. Composite >= 80: Strong Buy, >= 65: Buy, >= 45: Hold, >= 30: Sell, < 30: Strong Sell
3. Interest level: "Not Interesting" if composite < 40
4. Prices must be numbers with {currency_symbol}"""
        
        # Prepare messages for chat format
        messages = [
            {"role": "system", "content": "You are a stock analyst validating AI-generated composite scores. Output ONLY valid JSON."},
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
                max_new_tokens=250,  # Reduced for faster generation
                temperature=0.2,
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
        
        # Extract sentiment and composite scores from Qwen's response
        sentiment_score = analysis.get('sentiment_score', 50)
        composite_score = analysis.get('composite_score', 50)
        
        print(f"  Qwen sentiment: {sentiment_score}, Composite: {composite_score} -> {analysis.get('recommendation', 'N/A')}")
        
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
        fallback_composite = int((50 * 0.33 + fundamental_score * 0.33 + technical_score * 0.34))
        return {
            "sentiment_score": 50,
            "composite_score": fallback_composite,
            "interest_level": "Interesting",
            "recommendation": "Hold",
            "confidence_score": fallback_composite,
            "time_horizon": "Investment (Long-Term)",
            "reasoning": f"Analysis pending. Fundamental: {fundamental_score}, Technical: {technical_score}",
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
        
        print(f"  [OK] {ticker}: {analysis['recommendation']} ({analysis['time_horizon']})")
        
        # Small delay between analyses
        if idx < len(WATCHLIST):
            time.sleep(0.5)
    
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
