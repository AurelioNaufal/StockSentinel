import os
import json
import time
import yfinance as yf
import pandas as pd
import ta
from duckduckgo_search import DDGS
from newspaper import Article
import google.generativeai as genai
from datetime import datetime, timedelta

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

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
    "GGRM.JK",  # Gudang Garam
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
    "PFE",      # Pfizer
    "KO",       # Coca-Cola
    
    # Crypto (via Yahoo Finance)
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "BNB-USD",  # Binance Coin
    "XRP-USD",  # Ripple
    "ADA-USD",  # Cardano
    "DOGE-USD", # Dogecoin
    "SOL-USD",  # Solana
    "MATIC-USD",# Polygon
    "DOT-USD",  # Polkadot
    "AVAX-USD", # Avalanche
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
        
        # Get dividend info
        dividend_yield = 0
        try:
            info = stock.info
            dividend_yield = info.get('dividendYield', 0)
            if dividend_yield:
                dividend_yield = dividend_yield * 100  # Convert to percentage
        except:
            pass
        
        data = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'rsi': round(latest['RSI'], 2) if pd.notna(latest['RSI']) else None,
            'macd': round(latest['MACD'], 2) if pd.notna(latest['MACD']) else None,
            'macd_signal': round(latest['MACD_Signal'], 2) if pd.notna(latest['MACD_Signal']) else None,
            'sma_20': round(latest['SMA_20'], 2) if pd.notna(latest['SMA_20']) else None,
            'sma_50': round(latest['SMA_50'], 2) if pd.notna(latest['SMA_50']) else None,
            'atr': round(latest['ATR'], 2) if pd.notna(latest['ATR']) else None,
            'dividend_yield': round(dividend_yield, 2) if dividend_yield else 0,
            'volume': int(latest['Volume']),
        }
        
        return data
        
    except Exception as e:
        print(f"  Error fetching data for {ticker}: {e}")
        return None


def fetch_news(ticker):
    """Fetch recent news headlines and snippets for a ticker."""
    try:
        # Clean ticker for search (remove .JK suffix for better results)
        search_ticker = ticker.replace('.JK', '')
        
        # Search for news
        ddgs = DDGS()
        results = ddgs.news(f"{search_ticker} stock", max_results=5)
        
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


def analyze_with_gemini(ticker, stock_data, news):
    """Call Gemini API to analyze stock and return structured recommendation."""
    try:
        # Prepare prompt
        prompt = f"""
You are a professional stock analyst. Analyze the following data for {ticker} and provide investment recommendations in JSON format.

Technical Data:
- Current Price: ${stock_data['current_price']}
- RSI: {stock_data['rsi']}
- MACD: {stock_data['macd']}
- MACD Signal: {stock_data['macd_signal']}
- SMA 20: {stock_data['sma_20']}
- SMA 50: {stock_data['sma_50']}
- ATR: {stock_data['atr']}
- Dividend Yield: {stock_data['dividend_yield']}%
- Volume: {stock_data['volume']}

Recent News:
"""
        for idx, article in enumerate(news, 1):
            prompt += f"{idx}. {article['title']}\n   {article['snippet']}\n"
        
        prompt += """

Please provide your analysis in the following JSON format (respond ONLY with valid JSON, no markdown):
{
    "recommendation": "Strong Buy|Buy|Hold|Sell|Strong Sell",
    "confidence_score": 85,
    "time_horizon": "Scalp (Minutes-Hours)|Day Trade (Hours-Days)|Swing (Days-Weeks)|Position (Weeks-Months)|Long-term (Months-Years)",
    "reasoning": "Brief explanation of your recommendation based on technicals and news",
    "entry_zone": "Price range for entry (e.g., $150-$155)",
    "take_profit": "Target price for profit taking",
    "stop_loss": "Stop loss price to limit downside",
    "dividend_analysis": "Brief note on dividend if applicable, or 'N/A'"
}
"""
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Clean markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        analysis = json.loads(response_text.strip())
        
        return analysis
        
    except Exception as e:
        print(f"  Error analyzing with Gemini for {ticker}: {e}")
        # Return a fallback analysis
        return {
            "recommendation": "Hold",
            "confidence_score": 50,
            "time_horizon": "Position (Weeks-Months)",
            "reasoning": f"Analysis failed: {str(e)}",
            "entry_zone": f"${stock_data['current_price']}",
            "take_profit": f"${round(stock_data['current_price'] * 1.1, 2)}",
            "stop_loss": f"${round(stock_data['current_price'] * 0.95, 2)}",
            "dividend_analysis": f"{stock_data['dividend_yield']}%" if stock_data['dividend_yield'] > 0 else "N/A"
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
        
        # Analyze with Gemini
        analysis = analyze_with_gemini(ticker, stock_data, news)
        
        # Combine all data
        result = {
            **stock_data,
            **analysis,
            'news': news,
            'last_updated': datetime.now().isoformat()
        }
        
        results.append(result)
        
        print(f"  ✓ {ticker}: {analysis['recommendation']} ({analysis['time_horizon']})")
        
        # Rate limiting: Wait 4 seconds between Gemini calls (15 RPM = 4 seconds)
        if idx < len(WATCHLIST):
            print("  Waiting 4 seconds (rate limit)...")
            time.sleep(4)
    
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
