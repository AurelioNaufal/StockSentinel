# Sample Configuration File
# Copy this to config.py and customize as needed

# Gemini API Configuration
GEMINI_MODEL = 'gemini-1.5-flash'
GEMINI_RATE_LIMIT_DELAY = 4  # seconds between API calls

# Data Sources
STOCK_HISTORY_PERIOD = "60d"  # Period for technical analysis
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_SHORT = 20
SMA_LONG = 50
ATR_WINDOW = 14

# News Configuration
MAX_NEWS_ARTICLES = 5
NEWS_ARTICLES_TO_INCLUDE = 3

# Analysis Thresholds
HIGH_DIVIDEND_THRESHOLD = 2.0  # percentage
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Output Configuration
OUTPUT_DIR = "docs"
OUTPUT_FILE = "data.json"

# Custom Watchlist (optional - can override main.py)
CUSTOM_WATCHLIST = [
    # Add your custom tickers here
    # "AAPL",
    # "TSLA",
]
