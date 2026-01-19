# 📈 StockSentinel Interactive

An AI-powered, on-demand stock analysis platform hosted on GitHub Pages. Analyze major IDX, US, and Crypto assets with real-time recommendations powered by Google Gemini 1.5 Flash.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Gemini](https://img.shields.io/badge/AI-Gemini%201.5%20Flash-purple.svg)

## 🌟 Features

### Two Analysis Modes:

1. **Automated Top Stock Analysis** 
   - Runs manually via GitHub Actions
   - Analyzes 50+ predefined major assets (IDX, US, Crypto)
   - Generates comprehensive AI-driven recommendations
   - Outputs results to `data.json`

2. **Search in Cache**
   - Search any stock directly on the website
   - Instant results from cached `data.json`
   - Add more stocks by updating watchlist and running workflow

### Key Features:
- ✅ **Free & Local**: Powered by Qwen 2.5 - 1.5B Instruct (Hugging Face)
- ✅ **No API Keys**: Everything runs locally
- ✅ **Automated Scanning**: GitHub Actions workflow
- ✅ **Interactive UI**: Tailwind CSS + Alpine.js
- ✅ **Smart Filtering**: Strong Buys, Dividend Gems, Scalp/Day Trade
- ✅ **No Rate Limits**: Unlimited analysis
- ✅ **Dividend Insights**: Yield analysis for income investors
- ✅ **News Integration**: Latest market news for each stock
- ✅ **Technical Analysis**: RSI, MACD, SMA, ATR

## 🚀 Quick Start

### Prerequisites

- GitHub account
- Python 3.10+ (for local testing)
- 8GB+ RAM (for model)
- 5GB free disk space

### Setup Instructions

#### 1. Clone/Fork This Repository

```bash
git clone https://github.com/YOUR_USERNAME/StockSentinel.git
cd StockSentinel
```

#### 2. Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** First time will download Qwen model (~3GB). This is automatic.

#### 3. Enable GitHub Pages

1. Go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` (or `master`)
4. Folder: `/docs`
5. Click **Save**

#### 4. Run Analysis (Optional - Test Locally First)

1. Go to **Actions** tab in your repository
2. Click **Manual Stock Analysis Scan** workflow
3. Click **Run workflow**
4. Select branch (usually `main`)
5. Click **Run workflow**

The workflow will:
- Install dependencies
- Fetch data for all watchlist stocks
- Analyze each with Gemini AI
- Generate `docs/data.json`
- Commit and push results
- Update live website automatically

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (model downloads automatically first time)
python main.py
```

**First run:** Model downloads (~3GB), takes 10-15 minutes  
**Subsequent runs:** Model loads from cache, much faster

---

## 🎯 Usage

### Run Automated Analysis

#### Via GitHub Actions (Recommended)

### View Results

Visit your GitHub Pages URL:
```
https://YOUR_USERNAME.github.io/StockSentinel/
```

## 📁 Project Structure

```
StockSentinel/
├── .github/
│   └── workflows/
│       └── manual_scan.yml       # GitHub Actions workflow
├── docs/                         # GitHub Pages root
│   ├── index.html                # Main HTML
│   ├── script.js                 # Frontend logic
│   └── data.json                 # Generated stock data
├── main.py                       # Backend analysis script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Customize Watchlist

Edit the `WATCHLIST` array in `main.py`:

```python
WATCHLIST = [
    # Add your favorite stocks
    "AAPL",
    "TSLA",
    "BBCA.JK",
    "BTC-USD",
    # ... more tickers
]
```

### Adjust Rate Limiting

Default: 4 seconds between API calls (15 RPM free tier)

Edit in `main.py`:
```python
time.sleep(4)  # Change to your desired delay
```

Edit in `docs/script.js`:
```javascript
const RATE_LIMIT_DELAY = 4000;  // milliseconds
```

### Schedule Automatic Scans

Edit `.github/workflows/manual_scan.yml`:

```yaml
on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
    # - cron: '0 */6 * * *'  # Every 6 hours
    # - cron: '0 9 * * 1-5'  # Weekdays at 9 AM UTC
```

## 📊 Data Output Format

`docs/data.json` structure:

```json
{
  "last_updated": "2026-01-19T10:30:00",
  "total_assets": 45,
  "assets": [
    {
      "ticker": "AAPL",
      "current_price": 185.50,
      "recommendation": "Strong Buy",
      "confidence_score": 85,
      "time_horizon": "Position (Weeks-Months)",
      "reasoning": "Strong technical indicators...",
      "entry_zone": "$180-$185",
      "take_profit": "$200",
      "stop_loss": "$175",
      "dividend_yield": 0.52,
      "dividend_analysis": "Consistent dividend payer",
      "rsi": 58.5,
      "macd": 1.25,
      "macd_signal": 0.85,
      "sma_20": 182.30,
      "sma_50": 178.60,
      "atr": 3.45,
      "volume": 52000000,
      "news": [
        {
          "title": "Apple announces new...",
          "url": "https://...",
          "snippet": "..."
        }
      ],
      "last_updated": "2026-01-19T10:25:00"
    }
  ]
}
```

## 🎨 Frontend Features

### Filter Tabs
- **All Stocks**: View complete watchlist
- **Strong Buys**: High-confidence buy recommendations
- **Dividend Gems**: Stocks with >2% yield
- **Scalp/Day Trade**: Short-term opportunities

### Search Functionality
1. Enter any ticker symbol (e.g., AAPL, BBCA.JK, BTC-USD)
2. If cached → Instant results
3. If not cached → Live Gemini analysis
4. Rate-limited to prevent API abuse

### Stock Cards
- Color-coded recommendations
- Key metrics (RSI, Dividend, etc.)
- Entry/TP/SL levels
- Click for detailed analysis

## 🛡️ Gemini API Free Tier Limits

- **Requests**: 15 per minute (RPM)
- **Daily**: 1,500 requests
- **Rate limiting**: Built-in 4-second delays

**Tips**:
- Don't exceed watchlist of ~100 stocks per run
- Schedule scans wisely (daily or less frequent)
- Monitor your API usage at [Google AI Studio](https://makersuite.google.com/)

## 🐛 Troubleshooting

### GitHub Actions Not Running

1. Check **Actions** tab is enabled in repository settings
2. Verify `GEMINI_API_KEY` secret exists
3. Check workflow file syntax

### Frontend Not Loading Data

1. Verify `docs/data.json` exists
2. Check GitHub Pages is enabled
3. Check browser console for errors
4. Ensure API key is configured in `script.js`

### API Rate Limit Errors

1. Reduce watchlist size
2. Increase delay in `main.py` and `script.js`
3. Wait before running again

### No Stock Data Returned

1. Verify ticker symbol is correct (use Yahoo Finance format)
2. Check if market is open (for real-time data)
3. Some stocks may not have all data fields

## 📈 Adding More Features

### Add More Technical Indicators

Edit `main.py`:
```python
hist['BB_Upper'] = ta.volatility.BollingerBands(hist['Close']).bollinger_hband()
hist['Stochastic'] = ta.momentum.StochasticOscillator(hist['High'], hist['Low'], hist['Close']).stoch()
```

### Customize Gemini Prompts

Edit the prompt in `main.py` and `docs/script.js` to:
- Add more analysis criteria
- Change output format
- Include specific trading strategies

### Add Email Notifications

Use GitHub Actions to send emails on:
- Strong Buy signals
- High dividend finds
- Specific conditions

## 📝 License

MIT License - Feel free to use and modify!

## ⚠️ Disclaimer

**This platform is for educational and informational purposes only.**

- Not financial advice
- Past performance ≠ future results
- Always do your own research (DYOR)
- Consult a licensed financial advisor
- Trade at your own risk

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

Issues? Questions?
- Open a GitHub Issue
- Check existing issues first
- Provide detailed error messages

## 🙏 Acknowledgments

- **Google Gemini**: AI-powered analysis
- **Yahoo Finance**: Market data via `yfinance`
- **DuckDuckGo**: News search
- **Tailwind CSS**: Beautiful UI
- **Alpine.js**: Reactive frontend

---

**Happy Trading! 📊💰**

Remember: The best investment is the one you understand. Stay informed, stay cautious, and never invest more than you can afford to lose.
