# ⚡ Quick Start Guide - StockSentinel Interactive

Get up and running in 10 minutes!

## 📋 Prerequisites

- GitHub account (free)
- Google Gemini API key (free) - [Get it here](https://makersuite.google.com/app/apikey)
- Basic command line knowledge (optional for local testing)

## 🚀 5-Step Setup

### Step 1: Fork/Clone This Repository

**Option A: Using GitHub Web Interface**
1. Click the "Fork" button at the top-right
2. Wait for fork to complete
3. You now have your own copy!

**Option B: Using Git Command Line**
```bash
git clone https://github.com/YOUR_USERNAME/StockSentinel.git
cd StockSentinel
```

### Step 2: Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key (starts with `AIzaSy...`)
4. Keep it safe - you'll need it twice!

### Step 3: Add API Key to GitHub Secrets

1. Go to your repository on GitHub
2. Click **Settings** (tab at the top)
3. In left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Fill in:
   - **Name**: `GEMINI_API_KEY`
   - **Secret**: Paste your API key
6. Click **Add secret**

✅ This key is now secure and will be used by GitHub Actions.

### Step 4: Enable GitHub Pages

1. Still in **Settings**, scroll down to **Pages** (left sidebar)
2. Under "Source", select:
   - **Branch**: `main` (or `master`)
   - **Folder**: `/docs`
3. Click **Save**
4. Wait 1-2 minutes for deployment

Your site will be available at:
```
https://YOUR_USERNAME.github.io/StockSentinel/
```

### Step 5: Run Your First Analysis

1. Go to **Actions** tab
2. Click "Manual Stock Analysis Scan" workflow
3. Click **Run workflow** button (right side)
4. Select branch `main`
5. Click green **Run workflow** button
6. Wait 5-10 minutes for completion

**What happens:**
- Python script fetches data for 50+ stocks
- Each stock is analyzed by Gemini AI
- Results are saved to `docs/data.json`
- Website automatically updates!

---

## 🎉 You're Done!

Visit your GitHub Pages URL to see your live stock analysis platform!

## 🔍 Next Steps

### Enable Individual Stock Search

To enable the "search any stock" feature:

1. **Open your repository**
2. **Navigate to**: `docs/script.js`
3. **Click the pencil icon** (Edit this file)
4. **Find line 8**:
   ```javascript
   const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE';
   ```
5. **Replace with your actual API key**:
   ```javascript
   const GEMINI_API_KEY = 'AIzaSyXXXXXXXXXXXXXXXXXXX';
   ```
6. **Commit changes**: Scroll down, click "Commit changes"

⚠️ **Security Note**: This exposes your API key in the code. Only do this if:
- Your repository is **private**, OR
- You're comfortable with the risk (free tier has limits anyway)

For production, use a serverless proxy (see [README.md](README.md) for details).

### Customize Your Watchlist

1. **Edit** `main.py`
2. **Find the** `WATCHLIST` **array** (around line 16)
3. **Add/remove tickers**:
   ```python
   WATCHLIST = [
       # Your favorite stocks
       "AAPL",
       "TSLA",
       "NVDA",
       "BBCA.JK",  # Indonesian stocks
       "BTC-USD",  # Crypto
       # ... add more
   ]
   ```
4. **Commit changes**

### Schedule Automatic Updates

Want daily automatic scans?

1. **Edit** `.github/workflows/manual_scan.yml`
2. **Add schedule trigger**:
   ```yaml
   on:
     workflow_dispatch:  # Manual trigger
     schedule:
       - cron: '0 0 * * *'  # Daily at midnight UTC
   ```
3. **Commit changes**

Your analysis will now run automatically every day!

---

## 🐛 Common Issues & Quick Fixes

### Issue: "GitHub Actions not running"

**Fix**: Check if Actions are enabled
1. **Settings** → **Actions** → **General**
2. Ensure "Allow all actions" is selected
3. Save changes

### Issue: "Website shows 404"

**Fix**: Wait 2-3 minutes after enabling Pages
- GitHub Pages takes time to build
- Check Settings → Pages for deployment status
- Look for green checkmark

### Issue: "No data on website"

**Fix**: Run the workflow first!
1. Go to **Actions** tab
2. Run "Manual Stock Analysis Scan"
3. Wait for completion
4. Refresh your website

### Issue: "API key error in workflow"

**Fix**: Verify secret name
- Must be exactly `GEMINI_API_KEY`
- Case-sensitive!
- Re-add if needed

### Issue: "Search not working"

**Fix**: Add API key to script.js (see "Enable Individual Stock Search" above)

---

## 📱 Using the Platform

### Filter Stocks

Click the filter buttons:
- **All Stocks**: Complete watchlist
- **💎 Strong Buys**: High-confidence opportunities
- **💰 Dividend Gems**: Stocks with >2% yield
- **⚡ Scalp/Day Trade**: Short-term trades

### View Details

- Click any stock card
- See full analysis in modal
- View entry, take profit, stop loss
- Read AI reasoning
- Check recent news

### Search Any Stock

1. Type ticker in search box (e.g., `COIN`, `SHOP`)
2. Press Enter or click Search
3. Wait 4-8 seconds for AI analysis
4. View results in modal

---

## 📊 Understanding the Data

### Recommendation Levels

- **Strong Buy** 🟢: High confidence, strong signals
- **Buy** 🟢: Positive outlook, good entry
- **Hold** 🟡: Neutral, wait for confirmation
- **Sell** 🔴: Negative signals, consider exit
- **Strong Sell** 🔴: High confidence bearish

### Time Horizons

- **Scalp**: Minutes to hours (very short-term)
- **Day Trade**: Hours to days
- **Swing**: Days to weeks
- **Position**: Weeks to months
- **Long-term**: Months to years

### Technical Indicators

- **RSI**: 30 = oversold, 70 = overbought
- **MACD**: Trend strength indicator
- **SMA**: Moving averages for trend
- **ATR**: Volatility measure

---

## 🎓 Learning Resources

**Stock Analysis Basics**
- [Investopedia - Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [RSI Explained](https://www.investopedia.com/terms/r/rsi.asp)
- [MACD Guide](https://www.investopedia.com/terms/m/macd.asp)

**GitHub Actions**
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Cron Schedule Syntax](https://crontab.guru/)

**Google Gemini API**
- [Gemini API Docs](https://ai.google.dev/docs)
- [API Key Management](https://makersuite.google.com/)

---

## ⚠️ Important Reminders

### Not Financial Advice
This tool is for **educational purposes only**. Always:
- Do your own research (DYOR)
- Consult a licensed financial advisor
- Never invest more than you can afford to lose

### API Limits
Gemini Free Tier:
- 15 requests per minute
- 1,500 requests per day
- Built-in 4-second delays handle this

### Data Freshness
- Data is only updated when workflow runs
- Manual trigger = on-demand updates
- Scheduled = automatic updates at set times

---

## 🆘 Need Help?

1. **Check** [TESTING.md](TESTING.md) for detailed troubleshooting
2. **Read** [README.md](README.md) for full documentation
3. **Open an Issue** on GitHub with:
   - What you tried
   - Error messages
   - Screenshots (if applicable)

---

## 🎉 Success!

You now have a fully functional AI-powered stock analysis platform!

**Share your results:**
- Tweet your setup
- Share on Reddit (r/algotrading, r/stocks)
- Star this repo if you found it helpful!

**Happy Trading! 📈💰**

Remember: Smart traders use tools wisely. Stay informed, stay cautious, and always manage your risk.
