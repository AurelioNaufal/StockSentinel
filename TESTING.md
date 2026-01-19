# 🧪 Testing Guide for StockSentinel

## Local Testing (Before GitHub Deployment)

### 1. Test Backend Script Locally

```bash
# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key (Windows PowerShell)
$env:GEMINI_API_KEY='your-api-key-here'

# Or (Linux/Mac)
export GEMINI_API_KEY='your-api-key-here'

# Test with a small watchlist first
# Edit main.py and temporarily change WATCHLIST to:
WATCHLIST = ["AAPL", "MSFT", "BBCA.JK"]

# Run the script
python main.py
```

**Expected Output:**
```
StockSentinel Automated Analysis
==================================================
Analyzing 3 assets...
Started at: 2026-01-19 10:30:00

[1/3] Processing AAPL...
  ✓ AAPL: Strong Buy (Position (Weeks-Months))
  Waiting 4 seconds (rate limit)...
[2/3] Processing MSFT...
  ✓ MSFT: Buy (Long-term (Months-Years))
  Waiting 4 seconds (rate limit)...
[3/3] Processing BBCA.JK...
  ✓ BBCA.JK: Hold (Position (Weeks-Months))

==================================================
✓ Analysis complete! 3 assets analyzed.
✓ Results saved to: docs/data.json
Finished at: 2026-01-19 10:31:00
```

### 2. Verify Generated Data

```bash
# Check if data.json was created
ls docs/data.json

# View the content (formatted)
python -m json.tool docs/data.json
```

**Expected Structure:**
```json
{
  "last_updated": "2026-01-19T10:30:00.123456",
  "total_assets": 3,
  "assets": [
    {
      "ticker": "AAPL",
      "current_price": 185.50,
      "recommendation": "Strong Buy",
      ...
    }
  ]
}
```

### 3. Test Frontend Locally

```bash
# Option A: Using Python's built-in server
cd docs
python -m http.server 8000

# Then visit: http://localhost:8000
```

```bash
# Option B: Using VS Code Live Server extension
# 1. Install "Live Server" extension
# 2. Right-click on docs/index.html
# 3. Select "Open with Live Server"
```

**Test Checklist:**
- [ ] Page loads without errors
- [ ] Stock cards display correctly
- [ ] Filter tabs work (All, Strong Buys, etc.)
- [ ] Click on stock card opens modal
- [ ] Modal shows all details
- [ ] Search box is visible

### 4. Test Individual Stock Search (Frontend API)

**IMPORTANT**: Before testing search:
1. Edit `docs/script.js`
2. Replace `const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE';`
3. With your actual API key

**Test Steps:**
1. Open the local website
2. Enter a ticker NOT in your watchlist (e.g., "COIN")
3. Click Search or press Enter
4. Wait ~4-8 seconds (rate limit + API call)
5. Modal should open with live analysis

**Expected Behavior:**
- "Searching..." appears
- "COIN not in cache. Fetching live analysis from Gemini..." message
- After delay: "Live analysis for COIN complete!"
- Modal opens with analysis

**Common Issues:**
- ❌ "Gemini API key not configured" → Update script.js
- ❌ CORS errors → Expected (Yahoo Finance blocked), but Gemini should still work
- ❌ Rate limit errors → Wait 4 seconds between searches

## Testing GitHub Actions Workflow

### 1. Dry Run (Test Syntax)

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Test workflow syntax
act -n

# Run workflow locally (if act installed)
act workflow_dispatch -s GEMINI_API_KEY='your-key'
```

### 2. Push to GitHub and Test

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/StockSentinel.git
git branch -M main
git push -u origin main
```

### 3. Trigger Workflow Manually

1. Go to your GitHub repository
2. Click **Actions** tab
3. Select "Manual Stock Analysis Scan"
4. Click **Run workflow**
5. Select branch (main)
6. Click **Run workflow**

**Monitor Progress:**
- Watch workflow running in real-time
- Check for errors in logs
- Verify `docs/data.json` is updated after completion

### 4. Verify GitHub Pages

1. Wait 1-2 minutes after workflow completes
2. Visit: `https://YOUR_USERNAME.github.io/StockSentinel/`
3. Verify site loads and data is fresh

## Unit Testing (Optional)

Create `test_main.py`:

```python
import unittest
from unittest.mock import patch, MagicMock
import main

class TestStockSentinel(unittest.TestCase):
    
    @patch('main.yf.Ticker')
    def test_fetch_stock_data(self, mock_ticker):
        # Mock yfinance response
        mock_hist = MagicMock()
        mock_hist.empty = False
        # ... setup mock data
        
        result = main.fetch_stock_data('AAPL')
        self.assertIsNotNone(result)
        self.assertEqual(result['ticker'], 'AAPL')
    
    def test_fetch_news(self):
        news = main.fetch_news('AAPL')
        self.assertIsInstance(news, list)
        # News might be empty, but should be a list
    
    # Add more tests...

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m unittest test_main.py
```

## Performance Testing

### API Rate Limiting Test

```python
# Create test_rate_limit.py
import time
from main import analyze_with_gemini, fetch_stock_data

tickers = ["AAPL", "MSFT", "GOOGL"]
start_time = time.time()

for ticker in tickers:
    stock_data = fetch_stock_data(ticker)
    if stock_data:
        analyze_with_gemini(ticker, stock_data, [])
    time.sleep(4)  # Rate limit

elapsed = time.time() - start_time
print(f"Analyzed {len(tickers)} stocks in {elapsed:.2f} seconds")
print(f"Average: {elapsed/len(tickers):.2f} seconds per stock")
```

**Expected**: ~4 seconds per stock (after first one)

### Load Testing (Frontend)

Use browser DevTools:
1. Open `docs/index.html` in browser
2. Open DevTools (F12)
3. Go to Performance tab
4. Record page load
5. Check metrics:
   - First Contentful Paint < 1s
   - Time to Interactive < 2s
   - No JavaScript errors

## Troubleshooting Common Issues

### Issue: "No data available for TICKER"

**Causes:**
- Ticker symbol incorrect
- Market closed (no recent data)
- Stock delisted

**Solution:**
- Verify ticker on Yahoo Finance
- Use correct suffix (.JK for IDX, -USD for crypto)

### Issue: "Error analyzing with Gemini"

**Causes:**
- Invalid API key
- Rate limit exceeded
- API quota exhausted
- Network issue

**Solution:**
```python
# Add better error handling in main.py
except genai.errors.ResourceExhausted:
    print("  ⚠️ API quota exhausted. Wait or upgrade plan.")
except genai.errors.PermissionDenied:
    print("  ⚠️ Invalid API key. Check your credentials.")
```

### Issue: Frontend search not working

**Checklist:**
- [ ] API key set in script.js
- [ ] Browser console shows no errors
- [ ] Network tab shows API request
- [ ] CORS enabled on API endpoint (if using proxy)

### Issue: GitHub Actions failing

**Check:**
1. Workflow logs for specific error
2. GEMINI_API_KEY secret is set
3. requirements.txt is correct
4. Python version matches (3.10+)

## Performance Optimization Tips

### Reduce Analysis Time

```python
# In main.py, parallelize data fetching
from concurrent.futures import ThreadPoolExecutor

def analyze_batch(tickers, batch_size=5):
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        results = executor.map(fetch_stock_data, tickers)
    return [r for r in results if r]
```

### Cache News Articles

```python
# Cache news for 1 hour to reduce API calls
import pickle
from datetime import datetime, timedelta

def fetch_news_cached(ticker):
    cache_file = f"cache/{ticker}_news.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
            if datetime.now() - cached['time'] < timedelta(hours=1):
                return cached['data']
    
    news = fetch_news(ticker)
    with open(cache_file, 'wb') as f:
        pickle.dump({'time': datetime.now(), 'data': news}, f)
    return news
```

## Continuous Integration

Add `.github/workflows/test.yml`:

```yaml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/
```

---

## Ready for Production?

**Checklist:**
- [ ] Tested locally with real API key
- [ ] `docs/data.json` generates successfully
- [ ] Frontend displays data correctly
- [ ] Search functionality works
- [ ] GitHub Actions workflow runs without errors
- [ ] GitHub Pages site is accessible
- [ ] Rate limiting is working
- [ ] API key is in GitHub Secrets (not hardcoded)
- [ ] README.md is complete
- [ ] .gitignore excludes sensitive files

**If all checked ✅, you're ready to deploy!** 🚀
