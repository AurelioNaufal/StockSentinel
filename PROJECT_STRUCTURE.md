# 📂 Project Structure - StockSentinel Interactive

Complete overview of all files and their purposes.

---

## 🗂️ Directory Structure

```
StockSentinel/
│
├── .github/
│   └── workflows/
│       └── manual_scan.yml          # GitHub Actions workflow for automated analysis
│
├── docs/                            # GitHub Pages root directory
│   ├── index.html                   # Main website HTML
│   ├── script.js                    # Frontend JavaScript logic
│   └── data.json                    # Generated stock analysis data
│
├── .gitignore                       # Git ignore rules
├── CHECKLIST.md                     # Complete setup checklist
├── config.example.py                # Sample configuration file
├── DEPLOYMENT.md                    # GitHub Pages deployment guide
├── main.py                          # Backend Python analysis script
├── QUICKSTART.md                    # Quick 5-step setup guide
├── README.md                        # Main project documentation
├── requirements.txt                 # Python dependencies
└── TESTING.md                       # Testing and troubleshooting guide
```

---

## 📄 File Descriptions

### Core Files (Required)

#### `main.py`
**Purpose**: Backend stock analysis script  
**Language**: Python 3.10+  
**What it does**:
- Defines watchlist of stocks to analyze
- Fetches price data from Yahoo Finance
- Calculates technical indicators (RSI, MACD, SMA, ATR)
- Scrapes news using DuckDuckGo
- Calls Gemini AI for analysis
- Generates `docs/data.json` with results
- Implements rate limiting (4 seconds between API calls)

**Key Sections**:
- `WATCHLIST`: Array of 50+ stock tickers
- `fetch_stock_data()`: Gets price and technical data
- `fetch_news()`: Scrapes recent news headlines
- `analyze_with_gemini()`: AI analysis with structured prompt
- `main()`: Orchestrates the entire analysis process

**Run Locally**:
```powershell
$env:GEMINI_API_KEY='your-key'
python main.py
```

---

#### `requirements.txt`
**Purpose**: Python package dependencies  
**What it does**: Lists all required Python packages for the project

**Packages**:
- `yfinance`: Stock price data from Yahoo Finance
- `pandas`: Data manipulation
- `ta`: Technical analysis indicators
- `google-generativeai`: Gemini AI SDK
- `duckduckgo-search`: News search
- `newspaper3k`: Article text extraction
- `lxml_html_clean`: HTML parsing support
- `requests`: HTTP requests

**Install**:
```powershell
pip install -r requirements.txt
```

---

#### `.github/workflows/manual_scan.yml`
**Purpose**: GitHub Actions workflow configuration  
**Language**: YAML  
**What it does**:
- Defines automated workflow for stock analysis
- Triggered manually via Actions tab or on schedule
- Sets up Python environment
- Installs dependencies
- Runs `main.py` with API key from secrets
- Commits and pushes `docs/data.json` back to repo

**Trigger Methods**:
1. Manual: Actions tab → Run workflow
2. Scheduled: Cron syntax (e.g., daily at midnight)

**Key Steps**:
1. Checkout code
2. Setup Python 3.10
3. Install requirements
4. Run analysis script
5. Commit changes
6. Push to repository

---

#### `docs/index.html`
**Purpose**: Main website HTML structure  
**Language**: HTML5 with Tailwind CSS  
**What it does**:
- Displays stock analysis platform UI
- Loads Tailwind CSS (via CDN)
- Loads Alpine.js for reactivity
- Shows stock cards in a grid
- Provides filter tabs
- Search functionality
- Modal for detailed stock view
- Responsive design

**Key Sections**:
- **Header**: Title, last updated, total assets
- **Search Bar**: Input for ticker search
- **Filter Tabs**: All, Strong Buys, Dividend Gems, Scalp/Day Trade
- **Stock Grid**: Cards displaying each stock
- **Modal**: Detailed view with full analysis
- **Footer**: Credits and disclaimer

**Styling**: Uses Tailwind utility classes

---

#### `docs/script.js`
**Purpose**: Frontend JavaScript logic  
**Language**: JavaScript (ES6+)  
**What it does**:
- Implements Alpine.js app logic
- Loads `data.json` on page load
- Handles filter functionality
- Implements stock search
- Calls Gemini API for live analysis (if stock not cached)
- Rate limiting for API calls
- Modal display logic
- Price formatting utilities

**Key Functions**:
- `init()`: Initialize app, load data
- `loadData()`: Fetch and parse data.json
- `applyFilter()`: Filter stocks based on active tab
- `searchStock()`: Handle stock search
- `fetchGeminiAnalysis()`: Call Gemini API for live analysis
- `showDetails()`: Display stock details in modal
- `getRecommendationClass()`: CSS class for recommendation badges

**Configuration**:
```javascript
const GEMINI_API_KEY = 'YOUR_KEY_HERE'; // ⚠️ Update this!
const RATE_LIMIT_DELAY = 4000; // 4 seconds
```

---

#### `docs/data.json`
**Purpose**: Stock analysis data storage  
**Language**: JSON  
**What it does**:
- Stores all analyzed stock data
- Generated by `main.py`
- Read by `script.js` to display stocks
- Updated each time workflow runs

**Structure**:
```json
{
  "last_updated": "ISO timestamp",
  "total_assets": 50,
  "assets": [
    {
      "ticker": "AAPL",
      "current_price": 185.50,
      "recommendation": "Strong Buy",
      "confidence_score": 85,
      "time_horizon": "Position (Weeks-Months)",
      "reasoning": "...",
      "entry_zone": "$180-$185",
      "take_profit": "$200",
      "stop_loss": "$175",
      "dividend_yield": 0.52,
      "rsi": 58.5,
      "macd": 1.25,
      "news": [...],
      ...
    }
  ]
}
```

---

#### `.gitignore`
**Purpose**: Git ignore patterns  
**What it does**: Specifies files/folders Git should ignore

**Ignores**:
- Python cache (`__pycache__/`)
- Virtual environments (`venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Environment variables (`.env`)
- OS files (`.DS_Store`, `Thumbs.db`)

**Note**: Does NOT ignore `docs/data.json` (needed for website)

---

### Documentation Files

#### `README.md`
**Purpose**: Main project documentation  
**Sections**:
- Project overview
- Features
- Quick start guide
- Setup instructions
- Usage guide
- Configuration options
- Data format
- API limits
- Troubleshooting
- Contributing guidelines
- License and disclaimer

**When to read**: First time setting up the project

---

#### `QUICKSTART.md`
**Purpose**: Simplified 5-step setup guide  
**Target Audience**: Users who want to get started fast  
**Sections**:
- 5-step setup (fork, API key, secrets, pages, run)
- Next steps (customize, schedule)
- Common issues
- Platform usage tips

**When to read**: Want to deploy quickly without deep diving

---

#### `DEPLOYMENT.md`
**Purpose**: Detailed GitHub Pages deployment guide  
**Sections**:
- Two deployment methods (scratch vs fork)
- Step-by-step with screenshots
- Post-deployment configuration
- Updating the platform
- Customization options
- Monitoring
- Security best practices
- Troubleshooting

**When to read**: Ready to deploy to GitHub Pages

---

#### `TESTING.md`
**Purpose**: Testing and troubleshooting guide  
**Sections**:
- Local backend testing
- Local frontend testing
- Search functionality testing
- GitHub Actions testing
- Unit testing examples
- Performance testing
- Common issues and solutions
- CI/CD setup

**When to read**: Testing before deployment or debugging issues

---

#### `CHECKLIST.md`
**Purpose**: Comprehensive setup checklist  
**Sections**:
- Pre-deployment checklist
- Configuration checklist
- Testing checklist
- Security checklist
- Production checklist
- Monitoring checklist

**When to read**: Use as a reference during setup to ensure nothing is missed

---

### Optional/Example Files

#### `config.example.py`
**Purpose**: Sample configuration file  
**What it does**: Shows how to create a custom configuration

**Usage** (optional):
1. Copy to `config.py`
2. Customize values
3. Import in `main.py`

**Not used by default** - configuration is in `main.py`

---

## 🔄 File Relationships

### Data Flow

```
1. GitHub Actions Trigger
   ↓
2. manual_scan.yml runs
   ↓
3. main.py executes
   ↓
4. Fetches data (yfinance, news)
   ↓
5. Calls Gemini API for each stock
   ↓
6. Generates docs/data.json
   ↓
7. Commits and pushes to repo
   ↓
8. GitHub Pages rebuilds site
   ↓
9. index.html loads
   ↓
10. script.js fetches data.json
   ↓
11. Displays stocks on website
```

### User Interaction Flow

```
User visits site
   ↓
script.js loads data.json
   ↓
Displays stock cards
   ↓
User clicks filter
   ↓
applyFilter() runs
   ↓
Filtered stocks displayed
   ↓
User searches for ticker
   ↓
Check if in data.json
   ├─ Yes → Show cached data
   └─ No → Call Gemini API → Show live analysis
```

---

## 📦 File Sizes (Approximate)

| File | Size | Notes |
|------|------|-------|
| `main.py` | ~9 KB | Backend script |
| `index.html` | ~13 KB | Frontend HTML |
| `script.js` | ~10 KB | Frontend JS |
| `data.json` | ~50-200 KB | Depends on watchlist size |
| `requirements.txt` | <1 KB | Dependency list |
| `manual_scan.yml` | ~1 KB | Workflow config |
| **Total (without data.json)** | ~35 KB | Very lightweight! |

---

## 🔑 Critical Files (Do Not Delete)

These files are essential for the platform to work:

1. **main.py** - Without it, no data generation
2. **requirements.txt** - Without it, dependencies won't install
3. **docs/index.html** - Without it, no website
4. **docs/script.js** - Without it, no interactivity
5. **docs/data.json** - Without it, no stock data to display
6. **.github/workflows/manual_scan.yml** - Without it, no automation

---

## 🎨 Customizable Files

Feel free to modify these:

- **main.py** - Change watchlist, prompts, indicators
- **index.html** - Modify design, layout, colors
- **script.js** - Add features, change filters
- **config.example.py** - Create custom config
- **README.md** - Update documentation
- All markdown files - Improve documentation

---

## 🚫 Files NOT Needed in Repository

These should be gitignored:

- `venv/` or `env/` - Virtual environment
- `__pycache__/` - Python cache
- `.env` - Environment variables (use GitHub Secrets instead)
- `config.py` - Custom config (if it contains secrets)
- `.DS_Store` - macOS metadata
- `.vscode/` - VS Code settings (unless you want to share)

---

## 📊 File Dependencies

```
manual_scan.yml
    ↓ triggers
main.py
    ↓ requires
requirements.txt (packages)
    ↓ generates
docs/data.json
    ↓ consumed by
docs/script.js
    ↓ runs in
docs/index.html
```

---

## 🔧 Which File to Edit for...

| Task | File(s) to Edit |
|------|-----------------|
| Add/remove stocks | `main.py` (WATCHLIST) |
| Change AI prompt | `main.py` (analyze_with_gemini) |
| Modify website design | `docs/index.html` |
| Add filter tabs | `docs/index.html` + `docs/script.js` |
| Change colors/theme | `docs/index.html` (CSS styles) |
| Adjust rate limiting | `main.py` (time.sleep) + `docs/script.js` (RATE_LIMIT_DELAY) |
| Schedule automatic runs | `.github/workflows/manual_scan.yml` (cron) |
| Change frontend API key | `docs/script.js` (GEMINI_API_KEY) |
| Add technical indicators | `main.py` (fetch_stock_data) |
| Update documentation | `README.md`, `QUICKSTART.md`, etc. |

---

## 🆘 Backup Your Work

Before making major changes, backup these files:

1. `main.py` - Your watchlist and logic
2. `docs/script.js` - Your frontend logic
3. `docs/index.html` - Your design
4. `.github/workflows/manual_scan.yml` - Your workflow config

**Tip**: Use Git branches for experiments!

```powershell
git checkout -b my-experiment
# Make changes
git add .
git commit -m "Experimental feature"
# If good:
git checkout main
git merge my-experiment
# If bad:
git checkout main
git branch -D my-experiment
```

---

## ✅ File Checklist for Deployment

Before deploying, ensure these files exist and are configured:

- [ ] `main.py` with customized watchlist
- [ ] `requirements.txt` with all dependencies
- [ ] `.github/workflows/manual_scan.yml` with correct workflow
- [ ] `docs/index.html` (no changes needed, but verify it exists)
- [ ] `docs/script.js` with API key (if using search feature)
- [ ] `docs/data.json` (sample or generated)
- [ ] `.gitignore` to protect sensitive files
- [ ] `README.md` for documentation

---

## 📚 Learning Path

**New to the project?** Read files in this order:

1. **README.md** - Understand the project
2. **QUICKSTART.md** - Get started fast
3. **main.py** - Understand backend logic
4. **docs/index.html** - See website structure
5. **docs/script.js** - Understand frontend logic
6. **DEPLOYMENT.md** - Deploy to GitHub Pages
7. **TESTING.md** - Test and troubleshoot
8. **CHECKLIST.md** - Ensure nothing is missed

---

## 🎓 Advanced Topics

Once comfortable, explore:

- **Modifying Gemini prompts** for better analysis
- **Adding custom technical indicators**
- **Creating new filter categories**
- **Implementing caching** for faster performance
- **Setting up serverless proxy** for API security
- **Adding email notifications** for alerts
- **Integrating with other APIs** (news, social sentiment)

---

## 🤝 Contributing

If you improve any files:

1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request
5. Describe your changes clearly

---

## 📞 Support

**Questions about a specific file?**

- Check its section in README.md
- Look for comments in the file itself
- Check TESTING.md for troubleshooting
- Open a GitHub issue with file name in title

---

**Last Updated**: 2026-01-19  
**Project Version**: 1.0.0

---

This project structure is designed to be:
- ✅ Easy to understand
- ✅ Easy to customize
- ✅ Easy to deploy
- ✅ Well documented
- ✅ Maintainable

Happy coding! 🚀📈
