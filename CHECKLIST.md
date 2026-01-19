# ✅ Setup Checklist - StockSentinel Interactive

Use this checklist to ensure proper setup and deployment.

---

## 📦 Pre-Deployment Checklist

### Repository Setup
- [ ] Repository created on GitHub
- [ ] All files committed to repository
- [ ] Repository is set to Public or Private (decide based on API key security)

### API Configuration
- [ ] Gemini API key obtained from [Google AI Studio](https://makersuite.google.com/app/apikey)
- [ ] API key tested (works with a simple API call)
- [ ] API key added to GitHub Secrets as `GEMINI_API_KEY`

### File Verification
- [ ] `requirements.txt` exists with all dependencies
- [ ] `main.py` exists with watchlist configured
- [ ] `.github/workflows/manual_scan.yml` exists
- [ ] `docs/index.html` exists
- [ ] `docs/script.js` exists
- [ ] `docs/data.json` exists (sample or generated)
- [ ] `.gitignore` exists to protect sensitive files

---

## 🔧 Configuration Checklist

### Backend Configuration (main.py)
- [ ] `WATCHLIST` array customized with your preferred stocks
- [ ] Indonesian stocks use `.JK` suffix (e.g., `BBCA.JK`)
- [ ] Crypto uses `-USD` suffix (e.g., `BTC-USD`)
- [ ] Rate limiting delay set appropriately (default: 4 seconds)
- [ ] Output directory is `docs` (for GitHub Pages)

### Frontend Configuration (docs/script.js)
- [ ] `GEMINI_API_KEY` configured (if using client-side search)
  - [ ] For private repos: API key can be hardcoded
  - [ ] For public repos: Consider serverless proxy
- [ ] `RATE_LIMIT_DELAY` set to 4000ms (matches backend)
- [ ] API URL is correct (Gemini 1.5 Flash endpoint)

### GitHub Actions Configuration
- [ ] Workflow file is in `.github/workflows/` directory
- [ ] Workflow uses `workflow_dispatch` for manual trigger
- [ ] (Optional) Cron schedule configured for automatic runs
- [ ] Workflow has `contents: write` permission
- [ ] Python version set to 3.10 or higher

### GitHub Pages Configuration
- [ ] GitHub Pages enabled in repository settings
- [ ] Source branch set to `main` (or `master`)
- [ ] Source folder set to `/docs`
- [ ] Custom domain configured (optional)

---

## 🧪 Testing Checklist

### Local Backend Testing
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Environment variable `GEMINI_API_KEY` set
- [ ] Test run with small watchlist (3-5 stocks)
- [ ] `docs/data.json` generated successfully
- [ ] JSON file is valid (no syntax errors)
- [ ] All stocks have required fields

### Local Frontend Testing
- [ ] Local server running (Python HTTP server or Live Server)
- [ ] Website loads without errors
- [ ] Stock cards display correctly
- [ ] All filter tabs work (All, Strong Buys, Dividend Gems, Scalp)
- [ ] Modal opens when clicking stock card
- [ ] All data fields display correctly
- [ ] No console errors in browser DevTools

### Search Functionality Testing
- [ ] API key configured in `script.js`
- [ ] Search for stock in cache (instant results)
- [ ] Search for stock NOT in cache (live API call)
- [ ] Rate limiting works (4-second delay)
- [ ] Error messages display correctly
- [ ] Modal shows analysis results

### GitHub Actions Testing
- [ ] Workflow appears in Actions tab
- [ ] Manual trigger button is visible
- [ ] Workflow runs without errors
- [ ] All workflow steps complete successfully
- [ ] `docs/data.json` is committed and pushed
- [ ] No rate limit errors (15 RPM respected)

### GitHub Pages Testing
- [ ] Site is accessible at `https://USERNAME.github.io/StockSentinel/`
- [ ] Page loads completely (no 404 errors)
- [ ] CSS and JavaScript load correctly
- [ ] Data from `data.json` displays
- [ ] All interactive features work

---

## 🔒 Security Checklist

### API Key Security
- [ ] GitHub Secret `GEMINI_API_KEY` is set (for backend)
- [ ] API key NOT committed to repository in plain text
- [ ] `.gitignore` includes any local config files
- [ ] For public repos: Frontend API key secured via proxy OR users accept risk

### Repository Security
- [ ] Sensitive data NOT in commit history
- [ ] No hardcoded credentials in any file
- [ ] `.env` files (if used) are in `.gitignore`

### Data Privacy
- [ ] No personal financial data in repository
- [ ] Sample data is generic/public stocks
- [ ] Disclaimer included on website

---

## 🚀 Production Checklist

### Pre-Launch
- [ ] All tests pass locally
- [ ] GitHub Actions workflow runs successfully
- [ ] GitHub Pages site loads correctly
- [ ] All links work (no broken links)
- [ ] Mobile responsive (test on phone)
- [ ] Performance acceptable (page load < 3 seconds)

### Launch
- [ ] Final commit with production settings
- [ ] Push to GitHub
- [ ] Run workflow to generate fresh data
- [ ] Verify live site has fresh data
- [ ] Test all features on live site

### Post-Launch
- [ ] Share URL (if public project)
- [ ] Monitor GitHub Actions runs
- [ ] Check API usage in Google AI Studio
- [ ] Watch for errors in workflow logs
- [ ] Update data regularly (manual or scheduled)

---

## 📊 Monitoring Checklist

### Daily (if automated)
- [ ] Check if scheduled workflow ran successfully
- [ ] Verify data.json is up-to-date
- [ ] Check for any workflow errors

### Weekly
- [ ] Review API usage in Google AI Studio
- [ ] Check if approaching free tier limits (1,500/day)
- [ ] Verify all stocks in watchlist still valid
- [ ] Review site analytics (if enabled)

### Monthly
- [ ] Update watchlist with new stocks of interest
- [ ] Remove delisted or problematic tickers
- [ ] Review and optimize Gemini prompts
- [ ] Check for library updates (requirements.txt)

---

## 🐛 Troubleshooting Checklist

### If GitHub Actions Fails
- [ ] Check workflow logs for specific error
- [ ] Verify `GEMINI_API_KEY` secret exists
- [ ] Confirm Python version is 3.10+
- [ ] Check if API quota exceeded
- [ ] Verify requirements.txt is correct
- [ ] Check for network issues

### If Website Doesn't Load
- [ ] Wait 2-3 minutes after enabling Pages
- [ ] Verify GitHub Pages is enabled
- [ ] Check Pages deployment status
- [ ] Verify branch and folder are correct
- [ ] Check for 404 errors in browser DevTools

### If Data Not Updating
- [ ] Run workflow manually
- [ ] Check if workflow completed successfully
- [ ] Verify data.json was committed
- [ ] Clear browser cache
- [ ] Check file permissions on GitHub

### If Search Not Working
- [ ] Verify API key in script.js
- [ ] Check browser console for errors
- [ ] Verify Gemini API endpoint is correct
- [ ] Test with a known valid ticker
- [ ] Check rate limiting (wait 4 seconds between searches)

---

## 📚 Documentation Checklist

### README.md
- [ ] Complete setup instructions
- [ ] All features documented
- [ ] Screenshots/GIFs (optional but helpful)
- [ ] Troubleshooting section
- [ ] License information
- [ ] Disclaimer included

### QUICKSTART.md
- [ ] 5-step quick start guide
- [ ] Common issues and fixes
- [ ] Links to detailed docs

### TESTING.md
- [ ] Local testing instructions
- [ ] Unit testing examples
- [ ] Performance testing tips

### Code Comments
- [ ] main.py has clear comments
- [ ] script.js has clear comments
- [ ] Complex logic is explained

---

## 🎯 Optimization Checklist

### Performance
- [ ] Rate limiting optimized (not too slow, not too fast)
- [ ] Watchlist size reasonable (<100 stocks)
- [ ] Frontend loads quickly (<3 seconds)
- [ ] API calls are efficient

### Cost (Free Tier)
- [ ] Daily API calls < 1,500 (Gemini free tier)
- [ ] RPM < 15 (Gemini rate limit)
- [ ] No unnecessary API calls
- [ ] Caching implemented where possible

### User Experience
- [ ] UI is intuitive
- [ ] Loading states are clear
- [ ] Error messages are helpful
- [ ] Mobile-friendly design
- [ ] Filters work smoothly

---

## ✅ Final Verification

Before announcing your project:

- [ ] Everything above is checked ✅
- [ ] You've personally used the platform
- [ ] Search works for both cached and live stocks
- [ ] No errors in browser console
- [ ] No errors in GitHub Actions
- [ ] API usage is within free tier limits
- [ ] You're comfortable with API key setup (especially for public repos)

---

## 🎉 You're Ready!

If all items are checked, your StockSentinel Interactive platform is ready for use!

### Next Steps:
1. Share your platform (if public)
2. Start analyzing stocks!
3. Iterate and improve based on usage
4. Consider contributing improvements back

**Questions?** Check [README.md](README.md) or open a GitHub Issue.

---

**Last Updated**: 2026-01-19  
**Version**: 1.0.0
