# 🚀 Deployment Guide - GitHub Pages

Step-by-step guide to deploy StockSentinel Interactive to GitHub Pages.

---

## 📋 Prerequisites

Before deploying, ensure you have:
- [x] GitHub account
- [x] Git installed on your computer
- [x] Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- [x] All project files ready

---

## 🌐 Method 1: Deploy from Scratch (Recommended)

### Step 1: Create New Repository on GitHub

1. Go to [GitHub](https://github.com)
2. Click the **+** icon (top-right) → **New repository**
3. Fill in:
   - **Repository name**: `StockSentinel` (or your preferred name)
   - **Description**: "AI-powered stock analysis platform"
   - **Visibility**: 
     - **Public** (if you're okay with API key in frontend code)
     - **Private** (recommended for security)
4. **Do NOT** initialize with README, .gitignore, or license (we have our own)
5. Click **Create repository**

### Step 2: Push Your Code to GitHub

Open terminal/PowerShell in your project directory:

```powershell
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - StockSentinel Interactive"

# Add remote (replace YOUR_USERNAME and YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/StockSentinel.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Alternative: Using GitHub Desktop**
1. Open GitHub Desktop
2. File → Add Local Repository
3. Choose your StockSentinel folder
4. Click "Publish repository"
5. Choose Public or Private
6. Click "Publish repository"

### Step 3: Add Gemini API Key to GitHub Secrets

1. Go to your repository on GitHub
2. Click **Settings** tab
3. In left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Enter:
   - **Name**: `GEMINI_API_KEY` (must be exact)
   - **Secret**: Your Gemini API key (e.g., `AIzaSyXXXXXXX...`)
6. Click **Add secret**

✅ Your API key is now securely stored!

### Step 4: Enable GitHub Pages

1. Still in **Settings**, scroll down to **Pages** (left sidebar)
2. Under **Source**:
   - **Branch**: Select `main` (or `master` if that's your default)
   - **Folder**: Select `/docs`
3. Click **Save**

You'll see a message:
```
✓ Your site is ready to be published at https://YOUR_USERNAME.github.io/StockSentinel/
```

**Note**: First deployment takes 1-2 minutes. Subsequent updates are faster.

### Step 5: Run First Analysis

1. Go to **Actions** tab
2. You may see a prompt to enable Actions → Click **Enable**
3. Click on **Manual Stock Analysis Scan** workflow
4. Click **Run workflow** (green button on right)
5. Select branch: `main`
6. Click **Run workflow**

**What happens now:**
- GitHub Actions starts running
- Python environment is set up
- Dependencies are installed
- `main.py` runs and analyzes all stocks
- `docs/data.json` is generated and committed
- Changes are pushed back to repository

**Duration**: 5-10 minutes for full watchlist (50+ stocks)

### Step 6: Verify Deployment

1. Wait for workflow to complete (green checkmark)
2. Go to **Settings** → **Pages**
3. You should see:
   ```
   ✓ Your site is published at https://YOUR_USERNAME.github.io/StockSentinel/
   ```
4. Click the URL to visit your site

**Expected Result**: Your stock analysis platform loads with fresh data! 🎉

---

## 🌐 Method 2: Deploy from Fork

### Step 1: Fork the Repository

1. Go to the original StockSentinel repository
2. Click **Fork** button (top-right)
3. Choose your account
4. Wait for fork to complete

### Step 2: Clone Your Fork

```powershell
git clone https://github.com/YOUR_USERNAME/StockSentinel.git
cd StockSentinel
```

### Step 3: Follow Steps 3-6 from Method 1

(Add API key, enable Pages, run workflow, verify)

---

## 🔧 Post-Deployment Configuration

### Enable Search Functionality

To use the "search any stock" feature:

1. Navigate to your repository on GitHub
2. Go to `docs/script.js`
3. Click **Edit** (pencil icon)
4. Find line ~8:
   ```javascript
   const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE';
   ```
5. Replace with your actual API key:
   ```javascript
   const GEMINI_API_KEY = 'AIzaSyXXXXXXXXXXXXX';
   ```
6. Scroll down and click **Commit changes**

⚠️ **Security Warning**: 
- For **private repos**: This is safe
- For **public repos**: Your API key will be visible in the code
  - Risk is limited (free tier has built-in quotas)
  - For production, use a serverless proxy instead

### Customize Your Watchlist

1. Go to `main.py` on GitHub
2. Click **Edit**
3. Modify the `WATCHLIST` array:
   ```python
   WATCHLIST = [
       # Add your preferred stocks
       "AAPL",
       "TSLA",
       "BBCA.JK",  # Indonesian stocks
       "BTC-USD",  # Crypto
       # ... more tickers
   ]
   ```
4. **Commit changes**
5. Re-run the workflow to generate new data

---

## 🔄 Updating Your Platform

### Manual Updates

Every time you want fresh data:
1. Go to **Actions** tab
2. Click **Manual Stock Analysis Scan**
3. Click **Run workflow**
4. Wait for completion
5. Refresh your website

### Automatic Updates (Scheduled)

Edit `.github/workflows/manual_scan.yml`:

1. Click **Edit** on GitHub
2. Modify the `on:` section:
   ```yaml
   on:
     workflow_dispatch:  # Keep manual trigger
     schedule:
       - cron: '0 0 * * *'  # Add this: Daily at midnight UTC
   ```
3. **Commit changes**

**Cron Schedule Examples:**
```yaml
- cron: '0 */6 * * *'    # Every 6 hours
- cron: '0 9 * * 1-5'    # Weekdays at 9 AM UTC
- cron: '0 0,12 * * *'   # Twice daily (midnight & noon)
```

Use [Crontab Guru](https://crontab.guru/) to create custom schedules.

---

## 🎨 Customization Options

### Change Theme Colors

Edit `docs/index.html`:

```html
<!-- Find the gradient classes -->
<style>
    .gradient-bg {
        background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
    }
</style>
```

### Add More Filters

Edit `docs/index.html` and `docs/script.js`:

1. Add button in HTML:
   ```html
   <button @click="activeFilter = 'new-filter'">
       🚀 New Filter
   </button>
   ```

2. Add filter logic in script.js:
   ```javascript
   case 'new-filter':
       this.filteredStocks = this.stocks.filter(s => 
           // Your filter condition
       );
       break;
   ```

### Modify Card Layout

Edit `docs/index.html`:
- Change grid columns: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`
- Adjust spacing: `gap-6`
- Modify card design in the `<template>` section

---

## 📊 Monitoring Your Deployment

### Check Workflow Status

**Actions Tab:**
- ✅ Green checkmark = Success
- ❌ Red X = Failed
- 🟡 Yellow circle = Running

**Click on a workflow run to see:**
- Detailed logs
- Each step's output
- Any error messages

### Monitor API Usage

1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Check your API usage dashboard
3. Monitor:
   - Requests per day
   - Requests per minute
   - Quota remaining

**Free Tier Limits:**
- 15 requests per minute (RPM)
- 1,500 requests per day

**Tip**: With 50 stocks and 4-second delays:
- Time: ~3-4 minutes per run
- API calls: ~50 per run
- Daily capacity: ~30 runs per day

### Check Site Performance

**GitHub Pages Status:**
- Settings → Pages shows deployment status
- Green checkmark = Live
- Build time usually <1 minute

**Test Your Site:**
- Visit your URL
- Open DevTools (F12)
- Check Console for errors
- Check Network tab for failed requests

---

## 🐛 Common Deployment Issues

### Issue: "Workflow not found"

**Fix:**
1. Ensure `.github/workflows/manual_scan.yml` exists
2. File must be in exact path
3. YAML syntax must be valid (no tabs, proper indentation)

### Issue: "Secret not found"

**Fix:**
1. Verify secret name is exactly `GEMINI_API_KEY`
2. Check in Settings → Secrets and variables → Actions
3. Re-add secret if necessary

### Issue: "Permission denied" when pushing to GitHub

**Fix:**
1. Use HTTPS: `https://github.com/USER/REPO.git`
2. Or set up SSH keys: [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
3. Verify repository permissions

### Issue: "GitHub Pages not updating"

**Fix:**
1. Check if workflow committed changes
2. Clear browser cache (Ctrl+Shift+R)
3. Wait 1-2 minutes for Pages to rebuild
4. Check Settings → Pages for deployment errors

### Issue: "Rate limit exceeded"

**Fix:**
1. You hit Gemini's 15 RPM limit
2. Increase delay in main.py: `time.sleep(5)` or more
3. Reduce watchlist size
4. Wait and try again later

---

## 🔒 Security Best Practices

### For Private Repositories
- ✅ API key in script.js is relatively safe
- ✅ Only you and collaborators can see code
- ✅ GitHub Pages site is private (only you can access)

### For Public Repositories
- ⚠️ API key in script.js is visible to everyone
- ✅ Gemini free tier has built-in quotas (safety net)
- ✅ Backend API key (GitHub Secret) is always secure
- 🔐 **Recommended**: Use serverless proxy for frontend API

### Serverless Proxy Setup (Advanced)

**Vercel Function Example:**

1. Create `api/analyze.js`:
   ```javascript
   export default async function handler(req, res) {
       const { ticker } = req.body;
       const apiKey = process.env.GEMINI_API_KEY;
       
       // Call Gemini API server-side
       const response = await fetch(GEMINI_URL, {
           method: 'POST',
           headers: { /* ... */ },
           body: JSON.stringify({ /* ... */ })
       });
       
       const data = await response.json();
       res.json(data);
   }
   ```

2. Deploy to Vercel
3. Update `docs/script.js` to call your proxy
4. API key stays secure on server!

---

## ✅ Deployment Checklist

- [ ] Repository created on GitHub
- [ ] Code pushed to main branch
- [ ] `GEMINI_API_KEY` added to GitHub Secrets
- [ ] GitHub Pages enabled (source: main branch, /docs folder)
- [ ] First workflow run completed successfully
- [ ] `docs/data.json` generated and committed
- [ ] Website accessible at GitHub Pages URL
- [ ] All features working (filters, cards, modals)
- [ ] Search configured (API key in script.js)
- [ ] (Optional) Scheduled runs configured

---

## 🎉 You're Live!

Congratulations! Your StockSentinel Interactive platform is now live on GitHub Pages!

**Your URL**: `https://YOUR_USERNAME.github.io/StockSentinel/`

**Share it:**
- Tweet about it
- Share on LinkedIn
- Post on Reddit (r/algotrading, r/stocks)
- Add to your portfolio

**Next Steps:**
- Monitor workflow runs
- Update watchlist as needed
- Customize design
- Add new features
- Get feedback from users

---

## 📚 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Markdown Guide](https://www.markdownguide.org/)

---

**Need Help?** Open an issue on GitHub or check [TESTING.md](TESTING.md) for troubleshooting.

**Happy Deploying! 🚀📈**
