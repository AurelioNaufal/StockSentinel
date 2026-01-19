# 🔄 How to Update & Push to GitHub

## Quick Update Process

### Step 1: Check What Changed
```powershell
# See what files were modified
git status

# See the actual changes
git diff
```

### Step 2: Add Your Changes
```powershell
# Add all changed files
git add .

# Or add specific files
git add main.py
git add docs/script.js
```

### Step 3: Commit Your Changes
```powershell
# Commit with a descriptive message
git commit -m "Fix: Update to Gemini 2.0 Flash Exp model"

# Or for multiple changes
git commit -m "Update Gemini model and fix API endpoint"
```

### Step 4: Push to GitHub
```powershell
# Push to main branch
git push origin main

# Or if it's your default branch
git push
```

---

## 📝 Common Update Scenarios

### Updating the Watchlist
```powershell
# 1. Edit main.py (change WATCHLIST array)
# 2. Save the file
git add main.py
git commit -m "Update stock watchlist"
git push
```

### Fixing Code Issues
```powershell
# 1. Fix the code in your files
# 2. Test locally if possible
git add .
git commit -m "Fix: [describe what you fixed]"
git push
```

### Updating Documentation
```powershell
git add README.md
git commit -m "Update documentation"
git push
```

---

## 🔧 Complete Update Commands (Copy-Paste Ready)

### After Making Changes:
```powershell
git add .
git commit -m "Update: Gemini 2.0 Flash Exp model"
git push origin main
```

### If Push is Rejected:
```powershell
# Pull latest changes first
git pull origin main

# Then push again
git push origin main
```

---

## ✅ Verify Your Update

After pushing:

1. **Check GitHub Repository**
   - Go to your repo on GitHub
   - Verify files are updated
   - Check the commit message appears

2. **Check GitHub Actions**
   - Go to Actions tab
   - See if any workflows are running
   - Check for errors

3. **Check GitHub Pages**
   - Wait 1-2 minutes
   - Visit your site: `https://YOUR_USERNAME.github.io/StockSentinel/`
   - Hard refresh: `Ctrl + Shift + R`

---

## 🚀 Run Analysis After Update

If you updated main.py or workflow files:

1. Go to **Actions** tab on GitHub
2. Click **Manual Stock Analysis Scan**
3. Click **Run workflow**
4. Wait for completion (green checkmark ✅)
5. Check your website for updated data

---

## 🐛 Troubleshooting

### "Permission denied" Error
```powershell
# Make sure you're authenticated
# Use HTTPS with personal access token or SSH with keys
```

### "Nothing to commit" Message
```powershell
# No changes were made, or files not staged
git status  # Check what's going on
```

### "Merge conflict" Error
```powershell
# Pull first, resolve conflicts, then push
git pull origin main
# Fix conflicts in files
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

---

## 📋 Git Cheat Sheet

| Command | What It Does |
|---------|--------------|
| `git status` | See what changed |
| `git diff` | See exact changes |
| `git add .` | Stage all changes |
| `git add filename` | Stage specific file |
| `git commit -m "message"` | Save changes locally |
| `git push` | Upload to GitHub |
| `git pull` | Download from GitHub |
| `git log` | See commit history |

---

## ✨ Best Practices

1. **Commit often** - Small, focused commits are better
2. **Write clear messages** - Explain WHAT and WHY
3. **Test before pushing** - Run locally if possible
4. **Pull before pushing** - Avoid conflicts
5. **Use branches** - For big changes (optional)

### Good Commit Messages:
```
✅ "Fix: Update to Gemini 2.0 Flash Exp model"
✅ "Add: TSLA to watchlist"
✅ "Update: Increase rate limit delay to 5 seconds"

❌ "update"
❌ "fix stuff"
❌ "changes"
```

---

## 🎯 Quick Reference

**Made changes? Run these:**
```powershell
git add .
git commit -m "Describe your changes"
git push
```

**That's it!** ✅

---

**Need more help?** Check git documentation: https://git-scm.com/doc
