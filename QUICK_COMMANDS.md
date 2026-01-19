# 🚀 Quick Commands - Qwen Model Setup

## Install & Run

```powershell
# 1. Create virtual environment (first time only)
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run analysis (model downloads automatically)
python main.py

# 3. Push to GitHub
git add .
git commit -m "Migrate to Qwen local model"
git push origin main
```

---

## Test with Small Watchlist

Edit `main.py`, find WATCHLIST and change to:

```python
WATCHLIST = ["AAPL", "MSFT", "BTC-USD"]  # Just 3 stocks for testing
```

Then run:
```powershell
# Make sure venv is activated
venv\Scripts\activate

python main.py
```

---

## GPU Setup (10x Faster!)

```powershell
# Make sure venv is activated
venv\Scripts\activate

# Check if you have GPU
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# If yes, install CUDA PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Clear Model Cache (If Issues)

```powershell
# Windows
rmdir /s /q C:\Users\%USERNAME%\.cache\huggingface

# Then rerun
python main.py
```

---

## Common Errors

### Out of Memory
→ Reduce watchlist to 10-20 stocks

### Slow Performance
→ Use GPU or reduce watchlist

### Model Download Fails
→ Clear cache (see above)

---

## What to Expect

### First Run:
- Downloads PyTorch (~2GB)
- Downloads Qwen model (~3GB)
- Total: ~5GB, 10-20 minutes

### Subsequent Runs:
- Loads from cache
- Much faster (seconds)

### Analysis Time:
- **CPU**: ~1 minute per stock
- **GPU**: ~6 seconds per stock
- **GitHub Actions**: ~2 minutes per stock

---

## Files Changed

✅ `main.py` - Uses Qwen model now  
✅ `requirements.txt` - Added transformers, torch  
✅ `docs/index.html` - Shows "Qwen 2.5"  
✅ `docs/script.js` - Search disabled (cache only)  
✅ `.github/workflows/manual_scan.yml` - No API key needed  

---

## Need Help?

📖 **Full Guide**: [SETUP_QWEN.md](SETUP_QWEN.md)  
📖 **Migration Info**: [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)  
📖 **Main Docs**: [README.md](README.md)  

---

**Model**: Qwen 2.5 - 1.5B Instruct  
**No API Keys Required** ✅  
**Free Forever** ✅  
**Privacy** ✅
