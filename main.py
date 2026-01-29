import os
import json
import time
import yfinance as yf
import pandas as pd
import ta
from duckduckgo_search import DDGS
from newspaper import Article
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Configure Hugging Face Model (Qwen 2.5 - 3B Instruct)
print("Loading Qwen 2.5 - 3B Instruct model (better accuracy)...")
model_name = "Qwen/Qwen2.5-3B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map=None,
    low_cpu_mem_usage=True
)

model = model.to('cpu')
print("Model loaded successfully! Running on CPU.\n")

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
    "BBTN.JK",  # Bank Tabungan Negara
    "ADRO.JK",  # Adaro Energy
    "ANTM.JK",  # Aneka Tambang
    "INCO.JK",  # Vale Indonesia
    "AALI.JK",  # Astra Agro Lestari
    "INDR.JK",  # Indorama Synthetics
    "SIDO.JK",  # Sido Muncul
    "SSIA.JK",  # Surya Semesta Internusa
    "MBMA.JK",  # Mitrabara Adiperdana
    "TINS.JK",  # Timah
    "AMRT.JK",  # Sumber Alfaria Trijaya (Alfamart)
    "BUMI.JK",  # Bumi Resources
    "HEXA.JK",  # Hexindo Adiperkasa
    "BACA.JK",  # Bank Capital Indonesia
    "NRCA.JK",  # Nusa Raya Cipta
    "HALO.JK",  # Halo Teknologi Indonesia
    "DOID.JK",  # Delta Dunia Makmur
    "BLES.JK",  # Berkah Beton Sadaya
    
    # US Stocks
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Google
    "AMZN",     # Amazon
    "NVDA",     # Nvidia
    "TSLA",     # Tesla
    "META",     # Meta
    "V",        # Visa
    "WMT",      # Walmart
    "DIS",      # Disney
    "NFLX",     # Netflix
    
    # Crypto (via Yahoo Finance)
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "XRP-USD",  # Ripple
    "SOL-USD",  # Solana
    "GC=F",     # Gold Futures (XAU)
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
        
        # Determine currency based on ticker
        currency = 'IDR' if ticker.endswith('.JK') else 'USD'
        
        # Get dividend info
        dividend_yield = 0
        try:
            info = stock.info
            dividend_yield = info.get('dividendYield', 0)
            # Already in decimal form (e.g., 0.05 = 5%) no need to convert
        except:
            pass
        
        # Prepare historical data for different timeframes
        hist_1m = stock.history(period="1mo")
        hist_6m = stock.history(period="6mo")
        hist_max = stock.history(period="max")
        
        chart_data = {
            '1M': [],
            '6M': [],
            'MAX': []
        }
        
        # Process each timeframe
        for period_key, hist_period in [('1M', hist_1m), ('6M', hist_6m), ('MAX', hist_max)]:
            if not hist_period.empty:
                for idx in range(len(hist_period)):
                    row = hist_period.iloc[idx]
                    # Calculate SMA 20 if enough data
                    sma_20_val = None
                    if idx >= 19:
                        sma_20_val = round(hist_period['Close'].iloc[max(0, idx-19):idx+1].mean(), 2)
                    
                    chart_data[period_key].append({
                        'date': hist_period.index[idx].strftime('%Y-%m-%d'),
                        'price': round(row['Close'], 2),
                        'sma_20': sma_20_val
                    })
        
        data = {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'currency': currency,
            'rsi': round(latest['RSI'], 2) if pd.notna(latest['RSI']) else None,
            'macd': round(latest['MACD'], 2) if pd.notna(latest['MACD']) else None,
            'macd_signal': round(latest['MACD_Signal'], 2) if pd.notna(latest['MACD_Signal']) else None,
            'sma_20': round(latest['SMA_20'], 2) if pd.notna(latest['SMA_20']) else None,
            'sma_50': round(latest['SMA_50'], 2) if pd.notna(latest['SMA_50']) else None,
            'atr': round(latest['ATR'], 2) if pd.notna(latest['ATR']) else None,
            'dividend_yield': round(dividend_yield, 2) if dividend_yield else 0,
            'volume': int(latest['Volume']),
            'chart_data': chart_data
        }
        
        return data
        
    except Exception as e:
        print(f"  Error fetching data for {ticker}: {e}")
        return None


def fetch_news(ticker):
    """Fetch recent news headlines and snippets for a ticker."""
    try:
        # Clean ticker for search
        if ticker.endswith('.JK'):
            # Indonesian stocks - use company name mapping for better news
            stock_names = {
                'BBCA.JK': 'Bank BCA Indonesia',
                'BBRI.JK': 'Bank BRI Indonesia',
                'BMRI.JK': 'Bank Mandiri Indonesia',
                'TLKM.JK': 'Telkom Indonesia',
                'ASII.JK': 'Astra International',
                'UNVR.JK': 'Unilever Indonesia',
                'ICBP.JK': 'Indofood CBP',
                'INDF.JK': 'Indofood',
                'KLBF.JK': 'Kalbe Farma',
                'GGRM.JK': 'Gudang Garam',
                'BBTN.JK': 'BTN Bank',
                'ACES.JK': 'Ace Hardware Indonesia',
                'ADRO.JK': 'Adaro Energy',
                'ANTM.JK': 'Aneka Tambang',
                'INCO.JK': 'Vale Indonesia'
            }
            search_query = stock_names.get(ticker, ticker.replace('.JK', '')) + ' saham'
        else:
            search_query = f"{ticker} stock"
        
        # Search for news
        ddgs = DDGS()
        results = ddgs.news(search_query, max_results=10)
        
        news_items = []
        for result in results[:3]:  # Limit to 3 articles for speed
            news_items.append({
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('body', '')
            })
        
        return news_items
        
    except Exception as e:
        print(f"  Error fetching news for {ticker}: {e}")
        return []


def predict_price_trend(ticker, stock_data):
    """Predict 6-month price trend using Gradient Boosting with comprehensive feature engineering."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        
        # Get 2 years of historical data for better pattern recognition
        stock = yf.Ticker(ticker)
        hist_2y = stock.history(period="2y")
        
        if len(hist_2y) < 100:
            return None
        
        # Advanced feature engineering
        def create_features(data, lookback=60):
            features = pd.DataFrame()
            close = data['Close'].values
            volume = data['Volume'].values
            high = data['High'].values
            low = data['Low'].values
            
            # Price-based features
            features['price'] = close
            features['log_price'] = np.log(close + 1)
            
            # Multiple timeframe moving averages
            for window in [5, 10, 20, 50]:
                features[f'ma{window}'] = pd.Series(close).rolling(window=window, min_periods=1).mean().values
                features[f'price_to_ma{window}'] = close / features[f'ma{window}']
            
            # Exponential moving averages (more weight on recent prices)
            for span in [12, 26]:
                features[f'ema{span}'] = pd.Series(close).ewm(span=span, adjust=False).mean().values
            
            # Volatility features
            features['volatility_20'] = pd.Series(close).rolling(window=20, min_periods=1).std().values
            features['volatility_50'] = pd.Series(close).rolling(window=50, min_periods=1).std().values
            
            # Momentum indicators
            for period in [5, 10, 20]:
                features[f'momentum_{period}'] = pd.Series(close).pct_change(period).fillna(0).values
                features[f'roc_{period}'] = ((close - np.roll(close, period)) / np.roll(close, period))
            
            # Rate of change of momentum (acceleration)
            features['momentum_acceleration'] = pd.Series(features['momentum_10']).diff().fillna(0).values
            
            # RSI-like momentum
            returns = pd.Series(close).pct_change().fillna(0)
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.rolling(window=14, min_periods=1).mean()
            avg_loss = losses.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            features['rsi'] = (100 - (100 / (1 + rs))).values
            
            # Volume features
            features['volume'] = volume
            features['volume_ma5'] = pd.Series(volume).rolling(window=5, min_periods=1).mean().values
            features['volume_ratio'] = volume / (features['volume_ma5'] + 1)
            
            # Price range and spread
            features['high_low_range'] = high - low
            features['high_low_ratio'] = high / (low + 1)
            
            # Trend strength (ADX-like)
            features['trend_strength'] = np.abs(features['ma5'] - features['ma20']) / close
            
            # Cyclical features (time patterns)
            day_of_year = data.index.dayofyear.values
            features['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            features['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
            
            return features
        
        # Create comprehensive features
        features_df = create_features(hist_2y)
        
        # Create sequences for multi-step prediction
        sequence_length = 20  # Use last 20 days to predict next day
        X_sequences = []
        y_targets = []
        
        for i in range(sequence_length, len(features_df)):
            # Use statistical aggregations of recent sequence
            sequence = features_df.iloc[i-sequence_length:i]
            
            # Create aggregated features from sequence
            agg_features = []
            for col in sequence.columns:
                if col != 'price':
                    agg_features.extend([
                        sequence[col].mean(),
                        sequence[col].std(),
                        sequence[col].iloc[-1],  # Latest value
                    ])
            
            X_sequences.append(agg_features)
            y_targets.append(features_df.iloc[i]['price'])
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Gradient Boosting (better than Random Forest for time series)
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Multi-step prediction with model updates
        predictions = []
        prediction_upper = []
        prediction_lower = []
        prediction_dates = []
        last_date = hist_2y.index[-1]
        
        # Get baseline statistics for constraints
        base_price = stock_data['current_price']
        historical_returns = pd.Series(hist_2y['Close'].values).pct_change().dropna()
        daily_volatility = historical_returns.std()
        
        # Start with last known features
        current_features = features_df.tail(sequence_length).copy()
        
        for i in range(126):
            # Prepare sequence features
            sequence = current_features.tail(sequence_length)
            agg_features = []
            for col in sequence.columns:
                if col != 'price':
                    agg_features.extend([
                        sequence[col].mean(),
                        sequence[col].std(),
                        sequence[col].iloc[-1],
                    ])
            
            # Predict next price
            X_future = scaler.transform([agg_features])
            pred_price = model.predict(X_future)[0]
            
            # Apply soft constraint to prevent exponential growth
            # Allow max ±25% deviation from base price over 6 months
            max_deviation = base_price * 0.25
            time_factor = (i + 1) / 126  # 0 to 1 over 6 months
            max_price = base_price + (max_deviation * time_factor)
            min_price = base_price - (max_deviation * time_factor)
            
            # Apply constraint with smoothing
            if pred_price > max_price:
                pred_price = max_price - (max_price - pred_price) * 0.1  # Soft cap
            elif pred_price < min_price:
                pred_price = min_price + (pred_price - min_price) * 0.1  # Soft floor
            
            predictions.append(pred_price)
            
            # FIXED uncertainty band - does NOT grow over time to prevent Y-axis expansion
            # Use a constant 3% band around predictions
            uncertainty = base_price * 0.03  # Fixed 3% of base price
            
            # Apply bounds to BOTH prediction and uncertainty
            upper_bound = pred_price + uncertainty
            lower_bound = pred_price - uncertainty
            
            # CRITICAL: Enforce absolute bounds to prevent exponential growth
            # Bounds should never exceed ±20% from base price at any point
            max_allowed = base_price * 1.20  # +20% max
            min_allowed = base_price * 0.80  # -20% min
            
            upper_bound = min(upper_bound, max_allowed)
            lower_bound = max(lower_bound, min_allowed)
            
            # Ensure bounds don't cross or get inverted
            if upper_bound < lower_bound:
                upper_bound = lower_bound = pred_price
            
            prediction_upper.append(upper_bound)
            prediction_lower.append(lower_bound)
            
            # Calculate date
            days_ahead = i + 1
            approx_date = last_date + pd.Timedelta(days=int(days_ahead * 365 / 252))
            prediction_dates.append(approx_date.strftime('%Y-%m-%d'))
            
            # Update features with prediction
            new_row = pd.Series(dtype=float)
            new_row['price'] = pred_price
            new_row['log_price'] = np.log(pred_price + 1)
            
            # Update MAs based on recent predictions
            recent_prices = list(current_features.tail(50)['price']) + predictions[-50:]
            for window in [5, 10, 20, 50]:
                ma_val = np.mean(recent_prices[-window:])
                new_row[f'ma{window}'] = ma_val
                new_row[f'price_to_ma{window}'] = pred_price / ma_val
            
            # Update EMAs
            for span in [12, 26]:
                ema_val = pd.Series(recent_prices[-span:]).ewm(span=span, adjust=False).mean().iloc[-1]
                new_row[f'ema{span}'] = ema_val
            
            # Update volatility
            new_row['volatility_20'] = np.std(recent_prices[-20:])
            new_row['volatility_50'] = np.std(recent_prices[-50:]) if len(recent_prices) >= 50 else np.std(recent_prices)
            
            # Update momentum
            for period in [5, 10, 20]:
                if len(recent_prices) > period:
                    momentum = (recent_prices[-1] - recent_prices[-period-1]) / recent_prices[-period-1]
                    momentum = np.clip(momentum, -0.05, 0.05)  # Cap momentum at ±5%
                    new_row[f'momentum_{period}'] = momentum
                    new_row[f'roc_{period}'] = momentum
                else:
                    new_row[f'momentum_{period}'] = 0
                    new_row[f'roc_{period}'] = 0
            
            # Fill other features
            new_row['momentum_acceleration'] = 0
            new_row['rsi'] = 50  # Neutral
            new_row['volume'] = current_features['volume'].mean()
            new_row['volume_ma5'] = current_features['volume_ma5'].mean()
            new_row['volume_ratio'] = 1.0
            new_row['high_low_range'] = current_features['high_low_range'].mean()
            new_row['high_low_ratio'] = current_features['high_low_ratio'].mean()
            new_row['trend_strength'] = np.abs(new_row['ma5'] - new_row['ma20']) / pred_price
            
            # Time features
            day_of_year = (approx_date.timetuple().tm_yday)
            new_row['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            new_row['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)
            
            # Append and maintain sequence length
            current_features = pd.concat([current_features, pd.DataFrame([new_row])], ignore_index=True)
            if len(current_features) > sequence_length:
                current_features = current_features.tail(sequence_length)
        
        current_price = stock_data['current_price']
        predicted_6m = predictions[-1]
        price_change_pct = ((predicted_6m - current_price) / current_price) * 100
        
        # Backtesting with walk-forward validation
        backtest_results = None
        if len(X) >= 150:
            train_size = len(X) - 63
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            backtest_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            backtest_model.fit(X_train, y_train)
            predicted_test = backtest_model.predict(X_test)
            
            mape = np.mean(np.abs((y_test - predicted_test) / y_test)) * 100
            accuracy = max(0, 100 - mape)
            
            backtest_dates = [last_date - pd.Timedelta(days=(63-i)*365//252) for i in range(len(y_test))]
            backtest_results = {
                'dates': [d.strftime('%Y-%m-%d') for d in backtest_dates],
                'actual': [round(float(val), 2) for val in y_test],
                'predicted': [round(float(val), 2) for val in predicted_test],
                'mape': round(mape, 2),
                'accuracy': round(accuracy, 2)
            }
        else:
            accuracy = None
            
            # Prepare backtest data for visualization
            backtest_dates = [hist_1y.index[train_size + i].strftime('%Y-%m-%d') for i in range(len(y_test))]
            backtest_results = {
                'dates': backtest_dates,
                'actual': [round(float(val), 2) for val in y_test],
                'predicted': [round(float(val), 2) for val in predicted_test],
                'mape': round(mape, 2),
                'accuracy': round(accuracy, 2)
            }
        
        # Determine trend
        if price_change_pct > 15:
            trend = "Strong Uptrend"
        elif price_change_pct > 5:
            trend = "Uptrend"
        elif price_change_pct > -5:
            trend = "Sideways"
        elif price_change_pct > -15:
            trend = "Downtrend"
        else:
            trend = "Strong Downtrend"
        
        # Prepare prediction graph data (sample every 5 days to reduce data size)
        prediction_graph = []
        for i in range(0, len(predictions), 5):  # Every 5 days
            prediction_graph.append({
                'date': prediction_dates[i],
                'price': round(predictions[i], 2),
                'upper': round(prediction_upper[i], 2),
                'lower': round(prediction_lower[i], 2)
            })
        # Always include the last prediction
        if len(predictions) % 5 != 1:
            prediction_graph.append({
                'date': prediction_dates[-1],
                'price': round(predictions[-1], 2),
                'upper': round(prediction_upper[-1], 2),
                'lower': round(prediction_lower[-1], 2)
            })
        
        return {
            'predicted_6m_price': round(predicted_6m, 2),
            'price_change_pct': round(price_change_pct, 2),
            'trend': trend,
            'backtest_accuracy': round(accuracy, 2) if accuracy else None,
            'prediction_graph': prediction_graph,  # Array of {date, price}
            'backtest_results': backtest_results   # Actual vs predicted with MAPE
        }
        
    except Exception as e:
        print(f"  Prediction error: {e}")
        return None


def calculate_fundamental_score(ticker, stock_data):
    """Calculate fundamental score 0-100 including comprehensive valuation metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        scores = []
        
        # VALUATION METRICS (40% weight - most important for buy/sell decisions)
        
        # 1. P/E Ratio (lower is better, typical range 10-30)
        pe = info.get('trailingPE', None) or info.get('forwardPE', None)
        if pe and pe > 0:
            if pe < 15:
                scores.append(95)  # Undervalued
            elif pe < 25:
                scores.append(70)  # Fair
            elif pe < 40:
                scores.append(45)  # Expensive
            else:
                scores.append(25)  # Overvalued
        
        # 2. P/B Ratio (Price to Book) - lower is better
        pb = info.get('priceToBook', None)
        if pb and pb > 0:
            if pb < 1:
                scores.append(95)  # Undervalued (trading below book value)
            elif pb < 3:
                scores.append(70)  # Fair
            elif pb < 5:
                scores.append(45)  # Expensive
            else:
                scores.append(25)  # Overvalued
        
        # 3. PEG Ratio (P/E to Growth) - best overall valuation metric
        peg = info.get('pegRatio', None)
        if peg and peg > 0:
            if peg < 1:
                scores.append(95)  # Undervalued (growth justifies price)
            elif peg < 2:
                scores.append(70)  # Fair
            elif peg < 3:
                scores.append(45)  # Expensive
            else:
                scores.append(25)  # Overvalued
        
        # 4. P/S Ratio (Price to Sales) - lower is better
        ps = info.get('priceToSalesTrailing12Months', None)
        if ps and ps > 0:
            if ps < 1:
                scores.append(90)  # Undervalued
            elif ps < 3:
                scores.append(70)  # Fair
            elif ps < 5:
                scores.append(45)  # Expensive
            else:
                scores.append(25)  # Overvalued
        
        # PROFITABILITY METRICS (30% weight)
        
        # 5. ROE (Return on Equity) - higher is better
        roe = info.get('returnOnEquity', None)
        if roe:
            roe_pct = roe * 100
            if roe_pct > 20:
                scores.append(90)
            elif roe_pct > 15:
                scores.append(75)
            elif roe_pct > 10:
                scores.append(60)
            else:
                scores.append(40)
        
        # 6. Profit Margins (higher is better)
        profit_margin = info.get('profitMargins', None)
        if profit_margin:
            margin_pct = profit_margin * 100
            if margin_pct > 20:
                scores.append(90)
            elif margin_pct > 10:
                scores.append(70)
            elif margin_pct > 5:
                scores.append(50)
            else:
                scores.append(30)
        
        # FINANCIAL HEALTH (20% weight)
        
        # 7. Debt to Equity (lower is better)
        debt_to_equity = info.get('debtToEquity', None)
        if debt_to_equity is not None:
            if debt_to_equity < 50:
                scores.append(90)
            elif debt_to_equity < 100:
                scores.append(70)
            elif debt_to_equity < 200:
                scores.append(50)
            else:
                scores.append(30)
        
        # GROWTH METRICS (10% weight)
        
        # 8. Revenue Growth (YoY)
        revenue_growth = info.get('revenueGrowth', None)
        if revenue_growth:
            growth_pct = revenue_growth * 100
            if growth_pct > 20:
                scores.append(90)
            elif growth_pct > 10:
                scores.append(75)
            elif growth_pct > 0:
                scores.append(60)
            else:
                scores.append(40)
        
        # 9. Dividend Yield (bonus for income investors)
        if stock_data.get('dividend_yield', 0) > 3:
            scores.append(80)
        elif stock_data.get('dividend_yield', 0) > 1:
            scores.append(60)
        
        # Calculate weighted average
        if scores:
            base_score = round(np.mean(scores), 2)
            
            # Apply valuation status modifier (if we have it)
            valuation_status, _ = calculate_valuation_status(ticker)
            
            # Boost/penalize based on overall valuation
            if valuation_status == 'Undervalued':
                base_score = min(100, base_score + 5)  # +5 bonus
            elif valuation_status == 'Overvalued':
                base_score = max(0, base_score - 5)  # -5 penalty
            
            return round(base_score, 2)
        else:
            return 50  # Neutral if no data
            
    except Exception as e:
        print(f"  Fundamental scoring error: {e}")
        return 50


def calculate_valuation_status(ticker):
    """Determine if stock is undervalued, overvalued, or fairly valued based on multiple ratios."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        valuation_signals = []
        details = {}
        
        # 1. P/E Ratio Analysis
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if pe_ratio and pe_ratio > 0:
            details['pe_ratio'] = round(pe_ratio, 2)
            if pe_ratio < 15:
                valuation_signals.append(-1)  # Undervalued
                details['pe_signal'] = 'Low (Undervalued)'
            elif pe_ratio > 25:
                valuation_signals.append(1)  # Overvalued
                details['pe_signal'] = 'High (Overvalued)'
            else:
                valuation_signals.append(0)  # Fair
                details['pe_signal'] = 'Fair'
        
        # 2. P/B Ratio (Price-to-Book)
        pb_ratio = info.get('priceToBook')
        if pb_ratio and pb_ratio > 0:
            details['pb_ratio'] = round(pb_ratio, 2)
            if pb_ratio < 1:
                valuation_signals.append(-1)  # Undervalued
                details['pb_signal'] = 'Low (Undervalued)'
            elif pb_ratio > 3:
                valuation_signals.append(1)  # Overvalued
                details['pb_signal'] = 'High (Overvalued)'
            else:
                valuation_signals.append(0)
                details['pb_signal'] = 'Fair'
        
        # 3. PEG Ratio (P/E to Growth) - Best indicator if growth is positive
        peg_ratio = info.get('pegRatio')
        if peg_ratio and peg_ratio > 0:
            details['peg_ratio'] = round(peg_ratio, 2)
            if peg_ratio < 1:
                valuation_signals.append(-1)  # Undervalued
                details['peg_signal'] = 'Low (Undervalued)'
            elif peg_ratio > 2:
                valuation_signals.append(1)  # Overvalued
                details['peg_signal'] = 'High (Overvalued)'
            else:
                valuation_signals.append(0)
                details['peg_signal'] = 'Fair'
        
        # 4. Price to Sales
        ps_ratio = info.get('priceToSalesTrailing12Months')
        if ps_ratio and ps_ratio > 0:
            details['ps_ratio'] = round(ps_ratio, 2)
            if ps_ratio < 1:
                valuation_signals.append(-1)
                details['ps_signal'] = 'Low (Undervalued)'
            elif ps_ratio > 3:
                valuation_signals.append(1)
                details['ps_signal'] = 'High (Overvalued)'
            else:
                valuation_signals.append(0)
                details['ps_signal'] = 'Fair'
        
        # Calculate overall valuation status
        if len(valuation_signals) == 0:
            return 'Fair-Valued', details
        
        avg_signal = sum(valuation_signals) / len(valuation_signals)
        
        if avg_signal < -0.3:
            status = 'Undervalued'
        elif avg_signal > 0.3:
            status = 'Overvalued'
        else:
            status = 'Fair-Valued'
        
        return status, details
    
    except Exception as e:
        print(f"  Valuation calculation error: {e}")
        return 'Fair-Valued', {}


def calculate_technical_score(stock_data):
    """Calculate technical score 0-100 based on indicators."""
    scores = []
    
    # 1. RSI Score (30-70 is ideal, extremes are warning signs)
    rsi = stock_data.get('rsi')
    if rsi:
        if 40 <= rsi <= 60:
            scores.append(80)  # Neutral zone - good
        elif 30 <= rsi < 40:
            scores.append(90)  # Oversold - buy opportunity
        elif 60 < rsi <= 70:
            scores.append(60)  # Mildly overbought
        elif rsi < 30:
            scores.append(95)  # Very oversold - strong buy signal
        elif rsi > 70:
            scores.append(40)  # Overbought - caution
        else:
            scores.append(50)
    
    # 2. MACD Signal (bullish crossover is positive)
    macd = stock_data.get('macd')
    macd_signal = stock_data.get('macd_signal')
    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd > 0:
            scores.append(85)  # Bullish momentum
        elif macd > macd_signal:
            scores.append(70)  # Gaining strength
        elif macd < macd_signal and macd < 0:
            scores.append(40)  # Bearish momentum
        else:
            scores.append(55)
    
    # 3. Price vs Moving Averages
    price = stock_data.get('current_price')
    sma_20 = stock_data.get('sma_20')
    sma_50 = stock_data.get('sma_50')
    
    if price and sma_20 and sma_50:
        if price > sma_20 > sma_50:
            scores.append(85)  # Strong uptrend
        elif price > sma_20:
            scores.append(70)  # Above short-term MA
        elif price < sma_20 < sma_50:
            scores.append(40)  # Downtrend
        elif price < sma_20:
            scores.append(90)  # Below MA - potential buy
        else:
            scores.append(60)
    
    # 4. Volume Analysis (higher volume = more conviction)
    volume = stock_data.get('volume', 0)
    if volume > 0:
        # Placeholder: ideally compare to average volume
        scores.append(65)
    
    # Average all scores
    if scores:
        return round(np.mean(scores), 2)
    else:
        return 50


def calculate_composite_score(sentiment_score, fundamental_score, technical_score):
    """Calculate weighted composite score (33% sentiment, 33% fundamental, 34% technical)."""
    composite = (
        sentiment_score * 0.33 +      # Sentiment: 33%
        fundamental_score * 0.33 +     # Fundamentals: 33%
        technical_score * 0.34         # Technicals: 34%
    )
    
    # Determine suggested recommendation based on composite score
    if composite >= 80:
        suggested_rec = "Strong Buy"
    elif composite >= 65:
        suggested_rec = "Buy"
    elif composite >= 45:
        suggested_rec = "Hold"
    elif composite >= 30:
        suggested_rec = "Sell"
    else:
        suggested_rec = "Strong Sell"
    
    return round(composite, 2), suggested_rec


def analyze_with_qwen(ticker, stock_data, news):
    """Use Qwen model with AI-powered composite scoring to analyze stock."""
    try:
        # Calculate AI scores (fundamental + technical, sentiment from Qwen)
        print(f"  Calculating AI scores...")
        fundamental_score = calculate_fundamental_score(ticker, stock_data)
        technical_score = calculate_technical_score(stock_data)
        
        # Calculate valuation status
        print(f"  Analyzing valuation...")
        valuation_status, valuation_details = calculate_valuation_status(ticker)
        print(f"  Valuation: {valuation_status}")
        
        # Predict price trend (6 months)
        print(f"  Predicting 6-month price trend...")
        prediction = predict_price_trend(ticker, stock_data)
        
        print(f"  Scores - Fundamental: {fundamental_score}, Technical: {technical_score}")
        if prediction:
            print(f"  Prediction: {prediction['trend']} ({prediction['price_change_pct']:+.2f}%) to {prediction['predicted_6m_price']}")
            if prediction['backtest_accuracy']:
                print(f"  Backtest accuracy: {prediction['backtest_accuracy']:.2f}%")
        
        # Get currency symbol
        currency = stock_data.get('currency', 'USD')
        currency_symbol = 'Rp' if currency == 'IDR' else '$'
        
        # Format valuation details
        valuation_text = f"\nVALUATION ANALYSIS ({valuation_status}):\n"
        if valuation_details:
            if 'pe_ratio' in valuation_details:
                valuation_text += f"- P/E Ratio: {valuation_details['pe_ratio']} - {valuation_details.get('pe_signal', 'N/A')}\n"
            if 'pb_ratio' in valuation_details:
                valuation_text += f"- P/B Ratio: {valuation_details['pb_ratio']} - {valuation_details.get('pb_signal', 'N/A')}\n"
            if 'peg_ratio' in valuation_details:
                valuation_text += f"- PEG Ratio: {valuation_details['peg_ratio']} - {valuation_details.get('peg_signal', 'N/A')}\n"
            if 'ps_ratio' in valuation_details:
                valuation_text += f"- P/S Ratio: {valuation_details['ps_ratio']} - {valuation_details.get('ps_signal', 'N/A')}\n"
        else:
            valuation_text += "- Limited valuation data available\n"
        
        # Prepare prompt with AI scoring insights
        prediction_text = ""
        if prediction:
            prediction_text = f"\n6-MONTH PRICE PREDICTION:\n- Current: {currency_symbol}{stock_data['current_price']} → Predicted: {currency_symbol}{prediction['predicted_6m_price']}\n- Change: {prediction['price_change_pct']:+.2f}% ({prediction['trend']})\n- Backtest Accuracy: {prediction['backtest_accuracy']:.1f}%\n"
        
        prompt = f"""You are an expert stock analyst. Analyze {ticker} using the provided scoring system.

SCORING SYSTEM:
- Fundamental Score: {fundamental_score}/100 (PE, ROE, debt, margins, growth)
- Technical Score: {technical_score}/100 (RSI, MACD, moving averages)

TECHNICAL DETAILS:
- Price: {currency_symbol}{stock_data['current_price']}
- RSI: {stock_data['rsi']} | MACD: {stock_data['macd']} | Signal: {stock_data['macd_signal']}
- SMA(20): {currency_symbol}{stock_data['sma_20']} | SMA(50): {currency_symbol}{stock_data['sma_50']}
- Dividend Yield: {stock_data['dividend_yield']}%
{valuation_text}{prediction_text}
RECENT NEWS HEADLINES:
"""
        for idx, article in enumerate(news, 1):
            prompt += f"{idx}. {article['title']}\n"
        
        prompt += f"""

YOUR TASK:
1. Read and analyze the news sentiment comprehensively (0-100 score)
   - 0-30: Very negative
   - 30-45: Negative
   - 45-55: Neutral  
   - 55-70: Positive
   - 70-100: Very positive
2. Consider the valuation status ({valuation_status}) - adjust recommendation accordingly:
   - Undervalued + positive indicators → favor Buy
   - Overvalued + negative indicators → favor Sell
   - Fair-Valued → rely on technicals and sentiment
3. Consider the 6-month price prediction trend in your analysis
4. Calculate composite score: (sentiment*33% + fundamental*33% + technical*34%)
5. Provide recommendation based on composite score, predicted trend, AND valuation

RESPOND WITH VALID JSON ONLY (no markdown):
{{
  "sentiment_score": X (0-100, your analysis of news sentiment),
  "composite_score": Y (calculated: sentiment*0.33 + {fundamental_score}*0.33 + {technical_score}*0.34),
  "interest_level": "Interesting" or "Not Interesting",
  "recommendation": "Strong Buy" or "Buy" or "Hold" or "Sell" or "Strong Sell",
  "confidence_score": Z (same as composite_score),
  "time_horizon": "Scalp Trading (Minutes/Hours)" or "Day Trading (Days)" or "Investment (Long-Term)",
  "reasoning": "Brief explanation mentioning sentiment, fundamentals, technicals, valuation ({valuation_status}), and predicted trend.",
  "entry_zone": "{currency_symbol}X.XX-{currency_symbol}Y.YY",
  "take_profit": "{currency_symbol}X.XX",
  "stop_loss": "{currency_symbol}X.XX"
}}

GUIDELINES:
1. Analyze news carefully for sentiment (not just keywords)
2. Consider valuation: {valuation_status} stock should influence your recommendation
3. Consider predicted trend: Strong Uptrend/Uptrend = favor Buy, Downtrend = favor Sell
4. Composite >= 80: Strong Buy, >= 65: Buy, >= 45: Hold, >= 30: Sell, < 30: Strong Sell
4. Interest level: "Not Interesting" if composite < 40
5. Prices must be numbers with {currency_symbol}"""
        
        # Prepare messages for chat format
        messages = [
            {"role": "system", "content": "You are a stock analyst validating AI-generated composite scores. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate response with more constrained parameters
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=250,  # Reduced for faster generation
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Aggressive JSON extraction
        response_text = response_text.strip()
        # Remove markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]
        
        # Find JSON object
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            response_text = response_text[start_idx:end_idx]
        
        analysis = json.loads(response_text.strip())
        
        # Extract sentiment and composite scores from Qwen's response
        sentiment_score = analysis.get('sentiment_score', 50)
        composite_score = analysis.get('composite_score', 50)
        
        print(f"  Qwen sentiment: {sentiment_score}, Composite: {composite_score} -> {analysis.get('recommendation', 'N/A')}")
        
        # Validate and fix common issues
        currency_symbol = 'Rp' if stock_data.get('currency') == 'IDR' else '$'
        
        # Fix entry_zone if it's an object or array
        if isinstance(analysis.get('entry_zone'), (dict, list)):
            analysis['entry_zone'] = f"{currency_symbol}{stock_data['current_price']}-{currency_symbol}{round(stock_data['current_price'] * 1.02, 2)}"
        
        # Fix take_profit if it's an object or array
        if isinstance(analysis.get('take_profit'), (dict, list)):
            analysis['take_profit'] = f"{currency_symbol}{round(stock_data['current_price'] * 1.1, 2)}"
        
        # Fix stop_loss if it's an object or array
        if isinstance(analysis.get('stop_loss'), (dict, list)):
            analysis['stop_loss'] = f"{currency_symbol}{round(stock_data['current_price'] * 0.95, 2)}"
        
        # Add prediction data to analysis
        if prediction:
            analysis['predicted_6m_price'] = prediction['predicted_6m_price']
            analysis['price_change_pct'] = prediction['price_change_pct']
            analysis['predicted_trend'] = prediction['trend']
            analysis['backtest_accuracy'] = prediction['backtest_accuracy']
            analysis['prediction_graph'] = prediction['prediction_graph']
            analysis['backtest_results'] = prediction['backtest_results']
        
        # Add valuation data to analysis
        analysis['valuation_status'] = valuation_status
        analysis['valuation_details'] = valuation_details
        
        return analysis
        
    except Exception as e:
        print(f"  Error analyzing with Qwen for {ticker}: {e}")
        # Return a fallback analysis
        currency_symbol = 'Rp' if stock_data.get('currency') == 'IDR' else '$'
        fallback_composite = int((50 * 0.33 + fundamental_score * 0.33 + technical_score * 0.34))
        
        # Try to get prediction even if Qwen fails
        prediction_fallback = predict_price_trend(ticker, stock_data)
        
        fallback = {
            "sentiment_score": 50,
            "composite_score": fallback_composite,
            "interest_level": "Interesting",
            "recommendation": "Hold",
            "confidence_score": fallback_composite,
            "time_horizon": "Investment (Long-Term)",
            "reasoning": f"Analysis pending. Fundamental: {fundamental_score}, Technical: {technical_score}",
            "entry_zone": f"{currency_symbol}{stock_data['current_price']}",
            "take_profit": f"{currency_symbol}{round(stock_data['current_price'] * 1.08, 2)}",
            "stop_loss": f"{currency_symbol}{round(stock_data['current_price'] * 0.95, 2)}"
        }
        
        # Add prediction if available
        if prediction_fallback:
            fallback['predicted_6m_price'] = prediction_fallback['predicted_6m_price']
            fallback['price_change_pct'] = prediction_fallback['price_change_pct']
            fallback['predicted_trend'] = prediction_fallback['trend']
            fallback['backtest_accuracy'] = prediction_fallback['backtest_accuracy']
            fallback['prediction_graph'] = prediction_fallback['prediction_graph']
            fallback['backtest_results'] = prediction_fallback['backtest_results']
        
        return fallback


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
        
        # Analyze with Qwen
        analysis = analyze_with_qwen(ticker, stock_data, news)
        
        # Combine all data
        result = {
            **stock_data,
            **analysis,
            'news': news,
            'last_updated': datetime.now().isoformat()
        }
        
        results.append(result)
        
        print(f"  [OK] {ticker}: {analysis['recommendation']} ({analysis['time_horizon']})")
        
        # Small delay between analyses
        if idx < len(WATCHLIST):
            time.sleep(0.5)
    
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
