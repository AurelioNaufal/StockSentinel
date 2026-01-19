// StockSentinel Interactive - Frontend Logic
// ==========================================

// Configuration
const GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'; // ⚠️ For production, use a serverless proxy!
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-exp:generateContent';
const RATE_LIMIT_DELAY = 4000; // 4 seconds between API calls (15 RPM free tier)

// Alpine.js App
function stockApp() {
    return {
        // State
        stocks: [],
        filteredStocks: [],
        activeFilter: 'all',
        selectedStock: null,
        searchQuery: '',
        isLoading: true,
        isSearching: false,
        searchError: '',
        searchMessage: '',
        lastUpdated: 'Loading...',
        totalAssets: 0,
        
        // API Rate Limiting
        lastApiCall: 0,
        apiQueue: [],
        
        // Initialize
        async init() {
            await this.loadData();
            this.applyFilter();
        },
        
        // Load data.json
        async loadData() {
            try {
                const response = await fetch('data.json');
                if (!response.ok) {
                    throw new Error('Failed to load data');
                }
                
                const data = await response.json();
                this.stocks = data.assets || [];
                this.totalAssets = data.total_assets || this.stocks.length;
                
                // Format last updated date
                if (data.last_updated) {
                    const date = new Date(data.last_updated);
                    this.lastUpdated = date.toLocaleString();
                }
                
                this.isLoading = false;
                this.applyFilter();
                
            } catch (error) {
                console.error('Error loading data:', error);
                this.isLoading = false;
                alert('Failed to load stock data. Please try again later.');
            }
        },
        
        // Apply filter
        applyFilter() {
            switch (this.activeFilter) {
                case 'strong-buy':
                    this.filteredStocks = this.stocks.filter(s => 
                        s.recommendation === 'Strong Buy'
                    );
                    break;
                    
                case 'dividend':
                    this.filteredStocks = this.stocks.filter(s => 
                        s.dividend_yield > 2
                    );
                    break;
                    
                case 'scalp':
                    this.filteredStocks = this.stocks.filter(s => 
                        s.time_horizon && (
                            s.time_horizon.includes('Scalp') || 
                            s.time_horizon.includes('Day Trade')
                        )
                    );
                    break;
                    
                default:
                    this.filteredStocks = this.stocks;
            }
        },
        
        // Search stock
        async searchStock() {
            const ticker = this.searchQuery.trim().toUpperCase();
            
            if (!ticker) {
                this.searchError = 'Please enter a ticker symbol';
                return;
            }
            
            this.searchError = '';
            this.searchMessage = '';
            this.isSearching = true;
            
            // First, check if stock exists in loaded data
            const existingStock = this.stocks.find(s => 
                s.ticker.toUpperCase() === ticker
            );
            
            if (existingStock) {
                this.isSearching = false;
                this.showDetails(existingStock);
                this.searchMessage = `Found ${ticker} in cached data`;
                return;
            }
            
            // If not found, fetch from Gemini API
            this.searchMessage = `${ticker} not in cache. Fetching live analysis from Gemini...`;
            
            try {
                const analysis = await this.fetchGeminiAnalysis(ticker);
                
                if (analysis) {
                    this.showDetails(analysis);
                    this.searchMessage = `Live analysis for ${ticker} complete!`;
                } else {
                    this.searchError = `Unable to analyze ${ticker}. Please verify the ticker symbol.`;
                }
                
            } catch (error) {
                console.error('Search error:', error);
                this.searchError = `Error analyzing ${ticker}: ${error.message}`;
            } finally {
                this.isSearching = false;
            }
        },
        
        // Fetch analysis from Gemini API (client-side)
        async fetchGeminiAnalysis(ticker) {
            // Check API key
            if (GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY_HERE') {
                throw new Error('Gemini API key not configured. Please update script.js');
            }
            
            // Rate limiting: Ensure 4 seconds between calls
            const now = Date.now();
            const timeSinceLastCall = now - this.lastApiCall;
            
            if (timeSinceLastCall < RATE_LIMIT_DELAY) {
                const waitTime = RATE_LIMIT_DELAY - timeSinceLastCall;
                this.searchMessage = `Rate limit: Waiting ${Math.ceil(waitTime / 1000)} seconds...`;
                await this.sleep(waitTime);
            }
            
            try {
                // Fetch basic stock data from Yahoo Finance (if available via CORS proxy)
                // For simplicity, we'll just use the ticker and ask Gemini to do the analysis
                const stockData = await this.fetchStockPrice(ticker);
                
                // Prepare Gemini prompt
                const prompt = `
You are a professional stock analyst. Analyze ${ticker} and provide investment recommendations in JSON format.

${stockData.current_price ? `Current Price: $${stockData.current_price}` : 'Please fetch current market data.'}

Please provide your analysis in the following JSON format (respond ONLY with valid JSON, no markdown):
{
    "ticker": "${ticker}",
    "current_price": ${stockData.current_price || 0},
    "recommendation": "Strong Buy|Buy|Hold|Sell|Strong Sell",
    "confidence_score": 85,
    "time_horizon": "Scalp (Minutes-Hours)|Day Trade (Hours-Days)|Swing (Days-Weeks)|Position (Weeks-Months)|Long-term (Months-Years)",
    "reasoning": "Brief explanation of your recommendation based on current market conditions",
    "entry_zone": "Price range for entry",
    "take_profit": "Target price for profit taking",
    "stop_loss": "Stop loss price to limit downside",
    "dividend_yield": 0,
    "dividend_analysis": "Brief note on dividend if applicable, or 'N/A'",
    "rsi": null,
    "macd": null,
    "sma_20": null,
    "sma_50": null,
    "news": []
}

Focus on current market trends, technical levels, and fundamental analysis.
`;
                
                // Call Gemini API
                const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        contents: [{
                            parts: [{
                                text: prompt
                            }]
                        }]
                    })
                });
                
                this.lastApiCall = Date.now();
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error?.message || 'API request failed');
                }
                
                const data = await response.json();
                
                // Parse response
                let analysisText = data.candidates[0].content.parts[0].text;
                
                // Clean markdown code blocks if present
                analysisText = analysisText.trim();
                if (analysisText.startsWith('```json')) {
                    analysisText = analysisText.slice(7);
                }
                if (analysisText.startsWith('```')) {
                    analysisText = analysisText.slice(3);
                }
                if (analysisText.endsWith('```')) {
                    analysisText = analysisText.slice(0, -3);
                }
                
                const analysis = JSON.parse(analysisText.trim());
                
                return analysis;
                
            } catch (error) {
                console.error('Gemini API error:', error);
                throw error;
            }
        },
        
        // Fetch stock price (basic implementation - may need CORS proxy)
        async fetchStockPrice(ticker) {
            // This is a simplified version. In production, you'd need:
            // 1. A CORS proxy or serverless function
            // 2. Or rely entirely on Gemini to fetch the data
            
            try {
                // For demo purposes, we'll just return empty data
                // and let Gemini fetch the current price
                return {
                    ticker: ticker,
                    current_price: null
                };
            } catch (error) {
                console.error('Error fetching stock price:', error);
                return {
                    ticker: ticker,
                    current_price: null
                };
            }
        },
        
        // Show stock details
        showDetails(stock) {
            this.selectedStock = stock;
        },
        
        // Get recommendation CSS class
        getRecommendationClass(recommendation) {
            const rec = recommendation.toLowerCase().replace(/\s+/g, '-');
            return `recommendation-${rec}`;
        },
        
        // Format price
        formatPrice(price) {
            if (!price) return 'N/A';
            return '$' + parseFloat(price).toFixed(2);
        },
        
        // Sleep utility
        sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
    };
}

// Watch for filter changes
document.addEventListener('alpine:init', () => {
    Alpine.watch('activeFilter', () => {
        const app = Alpine.$data(document.querySelector('[x-data]'));
        if (app) {
            app.applyFilter();
        }
    });
});

// Service Worker for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('sw.js').catch(() => {
            // Silently fail if sw.js doesn't exist
        });
    });
}
