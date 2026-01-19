// StockSentinel Interactive - Frontend Logic
// ==========================================
// Now using local Qwen model (no API calls from frontend)

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
            
            // Check if stock exists in loaded data (cache only)
            const existingStock = this.stocks.find(s => 
                s.ticker.toUpperCase() === ticker
            );
            
            if (existingStock) {
                this.isSearching = false;
                this.showDetails(existingStock);
                this.searchMessage = `Found ${ticker} in cached data`;
                return;
            }
            
            // Stock not found - no live API with local model
            this.searchError = `${ticker} not found in cache. Run workflow to add more stocks.`;
            this.isSearching = false;
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
