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
        activeMarket: 'all',
        activeTimeHorizon: 'all',
        selectedStock: null,
        searchQuery: '',
        isLoading: true,
        isSearching: false,
        searchError: '',
        searchMessage: '',
        lastUpdated: 'Loading...',
        totalAssets: 0,
        currentChart: null,
        chartTimeframe: '6M',
        
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
        applyFilter(filter, market, timeHorizon) {
            if (filter) {
                this.activeFilter = filter;
            }
            if (market) {
                this.activeMarket = market;
            }
            if (timeHorizon) {
                this.activeTimeHorizon = timeHorizon;
            }
            
            let filtered = this.stocks;
            
            // Filter by time horizon
            if (this.activeTimeHorizon === 'scalp') {
                filtered = filtered.filter(s => s.time_horizon && s.time_horizon.includes('Scalp Trading'));
            } else if (this.activeTimeHorizon === 'day-swing') {
                filtered = filtered.filter(s => s.time_horizon && s.time_horizon.includes('Day Trading'));
            } else if (this.activeTimeHorizon === 'investment') {
                filtered = filtered.filter(s => s.time_horizon && s.time_horizon.includes('Investment'));
            }
            
            // Filter by market type
            if (this.activeMarket === 'idx') {
                filtered = filtered.filter(s => s.ticker.endsWith('.JK'));
            } else if (this.activeMarket === 'us') {
                filtered = filtered.filter(s => !s.ticker.endsWith('.JK') && !s.ticker.includes('-USD'));
            } else if (this.activeMarket === 'crypto') {
                filtered = filtered.filter(s => s.ticker.includes('-USD'));
            }
            
            // Filter by recommendation type
            switch (this.activeFilter) {
                case 'strong-buy':
                    filtered = filtered.filter(s => 
                        s.recommendation === 'Strong Buy'
                    );
                    break;
                    
                case 'dividend':
                    filtered = filtered.filter(s => 
                        s.dividend_yield > 2
                    );
                    break;
                    
                case 'scalp':
                    filtered = filtered.filter(s => 
                        s.time_horizon && s.time_horizon.includes('Scalp Trading')
                    );
                    break;
                    
                default:
                    // 'all' - already filtered by market
            }
            
            this.filteredStocks = filtered;
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
            // Cleanup previous chart if exists
            if (this.currentChart) {
                this.currentChart.destroy();
            }
        },
        
        // Render chart for stock
        renderChart(stock) {
            if (!stock.chart_data || !stock.chart_data[this.chartTimeframe]) return;
            
            const ctx = document.getElementById('chart-' + stock.ticker);
            if (!ctx) return;
            
            // Cleanup previous chart
            if (this.currentChart) {
                this.currentChart.destroy();
            }
            
            const chartData = stock.chart_data[this.chartTimeframe];
            const dates = chartData.map(d => d.date);
            const prices = chartData.map(d => d.price);
            const sma20 = chartData.map(d => d.sma_20);
            
            this.currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [
                        {
                            label: 'Price',
                            data: prices,
                            borderColor: 'rgb(99, 102, 241)',
                            backgroundColor: 'rgba(99, 102, 241, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.1
                        },
                        {
                            label: 'MA(20)',
                            data: sma20,
                            borderColor: 'rgb(234, 88, 12)',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            pointRadius: 0,
                            borderDash: [5, 5],
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    const currency = stock.currency === 'IDR' ? 'Rp ' : '$';
                                    label += currency + context.parsed.y.toLocaleString();
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            ticks: {
                                maxTicksLimit: 8
                            }
                        },
                        y: {
                            display: true,
                            ticks: {
                                callback: function(value) {
                                    return stock.currency === 'IDR' ? 
                                        'Rp ' + value.toLocaleString() : 
                                        '$' + value.toFixed(2);
                                }
                            }
                        }
                    }
                }
            });
        },
        
        // Get recommendation CSS class
        getRecommendationClass(recommendation) {
            const rec = recommendation.toLowerCase().replace(/\s+/g, '-');
            return `recommendation-${rec}`;
        },
        
        // Format price with currency detection
        formatPrice(price, currency) {
            if (!price) return 'N/A';
            const symbol = currency === 'IDR' ? 'Rp ' : '$';
            const formatted = parseFloat(price).toLocaleString('en-US', {
                minimumFractionDigits: currency === 'IDR' ? 0 : 2,
                maximumFractionDigits: currency === 'IDR' ? 0 : 2
            });
            return symbol + formatted;
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
