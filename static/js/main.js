// Main JavaScript file for Stock Forecasting Website

// Utility Functions
const utils = {
    // Format currency
    formatCurrency: (value, currency = 'USD') => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(value);
    },

    // Format percentage
    formatPercentage: (value) => {
        return `${value.toFixed(2)}%`;
    },

    // Format large numbers
    formatNumber: (value) => {
        if (value >= 1000000000) {
            return `$${(value / 1000000000).toFixed(2)}B`;
        } else if (value >= 1000000) {
            return `$${(value / 1000000).toFixed(2)}M`;
        }
        return `$${value.toFixed(2)}`;
    },

    // Show loading spinner
    showLoading: (elementId) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="text-center py-5">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading data... ⏳</p>
                </div>
            `;
        }
    },

    // Show error message
    showError: (elementId, message) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <i class="fas fa-exclamation-triangle"></i> ${message}
                </div>
            `;
        }
    },

    // Show success message
    showSuccess: (message) => {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
        alertDiv.style.zIndex = '9999';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);

        // Auto dismiss after 3 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 3000);
    },

    // Validate stock symbol
    validateSymbol: (symbol) => {
        return /^[A-Z]{1,5}$/.test(symbol.toUpperCase());
    },

    // Get color for change value
    getChangeColor: (change) => {
        return change >= 0 ? 'text-success' : 'text-danger';
    },

    // Get icon for change value
    getChangeIcon: (change) => {
        return change >= 0 ? '📈' : '📉';
    }
};

// API Helper Functions
const api = {
    // Generic fetch wrapper
    fetchData: async (url, options = {}) => {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Request failed');
            }

            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    // Get stock data
    getStockData: async (symbol) => {
        return await api.fetchData(`/stock/data/${symbol}`);
    },

    // Search stock
    searchStock: async (symbol) => {
        return await api.fetchData('/stock/search', {
            method: 'POST',
            body: JSON.stringify({ symbol: symbol })
        });
    },

    // Get prediction
    getPrediction: async (symbol, model, days) => {
        return await api.fetchData('/prediction/forecast', {
            method: 'POST',
            body: JSON.stringify({
                symbol: symbol,
                model: model,
                days: days
            })
        });
    }
};

// Chart Helper Functions
const charts = {
    // Default chart options
    defaultOptions: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            },
            tooltip: {
                mode: 'index',
                intersect: false
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    },

    // Create line chart
    createLineChart: (ctx, labels, datasets, title) => {
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                ...charts.defaultOptions,
                plugins: {
                    ...charts.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: title
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            }
        });
    },

    // Create bar chart
    createBarChart: (ctx, labels, data, title) => {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: title,
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                ...charts.defaultOptions,
                plugins: {
                    ...charts.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: title
                    }
                }
            }
        });
    }
};

// Session Storage Helper
const storage = {
    // Save selected stock
    saveStock: (symbol) => {
        sessionStorage.setItem('selectedStock', symbol);
    },

    // Get selected stock
    getStock: () => {
        return sessionStorage.getItem('selectedStock') || 'AAPL';
    },

    // Clear storage
    clear: () => {
        sessionStorage.clear();
    }
};

// Initialize tooltips and popovers
document.addEventListener('DOMContentLoaded', function() {
    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 100);
    });
});

// Auto-dismiss alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 150);
        }, 5000);
    });
});

// Export utilities
window.utils = utils;
window.api = api;
window.charts = charts;
window.storage = storage;