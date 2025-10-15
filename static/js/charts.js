// Chart.js Configuration and Helper Functions

// Global Chart Configuration
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = true;
Chart.defaults.plugins.legend.display = true;

// Color Schemes
const colorSchemes = {
    primary: {
        border: 'rgb(54, 162, 235)',
        background: 'rgba(54, 162, 235, 0.2)'
    },
    success: {
        border: 'rgb(75, 192, 192)',
        background: 'rgba(75, 192, 192, 0.2)'
    },
    danger: {
        border: 'rgb(255, 99, 132)',
        background: 'rgba(255, 99, 132, 0.2)'
    },
    warning: {
        border: 'rgb(255, 205, 86)',
        background: 'rgba(255, 205, 86, 0.2)'
    },
    info: {
        border: 'rgb(153, 102, 255)',
        background: 'rgba(153, 102, 255, 0.2)'
    }
};

// Stock Price Line Chart
function createStockPriceChart(ctx, dates, prices, symbol) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: `${symbol} Price`,
                data: prices,
                borderColor: colorSchemes.primary.border,
                backgroundColor: colorSchemes.primary.background,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} Stock Price History`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `Price: ${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return ' '+ value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Prediction Comparison Chart
function createPredictionChart(ctx, historicalDates, historicalPrices, forecastDates, forecastPrices, symbol, model) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [...historicalDates, ...forecastDates],
            datasets: [{
                label: 'Historical Prices',
                data: historicalPrices.map((price, idx) => ({
                    x: historicalDates[idx],
                    y: price
                })),
                borderColor: colorSchemes.success.border,
                backgroundColor: colorSchemes.success.background,
                fill: true,
                tension: 0.4
            }, {
                label: 'Predicted Prices',
                data: forecastPrices.map((price, idx) => ({
                    x: forecastDates[idx],
                    y: price
                })),
                borderColor: colorSchemes.danger.border,
                backgroundColor: colorSchemes.danger.background,
                borderDash: [5, 5],
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} - ${model} Prediction`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Volume Bar Chart
function createVolumeChart(ctx, dates, volumes, symbol) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Volume',
                data: volumes,
                backgroundColor: colorSchemes.info.background,
                borderColor: colorSchemes.info.border,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} Trading Volume`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return (value / 1000000).toFixed(1) + 'M';
                        }
                    }
                }
            }
        }
    });
}

// Technical Indicators Chart
function createTechnicalChart(ctx, dates, closePrice, sma20, sma50, symbol) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Close Price',
                data: closePrice,
                borderColor: colorSchemes.primary.border,
                backgroundColor: 'transparent',
                tension: 0.4
            }, {
                label: 'SMA 20',
                data: sma20,
                borderColor: colorSchemes.warning.border,
                backgroundColor: 'transparent',
                tension: 0.4,
                borderDash: [5, 5]
            }, {
                label: 'SMA 50',
                data: sma50,
                borderColor: colorSchemes.info.border,
                backgroundColor: 'transparent',
                tension: 0.4,
                borderDash: [5, 5]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} - Technical Indicators`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// RSI Chart
function createRSIChart(ctx, dates, rsiValues, symbol) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'RSI',
                data: rsiValues,
                borderColor: colorSchemes.danger.border,
                backgroundColor: colorSchemes.danger.background,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${symbol} - RSI Indicator`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: 70,
                            yMax: 70,
                            borderColor: 'red',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                content: 'Overbought (70)',
                                enabled: true
                            }
                        },
                        line2: {
                            type: 'line',
                            yMin: 30,
                            yMax: 30,
                            borderColor: 'green',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                content: 'Oversold (30)',
                                enabled: true
                            }
                        }
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// Export functions
window.chartHelpers = {
    createStockPriceChart,
    createPredictionChart,
    createVolumeChart,
    createTechnicalChart,
    createRSIChart,
    colorSchemes
};