// Types for better code organization
/**
 * @typedef {Object} Stock
 * @property {string} symbol
 * @property {string} name
 * @property {number} price
 * @property {number} change
 * @property {number} marketCap
 * @property {number} volume
 */

// Utility functions for formatting
const formatters = {
    currency: (number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(number);
    },

    largeNumber: (number) => {
        const tiers = [
            { threshold: 1e12, suffix: 'T', divisor: 1e12 },
            { threshold: 1e9, suffix: 'B', divisor: 1e9 },
            { threshold: 1e6, suffix: 'M', divisor: 1e6 }
        ];

        const tier = tiers.find(tier => number >= tier.threshold);
        if (tier) {
            return (number / tier.divisor).toFixed(2) + tier.suffix;
        }
        return number.toLocaleString();
    },

    percentageChange: (change) => {
        const isPositive = change >= 0;
        return `${isPositive ? '+' : ''}${change.toFixed(2)}%`;
    }
};

// Stock data management
class StockManager {
    constructor(stocks) {
        this.stocks = stocks;
    }

    createTableRow(stock) {
        const isPositive = stock.change >= 0;
        const changeClassName = isPositive ? 'text-green-600' : 'text-red-600';

        return `
            <tr class="hover:bg-gray-50 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap font-medium">${stock.symbol}</td>
                <td class="px-6 py-4 whitespace-nowrap">${stock.name}</td>
                <td class="px-6 py-4 whitespace-nowrap">${formatters.currency(stock.price)}</td>
                <td class="px-6 py-4 whitespace-nowrap ${changeClassName} font-medium">
                    ${formatters.percentageChange(stock.change)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">${formatters.largeNumber(stock.marketCap)}</td>
                <td class="px-6 py-4 whitespace-nowrap">${formatters.largeNumber(stock.volume)}</td>
            </tr>
        `;
    }

    updateTable() {
        const stockListTable = document.getElementById('stock-list-table');
        if (!stockListTable) {
            console.error('Stock list table not found');
            return;
        }

        const tbody = stockListTable.querySelector('tbody');
        if (!tbody) {
            console.error('Table body not found');
            return;
        }

        tbody.innerHTML = this.stocks.map(stock => this.createTableRow(stock)).join('');
    }

    // Method to add real-time updates (example implementation)
    startRealTimeUpdates(interval = 5000) {
        setInterval(() => {
            this.stocks = this.stocks.map(stock => ({
                ...stock,
                price: stock.price * (1 + (Math.random() * 0.02 - 0.01)),
                change: stock.change + (Math.random() * 0.4 - 0.2)
            }));
            this.updateTable();
        }, interval);
    }
}

// Initialize stock manager with data
const stockManager = new StockManager(stocks);

// Start updating when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    stockManager.updateTable();
    stockManager.startRealTimeUpdates();
});

// Export for testing
if (typeof module !== 'undefined') {
    module.exports = {
        formatters,
        StockManager
    };
}