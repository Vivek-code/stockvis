// Global Chart Instance
let marketChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    loadHistory();

    // Listen for ticker change to reload history
    document.getElementById('tickerSelect').addEventListener('change', loadHistory);

    // Load prediction history table
    loadPredictionHistory();

    // Load leaderboard
    loadMetrics();
});

async function loadPredictionHistory() {
    const tbody = document.getElementById('historyTableBody');
    try {
        const response = await fetch('/api/predictions');
        const data = await response.json();

        tbody.innerHTML = '';

        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No history yet.</td></tr>';
            return;
        }

        data.forEach(row => {
            const tr = document.createElement('tr');
            const date = new Date(row.created_at).toLocaleString();
            tr.innerHTML = `
                <td>${date}</td>
                <td class="fw-bold text-primary">${row.ticker}</td>
                <td><span class="badge bg-light text-dark border">${row.model.toUpperCase()}</span></td>
                <td class="text-end">₹${row.price.toFixed(2)}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (error) {
        console.error("Failed to load prediction history:", error);
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Error loading history</td></tr>';
    }
}

async function loadHistory() {
    const ticker = document.getElementById('tickerSelect').value;
    const predictBtn = document.getElementById('predictBtn');

    // Disable button while loading
    predictBtn.disabled = true;

    try {
        const response = await fetch(`/api/history?ticker=${ticker}&limit=100`);
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        updateChartHistory(data.dates, data.prices);

        // Clear previous predictions
        document.getElementById('predictionResultCard').classList.add('d-none');

    } catch (error) {
        console.error("Failed to load history:", error);
        alert("Failed to load historical data. Please check connection or ticker.");
    } finally {
        predictBtn.disabled = false;
    }
}

async function runPrediction() {
    const ticker = document.getElementById('tickerSelect').value;
    const model = document.getElementById('modelSelect').value;
    const days = document.getElementById('horizonInput').value;

    const btn = document.getElementById('predictBtn');
    const spinner = document.getElementById('loadingSpinner');
    const resultCard = document.getElementById('predictionResultCard');

    btn.disabled = true;
    spinner.classList.remove('d-none');
    resultCard.classList.add('d-none');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ticker, model, days })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Show result
        const lastPrice = data.last_close;
        const predictedPrice = data.predictions[0]; // First day prediction
        const lastDate = new Date(data.last_date);

        // Update Chart with Forecast
        updateChartForecast(data.predictions, lastDate);

        // Update Result Card
        document.getElementById('predictionValue').innerText = `₹${predictedPrice.toFixed(2)}`;

        // Calculate date of prediction
        const predDate = new Date(lastDate);
        predDate.setDate(predDate.getDate() + 1); // Next day predDate
        document.getElementById('predictionDate').innerText = `For ${new Date().toDateString()}`; // Using today as per user edit

        // Color coding based on movement
        if (predictedPrice > lastPrice) {
            resultCard.className = "card border-0 shadow-sm p-3 mt-3 bg-success text-white";
        } else {
            resultCard.className = "card border-0 shadow-sm p-3 mt-3 bg-danger text-white";
        }
        resultCard.classList.remove('d-none');

        // Refresh History Table
        loadPredictionHistory();

    } catch (error) {
        console.error("Prediction failed:", error);
        alert("Prediction failed. See console.");
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
    }
}

function initChart() {
    const ctx = document.getElementById('mainChart').getContext('2d');

    marketChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Historical Close',
                    data: [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Forecast',
                    data: [],
                    borderColor: '#ffc107',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 4,
                    pointBackgroundColor: '#fff'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    titleColor: '#000',
                    bodyColor: '#000',
                    borderColor: '#e9ecef',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function (context) {
                            return ' ₹' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                y: {
                    grid: {
                        color: '#f0f0f0'
                    },
                    ticks: {
                        callback: function (value) {
                            return '₹' + value;
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function updateChartHistory(dates, prices) {
    marketChart.data.labels = dates;
    marketChart.data.datasets[0].data = prices;
    // Clear forecast
    marketChart.data.datasets[1].data = new Array(dates.length).fill(null);
    marketChart.update();
}

function updateChartForecast(predictions, lastDateObj) {
    const currentLabels = marketChart.data.labels;
    const currentPrices = marketChart.data.datasets[0].data;
    const lastPrice = currentPrices[currentPrices.length - 1]; // Use as anchor

    // Generate new dates
    const newLabels = [...currentLabels];
    const forecastData = new Array(currentLabels.length).fill(null);

    // Anchor point (connect history to forecast)
    forecastData[currentLabels.length - 1] = lastPrice;

    let tempDate = new Date(lastDateObj);

    predictions.forEach((price) => {
        tempDate.setDate(tempDate.getDate() + 1);
        // If weekend, skip? Simple logic for now: just add days
        // Ideally checking for weekends.
        newLabels.push(tempDate.toISOString().split('T')[0]);
        forecastData.push(price);
    });

    marketChart.data.labels = newLabels;
    // Pad history dataset with nulls
    const historyPad = new Array(predictions.length).fill(null);
    // actually we don't need to pad history with nulls if we just set the data array
    // but we need to make sure the array length matches labels if we want to be safe, 
    // chart.js handles shorter arrays fine usually.

    marketChart.data.datasets[1].data = forecastData;
    marketChart.update();
}


// --- Leaderboard Logic ---
async function loadMetrics() {
    const tbody = document.getElementById('leaderboardBody');
    if (!tbody) return; // Guard clause

    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();

        tbody.innerHTML = '';

        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No metrics available. Train models first.</td></tr>';
            return;
        }

        // Sort by R2 descending initially
        data.sort((a, b) => b.r2 - a.r2);

        data.forEach(row => {
            const tr = document.createElement('tr');
            // Highlight high R2
            const r2Class = row.r2 > 0.8 ? 'text-success fw-bold' : (row.r2 < 0 ? 'text-danger' : '');

            tr.innerHTML = `
                <td class="fw-bold text-muted">${row.ticker}</td>
                <td><span class="badge bg-light text-dark border">${row.model.toUpperCase()}</span></td>
                <td class="text-end ${r2Class}">${(row.r2 * 100).toFixed(2)}%</td>
                <td class="text-end">${row.mape.toFixed(2)}%</td>
                <td class="text-end">${row.rmse.toFixed(4)}</td>
                <td class="text-end">${row.mae.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error loading metrics:', error);
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Failed to load metrics.</td></tr>';
    }
}

function sortTable(n) {
    var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
    table = document.getElementById("leaderboardTable");
    switching = true;
    dir = "asc";
    while (switching) {
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {
            shouldSwitch = false;
            x = rows[i].getElementsByTagName("TD")[n];
            y = rows[i + 1].getElementsByTagName("TD")[n];

            let valX = x.innerText;
            let valY = y.innerText;

            // Parse numbers for metric columns (2,3,4,5)
            if (n >= 2) {
                valX = parseFloat(valX.replace('%', ''));
                valY = parseFloat(valY.replace('%', ''));
            }

            if (dir == "asc") {
                if (valX > valY) { shouldSwitch = true; break; }
            } else if (dir == "desc") {
                if (valX < valY) { shouldSwitch = true; break; }
            }
        }
        if (shouldSwitch) {
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
            switchcount++;
        } else {
            if (switchcount == 0 && dir == "asc") {
                dir = "desc";
                switching = true;
            }
        }
    }
}
