// Global Chart Instance
let marketChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    loadHistory();

    // Listen for ticker change to reload history
    document.getElementById('tickerSelect').addEventListener('change', loadHistory);
});

async def loadHistory() {
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
        predDate.setDate(predDate.getDate() + 1); // Next day
        document.getElementById('predictionDate').innerText = `For ${predDate.toDateString()}`;

        // Color coding based on movement
        if (predictedPrice > lastPrice) {
            resultCard.className = "card border-0 shadow-sm p-3 mt-3 bg-success text-white";
        } else {
            resultCard.className = "card border-0 shadow-sm p-3 mt-3 bg-danger text-white";
        }
        resultCard.classList.remove('d-none');

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
