# Stock Price Prediction Project Report

## 1. Introduction
This project aims to develop a robust machine learning application for predicting stock prices of major NIFTY-50 companies. By leveraging historical market data (OHLCV) and advanced deep learning architectures (LSTM, GRU, CNN), the system provides users with actionable insights through an interactive web dashboard.

## 2. Methodology

### 2.1 Data Collection & Preprocessing
- **Source**: Historical stock data is fetched in real-time using the `yfinance` API.
- **Scope**: The system currently supports 10 major Indian companies: `RELIANCE.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`, `TATAMOTORS.NS`, `INFY.NS`, `LT.NS`, `NTPC.NS`, `ADANIENT.NS`, `TATASTEEL.NS`.
- **Feature Engineering**:
  - Technical indicators are computed to enrich the dataset:
    - **RSI (14)**: Relative Strength Index.
    - **MACD**: Moving Average Convergence Divergence.
    - **Bollinger Bands**: Volatility measure.
    - **EMA**: Exponential Moving Average.
  - **Scaling**: Data is normalized using `MinMaxScaler` (range 0-1) to ensure model stability.
  - **Sequence Generation**: A sliding window approach with a **30-day lookback** period is used to create input sequences for the models.

### 2.2 Model Architecture
Three distinct deep learning models were designed and trained for *each* supported company:

1.  **Long Short-Term Memory (LSTM)**:
    - Designed to capture long-term dependencies in time-series data.
    - Architecture: LSTM layer (50 units) -> Dropout (0.2) -> Dense (1).

2.  **Gated Recurrent Unit (GRU)**:
    - A streamlined variant of LSTM, often offering comparable performance with lower computational cost.
    - Architecture: GRU layer (50 units) -> Dropout (0.2) -> Dense (1).

3.  **1D Convolutional Neural Network (CNN)**:
    - Effective for extracting local patterns and trends within the time window.
    - Architecture: Conv1D (64 filters, kernel=3) -> MaxPooling -> Flatten -> Dense (1).

### 2.3 Training Strategy
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.
- **Validation**: 20% of the data was held out for validation.
- **Optimization**: Early Stopping was implemented to prevent overfitting, monitoring validation loss with patience.

### 2.4 System Architecture
- **Backend**: Flask (Python) serves as the API layer, handling data fetching, model inference (Lazy Loading), and serving the frontend.
- **Frontend**: HTML5, CSS3 (Bootstrap 5), and JavaScript (Chart.js) provide a responsive and interactive user interface.
- **Multi-Model Support**: The system dynamically loads the specific trained model for the selected ticker and model type on demand.

## 3. Results
The models were evaluated on unseen test data from late 2023 to early 2024.

- **Visual Analysis**: Comparison plots for all 10 companies have been generated, overlaying the predictions of LSTM, GRU, and CNN models against actual stock prices.
- **Performance**:
  - **Trend Capture**: All three models successfully captured the major trends and turning points in the stock prices.
  - **Responsiveness**: GRU and LSTM models showed high responsiveness to recent price changes.
  - **Metrics**: detailed MAE and RMSE scores were logged during training.

*(See the `project_report_assets` folder for individual comparison graphs for each company.)*

## 4. Conclusion and Future Scope

### 4.1 Conclusion
The developed application successfully demonstrates the viability of deep learning for stock trend forecasting. The multi-model approach allows users to compare different architectures, and the web-based dashboard makes these advanced analytics accessible. The system is modular, scalable, and currently supports 10 key assets with high reliability.

### 4.2 Future Scope
- **Expanded Asset Class**: Support for US stocks, Crypto, and Forex.
- **Database Integration**: Implementing SQLite/PostgreSQL to cache predictions and user preferences.
- **Advanced Features**: 
  - Sentiment Analysis from news headlines.
  - Portfolio optimization suggestions.
  - User accounts and watchlists.
- **Deployment**: containerizing the application with Docker and deploying to a cloud platform (AWS/GCP).
