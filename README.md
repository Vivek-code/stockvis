# 📈 StockVision: AI-Powered Financial Forecasting Ecosystem

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" />
</div>

## 📖 Overview

**StockVision** is a comprehensive, full-stack financial forecasting platform designed to predict stock trends by combining historical numerical data with real-time news sentiment. 

Unlike traditional singular-model predictors, StockVision leverages an **intelligent ensemble of deep learning architectures**—including custom Multi-Head Attention Transformers, GRU, LSTM, and 1D-CNN—working natively alongside a **HuggingFace FinBERT NLP pipeline** to deliver highly accurate and robust market insights.

## ✨ Key Features

- **🧠 Multi-Model Deep Learning Ensemble**: Utilizes a weighted voting strategy (Transformer 35%, LSTM 25%, GRU 25%, CNN 15%) to mitigate "single-model hallucination" and increase prediction accuracy.
- **📰 Real-Time Sentiment Analysis**: Integrates the `ProsusAI/finbert` model to recursively parse live news headlines from Yahoo Finance, extracting positive/negative/neutral market sentiment impact.
- **📊 Interactive "Reality Check" Dashboard**: A dedicated backtesting module that fetches past historical predictions and compares them directly against actual market closes using visually intuitive Chart.js graphs.
- **⚡ Full-Stack Flask Web App**: A completely responsive and interactive UI built with Flask, HTML/CSS, and JS that maps metrics natively and serves RESTful API endpoints.

## 🛠️ Tech Stack

- **Backend & API:** Python 3, Flask, REST API
- **Machine Learning & AI:** TensorFlow/Keras, PyTorch, HuggingFace Transformers (FinBERT), Scikit-Learn
- **Data Processing:** Pandas, Numpy, Pandas-TA (Technical Analysis), yfinance
- **Database:** SQLite (for chronological Reality Check tracking)
- **Frontend & Visualization:** HTML5, CSS3, JavaScript, Chart.js

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your local machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/StockVision.git
   cd StockVision
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize mock database (Optional - for Reality Check testing):**
   ```bash
   python populate_mock_db.py
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the web app:**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📁 Project Structure

- `app.py`: Main Flask application entry point and API routing.
- `train_models.py` / `retrain_transformer.py`: Scripts for training and fine-tuning the deep learning models.
- `model_definitions.py`: Core architecture definitions for the Transformer, GRU, LSTM, and CNN.
- `sentiment_analyzer.py`: HuggingFace FinBERT extraction and classification pipeline.
- `load_model_pkg.py`: Engine for loading models and executing the ensemble prediction logic.
- `features.py`: Logic for technical indicator feature injection and scaling.
- `populate_mock_db.py`: Script to generate chronological mock data for the Reality Check module.
- `templates/` & `static/`: HTML templates and CSS/JS assets for the interactive dashboard.

## ⚙️ How It Works

1. **Data Ingestion:** The app fetches real-time OHLCV data using the `yfinance` library.
2. **Feature Engineering:** Technical indicators (RSI, MACD, Bollinger Bands) and real-time sentiment scores from live news are injected into the dataset.
3. **Inference Engine:** The data is processed through the ensemble of models. The predictions are aggregated based on their predefined weights.
4. **Visualization:** The Flask backend serves the final prediction, alongside model-specific breakdowns and news sentiment, to the frontend dashboard.
5. **Reality Check:** Every prediction is stored in SQLite. Days later, users can view how well the AI predicted the actual closing prices.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/StockVision/issues).

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
