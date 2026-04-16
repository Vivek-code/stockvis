# StockVision — Project Enhancement & Architecture Report

## 1. Project Overview
Originally, the StockVision platform natively hosted three fundamental deep-learning topologies: **GRU**, **LSTM**, and **1D-CNN**. 

To elevate the application into an advanced, comprehensive financial prediction ecosystem, several major architectural milestones have been completed. This report documents these additions, detailing their purpose, design, and exact codebase locations.

---

## 2. The Transformer Model Architecture
**Purpose:** To leverage state-of-the-art Multi-Head Attention algorithms allowing the model to weigh different historical days dynamically instead of linearly, maximizing long-term sequential trend captures.

🟢 **Code Locations:**
* `model_definitions.py`: We engineered custom `TransformerEncoder` layers and embedded them into the core compilation stack, adjusting learning pipelines to utilize `Adam(lr=1e-4)`. 
* `train_models.py`: Added explicit logic blocks tracking the Transformer model configurations (150 epochs, batch size 16, and EarlyStopping mechanisms).
* `retrain_transformer.py`: A dedicated execution framework allowing isolated tuning of the Transformer module across NIFTY-50 tickers without disrupting baseline model caches.
* `templates/dashboard.html` & `app.py`: Embedded the Transformer dynamically into the global selection loops and API pipelines.

---

## 3. Real-Time Sentiment Analysis (FinBERT)
**Purpose:** The machine learning algorithms originally relied purely on numeric OHLCV structural data. We expanded the framework by extracting live news headlines directly from Yahoo Finance and passing them recursively through a HuggingFace Financial-BERT (FinBERT) Neural Network, deriving live Sentiment impact (-1.0 to 1.0).

🟢 **Code Locations:**
* `sentiment_analyzer.py` (NEW): Built the raw sequence-classification inference pipeline caching the `ProsusAI/finbert` CPU-weighted models. Includes extraction methodologies avoiding nested metadata lock-ups common in Yahoo Finance data structures.
* `features.py`: Added programmatic feature injection where `df['sentiment'] = 0.0` arrays align historical states to support live real-time variables inside scaled input tensors.
* `app.py`: Created `GET /api/sentiment/<ticker>` routing, mapping the NLP extraction limits securely for REST responses.
* `templates/dashboard.html`: Deployed an interactive Market Headlines Card showing Real-Time Badges (Positive/Negative/Neutral) natively alongside numeric metrics.

---

## 4. The Ensemble Strategy Integrator
**Purpose:** Built a robust voting system mitigating the risk of solitary neural hallucination by taking a generalized mathematical average of predictions modeled by all running architectures simultaneously. Weights applied: Transformer (35%), LSTM (25%), GRU (25%), CNN (15%). 

🟢 **Code Locations:**
* `load_model_pkg.py`: Added the `ensemble_predict()` operator logic. It routes dynamically through server memory tracking all distinct models, executing parallel inferences, evaluating against statistical weights, and executing custom scalar-tensor inverse transformations independently. 
* `app.py`: Upgraded `/api/predict` to flawlessly parse the "ensemble" database metric natively inside the temporal prediction `lookback` sequences across sequential forecast generation blocks.
* `templates/dashboard.html`: Unlocked inside the AI Model `<select>` block as an aggregate option explicitly mapped to the UI Leaderboards.

---

## 5. The "Reality Check" Metric Loop
**Purpose:** An entirely new Dashboard segment. This module is responsible for fetching historical stock predictions previously saved in our SQLite persistence databases up to 90 days ago, comparing those original predictions against the Actual close margins today.

🟢 **Code Locations:**
* `populate_mock_db.py` (NEW): Developed a realistic simulated-data generator modeling synthetic chronological noise mapping >500 rows precisely matching NIFTY constraints across past dates to fully activate the module visualization features.
* `app.py`: Created the `/reality-check` endpoint housing a dynamic, time-based local map caching system `YF_CACHE` preventing Yahoo Finance HTTP bans while calculating `mean_pct` tracking errors against local data natively.
* `templates/reality_check.html` (NEW): Developed the comprehensive Dashboard template hosting chronological Time-Series constraints mapping (Actual vs Predicted Prices). Mapped Chart.js line visualizations, KPI cards, sorting routines filtering Models/Tickers selectively, and a raw HTML `.csv` exporter module natively.
* `templates/base.html`: Extended global user navigation access allowing the module to be accessible securely inside the main project ecosystem.
