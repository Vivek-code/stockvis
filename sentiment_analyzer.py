"""
FinBERT-based sentiment analyzer for stock news headlines.
Uses yfinance to fetch news and ProsusAI/finbert for classification.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yfinance as yf

# Module-level cache for the FinBERT model and tokenizer
_finbert_tokenizer = None
_finbert_model = None
_pipeline_load_failed = False

def _load_pipeline():
    """
    Lazily loads the FinBERT tokenizer and model.
    Returns (tokenizer, model) or None if loading fails.
    """
    global _finbert_tokenizer, _finbert_model, _pipeline_load_failed

    if _finbert_tokenizer is not None and _finbert_model is not None:
        return _finbert_tokenizer, _finbert_model

    if _pipeline_load_failed:
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("Loading FinBERT model and tokenizer (first time only)...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert_model.eval()  # Set to evaluation mode
        print("FinBERT loaded successfully.")
        return _finbert_tokenizer, _finbert_model
    except Exception as e:
        print(f"Failed to load FinBERT: {e}")
        _pipeline_load_failed = True
        return None


def get_sentiment_score(ticker_symbol):
    """
    Fetches recent news headlines for a ticker and scores them using FinBERT.

    Args:
        ticker_symbol (str): e.g. 'RELIANCE.NS'

    Returns:
        tuple: (score: float, headlines: list)
            score is mean(positive - negative) across all headlines, range [-1.0, 1.0].
            headlines is a list of the parsed string titles.
            Returns (0.0, []) if no news found or any error occurs.
    """
    try:
        # Fetch news from yfinance
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news

        if not news:
            return 0.0, []

        # Extract headline titles — yfinance wraps news in a nested structure
        headlines = []
        for item in news:
            title = None
            if isinstance(item, dict):
                # Current yfinance format: item['content']['title']
                content = item.get('content')
                if isinstance(content, dict):
                    title = content.get('title')
                # Fallback to flat format (older yfinance versions)
                if not title:
                    title = item.get('title') or item.get('headline')
            if title:
                headlines.append(title)

        headlines = headlines[:5]  # Limit to 5 headlines to prevent very slow CPU inference

        # Load FinBERT
        models = _load_pipeline()
        if models is None:
            return 0.0, headlines
            
        tokenizer, model = models

        # Score each headline
        scores_list = []
        import torch
        for headline in headlines:
            try:
                # FinBERT max 512 tokens
                inputs = tokenizer(headline[:512], return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
                
                # ProsusAI/finbert labels: 0: positive, 1: negative, 2: neutral (usually in this order, but let's map safely)
                # get config mapping
                id2label = model.config.id2label
                
                label_scores = {id2label[i].lower(): probs[i] for i in range(len(probs))}
                
                pos = label_scores.get('positive', 0.0)
                neg = label_scores.get('negative', 0.0)
                scores_list.append(pos - neg)
            except Exception as e:
                print(f"Error analyzing headline: {e}")
                continue

        if not scores_list:
            return 0.0, headlines

        mean_score = sum(scores_list) / len(scores_list)
        return float(mean_score), headlines

    except Exception as e:
        print(f"Sentiment analysis error for {ticker_symbol}: {e}")
        return 0.0, []


def get_batch_sentiment(tickers_list):
    """
    Returns sentiment scores for multiple tickers.

    Args:
        tickers_list (list): List of ticker symbols.

    Returns:
        dict: {ticker: {'score': float, 'headlines_analyzed': int}}
    """
    results = {}
    for ticker in tickers_list:
        score, headlines = get_sentiment_score(ticker)
        results[ticker] = {
            'score': score,
            'headlines': headlines,
            'headlines_analyzed': len(headlines)
        }
    return results


if __name__ == "__main__":
    # Quick test
    score, count = get_sentiment_score("RELIANCE.NS")
    print(f"RELIANCE.NS — Sentiment: {score:.4f}, Headlines: {count}")
