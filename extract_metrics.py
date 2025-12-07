import os
import json
import pandas as pd

def extract_metrics(base_dir="models"):
    metrics_data = []
    
    # Traverse directory
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            try:
                with open(os.path.join(root, "config.json"), 'r') as f:
                    config = json.load(f)
                    
                metrics_data.append({
                    "Ticker": config.get("ticker", "Unknown"),
                    "Model": config.get("model_type", "Unknown").upper(),
                    "MAE": config.get("mae", 0.0),
                    "RMSE": config.get("rmse", 0.0),
                    "Date Trained": config.get("date_trained", "N/A")
                })
            except Exception as e:
                print(f"Error reading {root}: {e}")
                
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Filter out Unknown tickers (legacy models)
    df = df[df['Ticker'] != "Unknown"]
    
    # Sort
    df = df.sort_values(by=["Ticker", "Model"])
    
    # Generate Markdown Table Manually to avoid tabulate dependency
    md_content = "# Model Training Metrics\n\n"
    md_content += "This document lists the performance metrics (MAE and RMSE) for all trained models on the test set (2023-2024).\n\n"
    
    # Table Header
    headers = ["Ticker", "Model", "MAE", "RMSE", "Date Trained"]
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Table Rows
    for _, row in df.iterrows():
        line = f"| {row['Ticker']} | {row['Model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['Date Trained']} |"
        md_content += line + "\n"
    
    # Save to file
    with open("MODEL_METRICS.md", "w") as f:
        f.write(md_content)
        
    print("Successfully generated MODEL_METRICS.md")
    print(df)

if __name__ == "__main__":
    extract_metrics()
