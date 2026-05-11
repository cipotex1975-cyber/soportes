import sys
import os
from services.data_service import DataService
from services.ml_service import MLService
from services.visualization_service import VisualizationService
from services.evaluation_service import EvaluationService
import json

def run_analysis(symbols=["EURUSD=X", "GBPUSD=X", "USDJPY=X"], period="2y", interval="1d"):
    data_service = DataService()
    ml_service = MLService()
    viz_service = VisualizationService()
    eval_service = EvaluationService()

    results = []

    for symbol in symbols:
        try:
            print(f"\n--- Analyzing {symbol} ---")
            # 1. Fetch Data
            df = data_service.fetch_data(symbol, period=period, interval=interval)
            
            # 2. Detect Levels
            analysis = ml_service.detect_levels(df, symbol)
            
            # 3. Generate Charts
            png_path = viz_service.save_static_chart(df, analysis)
            html_path = viz_service.save_interactive_chart(df, analysis)
            
            print(f"S/R Levels detected: {len(analysis['supports'])} supports, {len(analysis['resistances'])} resistances")
            print(f"Charts saved: {png_path}, {html_path}")
            
            results.append(analysis)

        except Exception as e:
            print(f"Failed to analyze {symbol}: {e}")

    # Save summary results
    with open("data/analysis_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def evaluate_current_price(symbol, tolerance=0.5):
    data_service = DataService()
    ml_service = MLService()
    eval_service = EvaluationService()

    analysis = ml_service.load_levels(symbol)
    if not analysis:
        print(f"No previous analysis found for {symbol}. Running full analysis first...")
        run_analysis([symbol])
        analysis = ml_service.load_levels(symbol)

    current_price = data_service.get_latest_price(symbol)
    evaluation = eval_service.evaluate_price(symbol, current_price, analysis, tolerance)
    
    print("\n--- Price Evaluation ---")
    print(json.dumps(evaluation, indent=2))
    return evaluation

if __name__ == "__main__":
    # Default behavior: run analysis for major pairs
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "EURUSD=X"
        evaluate_current_price(symbol)
    else:
        run_analysis()
