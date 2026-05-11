from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.data_service import DataService
from services.ml_service import MLService
from services.evaluation_service import EvaluationService
import uvicorn

app = FastAPI(title="Forex AI Technical Analysis API")

# Initialize services
data_service = DataService()
ml_service = MLService()
eval_service = EvaluationService()

class AnalysisRequest(BaseModel):
    symbols: List[str] = ["EURUSD=X"]
    period: str = "2y"
    interval: str = "1d"

class EvaluationRequest(BaseModel):
    symbol: str
    tolerance_percent: float = 0.5

@app.get("/")
def read_root():
    return {"message": "Forex AI Analysis API is running"}

@app.post("/train")
def train_models(request: AnalysisRequest):
    results = []
    for symbol in request.symbols:
        try:
            df = data_service.fetch_data(symbol, request.period, request.interval)
            analysis = ml_service.detect_levels(df, symbol)
            results.append(analysis)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    return results

@app.get("/levels/{symbol}")
def get_levels(symbol: str):
    analysis = ml_service.load_levels(symbol)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis found for {symbol}. Run /train first.")
    return analysis

@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    analysis = ml_service.load_levels(request.symbol)
    if not analysis:
        # Try to train on the fly
        try:
            df = data_service.fetch_data(request.symbol)
            analysis = ml_service.detect_levels(df, request.symbol)
        except:
            raise HTTPException(status_code=404, detail=f"Could not analyze {request.symbol}")
    
    current_price = data_service.get_latest_price(request.symbol)
    evaluation = eval_service.evaluate_price(request.symbol, current_price, analysis, request.tolerance_percent)
    return evaluation

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
