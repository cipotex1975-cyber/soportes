from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.data_service import DataService
from services.ml_service import MLService
from services.evaluation_service import EvaluationService
from services.feature_service import FeatureService
from services.rl_service import RLService
import uvicorn

app = FastAPI(title="Advanced Financial AI Analysis API")

# Initialize services
data_service = DataService()
ml_service = MLService()
eval_service = EvaluationService()
feature_service = FeatureService()
rl_service = RLService()

class AnalysisRequest(BaseModel):
    symbols: List[str] = ["EURUSD=X"]
    period: str = "2y"
    interval: str = "1d"

class EvaluationRequest(BaseModel):
    symbol: str
    tolerance_percent: float = 0.5
    use_model: str = "all" # 'none', 'xgboost', 'rl', 'all'

@app.get("/")
def read_root():
    return {"message": "Advanced Financial AI Analysis API is running"}

@app.post("/analyze_symbol")
def analyze_symbol(request: EvaluationRequest):
    """
    Comprehensive analysis including indicators and ML levels.
    """
    try:
        df = data_service.fetch_data(request.symbol)
        df = feature_service.add_indicators(df)
        df = feature_service.detect_patterns(df)
        
        analysis = ml_service.detect_levels(df, request.symbol)
        current_price = data_service.get_latest_price(request.symbol)
        
        evaluation = eval_service.evaluate_price(
            request.symbol, current_price, analysis, df, ml_service=ml_service, use_model=request.use_model, tolerance_percent=request.tolerance_percent
        )
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_supervised")
def train_supervised(symbol: str):
    """
    Train XGBoost model for a symbol.
    """
    try:
        df = data_service.fetch_data(symbol,"5y","1d")
        df = feature_service.get_features_for_ml(df)
        # Placeholder for target generation logic
        y = (df['Close'].shift(-1) > df['Close']).astype(int)
        X = df.drop(columns=['Close'])
        ml_service.train_supervised_model(X, y, symbol)
        return {"status": "success", "symbol": symbol}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_rl")
def train_rl(symbol: str):
    """
    Train RL Agent for a symbol considering S/R levels.
    """
    try:
        df = data_service.fetch_data(symbol, "5y", "1d")
        df = feature_service.add_indicators(df)
        
        # Detectar niveles para que el agente aprenda de ellos
        levels = ml_service.detect_levels(df, symbol)
        
        rl_service.train_agent(df, symbol, levels)
        return {"status": "success", "symbol": symbol, "levels_detected": len(levels['supports']) + len(levels['resistances'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
