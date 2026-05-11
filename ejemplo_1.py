import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.ml_service import MLService
from services.evaluation_service import EvaluationService
from services.data_service import DataService

# 1. Cargar servicios
ml = MLService()
ev = EvaluationService()
ds = DataService()

# 2. Cargar el análisis previamente guardado
symbol = "EURUSD=X"
analysis = ml.load_levels(symbol)

# 3. Obtener precio actual y evaluar
current_price = ds.get_latest_price(symbol)
recommendation = ev.evaluate_price(symbol, current_price, analysis)

print(f"Sugerencia para {symbol}: {recommendation['suggested_action']}")
print(f"Sugerencia para {symbol}: {recommendation}")

