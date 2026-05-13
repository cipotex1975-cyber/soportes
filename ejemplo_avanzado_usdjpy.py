import json
import pandas as pd
from services.data_service import DataService
from services.ml_service import MLService
from services.feature_service import FeatureService
from services.evaluation_service import EvaluationService
from services.rl_service import RLService

def entrenar_agente_rl(symbol="USDJPY=X"):
    print(f"\n--- Entrenando Agente RL para {symbol} ---")
    data_service = DataService()
    feature_service = FeatureService()
    ml_service = MLService()
    rl_service = RLService()

    df = data_service.fetch_data(symbol, "15y", "1d")
    df = feature_service.add_indicators(df)
    levels = ml_service.detect_levels(df, symbol)
    
    rl_service.train_agent(df, symbol, levels)
    print(f"Entrenamiento completado. Modelo guardado en models/reinforcement_learning/")

def entrenar_agente_supervised(symbol="USDJPY=X"):
    print(f"\n--- Entrenando Modelo Supervisado (XGBoost) para {symbol} ---")
    data_service = DataService()
    feature_service = FeatureService()
    ml_service = MLService()

    # 1. Obtener datos y características
    df = data_service.fetch_data(symbol, "5y", "1d")
    df = feature_service.get_features_for_ml(df)
    
    # 2. Generar Target: 1 si el precio subió mañana, 0 si bajó
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = df.drop(columns=['Close'], errors='ignore')
    
    # 3. Entrenar
    ml_service.train_supervised_model(X, y, symbol)
    print(f"Entrenamiento XGBoost completado. Modelo guardado en models/")

def analizar_usdjpy(model_type="all"):
    # 1. Inicializar servicios
    data_service = DataService()
    ml_service = MLService()
    feature_service = FeatureService()
    eval_service = EvaluationService()

    symbol = "USDJPY=X"
    print(f"--- Iniciando análisis ({model_type}) para {symbol} ---")

    # 2. Obtener datos históricos
    df = data_service.fetch_data(symbol, period="2y", interval="1d")
    
    # 3. Aplicar Ingeniería de Características
    df = feature_service.get_features_for_ml(df)
    
    # 4. Detectar niveles
    analysis = ml_service.detect_levels(df, symbol)

    # 5. Evaluación avanzada (Eliges qué modelo usar: 'none', 'xgboost', 'rl', 'all')
    current_price = data_service.get_latest_price(symbol)
    evaluation = eval_service.evaluate_price(
        symbol, current_price, analysis, df, 
        ml_service=ml_service, use_model=model_type
    )

    # 6. Calcular métricas de trading si hay modelo XGBoost
    trading_metrics = {}
    model = ml_service.load_model(symbol, "xgboost")
    trained_features = ml_service.load_feature_names(symbol)
    
    if model and trained_features:
        try:
            # Usar datos recientes para simular
            test_df = df.tail(100).copy()  # Últimos 100 días
            
            # IMPORTANTE: Usar EXACTAMENTE las mismas features que se usaron en entrenamiento
            # Si faltan, llenarlas con 0; si sobran, ignorarlas
            X_test = pd.DataFrame(index=test_df.index)
            
            for feat in trained_features:
                if feat in test_df.columns:
                    X_test[feat] = test_df[feat]
                else:
                    # Feature no disponible en datos de análisis, llenar con promedio histórico
                    X_test[feat] = df[feat].mean() if feat in df.columns else 0
            
            # Garantizar que el orden y nombre de las columnas sea exacto
            X_test = X_test[trained_features]
            
            # Excluir última fila (no tiene target)
            predictions = model.predict_proba(X_test[:-1])[:, 1]
            actual_returns = (test_df['Close'].shift(-1) / test_df['Close'] - 1).dropna().values
            
            # Alinear longitudes
            min_len = min(len(predictions), len(actual_returns))
            predictions = predictions[:min_len]
            actual_returns = actual_returns[:min_len]
            
            if len(predictions) > 0:
                trading_metrics = eval_service.calculate_trading_metrics(predictions, actual_returns)
                print(f"[✓ Métricas de trading calculadas: {len(predictions)} predicciones]")
        except Exception as e:
            print(f"[Warning] No se pudieron calcular métricas de trading: {str(e)[:100]}")

    # 7. Mostrar resultado final (JSON avanzado)
    print("\n--- RESULTADO DE LA EVALUACIÓN ---")
    print(json.dumps(evaluation, indent=2))
    
    if trading_metrics:
        print("\n--- MÉTRICAS DE TRADING ---")
        print(json.dumps(trading_metrics, indent=2))

    # 7. Ejemplo de cómo se vería una recomendación
    if evaluation['suggested_action'] != "hold":
        print(f"\n🚀 OPORTUNIDAD DETECTADA: {evaluation['suggested_action'].upper()}")
        print(f"Fuerza del nivel cercano: {evaluation['level_strength'].upper()}")

if __name__ == "__main__":
    import sys
    # Ejemplo de uso:
    # py ejemplo_avanzado_usdjpy.py train_rl       -> Entrena Reinforcement Learning
    # py ejemplo_avanzado_usdjpy.py train_ml       -> Entrena XGBoost (Supervisado)
    # py ejemplo_avanzado_usdjpy.py analyze rl     -> Analiza usando solo RL
    # py ejemplo_avanzado_usdjpy.py analyze xgboost -> Analiza usando solo XGBoost
    # py ejemplo_avanzado_usdjpy.py analyze all   -> Analiza con todos los modelos
    # py ejemplo_avanzado_usdjpy.py analyze none   -> Analiza solo con estadísticas
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    
    if mode == "train_rl":
        entrenar_agente_rl()
    elif mode == "train_ml":
        entrenar_agente_supervised()
    else:
        model_type = sys.argv[2] if len(sys.argv) > 2 else "all"
        analizar_usdjpy(model_type)
