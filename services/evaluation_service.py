from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

class EvaluationService:
    def evaluate_price(
        self, 
        symbol: str, 
        current_price: float, 
        analysis: Dict[str, Any],
        df: pd.DataFrame,
        ml_service: Any = None,
        use_model: str = "all",
        tolerance_percent: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluación integral permitiendo elegir el motor de decisión:
        use_model: 'none', 'xgboost', 'rl', 'all'
        """
        supports = analysis.get('supports', [])
        resistances = analysis.get('resistances', [])
        
        all_levels = supports + resistances
        if not all_levels:
            return {"error": "No levels detected for this symbol"}

        nearest_level_obj = min(all_levels, key=lambda x: abs(x['price'] - current_price))
        nearest_level = nearest_level_obj['price']
        distance_pct = (abs(current_price - nearest_level) / current_price) * 100

        zone_type = "neutral"
        suggested_action = "hold"
        bounce_prob = 0.5
        breakout_prob = 0.5
        fake_breakout_risk = 0.1
        models_used = []

        # 1. Base Logic (Heuristics/Statistics)
        rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
        volume_ma = df['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in df.columns else 1
        current_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 1
        volume_confirmation = current_volume > volume_ma

        # 2. ML Model Logic (XGBoost)
        if ml_service and use_model in ["xgboost", "all"]:
            xgb_model = ml_service.load_model(symbol, "xgboost")
            if xgb_model:
                try:
                    from services.feature_service import FeatureService
                    fs = FeatureService()
                    X = fs.get_features_for_ml(df.tail(5)).tail(1).drop(columns=['Close'], errors='ignore')
                    prediction = xgb_model.predict(X)[0]
                    if prediction == 1: bounce_prob += 0.15
                    else: breakout_prob += 0.15
                    models_used.append("xgboost")
                except: pass

        # 3. RL Model Logic (PPO Agent)
        if ml_service and use_model in ["rl", "all"]:
            from services.rl_service import RLService
            rl_service = RLService()
            agent = rl_service.load_agent(symbol)
            if agent:
                try:
                    # Preparar la observación igual que en el entorno de entrenamiento
                    # Necesitamos pasarle los niveles actuales
                    clean_df = df.tail(1).copy()
                    sup_list = [l['price'] for l in supports]
                    res_list = [l['price'] for l in resistances]
                    clean_df['dist_to_sup'] = clean_df['Close'].apply(lambda x: min([abs(x - s) for s in sup_list]) / x if sup_list else 1.0)
                    clean_df['dist_to_res'] = clean_df['Close'].apply(lambda x: min([abs(x - r) for r in res_list]) / x if res_list else 1.0)
                    
                    obs = clean_df.values.astype(np.float32)
                    action, _ = agent.predict(obs)
                    
                    # 0: Hold, 1: Buy, 2: Sell
                    if action == 1: bounce_prob += 0.2
                    elif action == 2: bounce_prob -= 0.1 # Probabilidad de rebote baja si el agente vende
                    models_used.append(f"rl_agent(action={int(action)})")
                except: pass

        # 4. Decision Logic
        is_near = distance_pct <= tolerance_percent
        if is_near:
            if current_price > nearest_level: 
                zone_type = "support"
                if rsi < 40: bounce_prob += 0.1
                suggested_action = "possible_buy" if bounce_prob > 0.6 else "hold"
            else:
                zone_type = "resistance"
                if rsi > 60: bounce_prob += 0.1
                suggested_action = "possible_sell" if bounce_prob > 0.6 else "hold"
        else:
            if resistances and current_price > max([l['price'] for l in resistances]):
                zone_type = "breakout"
                if volume_confirmation: breakout_prob += 0.1
                suggested_action = "possible_buy" if breakout_prob > 0.6 else "wait_confirmation"
            elif supports and current_price < min([l['price'] for l in supports]):
                zone_type = "breakout"
                if volume_confirmation: breakout_prob += 0.1
                suggested_action = "possible_sell" if breakout_prob > 0.6 else "wait_confirmation"

        return {
            "symbol": symbol,
            "current_price": round(current_price, 5),
            "zone_type": zone_type,
            "nearest_level": round(nearest_level, 5),
            "level_strength": nearest_level_obj.get('strength_label', 'unknown'),
            "bounce_probability": round(min(bounce_prob, 0.99), 2),
            "breakout_probability": round(min(breakout_prob, 0.99), 2),
            "fake_breakout_risk": round(fake_breakout_risk, 2),
            "rsi": round(rsi, 2),
            "volume_confirmation": bool(volume_confirmation),
            "models_involved": models_used if models_used else ["statistics_only"],
            "suggested_action": suggested_action
        }
