import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.signal import argrelextrema
import joblib
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class MLService:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def detect_levels(self, df: pd.DataFrame, symbol: str, window: int = 5) -> Dict[str, Any]:
        """
        Detect support and resistance levels with enhanced strength classification.
        """
        # 1. Find local pivots
        df['min'] = df.iloc[argrelextrema(df.Low.values, np.less_equal, order=window)[0]]['Low']
        df['max'] = df.iloc[argrelextrema(df.High.values, np.greater_equal, order=window)[0]]['High']
        
        pivots = []
        for i in range(len(df)):
            if not np.isnan(df['min'].iloc[i]):
                pivots.append({'price': df['min'].iloc[i], 'index': i, 'vol': df['Volume'].iloc[i] if 'Volume' in df.columns else 1})
            if not np.isnan(df['max'].iloc[i]):
                pivots.append({'price': df['max'].iloc[i], 'index': i, 'vol': df['Volume'].iloc[i] if 'Volume' in df.columns else 1})

        if len(pivots) < 3:
            return {"supports": [], "resistances": [], "confidence": 0}

        pivot_prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        
        # 2. Clustering with DBSCAN
        price_range = df['High'].max() - df['Low'].min()
        eps = price_range * 0.015 
        db = DBSCAN(eps=eps, min_samples=2).fit(pivot_prices)
        
        clusters = {}
        for i, label in enumerate(db.labels_):
            if label == -1: continue
            if label not in clusters: clusters[label] = []
            clusters[label].append(pivots[i])

        levels = []
        current_time = datetime.now()
        for cluster in clusters.values():
            avg_price = np.mean([p['price'] for p in cluster])
            touches = len(cluster)
            total_vol = sum([p['vol'] for p in cluster])
            
            # Strength classification
            strength_score = touches * (1 + (total_vol / df['Volume'].mean() if 'Volume' in df.columns else 0))
            if strength_score > 10: strength = "very strong"
            elif strength_score > 5: strength = "strong"
            elif strength_score > 2: strength = "medium"
            else: strength = "weak"

            levels.append({
                "price": float(avg_price),
                "strength_label": strength,
                "strength_score": float(strength_score),
                "touches": touches,
                "min": float(np.min([p['price'] for p in cluster])),
                "max": float(np.max([p['price'] for p in cluster]))
            })

        current_price = df['Close'].iloc[-1]
        supports = sorted([l for l in levels if l['price'] < current_price], key=lambda x: x['price'], reverse=True)
        resistances = sorted([l for l in levels if l['price'] > current_price], key=lambda x: x['price'])

        result = {
            "symbol": symbol,
            "supports": supports,
            "resistances": resistances,
            "confidence": round(1.0 - (list(db.labels_).count(-1) / len(pivots)), 2) if pivots else 0,
            "analysis_date": datetime.now().strftime("%Y-%m-%d")
        }

        self._save_model(symbol, result, "levels")
        return result

    def train_supervised_model(self, X: pd.DataFrame, y: pd.Series, symbol: str):
        """
        Train an XGBoost model to predict bounce/breakout.
        """
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, learning_rate=0.05)
        model.fit(X, y)
        self._save_model(symbol, model, "xgboost")
        return model

    def train_lstm_model(self, data: np.array, symbol: str):
        """
        Train an LSTM model for price sequences.
        """
        # Note: Implementation requires tensorflow
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(data.shape[1], data.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        # model.fit(...) - Simplified for logic preservation
        self._save_model(symbol, model, "lstm")
        return model

    def _save_model(self, symbol: str, model: Any, type: str):
        suffix = "joblib" if type != "lstm" else "h5"
        model_path = os.path.join(self.models_dir, f"{symbol}_{type}.{suffix}")
        if type == "lstm":
            model.save(model_path)
        else:
            joblib.dump(model, model_path)

    def load_model(self, symbol: str, type: str) -> Optional[Any]:
        suffix = "joblib" if type != "lstm" else "h5"
        model_path = os.path.join(self.models_dir, f"{symbol}_{type}.{suffix}")
        if os.path.exists(model_path):
            if type == "lstm":
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            return joblib.load(model_path)
        return None
