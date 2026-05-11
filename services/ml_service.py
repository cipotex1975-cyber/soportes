import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.signal import argrelextrema
import joblib
import os
from typing import List, Dict, Any
from datetime import datetime

class MLService:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def detect_levels(self, df: pd.DataFrame, symbol: str, window: int = 5) -> Dict[str, Any]:
        """
        Detect support and resistance levels using pivots and DBSCAN clustering.
        """
        # 1. Find local pivots (min and max)
        # We use a window to define what a 'pivot' is
        df['min'] = df.iloc[argrelextrema(df.Low.values, np.less_equal, order=window)[0]]['Low']
        df['max'] = df.iloc[argrelextrema(df.High.values, np.greater_equal, order=window)[0]]['High']
        
        pivots = df['min'].dropna().tolist() + df['max'].dropna().tolist()
        pivots = np.array(pivots).reshape(-1, 1)

        if len(pivots) < 3:
            return {"supports": [], "resistances": [], "confidence": 0}

        # 2. Clustering with DBSCAN
        # eps is the distance between points to be considered in the same cluster
        # This needs to be relative to the price range
        price_range = df['High'].max() - df['Low'].min()
        eps = price_range * 0.015 # 1.5% of range as cluster threshold
        
        db = DBSCAN(eps=eps, min_samples=2).fit(pivots)
        
        clusters = {}
        for i, label in enumerate(db.labels_):
            if label == -1: continue # Noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(pivots[i][0])

        levels = []
        for cluster in clusters.values():
            levels.append({
                "price": float(np.mean(cluster)),
                "strength": len(cluster),
                "min": float(np.min(cluster)),
                "max": float(np.max(cluster))
            })

        # 3. Categorize into Support and Resistance based on current price
        current_price = df['Close'].iloc[-1]
        supports = sorted([l['price'] for l in levels if l['price'] < current_price], reverse=True)
        resistances = sorted([l['price'] for l in levels if l['price'] > current_price])

        # Confidence is a proxy based on how many clusters we found vs noise
        noise_count = list(db.labels_).count(-1)
        confidence = 1.0 - (noise_count / len(pivots)) if len(pivots) > 0 else 0

        result = {
            "symbol": symbol,
            "supports": supports,
            "resistances": resistances,
            "confidence": round(confidence, 2),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "raw_levels": levels
        }

        # Save result
        self._save_model(symbol, result)
        
        return result

    def _save_model(self, symbol: str, data: Dict):
        model_path = os.path.join(self.models_dir, f"{symbol}_levels.joblib")
        joblib.dump(data, model_path)

    def load_levels(self, symbol: str) -> Optional[Dict]:
        model_path = os.path.join(self.models_dir, f"{symbol}_levels.joblib")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
