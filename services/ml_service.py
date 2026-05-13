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

    
    def train_supervised_model(self, X, y, symbol: str):

        from xgboost import XGBClassifier

        from sklearn.model_selection import train_test_split

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
            roc_auc_score
        )

        # ======================================
        # Split temporal
        # ======================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False
        )

        # ======================================
        # Modelo
        # ======================================

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # ======================================
        # Entrenamiento
        # ======================================

        print(f"\nEntrenando XGBoost para {symbol}...\n")

        model.fit(
            X_train,
            y_train
        )

        # ======================================
        # Predicciones
        # ======================================

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        # ======================================
        # Métricas
        # ======================================

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_prob)

        print("\n==============================")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")
        print("==============================\n")

        print("Classification Report:\n")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:\n")
        print(confusion_matrix(y_test, y_pred))

        # ======================================
        # Feature importance
        # ======================================

        feature_importance = sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nTop Features:")

        for feature, importance in feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")

        # ======================================
        # Guardar
        # ======================================

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

    def _get_model_path(self, symbol: str, type: str) -> str:
        suffix = "joblib" if type != "lstm" else "h5"
        return os.path.join(self.models_dir, f"{symbol}_{type}.{suffix}")

    def _save_model(self, symbol: str, model: Any, type: str):
        model_path = self._get_model_path(symbol, type)
        if type == "lstm":
            model.save(model_path)
        else:
            joblib.dump(model, model_path)

    def model_exists(self, symbol: str, type: str) -> bool:
        model_path = self._get_model_path(symbol, type)
        return os.path.exists(model_path)

    def load_model(self, symbol: str, type: str) -> Optional[Any]:
        model_path = self._get_model_path(symbol, type)
        if not os.path.exists(model_path):
            print(f"[MLService] Modelo entrenado no encontrado: {model_path}")
            return None

        if type == "lstm":
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        return joblib.load(model_path)
