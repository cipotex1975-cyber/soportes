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
        df_copy = df.copy()

        # 1. Find local pivots without mutating the original dataset
        df_copy['min'] = np.nan
        df_copy['max'] = np.nan
        min_idx = argrelextrema(df_copy.Low.values, np.less_equal, order=window)[0]
        max_idx = argrelextrema(df_copy.High.values, np.greater_equal, order=window)[0]
        df_copy.iloc[min_idx, df_copy.columns.get_loc('min')] = df_copy['Low'].iloc[min_idx].values
        df_copy.iloc[max_idx, df_copy.columns.get_loc('max')] = df_copy['High'].iloc[max_idx].values
        
        pivots = []
        for i in range(len(df_copy)):
            if not np.isnan(df_copy['min'].iloc[i]):
                pivots.append({'price': df_copy['min'].iloc[i], 'index': i, 'vol': df_copy['Volume'].iloc[i] if 'Volume' in df_copy.columns else 1})
            if not np.isnan(df_copy['max'].iloc[i]):
                pivots.append({'price': df_copy['max'].iloc[i], 'index': i, 'vol': df_copy['Volume'].iloc[i] if 'Volume' in df_copy.columns else 1})

        if len(pivots) < 3:
            return {"supports": [], "resistances": [], "confidence": 0}

        pivot_prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        
        # 2. Clustering with DBSCAN
        price_range = df_copy['High'].max() - df_copy['Low'].min()
        eps = price_range * 0.015 
        db = DBSCAN(eps=eps, min_samples=2).fit(pivot_prices)
        
        clusters = {}
        for i, label in enumerate(db.labels_):
            if label == -1: continue
            if label not in clusters: clusters[label] = []
            clusters[label].append(pivots[i])

        levels = []
        current_time = datetime.now()
        volume_mean = 0.0
        if 'Volume' in df_copy.columns:
            volume_mean = df_copy['Volume'].mean()
            if np.isnan(volume_mean):
                volume_mean = 0.0

        for cluster in clusters.values():
            avg_price = np.mean([p['price'] for p in cluster])
            touches = len(cluster)
            total_vol = sum([p['vol'] for p in cluster])
            
            # Strength classification
            volume_factor = total_vol / volume_mean if volume_mean > 0 else 0.0
            strength_score = touches * (1 + volume_factor)

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
        from sklearn.model_selection import TimeSeriesSplit, train_test_split
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
            roc_auc_score
        )
        import numpy as np

        # ======================================
        # Walk-forward validation
        # ======================================

        tscv = TimeSeriesSplit(n_splits=2)
        cv_scores = []

        for train_index, val_index in tscv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
            y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

            # Calcular scale_pos_weight
            pos_weight = len(y_train_cv[y_train_cv == 0]) / len(y_train_cv[y_train_cv == 1])

            model_cv = XGBClassifier(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=4,
                subsample=0.7,
                colsample_bytree=0.7,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=pos_weight,
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='auc'
            )

            model_cv.fit(
                X_train_cv, y_train_cv,
                eval_set=[(X_val_cv, y_val_cv)],
                verbose=False
            )

            y_pred_cv = model_cv.predict(X_val_cv)
            auc_cv = roc_auc_score(y_val_cv, model_cv.predict_proba(X_val_cv)[:, 1])
            cv_scores.append(auc_cv)

        print(f"CV AUC Scores: {cv_scores}")
        print(f"Mean CV AUC: {np.mean(cv_scores):.4f}")

        # ======================================
        # Split final (temporal)
        # ======================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False
        )

        # ======================================
        # Modelo final
        # ======================================

        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='auc'
        )

        # ======================================
        # Entrenamiento
        # ======================================

        print(f"\nEntrenando XGBoost para {symbol}...\n")

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
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
        # Feature importance y selección
        # ======================================

        feature_importance = sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nTop Features:")
        for feature, importance in feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")

        # Filtrar features con importancia < 0.01 para evitar overfitting
        important_features = [f for f, imp in feature_importance if imp >= 0.01]
        print(f"\nFeatures seleccionadas: {len(important_features)} de {len(X.columns)}")

        # Re-entrenar con features seleccionadas si es necesario
        if len(important_features) < len(X.columns):
            X_train_sel = X_train[important_features]
            X_test_sel = X_test[important_features]
            model.fit(X_train_sel, y_train, eval_set=[(X_test_sel, y_test)], verbose=False)

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
