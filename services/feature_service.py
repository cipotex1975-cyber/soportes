import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, List

class FeatureService:
    def __init__(self):
        pass

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using pandas_ta.
        """
        # Trend
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        # Momentum
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        
        # Volatility
        df.ta.atr(length=14, append=True)
        
        # Volume
        if 'Volume' in df.columns:
            df.ta.obv(append=True)
            
        return df

    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns.
        """
        # Simple manual detection for key patterns
        # Hammer: Small body, long lower wick, little/no upper wick
        body = abs(df['Close'] - df['Open'])
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        df['pattern_hammer'] = (lower_wick > 2 * body) & (upper_wick < 0.1 * body)
        
        # Doji: Open and Close are very close
        df['pattern_doji'] = body < 0.1 * (df['High'] - df['Low'])
        
        # Engulfing: Current body covers previous body
        df['pattern_engulfing_bull'] = (df['Close'] > df['Open']) & (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
        df['pattern_engulfing_bear'] = (df['Close'] < df['Open']) & (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
        
        return df

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced trading features for better ML performance.
        """
        # Momentum real
        df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        df['log_return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
        df['momentum_20d'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        
        # Volatilidad
        df['rolling_std_20'] = df['log_return_1d'].rolling(20).std()
        df['rolling_std_60'] = df['log_return_1d'].rolling(60).std()
        
        # BBands - handle potential missing columns
        try:
            bbands = df.ta.bbands(length=20)
            if bbands is not None:
                df['bb_width'] = (bbands.iloc[:, 2] - bbands.iloc[:, 0]) / bbands.iloc[:, 1].replace(0, 1)
        except:
            df['bb_width'] = 0
        
        # Microestructura de velas
        body = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        df['body_ratio'] = body / total_range.replace(0, 1)
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['upper_wick_ratio'] = upper_wick / total_range.replace(0, 1)
        df['lower_wick_ratio'] = lower_wick / total_range.replace(0, 1)
        
        # Tendencia
        if 'EMA_20' in df.columns:
            df['ema_20_slope'] = df['EMA_20'] - df['EMA_20'].shift(10)
        else:
            df['ema_20_slope'] = 0
            
        if 'EMA_200' in df.columns:
            df['distance_to_ema200'] = (df['Close'] - df['EMA_200']) / df['EMA_200'].replace(0, 1)
        else:
            df['distance_to_ema200'] = 0
            
        try:
            df.ta.adx(length=14, append=True)
        except:
            pass
        
        # Volumen - Mejorado para manejar Forex/Criptos sin volumen
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            vol_mean = df['Volume'].rolling(20).mean()
            df['relative_volume'] = df['Volume'] / vol_mean.replace(0, 1)
            df['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).quantile(0.95)).astype(int)
        else:
            df['relative_volume'] = 0
            df['volume_spike'] = 0
        
        # Breakout strength
        df['breakout_up'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)
        
        # Regime detection
        df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_60'].replace(0, 1)
        df['regime_trend'] = (df['volatility_ratio'] > 1).astype(int)
        
        # Rellenar NaNs de indicadores iniciales con 0 antes del dropna final
        # para no perder el dataset completo si solo fallan los primeros registros
        cols_to_fix = ['ema_20_slope', 'distance_to_ema200', 'relative_volume', 'volatility_ratio']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def get_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for supervised learning.
        """
        df = df.copy()
        df = self.add_indicators(df)
        df = self.detect_patterns(df)
        df = self.add_advanced_features(df)
        
        # Dropna() al final, pero ahora es más seguro porque inicializamos 
        # con 0 las columnas críticas si el volumen falla.
        df_cleaned = df.dropna()
        
        if len(df_cleaned) == 0:
            print("[Warning] Dataset vacío tras dropna. Reintentando con fillna(0).")
            return df.fillna(0)
            
        return df_cleaned
