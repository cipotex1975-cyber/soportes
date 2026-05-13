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
        prev_body = body.shift(1)
        df['pattern_engulfing_bull'] = (df['Close'] > df['Open']) & (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
        df['pattern_engulfing_bear'] = (df['Close'] < df['Open']) & (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
        
        return df
        
        return df
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
        df['bb_width'] = (df.ta.bbands(length=20).iloc[:, 1] - df.ta.bbands(length=20).iloc[:, 2]) / df.ta.bbands(length=20).iloc[:, 0]
        
        # Microestructura de velas
        body = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        df['body_ratio'] = body / total_range.replace(0, 1)  # Evitar división por cero
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['upper_wick_ratio'] = upper_wick / total_range.replace(0, 1)
        df['lower_wick_ratio'] = lower_wick / total_range.replace(0, 1)
        
        # Tendencia
        df['ema_20_slope'] = df['EMA_20'] - df['EMA_20'].shift(10)
        df['distance_to_ema200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
        df.ta.adx(length=14, append=True)  # ADX para fuerza de tendencia
        
        # Volumen
        if 'Volume' in df.columns:
            df['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).quantile(0.95)).astype(int)
        
        # Breakout strength
        df['breakout_up'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)
        
        # Regime detection (simple: trend if volatility ratio > 1)
        df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_60']
        df['regime_trend'] = (df['volatility_ratio'] > 1).astype(int)
        
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
        df['bb_width'] = (df.ta.bbands(length=20).iloc[:, 1] - df.ta.bbands(length=20).iloc[:, 2]) / df.ta.bbands(length=20).iloc[:, 0]
        
        # Microestructura de velas
        body = abs(df['Close'] - df['Open'])
        total_range = df['High'] - df['Low']
        df['body_ratio'] = body / total_range.replace(0, 1)  # Evitar división por cero
        upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
        df['upper_wick_ratio'] = upper_wick / total_range.replace(0, 1)
        df['lower_wick_ratio'] = lower_wick / total_range.replace(0, 1)
        
        # Tendencia
        df['ema_20_slope'] = df['EMA_20'] - df['EMA_20'].shift(10)
        df['distance_to_ema200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
        df.ta.adx(length=14, append=True)  # ADX para fuerza de tendencia
        
        # Volumen
        if 'Volume' in df.columns:
            df['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).quantile(0.95)).astype(int)
        
        # Breakout strength
        df['breakout_up'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)
        
        # Regime detection (simple: trend if volatility ratio > 1)
        df['volatility_ratio'] = df['rolling_std_20'] / df['rolling_std_60']
        df['regime_trend'] = (df['volatility_ratio'] > 1).astype(int)
        
        return df
        """
        Prepare features for supervised learning.
        """
        df = self.add_indicators(df)
        df = self.detect_patterns(df)
        df = self.add_advanced_features(df)
        
        # Add target: 1 if price goes up by X% in next N bars, -1 if down, 0 otherwise
        # This is for training supervised models later
        return df.dropna()

    def get_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for supervised learning.
        """
        df = self.add_indicators(df)
        df = self.detect_patterns(df)
        df = self.add_advanced_features(df)
        
        # Add target: 1 if price goes up by X% in next N bars, -1 if down, 0 otherwise
        # This is for training supervised models later
        return df.dropna()
