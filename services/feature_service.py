import pandas as pd
import pandas_ta as ta
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

    def get_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for supervised learning.
        """
        df = self.add_indicators(df)
        df = self.detect_patterns(df)
        
        # Add target: 1 if price goes up by X% in next N bars, -1 if down, 0 otherwise
        # This is for training supervised models later
        return df.dropna()
