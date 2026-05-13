import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class FakeBreakoutService:
    """
    Servicio para calcular la probabilidad de que una ruptura sea falsa (Fake Breakout).
    Basado en múltiples señales técnicas y de volumen.
    """

    def calculate_fake_breakout_risk(self, df: pd.DataFrame, breakout_level: float, direction: str) -> float:
        """
        Calcula el riesgo acumulativo de un fake breakout entre 0.0 y 1.0.
        direction: 'up' (resistencia rota) o 'down' (soporte roto)
        """
        if df.empty or len(df) < 20:
            return 0.1  # Valor por defecto si no hay suficientes datos

        fake_breakout_score = 0.0
        current_price = df['Close'].iloc[-1]

        # 1. Confirmación por volumen (0.25)
        if not self.evaluate_volume_confirmation(df):
            fake_breakout_score += 0.25

        # 2. Tiempo fuera del rango (0.30)
        if self.evaluate_time_outside(df, breakout_level, direction) < 2:
            fake_breakout_score += 0.30

        # 3. Fuerza de la ruptura (0.15)
        strength = abs(current_price - breakout_level) / breakout_level
        if strength < 0.001:
            fake_breakout_score += 0.15

        # 4. Divergencia RSI (0.20)
        if self.detect_rsi_divergence(df, direction):
            fake_breakout_score += 0.20

        # 5. Confirmación MACD (0.15)
        if not self.evaluate_macd_confirmation(df, direction):
            fake_breakout_score += 0.15

        # 6. Patrones de velas de rechazo (0.25)
        if self.detect_rejection_candle(df, direction):
            fake_breakout_score += 0.25

        # 7. Momentum del precio (0.10)
        if self.evaluate_momentum(df, direction) == "weak":
            fake_breakout_score += 0.10

        return min(fake_breakout_score, 1.0)

    def evaluate_volume_confirmation(self, df: pd.DataFrame) -> bool:
        """Valida si la ruptura ocurre con volumen superior a la media de 20."""
        if 'Volume' not in df.columns:
            return True
        
        volume_ma = df['Volume'].rolling(20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        return current_volume >= volume_ma

    def evaluate_time_outside(self, df: pd.DataFrame, breakout_level: float, direction: str) -> int:
        """Cuenta cuántas velas consecutivas han cerrado fuera del nivel."""
        count = 0
        closes = df['Close'].values[::-1] # Invertido para contar desde la actual
        
        for close in closes:
            if direction == "up" and close > breakout_level:
                count += 1
            elif direction == "down" and close < breakout_level:
                count += 1
            else:
                break
        return count

    def detect_rsi_divergence(self, df: pd.DataFrame, direction: str) -> bool:
        """Detecta divergencias simples en el RSI."""
        rsi_col = [c for c in df.columns if 'RSI' in c]
        if not rsi_col:
            return False
        
        rsi = df[rsi_col[0]]
        close = df['Close']
        
        if len(df) < 5:
            return False

        # Simplificado: comparar picos recientes
        if direction == "up": # Buscamos divergencia bajista
            # Precio hace nuevo máximo pero RSI no
            if close.iloc[-1] > close.iloc[-3:].max() and rsi.iloc[-1] < rsi.iloc[-3:].max():
                return True
        else: # Buscamos divergencia alcista
            # Precio hace nuevo mínimo pero RSI no
            if close.iloc[-1] < close.iloc[-3:].min() and rsi.iloc[-1] > rsi.iloc[-3:].min():
                return True
        
        return False

    def evaluate_macd_confirmation(self, df: pd.DataFrame, direction: str) -> bool:
        """Valida si el MACD tiene momentum a favor de la ruptura."""
        # pandas_ta usualmente usa MACD_12_26_9, MACDh_12_26_9 (hist), MACDs_12_26_9 (signal)
        macd_h = [c for c in df.columns if 'MACDh' in c]
        if not macd_h:
            return True # No podemos validar, asumimos OK
        
        hist = df[macd_h[0]]
        if direction == "up":
            # Para ruptura alcista, el histograma debería ser positivo y preferiblemente creciente
            return hist.iloc[-1] > 0
        else:
            # Para ruptura bajista, el histograma debería ser negativo
            return hist.iloc[-1] < 0

    def detect_rejection_candle(self, df: pd.DataFrame, direction: str) -> bool:
        """Detecta velas de rechazo como Pin Bar o Shooting Star."""
        if len(df) < 2:
            return False
            
        last = df.iloc[-1]
        body = abs(last['Close'] - last['Open'])
        upper_wick = last['High'] - max(last['Open'], last['Close'])
        lower_wick = min(last['Open'], last['Close']) - last['Low']
        range_total = last['High'] - last['Low']
        
        if range_total == 0: return False

        if direction == "up":
            # Buscamos Shooting Star o Pin Bar bajista (mecha superior larga)
            if upper_wick > 2 * body and lower_wick < 0.2 * upper_wick:
                return True
            # Bearish Engulfing
            if last['Close'] < last['Open'] and df['Close'].iloc[-2] > df['Open'].iloc[-2]:
                if last['Open'] >= df['Close'].iloc[-2] and last['Close'] <= df['Open'].iloc[-2]:
                    return True
        else:
            # Buscamos Hammer o Pin Bar alcista (mecha inferior larga)
            if lower_wick > 2 * body and upper_wick < 0.2 * lower_wick:
                return True
            # Bullish Engulfing
            if last['Close'] > last['Open'] and df['Close'].iloc[-2] < df['Open'].iloc[-2]:
                if last['Open'] <= df['Close'].iloc[-2] and last['Close'] >= df['Open'].iloc[-2]:
                    return True
        
        # Doji (indecisión)
        if body < 0.1 * range_total:
            return True

        return False

    def evaluate_momentum(self, df: pd.DataFrame, direction: str) -> str:
        """Evalúa la velocidad del movimiento reciente."""
        if len(df) < 5:
            return "normal"
            
        returns = df['Close'].pct_change(3).iloc[-1]
        
        if direction == "up":
            return "strong" if returns > 0.002 else "weak"
        else:
            return "strong" if returns < -0.002 else "weak"
