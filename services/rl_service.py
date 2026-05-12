import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import os
from typing import Dict, Any

class TradingEnv(gym.Env):
    """
    Entorno de trading avanzado que considera indicadores técnicos y niveles de S/R.
    """
    def __init__(self, df: pd.DataFrame, levels: Dict[str, Any]):
        super(TradingEnv, self).__init__()
        
        # Preparar datos: Calculamos distancias a niveles de S/R para cada fila
        self.df = df.copy()
        self.supports = [l['price'] for l in levels.get('supports', [])]
        self.resistances = [l['price'] for l in levels.get('resistances', [])]
        
        # Añadir características de proximidad a niveles
        self.df['dist_to_sup'] = self.df['Close'].apply(
            lambda x: min([abs(x - s) for s in self.supports]) / x if self.supports else 1.0
        )
        self.df['dist_to_res'] = self.df['Close'].apply(
            lambda x: min([abs(x - r) for r in self.resistances]) / x if self.resistances else 1.0
        )
        
        self.current_step = 0
        
        # Acciones: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # El estado incluye todas las columnas del DF (Indicadores + Distancias S/R)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.df.columns),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 2
        
        # Recompensa: Diferencia de precio al siguiente paso
        reward = 0
        current_close = self.df.iloc[self.current_step]['Close']
        next_close = self.df.iloc[self.current_step + 1]['Close']
        diff = next_close - current_close
        
        if action == 1: # Buy
            reward = diff
        elif action == 2: # Sell
            reward = -diff
            
        # Penalización por inactividad si el mercado se mueve mucho (opcional)
        if action == 0:
            reward = -abs(diff) * 0.1 

        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, float(reward), done, False, {}

class RLService:
    def __init__(self, models_dir: str = "models/reinforcement_learning"):
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_agent(self, df: pd.DataFrame, symbol: str, levels: Dict[str, Any]):
        """
        Entrena el agente PPO usando los datos y los niveles de S/R detectados.
        """
        # Limpiar datos para el entorno (eliminar NaNs de indicadores)
        clean_df = df.dropna()
        
        env = TradingEnv(clean_df, levels)
        
        # MlpPolicy es ideal para datos tabulares/indicadores
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        
        print(f"Entrenando agente RL para {symbol} con niveles de S/R...")
        model.learn(total_timesteps=20000) # Aumentado para mejor aprendizaje
        
        model_path = os.path.join(self.models_dir, f"{symbol}_ppo")
        model.save(model_path)
        return model

    def load_agent(self, symbol: str):
        model_path = os.path.join(self.models_dir, f"{symbol}_ppo.zip")
        if os.path.exists(model_path):
            return PPO.load(model_path)
        return None
