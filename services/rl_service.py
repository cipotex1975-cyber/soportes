import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import os
from typing import Dict, Any, Optional

class TradingEnv(gym.Env):
    """
    Entorno de trading RL más realista:
    - Maneja posiciones
    - Reward basado en PnL real
    - Balance y equity
    - Observaciones normalizadas externamente con VecNormalize
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        levels: Dict[str, Any],
        initial_balance: float = 10000,
        max_episode_steps: int = 500
    ):
        super(TradingEnv, self).__init__()

        self.df = df.copy().reset_index(drop=True)

        # =========================
        # Niveles soporte/resistencia
        # =========================
        self.supports = [l['price'] for l in levels.get('supports', [])]
        self.resistances = [l['price'] for l in levels.get('resistances', [])]

        # =========================
        # Features S/R
        # =========================
        self.df['dist_to_sup'] = self.df['Close'].apply(
            lambda x: (
                min([abs(x - s) for s in self.supports]) / x
                if self.supports else 1.0
            )
        )

        self.df['dist_to_res'] = self.df['Close'].apply(
            lambda x: (
                min([abs(x - r) for r in self.resistances]) / x
                if self.resistances else 1.0
            )
        )

        # =========================
        # Verificación mínima
        # =========================
        if len(self.df) < 100:
            raise ValueError(
                f"Dataset demasiado pequeño después de dropna(): {len(self.df)} filas"
            )

        # =========================
        # Configuración episodio
        # =========================
        self.initial_balance = initial_balance
        self.max_episode_steps = min(max_episode_steps, len(self.df) - 1)

        # =========================
        # Estado interno
        # =========================
        self.current_step = 0
        self.balance = initial_balance
        self.net_worth = initial_balance

        # Posiciones
        self.position = 0
        # 0 = flat
        # 1 = long

        self.entry_price = 0.0

        # =========================
        # Acciones
        # =========================
        # 0 = Hold
        # 1 = Buy/Open Long
        # 2 = Sell/Close Long
        self.action_space = spaces.Discrete(3)

        # =========================
        # Observaciones
        # =========================
        self.feature_columns = list(self.df.columns)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) + 3,),
            dtype=np.float32
        )

    def _get_observation(self):

        market_obs = self.df.iloc[self.current_step].values.astype(np.float32)

        agent_obs = np.array([
            self.balance / self.initial_balance,
            float(self.position),
            self.net_worth / self.initial_balance
        ], dtype=np.float32)

        obs = np.concatenate([market_obs, agent_obs])

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

        self.position = 0
        self.entry_price = 0.0

        obs = self._get_observation()

        info = {}

        return obs, info

    def step(self, action):

        terminated = False
        truncated = False

        reward = 0.0

        current_price = float(
            self.df.iloc[self.current_step]['Close']
        )

        # =====================================
        # ACCIÓN: BUY
        # =====================================
        if action == 1:

            # Abrir LONG solo si estamos flat
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price

        # =====================================
        # ACCIÓN: SELL
        # =====================================
        elif action == 2:

            # Cerrar LONG
            if self.position == 1:

                pnl = current_price - self.entry_price

                reward = pnl

                self.balance += pnl
                self.net_worth = self.balance

                self.position = 0
                self.entry_price = 0.0

        # =====================================
        # Reward pequeño por mantener posición rentable
        # =====================================
        if self.position == 1:

            unrealized_pnl = current_price - self.entry_price

            reward += unrealized_pnl * 0.001

        # =====================================
        # Avanzar step
        # =====================================
        self.current_step += 1

        # =====================================
        # Finalización episodio
        # =====================================
        if self.current_step >= self.max_episode_steps:
            terminated = True

        if self.current_step >= len(self.df) - 1:
            truncated = True

        # =====================================
        # Penalización por quiebra
        # =====================================
        if self.balance <= self.initial_balance * 0.5:
            reward -= 100
            terminated = True

        obs = self._get_observation()

        info = {
            "balance": self.balance,
            "net_worth": self.net_worth,
            "position": self.position
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):

        print(
            f"Step: {self.current_step} | "
            f"Balance: {self.balance:.2f} | "
            f"Position: {self.position}"
        )


class RLService:

    def __init__(self, models_dir: str = "models/reinforcement_learning"):

        self.models_dir = models_dir

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def train_agent(
        self,
        df: pd.DataFrame,
        symbol: str,
        levels: Dict[str, Any]
    ):

        # =====================================
        # Limpieza de datos
        # =====================================
        clean_df = df.dropna().copy()

        print(f"Filas disponibles para entrenamiento: {len(clean_df)}")

        if len(clean_df) < 100:
            raise ValueError(
                "Muy pocos datos tras dropna(). "
                "Revisa indicadores o dataset."
            )

        # =====================================
        # Crear entorno
        # =====================================
        def make_env():
            return TradingEnv(
                clean_df,
                levels,
                initial_balance=10000,
                max_episode_steps=500
            )

        env = DummyVecEnv([make_env])

        # =====================================
        # Normalización
        # =====================================
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0
        )

        # =====================================
        # Modelo PPO
        # =====================================
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.2,
            tensorboard_log="./tensorboard_logs/"
        )

        print(f"\nEntrenando agente RL para {symbol}...\n")

        model.learn(
            total_timesteps=200000
        )

        # =====================================
        # Evaluación
        # =====================================
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=10
        )

        print("\n===================================")
        print(f"Mean reward: {mean_reward:.2f}")
        print(f"Std reward : {std_reward:.2f}")
        print("===================================\n")

        # =====================================
        # Guardar modelo
        # =====================================
        model_path = os.path.join(
            self.models_dir,
            f"{symbol}_ppo"
        )

        norm_path = os.path.join(
            self.models_dir,
            f"{symbol}_vecnormalize.pkl"
        )

        model.save(model_path)
        env.save(norm_path)

        print(f"Modelo guardado en: {model_path}")
        print(f"Normalizador guardado en: {norm_path}")

        return model

    def _get_agent_path(self, symbol: str) -> Optional[str]:

        base_path = os.path.join(
            self.models_dir,
            f"{symbol}_ppo"
        )

        zip_path = f"{base_path}.zip"

        if os.path.exists(zip_path):
            return zip_path

        if os.path.exists(base_path):
            return base_path

        return None

    def load_agent(self, symbol: str):

        model_path = self._get_agent_path(symbol)

        if not model_path:
            print(
                f"[RLService] "
                f"Agente RL no encontrado para {symbol}"
            )
            return None

        model = PPO.load(model_path)

        return model