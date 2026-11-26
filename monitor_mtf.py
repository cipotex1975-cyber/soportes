import argparse
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

SYMBOLS = ["XAU/USD", "USD/CAD", "USD/JPY"]

# Parámetros de estrategia
PIVOT_LEN_4H = 5
ATR_MULT_ZONE = 0.5
MAX_ZONES = 6
WICK_RATIO = 2.0
VALID_WINDOW_5M = 6
RR_RATIO = 2.0
SL_BUFFER_ATR = 0.3

# Zona horaria El Salvador (GMT-6, sin horario de verano)
EL_SALVADOR_TZ = timezone(timedelta(hours=-6))

_req_timestamps: list[float] = []


def _rate_limit_guard() -> None:
    """Evita superar el límite gratuito de TwelveData (8 peticiones/minuto)."""
    now = time.time()
    _req_timestamps[:] = [t for t in _req_timestamps if now - t < 62]
    if len(_req_timestamps) >= 7:
        wait = 62 - (now - _req_timestamps[0])
        if wait > 0:
            print(f"[Rate Limit] Esperando {wait:.0f}s para no exceder 8 req/min...")
            time.sleep(wait)
        now = time.time()
        _req_timestamps[:] = [t for t in _req_timestamps if now - t < 62]
    _req_timestamps.append(time.time())


def in_session(now_sv: datetime) -> bool:
    return 6 <= now_sv.hour < 20


def send_alert(message: str) -> None:
    print(f"\n[ALERTA] {message}\n")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        except Exception as exc:
            print(f"[ERROR Telegram] {exc}")

    if DISCORD_WEBHOOK_URL:
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
        except Exception as exc:
            print(f"[ERROR Discord] {exc}")


def fetch_ohlc(symbol: str, interval: str, outputsize: int = 120) -> Optional[pd.DataFrame]:
    _rate_limit_guard()
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if "values" not in data:
        print(f"[ERROR TwelveData {symbol} {interval}] {data.get('message', data)}")
        return None

    df = pd.DataFrame(data["values"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.reset_index(drop=True)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_4h_zones(df4h: pd.DataFrame) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    atr = calc_atr(df4h, 14)
    res_zones: list[tuple[float, float]] = []
    sup_zones: list[tuple[float, float]] = []

    n = PIVOT_LEN_4H
    for i in range(n, len(df4h) - n):
        cur_atr = atr.iloc[i]
        if np.isnan(cur_atr):
            continue
        half_w = cur_atr * ATR_MULT_ZONE * 0.5

        window_high = df4h["high"].iloc[i - n : i + n + 1]
        if df4h["high"].iloc[i] == window_high.max():
            res_zones.append((df4h["high"].iloc[i] - half_w, df4h["high"].iloc[i] + half_w))

        window_low = df4h["low"].iloc[i - n : i + n + 1]
        if df4h["low"].iloc[i] == window_low.min():
            sup_zones.append((df4h["low"].iloc[i] - half_w, df4h["low"].iloc[i] + half_w))

    return res_zones[-MAX_ZONES:], sup_zones[-MAX_ZONES:]


def price_in_zones(low: float, high: float, zones: list[tuple[float, float]]) -> bool:
    return any(high >= bot and low <= top for bot, top in zones)


def analyze_symbol(symbol: str, state: dict) -> None:
    sym_state = state.setdefault(
        symbol,
        {
            "last_15m_time": None,
            "last_5m_time": None,
            "bear_rej_level": None,
            "bull_rej_level": None,
            "bars_bear": 999,
            "bars_bull": 999,
            "last_4h_fetch": 0.0,
            "last_15m_fetch": 0.0,
            "df15m": None,
            "res_zones": [],
            "sup_zones": [],
        },
    )

    # Refrescar zonas 4H solo cada 60 minutos
    if time.time() - sym_state["last_4h_fetch"] >= 3600 or not sym_state["res_zones"]:
        df4h = fetch_ohlc(symbol, "4h", 120)
        if df4h is not None:
            sym_state["res_zones"], sym_state["sup_zones"] = detect_4h_zones(df4h)
            sym_state["last_4h_fetch"] = time.time()

    # Refrescar velas 15M cada 15 minutos (900s) para ahorrar créditos
    if time.time() - sym_state["last_15m_fetch"] >= 900 or sym_state["df15m"] is None:
        df15m_new = fetch_ohlc(symbol, "15min", 40)
        if df15m_new is not None:
            sym_state["df15m"] = df15m_new
            sym_state["last_15m_fetch"] = time.time()

    df15m = sym_state["df15m"]
    df5m = fetch_ohlc(symbol, "5min", 40)
    if df15m is None or df5m is None:
        return

    res_zones, sup_zones = sym_state["res_zones"], sym_state["sup_zones"]

    # Evaluar última vela cerrada de 15M (penúltima fila si la última está en formación)
    c15 = df15m.iloc[-2]
    body15 = max(abs(c15["close"] - c15["open"]), 1e-6)
    upper_wick = c15["high"] - max(c15["open"], c15["close"])
    lower_wick = min(c15["open"], c15["close"]) - c15["low"]

    bear_rej = (
        price_in_zones(c15["low"], c15["high"], res_zones)
        and upper_wick >= body15 * WICK_RATIO
        and (c15["close"] < c15["open"] or upper_wick > lower_wick * 2)
    )
    bull_rej = (
        price_in_zones(c15["low"], c15["high"], sup_zones)
        and lower_wick >= body15 * WICK_RATIO
        and (c15["close"] > c15["open"] or lower_wick > upper_wick * 2)
    )

    if sym_state["last_15m_time"] != c15["datetime"]:
        sym_state["last_15m_time"] = c15["datetime"]
        if bear_rej:
            sym_state["bear_rej_level"] = c15["high"]
            sym_state["bars_bear"] = 0
            send_alert(f"🟠 RECHAZO BAJISTA 15M | {symbol} | Precio: {c15['close']}")
        if bull_rej:
            sym_state["bull_rej_level"] = c15["low"]
            sym_state["bars_bull"] = 0
            send_alert(f"🔵 RECHAZO ALCISTA 15M | {symbol} | Precio: {c15['close']}")

    # Evaluar confirmación en 5M (última vela cerrada según timestamp de TwelveData)
    atr5 = calc_atr(df5m, 14).iloc[-2]
    c5 = df5m.iloc[-2]
    prev5 = df5m.iloc[-3]

    if sym_state["last_5m_time"] != c5["datetime"]:
        sym_state["last_5m_time"] = c5["datetime"]

        if sym_state["bars_bear"] < VALID_WINDOW_5M and sym_state["bear_rej_level"] is not None:
            sym_state["bars_bear"] += 1
            if c5["close"] < c5["open"] and c5["close"] < prev5["low"]:
                sl = sym_state["bear_rej_level"] + (atr5 * SL_BUFFER_ATR if not np.isnan(atr5) else 0)
                risk = max(sl - c5["close"], 1e-5)
                tp = c5["close"] - risk * RR_RATIO
                sym_state["bars_bear"] = 999
                send_alert(
                    f"🔴 ENTRADA SHORT 5M | {symbol}\n"
                    f"Entrada: {c5['close']:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}"
                )

        if sym_state["bars_bull"] < VALID_WINDOW_5M and sym_state["bull_rej_level"] is not None:
            sym_state["bars_bull"] += 1
            if c5["close"] > c5["open"] and c5["close"] > prev5["high"]:
                sl = sym_state["bull_rej_level"] - (atr5 * SL_BUFFER_ATR if not np.isnan(atr5) else 0)
                risk = max(c5["close"] - sl, 1e-5)
                tp = c5["close"] + risk * RR_RATIO
                sym_state["bars_bull"] = 999
                send_alert(
                    f"🟢 ENTRADA LONG 5M | {symbol}\n"
                    f"Entrada: {c5['close']:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}"
                )


def _start_health_server() -> None:
    """Mini servidor HTTP para que Render (Web Service gratuito) mantenga el servicio activo."""
    port = int(os.getenv("PORT", "10000"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[HTTP] Health server escuchando en puerto {port}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor MTF (4H -> 15M -> 5M) con TwelveData")
    parser.add_argument("--once", action="store_true", help="Ejecutar una sola vez y salir")
    parser.add_argument("--force", action="store_true", help="Ignorar filtro de horario (8am-5pm GMT-6)")
    args = parser.parse_args()

    if not TWELVEDATA_API_KEY:
        print("Error: Configura TWELVEDATA_API_KEY en tu archivo .env")
        return

    if not args.once:
        _start_health_server()

    state: dict = {}
    while True:
        now_sv = datetime.now(EL_SALVADOR_TZ)
        if args.force or in_session(now_sv):
            print(f"[{now_sv.strftime('%Y-%m-%d %H:%M:%S')} GMT-6] Analizando {', '.join(SYMBOLS)}...")
            for sym in SYMBOLS:
                analyze_symbol(sym, state)
        else:
            print(f"[{now_sv.strftime('%Y-%m-%d %H:%M:%S')} GMT-6] Fuera de horario (06:00-20:00). Esperando...")

        if args.once:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
