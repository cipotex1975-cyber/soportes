import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
DEFAULT_SYMBOLS = ["XAU/USD", "USD/CAD", "USD/JPY"]
INTERVALS = ["4h", "15min", "5min"]

_req_timestamps: list[float] = []


def _rate_limit_guard() -> None:
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


def download_series(symbol: str, interval: str, years: float) -> pd.DataFrame:
    target_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=int(years * 365))
    end_date = None
    chunks: list[pd.DataFrame] = []

    while True:
        _rate_limit_guard()
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": 5000,
            "apikey": TWELVEDATA_API_KEY,
            "order": "ASC",
        }
        if end_date:
            params["end_date"] = end_date

        resp = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=20)
        data = resp.json()
        if "values" not in data or not data["values"]:
            if "message" in data:
                print(f"  [Aviso {symbol} {interval}] {data['message']}")
            break

        df = pd.DataFrame(data["values"])
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["datetime"] = pd.to_datetime(df["datetime"])
        chunks.append(df)

        oldest = df["datetime"].iloc[0]
        print(f"  {symbol} {interval}: descargadas {len(df)} velas (desde {oldest})")
        if oldest <= target_start or len(df) < 5000:
            break
        end_date = (oldest - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

    if not chunks:
        return pd.DataFrame()

    full = pd.concat(chunks, ignore_index=True)
    full = full.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargar datos históricos de TwelveData para backtest")
    parser.add_argument("--years", type=float, default=1.0, help="Años de historia a descargar (ej. 1, 2)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Lista de pares (ej. XAU/USD EUR/USD)")
    args = parser.parse_args()

    if not TWELVEDATA_API_KEY:
        print("Error: Configura TWELVEDATA_API_KEY en tu archivo .env")
        return

    os.makedirs("data", exist_ok=True)
    for sym in args.symbols:
        sym_safe = sym.replace("/", "_")
        for interval in INTERVALS:
            print(f"Descargando {sym} ({interval}) - {args.years} año(s)...")
            df = download_series(sym, interval, args.years)
            if not df.empty:
                out_path = os.path.join("data", f"{sym_safe}_{interval}.csv")
                df.to_csv(out_path, index=False)
                print(f"  -> Guardado en {out_path} ({len(df)} velas)")


if __name__ == "__main__":
    main()
