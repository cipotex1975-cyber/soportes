import argparse
import os
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = ["XAU/USD", "USD/CAD", "USD/JPY"]

# Parámetros de estrategia (idénticos a monitor_mtf.py)
PIVOT_LEN_4H = 5
ATR_MULT_ZONE = 0.5
MAX_ZONES = 6
WICK_RATIO = 2.0
VALID_WINDOW_5M = 6
LIMIT_ACTIVATION_WINDOW = 6
RR_RATIO = 2.0
SL_BUFFER_ATR = 0.3
LIMIT_RETRACE = 0.40


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def precompute_4h_zones(df4h: pd.DataFrame) -> list[tuple[pd.Timestamp, list[tuple[float, float]], list[tuple[float, float]]]]:
    """Precalcula zonas 4H sin look-ahead: un pivote en i solo se conoce en i + PIVOT_LEN_4H."""
    atr = calc_atr(df4h, 14)
    res_zones: list[tuple[float, float]] = []
    sup_zones: list[tuple[float, float]] = []
    snapshots = []

    n = PIVOT_LEN_4H
    for idx in range(len(df4h)):
        piv_i = idx - n
        if piv_i >= n:
            cur_atr = atr.iloc[piv_i]
            if not np.isnan(cur_atr):
                half_w = cur_atr * ATR_MULT_ZONE * 0.5
                if df4h["high"].iloc[piv_i] == df4h["high"].iloc[piv_i - n : piv_i + n + 1].max():
                    res_zones.append((df4h["high"].iloc[piv_i] - half_w, df4h["high"].iloc[piv_i] + half_w))
                    res_zones = res_zones[-MAX_ZONES:]
                if df4h["low"].iloc[piv_i] == df4h["low"].iloc[piv_i - n : piv_i + n + 1].min():
                    sup_zones.append((df4h["low"].iloc[piv_i] - half_w, df4h["low"].iloc[piv_i] + half_w))
                    sup_zones = sup_zones[-MAX_ZONES:]
        snapshots.append((df4h["datetime"].iloc[idx], list(res_zones), list(sup_zones)))
    return snapshots


def price_in_zones(low: float, high: float, zones: list[tuple[float, float]]) -> bool:
    return any(high >= bot and low <= top for bot, top in zones)


def load_csv(symbol: str, interval: str) -> pd.DataFrame:
    path = os.path.join("data", f"{symbol.replace('/', '_')}_{interval}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def simulate_symbol(symbol: str) -> tuple[dict, list[dict]]:
    df4h = load_csv(symbol, "4h")
    df15m = load_csv(symbol, "15min")
    df5m = load_csv(symbol, "5min")
    if df4h.empty or df15m.empty or df5m.empty:
        return {}, []

    # TwelveData devuelve timestamps en UTC por defecto; convertir a El Salvador (GMT-6)
    for df in (df4h, df15m, df5m):
        df["dt_sv"] = df["datetime"] - timedelta(hours=6)

    zones_4h = precompute_4h_zones(df4h)
    df5m["atr"] = calc_atr(df5m, 14)

    z_idx = 0
    m15_idx = 0
    rej_15m_count = 0
    limits_created = 0
    trades: list[dict] = []

    bear_rej_level = None
    bull_rej_level = None
    bars_bear = 999
    bars_bull = 999

    i = 2
    while i < len(df5m) - 1:
        c5 = df5m.iloc[i]
        prev5 = df5m.iloc[i - 1]
        cur_time = c5["datetime"]

        # Avanzar zonas 4H conocidas hasta cur_time
        while z_idx + 1 < len(zones_4h) and zones_4h[z_idx + 1][0] <= cur_time:
            z_idx += 1
        _, res_zones, sup_zones = zones_4h[z_idx]

        # Avanzar velas 15M cerradas hasta cur_time
        while m15_idx + 1 < len(df15m) and df15m["datetime"].iloc[m15_idx + 1] <= cur_time:
            m15_idx += 1
            c15 = df15m.iloc[m15_idx]
            if 6 <= c15["dt_sv"].hour < 20:
                body15 = max(abs(c15["close"] - c15["open"]), 1e-6)
                uw = c15["high"] - max(c15["open"], c15["close"])
                lw = min(c15["open"], c15["close"]) - c15["low"]
                if price_in_zones(c15["low"], c15["high"], res_zones) and uw >= body15 * WICK_RATIO and (c15["close"] < c15["open"] or uw > lw * 2):
                    bear_rej_level = c15["high"]
                    bars_bear = 0
                    rej_15m_count += 1
                if price_in_zones(c15["low"], c15["high"], sup_zones) and lw >= body15 * WICK_RATIO and (c15["close"] > c15["open"] or lw > uw * 2):
                    bull_rej_level = c15["low"]
                    bars_bull = 0
                    rej_15m_count += 1

        if not (6 <= c5["dt_sv"].hour < 20):
            i += 1
            continue

        atr5 = c5["atr"] if not np.isnan(c5["atr"]) else 0.0
        signal = None

        if bars_bear < VALID_WINDOW_5M and bear_rej_level is not None:
            bars_bear += 1
            if c5["close"] < c5["open"] and c5["close"] < prev5["low"]:
                entry = c5["low"] + (c5["high"] - c5["low"]) * LIMIT_RETRACE
                sl = bear_rej_level + atr5 * SL_BUFFER_ATR
                risk = max(sl - entry, 1e-5)
                tp = entry - risk * RR_RATIO
                bars_bear = 999
                signal = ("SHORT", entry, sl, tp)

        elif bars_bull < VALID_WINDOW_5M and bull_rej_level is not None:
            bars_bull += 1
            if c5["close"] > c5["open"] and c5["close"] > prev5["high"]:
                entry = c5["high"] - (c5["high"] - c5["low"]) * LIMIT_RETRACE
                sl = bull_rej_level - atr5 * SL_BUFFER_ATR
                risk = max(entry - sl, 1e-5)
                tp = entry + risk * RR_RATIO
                bars_bull = 999
                signal = ("LONG", entry, sl, tp)

        if signal:
            side, entry, sl, tp = signal
            limits_created += 1
            # Buscar activación de orden límite en las siguientes LIMIT_ACTIVATION_WINDOW velas
            activated_at = None
            for k in range(i + 1, min(i + 1 + LIMIT_ACTIVATION_WINDOW, len(df5m))):
                bar = df5m.iloc[k]
                if side == "LONG":
                    if bar["low"] <= sl:
                        break
                    if bar["low"] <= entry <= bar["high"] or bar["low"] <= entry:
                        activated_at = k
                        break
                else:
                    if bar["high"] >= sl:
                        break
                    if bar["low"] <= entry <= bar["high"] or bar["high"] >= entry:
                        activated_at = k
                        break

            if activated_at is not None:
                outcome = None
                exit_time = None
                for k in range(activated_at + 1, len(df5m)):
                    bar = df5m.iloc[k]
                    if side == "LONG":
                        if bar["low"] <= sl:
                            outcome = -1.0
                            exit_time = bar["dt_sv"]
                            break
                        if bar["high"] >= tp:
                            outcome = RR_RATIO
                            exit_time = bar["dt_sv"]
                            break
                    else:
                        if bar["high"] >= sl:
                            outcome = -1.0
                            exit_time = bar["dt_sv"]
                            break
                        if bar["low"] <= tp:
                            outcome = RR_RATIO
                            exit_time = bar["dt_sv"]
                            break

                if outcome is not None:
                    trades.append({
                        "symbol": symbol,
                        "side": side,
                        "entry_time": df5m["dt_sv"].iloc[activated_at],
                        "exit_time": exit_time,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "r_result": outcome,
                    })
        i += 1

    info = {
        "symbol": symbol,
        "start_date": str(df5m["dt_sv"].iloc[0]),
        "end_date": str(df5m["dt_sv"].iloc[-1]),
        "bars_5m": len(df5m),
        "rej_15m": rej_15m_count,
        "limits_created": limits_created,
        "trades_activated": len(trades),
    }
    return info, trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest MTF (4H -> 15M -> 5M)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Pares a simular")
    parser.add_argument("--capital", type=float, default=1000.0, help="Capital inicial ($)")
    parser.add_argument("--risk", type=float, default=0.01, help="Riesgo por operación (ej. 0.01 = 1%)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    all_trades: list[dict] = []
    infos: list[dict] = []

    for sym in args.symbols:
        print(f"Simulando {sym}...")
        info, trades = simulate_symbol(sym)
        if not info:
            print(f"  [Aviso] Faltan datos CSV para {sym} en data/")
            continue
        infos.append(info)
        all_trades.extend(trades)

    if not all_trades:
        print("No se generaron operaciones. Asegúrate de ejecutar primero download_data.py.")
        return

    trades_df = pd.DataFrame(all_trades).sort_values("exit_time").reset_index(drop=True)

    # Curva de capital
    capital = args.capital
    equity = [capital]
    dates = [trades_df["entry_time"].iloc[0]]
    peak = capital
    max_dd_pct = 0.0

    for _, t in trades_df.iterrows():
        pnl = capital * args.risk * t["r_result"]
        capital += pnl
        equity.append(capital)
        dates.append(t["exit_time"])
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100.0
        max_dd_pct = max(max_dd_pct, dd)

    # Gráfico de curva de capital
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    fig.patch.set_facecolor("#131722")
    ax.set_facecolor("#131722")
    ax.plot(dates, equity, color="#26a69a", linewidth=1.8)
    ax.axhline(args.capital, color="#787b86", linestyle="--", linewidth=1.0)
    ax.set_title(f"Curva de Capital (Inicial: ${args.capital:,.0f} | Riesgo: {args.risk*100:.1f}%)", color="white", fontweight="bold")
    ax.tick_params(colors="#b2b5be", labelsize=8)
    ax.grid(color="#2a2e39", linestyle=":", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#363a45")
    plt.tight_layout()
    fig.savefig("results/equity_curve.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Reporte de texto
    lines = [
        "=== REPORTE DE BACKTEST MTF (4H -> 15M -> 5M) ===",
        f"Capital Inicial: ${args.capital:,.2f} | Riesgo/Op: {args.risk*100:.1f}% | LIMIT_RETRACE: {LIMIT_RETRACE*100:.0f}% | RR: 1:{RR_RATIO}",
        "",
        "--- RESULTADOS POR PAR ---",
    ]
    for info in infos:
        sym = info["symbol"]
        sub = trades_df[trades_df["symbol"] == sym]
        wins = (sub["r_result"] > 0).sum()
        wr = (wins / len(sub) * 100) if len(sub) else 0.0
        net_r = sub["r_result"].sum()
        lines.append(
            f"{sym} | Datos: {info['start_date']} a {info['end_date']} ({info['bars_5m']} velas 5M)\n"
            f"  Rechazos 15M: {info['rej_15m']} | Límites creadas: {info['limits_created']} | Activadas: {len(sub)} | Win Rate: {wr:.1f}% | Neto: {net_r:+.1f}R"
        )

    total_wins = (trades_df["r_result"] > 0).sum()
    total_losses = (trades_df["r_result"] < 0).sum()
    gross_win = trades_df.loc[trades_df["r_result"] > 0, "r_result"].sum()
    gross_loss = abs(trades_df.loc[trades_df["r_result"] < 0, "r_result"].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    ret_pct = (capital - args.capital) / args.capital * 100.0

    lines.extend([
        "",
        "--- RESUMEN GLOBAL ---",
        f"Total Operaciones: {len(trades_df)} (Ganadas: {total_wins}, Perdidas: {total_losses})",
        f"Win Rate Global: {total_wins / len(trades_df) * 100:.1f}%",
        f"Profit Factor: {pf:.2f}",
        f"Capital Final: ${capital:,.2f} ({ret_pct:+.2f}%)",
        f"Max Drawdown: {max_dd_pct:.2f}%",
    ])

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open("results/reporte_backtest.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    trades_df.to_csv("results/trades.csv", index=False)
    print("\nArchivos guardados en results/: reporte_backtest.txt, trades.csv, equity_curve.png")


if __name__ == "__main__":
    main()
