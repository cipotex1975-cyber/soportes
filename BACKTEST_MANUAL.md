# Manual del Simulador (Backtest MTF)

Este manual explica cómo descargar datos históricos y simular la estrategia (`4H -> 15M -> 5M` con **Buy/Sell Limit al 40%**) sin afectar el bot en vivo de Render.

---

## Paso 1: Descargar Datos Históricos (`download_data.py`)

El script `download_data.py` descarga velas de `4h`, `15min` y `5min` desde TwelveData y las guarda en la carpeta `data/` en formato CSV.

> **Nota:** Solo necesitas descargar los datos **una vez**. Después podrás correr el backtest todas las veces que quieras sin gastar créditos de la API.

### Ejemplos de uso

1. **Descargar 1 año para los pares por defecto (`XAU/USD`, `USD/CAD`, `USD/JPY`):**
   ```powershell
   python download_data.py
   ```
   *(Consume ~63 créditos de TwelveData en total).*

2. **Descargar 2 años de historia:**
   ```powershell
   python download_data.py --years 2
   ```

3. **Descargar pares personalizados (ej. agregando `EUR/USD` y `GBP/USD`):**
   ```powershell
   python download_data.py --years 1 --symbols XAU/USD USD/CAD EUR/USD GBP/USD
   ```

---

## Paso 2: Ejecutar el Backtest (`backtest.py`)

Una vez descargados los CSV en `data/`, ejecuta `backtest.py` (no consume créditos de TwelveData).

### Ejemplos de uso

1. **Simulación estándar (`$1,000` iniciales, `1%` de riesgo por operación):**
   ```powershell
   python backtest.py
   ```

2. **Simular con otro capital inicial y otro porcentaje de riesgo (ej. `$5,000` y `0.5%`):**
   ```powershell
   python backtest.py --capital 5000 --risk 0.005
   ```

3. **Simular solo pares específicos:**
   ```powershell
   python backtest.py --symbols XAU/USD EUR/USD
   ```

---

## Paso 3: Revisar los Resultados (`results/`)

Al finalizar, el simulador crea la carpeta `results/` con 3 archivos:

| Archivo | Qué contiene |
| :--- | :--- |
| **`results/reporte_backtest.txt`** | Resumen por cada par (fechas testeadas, cantidad de velas, rechazos `15M`, órdenes límite creadas/activadas, Win Rate, retorno en `R`) y resumen global (Profit Factor, Capital Final, Max Drawdown). |
| **`results/trades.csv`** | Lista detallada de cada operación simulada (par, dirección `LONG`/`SHORT`, fecha de entrada/salida, precios `entry`, `sl`, `tp` y resultado en `R`). |
| **`results/equity_curve.png`** | Gráfico de la curva de evolución de tu capital desde `$1,000` a lo largo de todo el periodo simulado. |
