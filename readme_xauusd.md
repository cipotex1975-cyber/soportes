# XAUUSD MTF S/R (4H -> 15M -> 5M) en Pine Script

Este indicador implementa la lógica multi-timeframe para `XAUUSD` en TradingView:

1. **4H (`240`):** Detecta pivotes (`ta.pivothigh` / `ta.pivotlow`) y dibuja zonas de soporte (verde) y resistencia (rojo) con ancho basado en `ATR(14)`.
2. **15M (`15`):** Monitorea si la vela de 15 minutos toca una zona de 4H y deja una mecha de rechazo (`wickRatio >= 2.0`). Emite etiqueta `Rej 15M` y alerta JSON.
3. **5M (gráfico actual):** Tras un rechazo en 15M, abre una ventana de confirmación (por defecto 6 velas de 5M). Si una vela de 5M rompe el máximo/mínimo de la vela anterior en dirección del rechazo, marca `LONG 5M` o `SHORT 5M` con **Entrada**, **SL** y **TP** (`1:2` por defecto) y emite alerta JSON.

## Cómo usarlo en TradingView
1. Abre **TradingView** en el gráfico de `XAUUSD` en temporalidad **5 minutos (5m)**.
2. Abre el **Pine Editor**, pega el contenido de `xauusd_mtf_sr.pine` y haz clic en **Add to chart**.
3. Para enviar las alertas a **Telegram**:
   - Crea una alerta en TradingView seleccionando este indicador y la condición **Any alert() function call**.
   - Activa **Webhook URL** apuntando a tu bot/relay de Telegram (por ejemplo, un servicio en Cloudflare Workers, Make, Zapier o un script propio que reenvíe el JSON al `sendMessage` de Telegram).
