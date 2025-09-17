# Crypto LSTM Predictor

Aplikace pro predikci hodinového vývoje ceny **Bitcoinu (BTC)** pomocí **LSTM** neuronové sítě a 18 technických indikátorů.  
Data jsou stahována z **Binance API** (ohlcv, objem) a **Blockchain.com API** (on-chain metriky).  
Každý vstupní indikátor i výstup modelu má konfigurovatelnou **váhu**, nastavitelnou v YAML konfiguračních souborech.

---

## Hlavní funkce
- 📊 **Datové zdroje**: Binance (historická a aktuální 1h OHLCV data), Blockchain.com (on-chain metriky).
- 🛠️ **Feature engineering**: 18 základních technických indikátorů (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, CMF aj.).
- 🧠 **Model**: LSTM (PyTorch) s vícevrstvou architekturou, dropout, konfigurovatelná délka sekvence.
- ⚖️ **Váhy**: 
  - Vstupní váhy – každému indikátoru lze nastavit relativní důležitost.
  - Výstupní váhy – ztrátová funkce kombinuje regresní (Δ% změna ceny) a klasifikační (↑ / ↓ / flat) predikci.
- 🔍 **Validace**: Time-series split, early stopping, metriky (RMSE, MAPE, directional accuracy, Sharpe ratio).
- 📈 **Backtesting**: jednoduchý backtest long/short strategie se započítáním poplatků a slippage.

---

## Struktura repozitáře
