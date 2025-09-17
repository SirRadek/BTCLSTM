# Crypto LSTM Predictor

Aplikace pro predikci hodinového vývoje ceny **Bitcoinu (BTC)** pomocí **LSTM** neuronové sítě a základních technických indikátorů.

Data lze stahovat z veřejných API burzy **Binance** a portálu **Blockchain.com**. Každému vstupnímu indikátoru i výstupu modelu lze přiřadit váhu, která je definována v YAML konfiguračních souborech.

---

## Struktura repozitáře

```
.
├── btclstm/               # Python balíček s implementací datových zdrojů, indikátorů a modelu
│   ├── data/              # Modul pro komunikaci s Binance a Blockchain.com API
│   ├── features/          # Výpočet technických indikátorů
│   ├── metrics/           # Vyhodnocovací metriky (RMSE, MAPE, directional accuracy, Sharpe ratio)
│   ├── config.py          # Načítání YAML konfigurací
│   ├── dataset.py         # Tvorba sekvenčních datových sad pro LSTM
│   ├── model.py           # Definice LSTM prediktoru v PyTorch
│   └── training.py        # Logika trénování, validace a výpočtu metrik
├── config/
│   └── default.yaml       # Výchozí konfigurace dat, tréninku a vah indikátorů
├── main.py                # Vstupní skript pipeline
├── requirements.txt       # Seznam Python závislostí
└── README.md              # Tento dokument
```

---

## Instalace

Doporučený postup je vytvořit virtuální prostředí a nainstalovat požadované balíčky:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Spuštění pipeline

Základní běh s výchozí konfigurací a stahováním dat z API:

```bash
python main.py
```

Pokud není k dispozici připojení k internetu, lze použít syntetická data:

```bash
python main.py --offline
```

Parametr `--config` umožňuje zadat alternativní YAML konfiguraci.

---

## Konfigurace

Konfigurace je uložena v `config/default.yaml` a zahrnuje:

- **data** – volba symbolu, timeframe, počtu svíček a nastavení offline režimu.
- **training** – délka sekvence, horizon predikce, architektura LSTM (počet vrstev, velikost skrytého stavu, dropout), počet epoch, batch size, rychlost učení a poměr trénink/validace.
- **weights** – váhy vybraných indikátorů a relativní důležitost regresní vs. klasifikační složky výstupu.

---

## Poznámky

- Modul `btclstm.data` využívá veřejná REST API; při volání v `main.py` jsou ošetřeny výpadky a v případě chyby se přepne do offline režimu.
- Výpočet technických indikátorů je realizován pomocí knihoven `pandas` a `numpy` a zahrnuje SMA, EMA, RSI, MACD, Bollingerova pásma, ATR, OBV a CMF.
- Tréninkový proces je implementován v PyTorch a vrací základní metriky modelu, které se vypíší do logu.

