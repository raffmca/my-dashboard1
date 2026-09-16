# Gamma Surface

Standalone Streamlit dashboard for public options-chain data. It shows dealer gamma exposure by strike, net gamma regime, gamma flip, call wall, put wall, and max-pain positioning.

## Version 1.1

- Reference-style two-sided call/put gamma chart with centered zero line
- Readable Max Pain node, guide, highlighted row, and summary card
- Focused strike matrix with right-aligned GEX columns
- Yahoo-derived ATM IV, expected move, Volt, and Delta Grower cards
- Negative gamma is red; positive gamma is green

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app uses Yahoo Finance through `yfinance`; quotes and option chains may be delayed and can be unavailable outside market hours. No API key is required.

## Deploy

On Streamlit Community Cloud, select `app.py` as the main file and `requirements.txt` as the dependency file.
