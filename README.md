# Gamma Surface

Standalone Streamlit dashboard for public options-chain data. It shows dealer gamma exposure by strike, net gamma regime, gamma flip, call wall, put wall, and max-pain positioning.

## Version 1.1

- Reference-style two-sided call/put gamma chart with centered zero line
- Readable Max Pain node, guide, highlighted row, and summary card
- Focused strike matrix with right-aligned GEX columns
- Yahoo-derived ATM IV, expected move, Volt, and Delta Grower cards
- Negative gamma is red; positive gamma is green

## Version 1.2

- CTA trend state is red for down/mixed and green for up
- Historical signal tests below 50% are highlighted red
- Removed the Position Plan section for a cleaner institutional view

## Top-20 expiry scanner

Use **Scan top 20 optionable** in the sidebar to rank the liquid S&P universe by near-term option dollar volume, open interest, and bid/ask quality. The scanner supports exact `TODAY` and `FRIDAY` expirations. Yahoo may not publish an exact same-day chain for every symbol; those symbols are skipped instead of being replaced with a different expiry.

## Actionable signal board

The **Actionable signals** tab turns the levels into a conditional playbook rather than a score: `LONG BREAKOUT`, `SHORT BREAKDOWN`, `RANGE / FADE`, or `WAIT`. Each result includes the reasons, trigger, invalidation, target, Gamma Flip, walls, relative volume, and ATR. Today uses `QQQ`, `SPY`, `IWM`, and Yahoo's `^SPX` mapping for `SPXW`; Friday uses the broader S&P optionable universe.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app uses Yahoo Finance through `yfinance`; quotes and option chains may be delayed and can be unavailable outside market hours. No API key is required.

## Optional password protection

In Streamlit Community Cloud, open the app's **Settings > Secrets** and add:

```toml
dashboard_password = "choose-a-strong-password"
```

Save the secret and reboot the app. The dashboard will then show an unlock screen. If `dashboard_password` is absent, password protection is disabled for local testing.

## Deploy

On Streamlit Community Cloud, select `app.py` as the main file and `requirements.txt` as the dependency file.
