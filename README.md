# NBA Playoff Predictor — Streamlit App

A professional Streamlit dashboard for NBA playoff matchup predictions.

## What is included

- `app.py` — Streamlit dashboard
- `nba_playoff_model.pkl` — trained XGBoost playoff model
- `feature_columns.pkl` — ordered model feature list
- `historical_playoff_games.csv` — historical training/diagnostic data
- `team_stats_template.csv` — editable all-team stats template
- `requirements.txt` — Python dependencies for deployment
- `runtime.txt` — recommended Python runtime
- `.streamlit/config.toml` — app theme configuration

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```




