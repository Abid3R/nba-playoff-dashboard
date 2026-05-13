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

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Team stats CSV schema

Recommended columns:

```text
team,abbr,conference,division,seed,wins,losses,win_pct,ppg,net_rating
```

The app can also recognize friendly aliases such as `team_name`, `rank`, `wpct`, `net_rtg`, and `points_per_game`.

## Deployment steps

1. Create a GitHub repository.
2. Upload all files in this folder.
3. Go to Streamlit Community Cloud.
4. Select **Create app**.
5. Choose your GitHub repository, branch, and `app.py` as the main file.
6. Click **Deploy**.
7. If build errors occur, open logs and verify that `requirements.txt` and all model/data files are in the same directory as `app.py`.

## Important note

The bundled `team_stats_template.csv` is a starter table for demos. Replace it with current official team statistics before presenting predictions as current-season analysis.
