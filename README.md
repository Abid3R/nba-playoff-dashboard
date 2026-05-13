# NBA Playoff Intelligence Dashboard

A professional Streamlit website for NBA team stats, team comparisons, matchup predictions, and playoff simulation.

## Features

- Live NBA team stats through `nba_api`
- Professional dark UI with animated CSS hero/cards
- All-team data table and filtering
- Team explorer with radar comparison
- Any-team playoff series predictor
- Auto playoff bracket simulator
- Model lab with accuracy and feature importance
- CSV download
- Streamlit Community Cloud ready

## Project files

```text
nba_playoff_website/
├── app.py
├── data_utils.py
├── model_utils.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

## Run locally

```bash
cd nba_playoff_website
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a free GitHub account.
2. Create a new public repository, for example: `nba-playoff-dashboard`.
3. Upload every file in this folder to that repository.
4. Go to Streamlit Community Cloud.
5. Click **New app**.
6. Choose your GitHub repository.
7. Set the main file path to:

```text
app.py
```

8. Click **Deploy**.

## How to replace the model with your final version

The current app trains the compact model from your notebook structure automatically.
To improve it:

1. Expand `create_training_data_from_series()` inside `model_utils.py` with more verified playoff series.
2. Keep the same feature columns, or update `FEATURE_COLS` and `build_prediction_features()` together.
3. Redeploy the app on Streamlit Cloud.

## Important note

NBA API requests can occasionally be blocked or rate-limited on free hosting. If that happens, the app automatically displays demo data so the website still loads.
