# Step-by-step deployment guide

## 1. Test locally

```bash
cd nba_playoff_streamlit_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
cd nba_playoff_streamlit_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 2. Create a GitHub repository

Create a new repository, then upload these files:

```text
app.py
requirements.txt
runtime.txt
nba_playoff_model.pkl
feature_columns.pkl
historical_playoff_games.csv
team_stats_template.csv
.streamlit/config.toml
README.md
```

## 3. Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **Create app**.
4. Select the repository.
5. Select the branch, usually `main`.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

## 4. Update the app after deployment

Whenever you change code or data:

```bash
git add .
git commit -m "Update playoff dashboard"
git push
```

Streamlit Community Cloud will rebuild from the latest GitHub version.

## 5. Common fixes

- **ModuleNotFoundError**: add the missing package to `requirements.txt`.
- **FileNotFoundError**: make sure all `.pkl` and `.csv` files are in the same directory as `app.py`.
- **Model pickle error**: keep `xgboost`, `scikit-learn`, `numpy`, and `pandas` pinned in `requirements.txt`.
- **Bad predictions**: replace `team_stats_template.csv` with current, verified team stats.
