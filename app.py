from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "nba_playoff_model.pkl"
FEATURES_PATH = APP_DIR / "feature_columns.pkl"
HISTORICAL_PATH = APP_DIR / "historical_playoff_games.csv"
TEAM_TEMPLATE_PATH = APP_DIR / "team_stats_template.csv"

REQUIRED_TEAM_COLUMNS = ["team", "conference", "seed", "wins", "losses", "win_pct", "ppg", "net_rating"]

NBA_TEAMS = [
    ("Atlanta Hawks", "ATL", "East", "Southeast"),
    ("Boston Celtics", "BOS", "East", "Atlantic"),
    ("Brooklyn Nets", "BKN", "East", "Atlantic"),
    ("Charlotte Hornets", "CHA", "East", "Southeast"),
    ("Chicago Bulls", "CHI", "East", "Central"),
    ("Cleveland Cavaliers", "CLE", "East", "Central"),
    ("Dallas Mavericks", "DAL", "West", "Southwest"),
    ("Denver Nuggets", "DEN", "West", "Northwest"),
    ("Detroit Pistons", "DET", "East", "Central"),
    ("Golden State Warriors", "GSW", "West", "Pacific"),
    ("Houston Rockets", "HOU", "West", "Southwest"),
    ("Indiana Pacers", "IND", "East", "Central"),
    ("LA Clippers", "LAC", "West", "Pacific"),
    ("Los Angeles Lakers", "LAL", "West", "Pacific"),
    ("Memphis Grizzlies", "MEM", "West", "Southwest"),
    ("Miami Heat", "MIA", "East", "Southeast"),
    ("Milwaukee Bucks", "MIL", "East", "Central"),
    ("Minnesota Timberwolves", "MIN", "West", "Northwest"),
    ("New Orleans Pelicans", "NOP", "West", "Southwest"),
    ("New York Knicks", "NYK", "East", "Atlantic"),
    ("Oklahoma City Thunder", "OKC", "West", "Northwest"),
    ("Orlando Magic", "ORL", "East", "Southeast"),
    ("Philadelphia 76ers", "PHI", "East", "Atlantic"),
    ("Phoenix Suns", "PHX", "West", "Pacific"),
    ("Portland Trail Blazers", "POR", "West", "Northwest"),
    ("Sacramento Kings", "SAC", "West", "Pacific"),
    ("San Antonio Spurs", "SAS", "West", "Southwest"),
    ("Toronto Raptors", "TOR", "East", "Atlantic"),
    ("Utah Jazz", "UTA", "West", "Northwest"),
    ("Washington Wizards", "WAS", "East", "Southeast"),
]


st.set_page_config(
    page_title="NBA Playoff Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --card-bg: rgba(255,255,255,0.78);
            --card-border: rgba(148,163,184,0.25);
            --muted: #64748b;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero {
            border-radius: 28px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.3rem;
            background:
                radial-gradient(circle at top left, rgba(96,165,250,0.35), transparent 30%),
                linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #111827 100%);
            color: white;
            box-shadow: 0 24px 70px rgba(15,23,42,0.25);
        }

        .eyebrow {
            letter-spacing: .14em;
            text-transform: uppercase;
            font-size: .78rem;
            color: #bfdbfe;
            font-weight: 700;
            margin-bottom: .45rem;
        }

        .hero h1 {
            font-size: clamp(2rem, 4.4vw, 4.5rem);
            margin: 0;
            line-height: 0.95;
            font-weight: 900;
        }

        .hero p {
            color: #dbeafe;
            max-width: 900px;
            font-size: 1.05rem;
            margin-top: 1rem;
        }

        .card {
            border-radius: 22px;
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            box-shadow: 0 18px 45px rgba(15,23,42,0.08);
        }

        .team-card {
            border-radius: 24px;
            padding: 1.35rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
            border: 1px solid rgba(148,163,184,.28);
            box-shadow: 0 16px 48px rgba(15,23,42,0.08);
            min-height: 180px;
        }

        .team-name {
            font-size: 1.35rem;
            font-weight: 850;
            color: #0f172a;
            margin-bottom: .15rem;
        }

        .small-muted {
            color: var(--muted);
            font-size: .9rem;
        }

        .winner-banner {
            padding: 1.1rem 1.35rem;
            border-radius: 24px;
            color: white;
            background: linear-gradient(135deg, #16a34a, #0f766e);
            box-shadow: 0 18px 45px rgba(22, 163, 74, .25);
            margin: .5rem 0 1.25rem 0;
        }

        .winner-banner h2 {
            margin: 0;
            font-size: 1.5rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148,163,184,.25);
            padding: 1rem 1.05rem;
            border-radius: 20px;
            box-shadow: 0 8px 26px rgba(15,23,42,.06);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: .75rem 1rem;
            background: #f8fafc;
        }

        .stTabs [aria-selected="true"] {
            background: #dbeafe;
            color: #1d4ed8;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading playoff model...")
def load_model_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    return model, list(feature_columns)


@st.cache_data(show_spinner=False)
def load_historical_data() -> pd.DataFrame:
    return pd.read_csv(HISTORICAL_PATH)


@st.cache_data(show_spinner=False)
def load_template_team_stats() -> pd.DataFrame:
    if TEAM_TEMPLATE_PATH.exists():
        return pd.read_csv(TEAM_TEMPLATE_PATH)

    rows = []
    for idx, (team, abbr, conference, division) in enumerate(NBA_TEAMS):
        seed = (idx % 15) + 1
        win_pct = round(0.72 - (seed - 1) * 0.023, 3)
        wins = int(round(win_pct * 82))
        rows.append(
            {
                "team": team,
                "abbr": abbr,
                "conference": conference,
                "division": division,
                "seed": seed,
                "wins": wins,
                "losses": 82 - wins,
                "win_pct": win_pct,
                "ppg": round(118 - (seed - 1) * 0.35, 1),
                "net_rating": round(8.0 - (seed - 1) * 0.85, 1),
            }
        )
    return pd.DataFrame(rows)


def clean_numeric(series: pd.Series, default: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(default)


def standardize_team_stats(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Friendly aliases for uploaded files.
    aliases = {
        "name": "team",
        "team_name": "team",
        "conf": "conference",
        "rank": "seed",
        "offense_ppg": "ppg",
        "points_per_game": "ppg",
        "net": "net_rating",
        "netrtg": "net_rating",
        "net_rtg": "net_rating",
        "wpct": "win_pct",
        "win_percentage": "win_pct",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    if "team" not in df.columns:
        df["team"] = [f"Team {i+1}" for i in range(len(df))]
    if "abbr" not in df.columns:
        df["abbr"] = df["team"].astype(str).str.upper().str.slice(0, 3)
    if "conference" not in df.columns:
        df["conference"] = "NBA"
    if "division" not in df.columns:
        df["division"] = "Unknown"
    if "seed" not in df.columns:
        df["seed"] = df.groupby("conference").cumcount() + 1
    if "wins" not in df.columns:
        df["wins"] = 41
    if "losses" not in df.columns:
        df["losses"] = 41
    if "ppg" not in df.columns:
        df["ppg"] = 114.0
    if "net_rating" not in df.columns:
        df["net_rating"] = 0.0

    df["wins"] = clean_numeric(df["wins"], 41).clip(0, 82)
    df["losses"] = clean_numeric(df["losses"], 41).clip(0, 82)
    games = (df["wins"] + df["losses"]).replace(0, np.nan)

    if "win_pct" in df.columns:
        df["win_pct"] = clean_numeric(df["win_pct"], np.nan)
        df.loc[df["win_pct"] > 1, "win_pct"] = df.loc[df["win_pct"] > 1, "win_pct"] / 100
        df["win_pct"] = df["win_pct"].fillna(df["wins"] / games)
    else:
        df["win_pct"] = df["wins"] / games

    df["win_pct"] = df["win_pct"].fillna(0.5).clip(0, 1)
    df["seed"] = clean_numeric(df["seed"], 8).round().astype(int).clip(1, 30)
    df["ppg"] = clean_numeric(df["ppg"], 114.0)
    df["net_rating"] = clean_numeric(df["net_rating"], 0.0)
    df["team"] = df["team"].astype(str)
    df["abbr"] = df["abbr"].astype(str)
    df["conference"] = df["conference"].astype(str)
    df["division"] = df["division"].astype(str)

    wanted = ["team", "abbr", "conference", "division", "seed", "wins", "losses", "win_pct", "ppg", "net_rating"]
    return df[wanted].sort_values(["conference", "seed", "team"]).reset_index(drop=True)


def get_team_row(team_stats: pd.DataFrame, team_name: str) -> pd.Series:
    return team_stats.loc[team_stats["team"] == team_name].iloc[0]


def build_feature_row(team_a: pd.Series, team_b: pd.Series, home_court_mode: str, feature_columns: list[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    a_wp = float(team_a["win_pct"])
    b_wp = float(team_b["win_pct"])
    a_seed = int(team_a["seed"])
    b_seed = int(team_b["seed"])

    if home_court_mode == "Team A":
        has_home_court = 1
    elif home_court_mode == "Team B":
        has_home_court = 0
    elif home_court_mode == "Neutral":
        has_home_court = 0
    else:
        has_home_court = int(a_seed <= b_seed)

    higher_seed_win_pct = a_wp if a_seed <= b_seed else b_wp
    lower_seed_win_pct = b_wp if a_seed <= b_seed else a_wp

    features = {
        "win_pct_diff": a_wp - b_wp,
        "seed_diff": b_seed - a_seed,
        "ppg_diff": float(team_a["ppg"]) - float(team_b["ppg"]),
        "net_rating_diff": float(team_a["net_rating"]) - float(team_b["net_rating"]),
        "has_home_court": has_home_court,
        "higher_seed_win_pct": higher_seed_win_pct,
        "lower_seed_win_pct": lower_seed_win_pct,
        "combined_win_pct": a_wp + b_wp,
        "team_a_win_pct": a_wp,
        "team_b_win_pct": b_wp,
    }

    row = {col: features.get(col, 0.0) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns), features


def predict_matchup(model, feature_frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(feature_frame)[0][1])
    prediction = model.predict(feature_frame)[0]
    return float(prediction)


def probability_label(probability: float) -> str:
    edge = abs(probability - 0.5)
    if edge < 0.06:
        return "Toss-up"
    if edge < 0.15:
        return "Lean"
    if edge < 0.25:
        return "Clear edge"
    return "Strong favorite"


def metric_delta(value: float, signed: bool = True) -> str:
    if signed:
        return f"{value:+.2f}"
    return f"{value:.2f}"


def build_probability_gauge(team_a: str, team_b: str, prob_a: float) -> go.Figure:
    winner = team_a if prob_a >= 0.5 else team_b
    prob_winner = max(prob_a, 1 - prob_a)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_a * 100,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": f"{team_a} win probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 35], "color": "rgba(239,68,68,0.20)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.22)"},
                    {"range": [65, 100], "color": "rgba(34,197,94,0.24)"},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": prob_a * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=315,
        margin=dict(l=20, r=20, t=60, b=10),
        annotations=[
            dict(
                text=f"{winner}: {prob_winner:.1%} · {probability_label(prob_a)}",
                x=0.5,
                y=0.03,
                showarrow=False,
                font=dict(size=15),
            )
        ],
    )
    return fig


def render_team_card(label: str, row: pd.Series):
    st.markdown(
        f"""
        <div class="team-card">
            <div class="small-muted">{label}</div>
            <div class="team-name">{row['team']}</div>
            <div class="small-muted">{row['conference']} · {row['division']} · Seed {int(row['seed'])}</div>
            <hr style="border:0;border-top:1px solid rgba(148,163,184,.24);margin:1rem 0;">
            <div style="display:grid;grid-template-columns: repeat(3, 1fr);gap:.85rem;">
                <div><div class="small-muted">Record</div><strong>{int(row['wins'])}-{int(row['losses'])}</strong></div>
                <div><div class="small-muted">Win %</div><strong>{float(row['win_pct']):.3f}</strong></div>
                <div><div class="small-muted">PPG</div><strong>{float(row['ppg']):.1f}</strong></div>
                <div><div class="small-muted">Net Rating</div><strong>{float(row['net_rating']):+.1f}</strong></div>
                <div><div class="small-muted">Abbr</div><strong>{row['abbr']}</strong></div>
                <div><div class="small-muted">Seed</div><strong>{int(row['seed'])}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">NBA Playoff Intelligence Suite</div>
            <h1>Playoff Predictor</h1>
            <p>
                A polished Streamlit dashboard for team stats, model diagnostics, and head-to-head playoff predictions.
                Upload current team data, tune matchups, and turn your trained model into a professional web app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


model, feature_columns = load_model_artifacts()
historical_df = load_historical_data()

if "team_stats" not in st.session_state:
    st.session_state["team_stats"] = standardize_team_stats(load_template_team_stats())

with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Use the template for demos, or upload current team stats for production predictions.")

    uploaded_stats = st.file_uploader(
        "Upload team stats CSV",
        type=["csv"],
        help="Recommended columns: team, conference, seed, wins, losses, win_pct, ppg, net_rating.",
    )

    if uploaded_stats is not None:
        try:
            st.session_state["team_stats"] = standardize_team_stats(pd.read_csv(uploaded_stats))
            st.success("Team stats uploaded and standardized.")
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

    st.download_button(
        "Download team stats template",
        data=standardize_team_stats(load_template_team_stats()).to_csv(index=False).encode("utf-8"),
        file_name="team_stats_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Prediction setup")
    conference_filter = st.selectbox("Team list filter", ["All"] + sorted(st.session_state["team_stats"]["conference"].dropna().unique().tolist()))
    home_court_label = st.selectbox("Home-court setting", ["Use seed logic", "Team A", "Team B", "Neutral"])
    show_advanced = st.toggle("Show advanced inputs", value=False)

team_stats = standardize_team_stats(st.session_state["team_stats"])
if conference_filter != "All":
    selector_stats = team_stats[team_stats["conference"] == conference_filter]
else:
    selector_stats = team_stats

if len(selector_stats) < 2:
    st.warning("Please provide at least two teams to compare.")
    st.stop()

render_hero()

with st.expander("Update the team data used by the dashboard", expanded=False):
    st.info(
        "The bundled table is a polished starter template. Replace it with official current stats before presenting predictions as current-season analysis."
    )
    edited = st.data_editor(
        team_stats,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "team": st.column_config.TextColumn("Team", required=True),
            "conference": st.column_config.SelectboxColumn("Conference", options=["East", "West", "NBA"], required=True),
            "seed": st.column_config.NumberColumn("Seed", min_value=1, max_value=30, step=1),
            "wins": st.column_config.NumberColumn("Wins", min_value=0, max_value=82, step=1),
            "losses": st.column_config.NumberColumn("Losses", min_value=0, max_value=82, step=1),
            "win_pct": st.column_config.NumberColumn("Win %", min_value=0.0, max_value=1.0, step=0.001, format="%.3f"),
            "ppg": st.column_config.NumberColumn("PPG", min_value=80.0, max_value=150.0, step=0.1, format="%.1f"),
            "net_rating": st.column_config.NumberColumn("Net Rating", min_value=-30.0, max_value=30.0, step=0.1, format="%+.1f"),
        },
    )
    st.session_state["team_stats"] = standardize_team_stats(edited)
    team_stats = st.session_state["team_stats"]

team_names = selector_stats["team"].tolist()
default_a = 0
default_b = 1 if len(team_names) > 1 else 0

tab_predict, tab_teams, tab_model, tab_deploy = st.tabs(
    ["Prediction Studio", "Team Data & Stats", "Model Intelligence", "Deployment Guide"]
)

with tab_predict:
    left, right = st.columns([1, 1])
    with left:
        team_a_name = st.selectbox("Team A", team_names, index=default_a)
    with right:
        available_b = [team for team in team_names if team != team_a_name]
        team_b_name = st.selectbox("Team B", available_b, index=min(default_b - 1, len(available_b) - 1))

    team_a = get_team_row(team_stats, team_a_name)
    team_b = get_team_row(team_stats, team_b_name)

    if show_advanced:
        st.markdown("#### Scenario overrides")
        ca, cb, cc, cd = st.columns(4)
        with ca:
            team_a_wp_adj = st.slider("Team A win % adjustment", -0.100, 0.100, 0.0, 0.005)
        with cb:
            team_b_wp_adj = st.slider("Team B win % adjustment", -0.100, 0.100, 0.0, 0.005)
        with cc:
            team_a_net_adj = st.slider("Team A net rating adjustment", -5.0, 5.0, 0.0, 0.1)
        with cd:
            team_b_net_adj = st.slider("Team B net rating adjustment", -5.0, 5.0, 0.0, 0.1)

        team_a = team_a.copy()
        team_b = team_b.copy()
        team_a["win_pct"] = float(np.clip(team_a["win_pct"] + team_a_wp_adj, 0, 1))
        team_b["win_pct"] = float(np.clip(team_b["win_pct"] + team_b_wp_adj, 0, 1))
        team_a["net_rating"] = float(team_a["net_rating"] + team_a_net_adj)
        team_b["net_rating"] = float(team_b["net_rating"] + team_b_net_adj)

    c1, c2 = st.columns(2)
    with c1:
        render_team_card("Team A", team_a)
    with c2:
        render_team_card("Team B", team_b)

    feature_frame, matchup_features = build_feature_row(team_a, team_b, home_court_label, feature_columns)
    prob_a = predict_matchup(model, feature_frame)
    prob_b = 1 - prob_a
    winner = team_a_name if prob_a >= 0.5 else team_b_name
    winner_prob = max(prob_a, prob_b)

    st.markdown(
        f"""
        <div class="winner-banner">
            <div class="small-muted" style="color:#dcfce7;">Projected winner</div>
            <h2>{winner} · {winner_prob:.1%} win probability</h2>
            <div>{probability_label(prob_a)} based on the selected matchup inputs.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Team A probability", f"{prob_a:.1%}")
    m2.metric("Team B probability", f"{prob_b:.1%}")
    m3.metric("Win % edge", metric_delta(matchup_features["win_pct_diff"]))
    m4.metric("Net rating edge", metric_delta(matchup_features["net_rating_diff"]))

    ga, gb = st.columns([1.1, 0.9])
    with ga:
        st.plotly_chart(build_probability_gauge(team_a_name, team_b_name, prob_a), use_container_width=True)
    with gb:
        feature_view = pd.DataFrame(
            [
                {"Driver": "Win percentage edge", "Value": matchup_features["win_pct_diff"], "Interpretation": "Positive favors Team A"},
                {"Driver": "Seed edge", "Value": matchup_features["seed_diff"], "Interpretation": "Positive means Team A has the better seed"},
                {"Driver": "PPG edge", "Value": matchup_features["ppg_diff"], "Interpretation": "Positive favors Team A"},
                {"Driver": "Net rating edge", "Value": matchup_features["net_rating_diff"], "Interpretation": "Positive favors Team A"},
                {"Driver": "Home court flag", "Value": matchup_features["has_home_court"], "Interpretation": "1 means Team A has home court"},
            ]
        )
        st.markdown("#### Matchup drivers")
        st.dataframe(feature_view, use_container_width=True, hide_index=True)

    st.markdown("#### Exact model input row")
    st.dataframe(feature_frame, use_container_width=True, hide_index=True)

with tab_teams:
    st.markdown("### Full team table")
    st.caption("Sort, filter, upload, or edit this table to power the prediction studio.")
    st.dataframe(team_stats, use_container_width=True, hide_index=True)

    avg_win = team_stats["win_pct"].mean()
    avg_ppg = team_stats["ppg"].mean()
    avg_net = team_stats["net_rating"].mean()
    top_net = team_stats.sort_values("net_rating", ascending=False).iloc[0]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Teams tracked", f"{len(team_stats):,}")
    s2.metric("Average win %", f"{avg_win:.3f}")
    s3.metric("Average PPG", f"{avg_ppg:.1f}")
    s4.metric("Top net rating", f"{top_net['team']}", f"{float(top_net['net_rating']):+.1f}")

    chart_df = team_stats.copy()
    chart_df["record"] = chart_df["wins"].astype(int).astype(str) + "-" + chart_df["losses"].astype(int).astype(str)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            chart_df,
            x="win_pct",
            y="net_rating",
            size="ppg",
            color="conference",
            hover_name="team",
            hover_data=["seed", "record", "ppg"],
            title="Win percentage vs. net rating",
        )
        fig.update_layout(height=440, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_chart = chart_df.sort_values("net_rating", ascending=True).tail(15)
        fig = px.bar(
            top_chart,
            x="net_rating",
            y="team",
            orientation="h",
            color="conference",
            title="Top teams by net rating",
            hover_data=["win_pct", "ppg", "seed"],
        )
        fig.update_layout(height=440, yaxis_title="", xaxis_title="Net rating", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        chart_df.sort_values("ppg", ascending=False),
        x="team",
        y="ppg",
        color="conference",
        title="Points per game by team",
        hover_data=["win_pct", "net_rating", "seed"],
    )
    fig.update_layout(height=470, xaxis_tickangle=-45, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.markdown("### Model diagnostics")
    st.caption("Diagnostics are computed from the uploaded historical data and trained artifact.")

    target_col = "label" if "label" in historical_df.columns else None
    X_hist = historical_df[feature_columns].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical rows", f"{len(historical_df):,}")
    c2.metric("Model features", f"{len(feature_columns):,}")
    c3.metric("Model class", model.__class__.__name__)
    if target_col is not None and hasattr(model, "predict"):
        y_true = historical_df[target_col]
        y_pred = model.predict(X_hist)
        in_sample_accuracy = float((y_pred == y_true).mean())
        c4.metric("In-sample accuracy", f"{in_sample_accuracy:.1%}")
    else:
        c4.metric("In-sample accuracy", "N/A")

    st.info(
        "Treat in-sample accuracy as a sanity check, not a true out-of-sample validation score. "
        "For production, validate against seasons the model has never seen."
    )

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Model feature importance",
        )
        fig.update_layout(height=440, yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    dist_cols = [c for c in ["win_pct_diff", "seed_diff", "ppg_diff", "net_rating_diff"] if c in historical_df.columns]
    if dist_cols:
        st.markdown("#### Historical feature distributions")
        selected_feature = st.selectbox("Feature", dist_cols)
        fig = px.histogram(
            historical_df,
            x=selected_feature,
            color=target_col if target_col else None,
            nbins=16,
            title=f"Distribution: {selected_feature}",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Historical data preview")
    st.dataframe(historical_df, use_container_width=True, hide_index=True)

with tab_deploy:
    st.markdown(
        """
        ### Deploy this app

        **Project files expected in your repository**

        ```
        .
        ├── app.py
        ├── nba_playoff_model.pkl
        ├── feature_columns.pkl
        ├── historical_playoff_games.csv
        ├── team_stats_template.csv
        ├── requirements.txt
        ├── runtime.txt
        └── .streamlit/
            └── config.toml
        ```

        **Local run**

        ```bash
        python -m venv .venv
        source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
        pip install -r requirements.txt
        streamlit run app.py
        ```

        **Production checklist**

        1. Replace `team_stats_template.csv` with current official team statistics.
        2. Keep the model artifacts in the same folder as `app.py`.
        3. Push the project to a GitHub repository.
        4. In Streamlit Community Cloud, create a new app from the repository, select the branch, set `app.py` as the main file, and deploy.
        5. If the app fails to build, open deployment logs and check missing package or path errors first.
        """
    )
