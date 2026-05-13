from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_utils import fetch_team_stats
from model_utils import train_model, predict_series, get_feature_importance

st.set_page_config(
    page_title="NBA Playoff Intelligence",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at top left, rgba(245, 158, 11, 0.13), transparent 30%),
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 28%),
                linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(2,6,23,.98));
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 28px;
    padding: 42px 42px;
    background: linear-gradient(135deg, rgba(30,41,59,.92), rgba(15,23,42,.82));
    box-shadow: 0 24px 70px rgba(0,0,0,.28);
    animation: fadeSlide 850ms ease-out;
}

.hero:before {
    content: '';
    position: absolute;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(249,115,22,.34), transparent 65%);
    right: -150px;
    top: -180px;
    animation: pulseGlow 3s infinite alternate ease-in-out;
}

.hero h1 {
    font-size: clamp(2.3rem, 5vw, 4.7rem);
    line-height: .95;
    margin: 0;
    letter-spacing: -0.06em;
    font-weight: 800;
    color: #FFFFFF;
}

.hero p {
    color: #CBD5E1;
    font-size: 1.08rem;
    max-width: 820px;
    margin-top: 18px;
}

.badge {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    color: #FDBA74;
    background: rgba(251,146,60,.12);
    border: 1px solid rgba(251,146,60,.30);
    font-weight: 700;
    font-size: .82rem;
    letter-spacing: .04em;
    text-transform: uppercase;
    margin-bottom: 18px;
}

.stat-card {
    background: linear-gradient(180deg, rgba(30,41,59,.94), rgba(15,23,42,.90));
    border: 1px solid rgba(148, 163, 184, .18);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 14px 44px rgba(0,0,0,.22);
    transition: transform .25s ease, border-color .25s ease;
    animation: fadeSlide 650ms ease-out;
}
.stat-card:hover { transform: translateY(-4px); border-color: rgba(251,146,60,.45); }
.stat-label { color: #94A3B8; font-size: .86rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.stat-value { color: #F8FAFC; font-size: 2rem; font-weight: 800; margin-top: 4px; }
.stat-caption { color: #CBD5E1; font-size: .86rem; margin-top: 4px; }

.prediction-box {
    border-radius: 28px;
    padding: 28px;
    background: linear-gradient(135deg, rgba(234,88,12,.18), rgba(37,99,235,.14));
    border: 1px solid rgba(251,146,60,.28);
    box-shadow: 0 18px 60px rgba(0,0,0,.25);
    animation: fadeSlide 650ms ease-out;
}

.team-chip {
    display:inline-block;
    border-radius:999px;
    padding: 7px 11px;
    margin: 4px 4px 4px 0;
    background: rgba(148,163,184,.10);
    border: 1px solid rgba(148,163,184,.20);
    color:#E2E8F0;
    font-size:.84rem;
}

.small-note { color: #94A3B8; font-size: .88rem; }

@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseGlow {
    from { transform: scale(.92); opacity: .55; }
    to { transform: scale(1.08); opacity: .95; }
}

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(15,23,42,.72);
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,.20);
    color: #E2E8F0;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] { background: rgba(249,115,22,.22); border-color: rgba(249,115,22,.45); }

[data-testid="stMetric"] {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 18px;
    padding: 16px;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_stats(season: str, season_type: str):
    return fetch_team_stats(season, season_type)


@st.cache_resource(show_spinner=False)
def load_model():
    return train_model()


def team_row(df: pd.DataFrame, team_abbr: str) -> dict:
    row = df.loc[df["TEAM_ABBREVIATION"] == team_abbr].iloc[0]
    return row.to_dict()


def display_card(label: str, value: str, caption: str):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_gauge(team_a: str, team_b: str, a_prob: float, b_prob: float):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[a_prob], y=[team_a], orientation="h", name=team_a,
            text=[f"{a_prob:.1f}%"], textposition="inside"
        )
    )
    fig.add_trace(
        go.Bar(
            x=[b_prob], y=[team_b], orientation="h", name=team_b,
            text=[f"{b_prob:.1f}%"], textposition="inside"
        )
    )
    fig.update_layout(
        barmode="stack",
        height=220,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        xaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,.16)", title="Win probability"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", y=1.18),
    )
    st.plotly_chart(fig, use_container_width=True)


def create_bracket(df: pd.DataFrame, model_bundle, conference: str):
    conf_df = df[df["CONFERENCE"] == conference].sort_values("SEED").head(8).copy()
    if len(conf_df) < 8:
        return None, []

    pairings = [(1, 8), (4, 5), (3, 6), (2, 7)]
    round_results = []
    round1_winners = []

    for seed_a, seed_b in pairings:
        a = conf_df[conf_df["SEED"] == seed_a].iloc[0]
        b = conf_df[conf_df["SEED"] == seed_b].iloc[0]
        result = predict_series(
            model_bundle,
            a["TEAM_ABBREVIATION"], b["TEAM_ABBREVIATION"],
            a.to_dict(), b.to_dict(),
        )
        round_results.append({"round": "Round 1", "matchup": f"{a['TEAM_ABBREVIATION']} vs {b['TEAM_ABBREVIATION']}", **result})
        winner = a if result["predicted_winner"] == a["TEAM_ABBREVIATION"] else b
        round1_winners.append(winner)

    semis = [(round1_winners[0], round1_winners[1]), (round1_winners[2], round1_winners[3])]
    finals_winners = []
    for a, b in semis:
        result = predict_series(model_bundle, a["TEAM_ABBREVIATION"], b["TEAM_ABBREVIATION"], a.to_dict(), b.to_dict())
        round_results.append({"round": "Semifinals", "matchup": f"{a['TEAM_ABBREVIATION']} vs {b['TEAM_ABBREVIATION']}", **result})
        winner = a if result["predicted_winner"] == a["TEAM_ABBREVIATION"] else b
        finals_winners.append(winner)

    a, b = finals_winners
    result = predict_series(model_bundle, a["TEAM_ABBREVIATION"], b["TEAM_ABBREVIATION"], a.to_dict(), b.to_dict())
    round_results.append({"round": "Conference Final", "matchup": f"{a['TEAM_ABBREVIATION']} vs {b['TEAM_ABBREVIATION']}", **result})

    champion = result["predicted_winner"]
    return champion, round_results


with st.sidebar:
    st.markdown("### 🏀 NBA Playoff Intelligence")
    st.caption("Professional Streamlit dashboard powered by your matchup model.")
    season = st.selectbox("Season", ["2025-26", "2024-25", "2023-24"], index=0)
    season_type = st.selectbox("Season type", ["Regular Season", "Playoffs"], index=0)
    page = st.radio(
        "Navigation",
        ["Overview", "Team Explorer", "Matchup Predictor", "Playoff Simulator", "Model Lab", "Download"],
        index=0,
    )
    st.markdown("---")
    st.caption("Tip: On Streamlit Cloud, the NBA API may occasionally rate-limit. The app automatically falls back to demo data.")

stats_df, is_live, data_message = load_stats(season, season_type)
model_bundle = load_model()

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <span class="badge">Live stats • Playoff model • Interactive scouting</span>
            <h1>NBA Playoff Intelligence Dashboard</h1>
            <p>Explore all NBA teams, compare advanced team stats, and run playoff-series predictions using your matchup model. Built for clean storytelling, fast analysis, and free deployment on Streamlit Cloud.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if is_live:
        st.success(data_message)
    else:
        st.warning(data_message)

    best_team = stats_df.sort_values("W_PCT", ascending=False).iloc[0]
    best_offense = stats_df.sort_values("PTS", ascending=False).iloc[0]
    best_net = stats_df.sort_values("NET_RATING", ascending=False).iloc[0]
    most_ast = stats_df.sort_values("AST", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        display_card("Teams Loaded", f"{len(stats_df)}", "All NBA teams")
    with c2:
        display_card("Best Record", best_team["TEAM_ABBREVIATION"], f"{best_team['WIN_PCT_DISPLAY']:.1f}% win rate")
    with c3:
        display_card("Top Scoring", best_offense["TEAM_ABBREVIATION"], f"{best_offense['PTS']:.1f} PPG")
    with c4:
        display_card("Best Net Rating", best_net["TEAM_ABBREVIATION"], f"{best_net['NET_RATING']:.1f}")

    st.write("")
    col_a, col_b = st.columns([1.15, .85])
    with col_a:
        top_net = stats_df.sort_values("NET_RATING", ascending=False).head(10)
        fig = px.bar(
            top_net,
            x="TEAM_ABBREVIATION",
            y="NET_RATING",
            text="NET_RATING",
            title="Top 10 Teams by Net Rating",
            hover_data=["TEAM_NAME", "W_PCT", "PTS", "DEF_RATING"],
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.45)", font_color="#F8FAFC")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.scatter(
            stats_df,
            x="OFF_RATING",
            y="DEF_RATING",
            size="W_PCT",
            color="CONFERENCE",
            hover_name="TEAM_NAME",
            text="TEAM_ABBREVIATION",
            title="Offense vs Defense Map",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.45)", font_color="#F8FAFC")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Team Explorer":
    st.title("Team Explorer")
    st.caption("Filter, rank, and inspect all team-level NBA stats.")
    left, right = st.columns([.35, .65])
    with left:
        conf = st.selectbox("Conference filter", ["All", "East", "West"])
        sort_by = st.selectbox("Sort teams by", ["W_PCT", "NET_RATING", "PTS", "DEF_RATING", "AST", "REB"])
        ascending = True if sort_by == "DEF_RATING" else False
        filtered = stats_df if conf == "All" else stats_df[stats_df["CONFERENCE"] == conf]
        filtered = filtered.sort_values(sort_by, ascending=ascending)
        selected = st.selectbox("Select a team", filtered["TEAM_ABBREVIATION"].tolist())
        row = team_row(stats_df, selected)
        st.markdown(
            f"""
            <div class="prediction-box">
                <h2 style="margin-top:0">{row['TEAM_NAME']} ({selected})</h2>
                <span class="team-chip">{row['CONFERENCE']} #{row['SEED']}</span>
                <span class="team-chip">Win% {row['WIN_PCT_DISPLAY']:.1f}</span>
                <span class="team-chip">Net {row['NET_RATING']:.1f}</span>
                <span class="team-chip">PPG {row['PTS']:.1f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Offensive Rating", f"{row['OFF_RATING']:.1f}")
        st.metric("Defensive Rating", f"{row['DEF_RATING']:.1f}", help="Lower defensive rating is better.")
    with right:
        display_cols = ["SEED", "CONFERENCE", "TEAM_ABBREVIATION", "TEAM_NAME", "W", "L", "WIN_PCT_DISPLAY", "PTS", "REB", "AST", "TOV", "FG_PCT_DISPLAY", "FG3_PCT_DISPLAY", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
        existing_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[existing_cols], use_container_width=True, hide_index=True)

        radar_metrics = ["PTS", "REB", "AST", "NET_RATING", "OFF_RATING"]
        selected_row = pd.DataFrame([row])
        league_avg = stats_df[radar_metrics].mean().to_dict()
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=[row[m] for m in radar_metrics], theta=radar_metrics, fill="toself", name=selected))
        radar.add_trace(go.Scatterpolar(r=[league_avg[m] for m in radar_metrics], theta=radar_metrics, fill="toself", name="League Avg"))
        radar.update_layout(
            polar=dict(bgcolor="rgba(15,23,42,.25)"),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            height=430,
            title=f"{selected} vs League Average",
        )
        st.plotly_chart(radar, use_container_width=True)

elif page == "Matchup Predictor":
    st.title("Matchup Predictor")
    st.caption("Choose any two NBA teams and predict a playoff series winner.")

    col1, col2 = st.columns(2)
    team_options = stats_df["TEAM_ABBREVIATION"].tolist()
    with col1:
        team_a = st.selectbox("Team A", team_options, index=0)
    with col2:
        default_b = 1 if len(team_options) > 1 else 0
        team_b = st.selectbox("Team B", team_options, index=default_b)

    if team_a == team_b:
        st.error("Please select two different teams.")
    else:
        a = team_row(stats_df, team_a)
        b = team_row(stats_df, team_b)
        result = predict_series(model_bundle, team_a, team_b, a, b)

        st.markdown(
            f"""
            <div class="prediction-box">
                <h2 style="margin:0 0 8px 0">Predicted Winner: {result['predicted_winner']}</h2>
                <p style="margin:0;color:#CBD5E1">Model confidence: <strong>{result['confidence']:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        probability_gauge(team_a, team_b, result["team_a_win_prob"], result["team_b_win_prob"])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(f"{team_a} Win Probability", f"{result['team_a_win_prob']:.1f}%")
        with c2:
            st.metric(f"{team_b} Win Probability", f"{result['team_b_win_prob']:.1f}%")
        with c3:
            st.metric("Home Court Team", team_a if result["features"]["has_home_court"] == 1 else team_b)

        st.subheader("Feature Differences Used by Model")
        feat_df = pd.DataFrame([result["features"]]).T.reset_index()
        feat_df.columns = ["Feature", "Value"]
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

elif page == "Playoff Simulator":
    st.title("Playoff Simulator")
    st.caption("Auto-selects top 8 teams by conference seed and simulates a clean playoff bracket using your matchup model.")

    east_champ, east_results = create_bracket(stats_df, model_bundle, "East")
    west_champ, west_results = create_bracket(stats_df, model_bundle, "West")

    if not east_results or not west_results:
        st.error("Not enough conference data to simulate the bracket.")
    else:
        col_e, col_w = st.columns(2)
        with col_e:
            st.subheader("Eastern Conference")
            st.success(f"Projected East Champion: {east_champ}")
            st.dataframe(pd.DataFrame(east_results)[["round", "matchup", "predicted_winner", "confidence"]], use_container_width=True, hide_index=True)
        with col_w:
            st.subheader("Western Conference")
            st.success(f"Projected West Champion: {west_champ}")
            st.dataframe(pd.DataFrame(west_results)[["round", "matchup", "predicted_winner", "confidence"]], use_container_width=True, hide_index=True)

        east_row = team_row(stats_df, east_champ)
        west_row = team_row(stats_df, west_champ)
        finals = predict_series(model_bundle, east_champ, west_champ, east_row, west_row)
        st.markdown(
            f"""
            <div class="hero">
                <span class="badge">NBA Finals Projection</span>
                <h1>{finals['predicted_winner']} wins the title</h1>
                <p>{east_champ} vs {west_champ} • {finals['predicted_winner']} has {finals['confidence']:.1f}% projected series confidence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        probability_gauge(east_champ, west_champ, finals["team_a_win_prob"], finals["team_b_win_prob"])

elif page == "Model Lab":
    st.title("Model Lab")
    st.caption("Audit the model, feature importance, and training setup.")
    st.warning("This website uses the compact matchup model structure from your notebook. For a production-grade NBA forecast, expand the training set with verified historical playoff series and injury/context features.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", model_bundle.algorithm_name)
    c2.metric("Holdout Accuracy", f"{model_bundle.training_accuracy * 100:.1f}%")
    if pd.notna(model_bundle.cv_mean):
        c3.metric("5-Fold CV", f"{model_bundle.cv_mean * 100:.1f}% ± {model_bundle.cv_std * 100:.1f}%")
    else:
        c3.metric("5-Fold CV", "N/A")

    importance = get_feature_importance(model_bundle)
    fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Model Feature Importance")
    fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,.45)", font_color="#F8FAFC", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("What the model is doing"):
        st.write(
            "The model predicts whether Team A beats Team B in a playoff series. It uses differences between the two teams, including win percentage, seed, points per game, net rating, and home-court advantage. The output is a probability for each team."
        )

elif page == "Download":
    st.title("Download Data")
    st.caption("Export the currently loaded NBA team stats from the app.")
    csv = stats_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download team stats CSV", data=csv, file_name=f"nba_team_stats_{season}.csv", mime="text/csv")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

st.markdown("<br><div class='small-note'>Built with Streamlit • Data via nba_api when available • Predictions from your notebook-style matchup model</div>", unsafe_allow_html=True)
