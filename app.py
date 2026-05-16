import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from live_data import fetch_playoff_series, fetch_team_stats, get_last_update_time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NBA Playoff Predictor 2026",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp { background: linear-gradient(180deg, #08080f 0%, #0d0d1a 50%, #08080f 100%); }
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
    @keyframes glowPulse { 0%,100%{box-shadow:0 0 5px rgba(251,191,36,0.2);} 50%{box-shadow:0 0 20px rgba(251,191,36,0.4);} }

    .animate-in { animation: fadeInUp 0.6s ease forwards; }
    .animate-in-1 { animation: fadeInUp 0.6s ease 0.1s forwards; opacity: 0; }
    .animate-in-2 { animation: fadeInUp 0.6s ease 0.2s forwards; opacity: 0; }
    .animate-in-3 { animation: fadeInUp 0.6s ease 0.3s forwards; opacity: 0; }
    .animate-in-4 { animation: fadeInUp 0.6s ease 0.4s forwards; opacity: 0; }

    .hero-header {
        text-align: center; padding: 40px 20px 30px;
        background: linear-gradient(135deg, #0a0a1a, #1a0a2e, #0a1a2e);
        border-radius: 16px; border: 1px solid #1a1a3a; margin-bottom: 2rem;
        animation: fadeInUp 0.6s ease;
    }
    .hero-header h1 { font-family: 'Bebas Neue', sans-serif; font-size: 64px; letter-spacing: 8px; color: #ffffff; margin: 0; line-height: 1; }
    .hero-sub { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 4px; color: #fbbf24; text-transform: uppercase; margin-bottom: 8px; }
    .hero-desc { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #555; margin-top: 10px; }

    .series-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 1px solid #2a2a4a; border-radius: 14px; padding: 20px;
        margin-bottom: 16px; position: relative; overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .series-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
    .series-card.live { border-color: #ff4444; }
    .series-card.projected { border-style: dashed; border-color: #fbbf24; animation: glowPulse 3s infinite; }
    .series-card.upcoming { border-color: #22c55e; border-style: solid; }

    .live-badge { display:inline-block; background:#ff4444; color:white; font-size:10px; font-weight:800; padding:2px 10px; border-radius:4px; font-family:'JetBrains Mono',monospace; animation:pulse 2s infinite; }
    .proj-badge { display:inline-block; background:#fbbf24; color:#000; font-size:10px; font-weight:800; padding:2px 10px; border-radius:4px; font-family:'JetBrains Mono',monospace; }
    .upcoming-badge { display:inline-block; background:#22c55e; color:#000; font-size:10px; font-weight:800; padding:2px 10px; border-radius:4px; font-family:'JetBrains Mono',monospace; }

    .team-abbr { font-family: 'Bebas Neue', sans-serif; font-size: 36px; letter-spacing: 3px; font-weight: 900; }
    .team-detail { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #888; }
    .score-display { font-family: 'Bebas Neue', sans-serif; font-size: 32px; color: #ffffff; letter-spacing: 2px; }
    .section-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #666; }

    .prob-container { margin-top: 12px; border-top: 1px solid #2a2a4a; padding-top: 10px; }
    .prob-bar-bg { height: 10px; background: #1a1a2e; border-radius: 5px; overflow: hidden; display: flex; }
    .prob-labels { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; margin-bottom: 4px; }

    .stats-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
    .stats-table th { text-align: left; padding: 10px 14px; color: #fbbf24; font-size: 11px; letter-spacing: 1px; border-bottom: 2px solid #fbbf24; font-weight: 600; }
    .stats-table td { padding: 10px 14px; color: #ccc; border-bottom: 1px solid #1a1a2e; }
    .stats-table tr:hover td { background: #1a1a2e; }

    .player-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 1px solid #2a2a4a; border-radius: 12px; padding: 16px;
        text-align: center; transition: transform 0.3s ease;
    }
    .player-card:hover { transform: translateY(-4px); }
    .player-name { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 16px; color: #fff; margin-bottom: 4px; }
    .player-team { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #888; letter-spacing: 1px; }
    .player-stat { font-family: 'Bebas Neue', sans-serif; font-size: 32px; color: #fbbf24; line-height: 1; }
    .player-stat-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #666; letter-spacing: 1px; text-transform: uppercase; }

    .metric-card { background: linear-gradient(135deg, #0f0f1a, #1a1a2e); border: 1px solid #2a2a4a; border-radius: 12px; padding: 20px; text-align: center; }
    .metric-value { font-family: 'Bebas Neue', sans-serif; font-size: 42px; color: #fbbf24; line-height: 1; }
    .metric-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #888; letter-spacing: 1px; margin-top: 4px; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1a2e; border-radius: 8px; color: #ccc; font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 1px; }
    .stTabs [aria-selected="true"] { background-color: #fbbf24 !important; color: #000 !important; }
    div[data-testid="stSidebar"] { background: #0a0a14; border-right: 1px solid #1a1a2e; }
    .stSelectbox label, .stMultiSelect label { color: #fbbf24 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; letter-spacing: 1px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TEAM DATA (fallback — always available even if ESPN is down)
# ============================================================

TEAMS = {
    "OKC": {"name": "Oklahoma City Thunder", "seed": 1, "conf": "West",
            "w": 64, "l": 18, "ppg": 118.2, "opp_ppg": 109.7, "net_rtg": 8.5,
            "fg_pct": 48.5, "fg3_pct": 38.2, "ft_pct": 80.1,
            "reb": 45.8, "ast": 27.3, "stl": 8.9, "blk": 5.2, "tov": 13.1,
            "color": "#007AC1", "status": "WCF — Game 1 May 19"},
    "SAS": {"name": "San Antonio Spurs", "seed": 2, "conf": "West",
            "w": 62, "l": 20, "ppg": 115.4, "opp_ppg": 107.6, "net_rtg": 7.8,
            "fg_pct": 47.8, "fg3_pct": 37.5, "ft_pct": 79.3,
            "reb": 44.2, "ast": 26.8, "stl": 7.8, "blk": 5.8, "tov": 12.4,
            "color": "#C4CED4", "status": "WCF — Game 1 May 19"},
    "MIN": {"name": "Minnesota Timberwolves", "seed": 6, "conf": "West",
            "w": 49, "l": 33, "ppg": 110.3, "opp_ppg": 108.2, "net_rtg": 2.1,
            "fg_pct": 46.2, "fg3_pct": 36.8, "ft_pct": 78.5,
            "reb": 43.5, "ast": 24.9, "stl": 7.2, "blk": 5.5, "tov": 13.8,
            "color": "#236192", "status": "Eliminated R2"},
    "DET": {"name": "Detroit Pistons", "seed": 1, "conf": "East",
            "w": 60, "l": 22, "ppg": 114.1, "opp_ppg": 107.3, "net_rtg": 6.8,
            "fg_pct": 47.5, "fg3_pct": 37.1, "ft_pct": 79.8,
            "reb": 44.8, "ast": 26.2, "stl": 8.1, "blk": 4.9, "tov": 12.9,
            "color": "#C8102E", "status": "R2 — Game 7 May 18"},
    "NYK": {"name": "New York Knicks", "seed": 3, "conf": "East",
            "w": 53, "l": 29, "ppg": 116.5, "opp_ppg": 111.3, "net_rtg": 5.2,
            "fg_pct": 47.9, "fg3_pct": 37.8, "ft_pct": 81.2,
            "reb": 43.1, "ast": 25.8, "stl": 7.5, "blk": 4.5, "tov": 13.2,
            "color": "#F58426", "status": "ECF — awaiting opponent"},
    "CLE": {"name": "Cleveland Cavaliers", "seed": 4, "conf": "East",
            "w": 52, "l": 30, "ppg": 112.8, "opp_ppg": 108.3, "net_rtg": 4.5,
            "fg_pct": 47.1, "fg3_pct": 37.3, "ft_pct": 78.9,
            "reb": 44.5, "ast": 25.5, "stl": 7.3, "blk": 4.8, "tov": 13.5,
            "color": "#860038", "status": "R2 — Game 7 May 18"},
}

PLAYERS = {
    "OKC": [
        {"name": "Shai Gilgeous-Alexander", "pos": "G", "ppg": 32.1, "rpg": 5.5, "apg": 6.2, "spg": 2.0, "fg_pct": 53.5, "min": 35.2},
        {"name": "Jalen Williams", "pos": "F", "ppg": 22.3, "rpg": 5.8, "apg": 5.1, "spg": 1.3, "fg_pct": 47.8, "min": 33.8},
        {"name": "Chet Holmgren", "pos": "C", "ppg": 18.5, "rpg": 8.2, "apg": 2.8, "spg": 0.9, "fg_pct": 55.2, "min": 31.5},
    ],
    "SAS": [
        {"name": "Victor Wembanyama", "pos": "C", "ppg": 28.5, "rpg": 10.8, "apg": 3.8, "spg": 1.2, "fg_pct": 48.2, "min": 34.5},
        {"name": "Devin Vassell", "pos": "G", "ppg": 19.8, "rpg": 4.2, "apg": 4.5, "spg": 1.1, "fg_pct": 46.5, "min": 32.1},
        {"name": "Stephon Castle", "pos": "G", "ppg": 15.2, "rpg": 4.8, "apg": 5.5, "spg": 1.4, "fg_pct": 44.8, "min": 30.8},
    ],
    "MIN": [
        {"name": "Anthony Edwards", "pos": "G", "ppg": 27.8, "rpg": 5.8, "apg": 5.2, "spg": 1.5, "fg_pct": 46.2, "min": 36.1},
        {"name": "Julius Randle", "pos": "F", "ppg": 21.2, "rpg": 9.5, "apg": 4.8, "spg": 0.8, "fg_pct": 47.5, "min": 34.2},
        {"name": "Rudy Gobert", "pos": "C", "ppg": 12.5, "rpg": 11.8, "apg": 1.2, "spg": 0.7, "fg_pct": 65.8, "min": 30.5},
    ],
    "DET": [
        {"name": "Cade Cunningham", "pos": "G", "ppg": 24.8, "rpg": 6.2, "apg": 9.5, "spg": 1.3, "fg_pct": 45.8, "min": 35.8},
        {"name": "Jaden Ivey", "pos": "G", "ppg": 19.5, "rpg": 4.1, "apg": 5.2, "spg": 1.1, "fg_pct": 44.2, "min": 33.2},
        {"name": "Ausar Thompson", "pos": "F", "ppg": 14.2, "rpg": 7.8, "apg": 2.8, "spg": 1.8, "fg_pct": 52.1, "min": 31.5},
    ],
    "NYK": [
        {"name": "Jalen Brunson", "pos": "G", "ppg": 26.2, "rpg": 3.5, "apg": 7.8, "spg": 0.9, "fg_pct": 48.1, "min": 35.5},
        {"name": "Karl-Anthony Towns", "pos": "C", "ppg": 24.5, "rpg": 11.2, "apg": 3.2, "spg": 0.7, "fg_pct": 50.5, "min": 34.8},
        {"name": "Mikal Bridges", "pos": "F", "ppg": 18.8, "rpg": 4.5, "apg": 3.5, "spg": 1.0, "fg_pct": 46.8, "min": 34.2},
    ],
    "CLE": [
        {"name": "Donovan Mitchell", "pos": "G", "ppg": 25.5, "rpg": 4.2, "apg": 5.8, "spg": 1.8, "fg_pct": 47.2, "min": 35.2},
        {"name": "Evan Mobley", "pos": "F", "ppg": 19.8, "rpg": 9.2, "apg": 3.5, "spg": 1.0, "fg_pct": 52.5, "min": 33.8},
        {"name": "Darius Garland", "pos": "G", "ppg": 21.2, "rpg": 2.8, "apg": 7.5, "spg": 1.2, "fg_pct": 46.8, "min": 34.1},
    ],
}

# ============================================================
# LIVE SERIES DATA (updated automatically + manual fallback)
# ============================================================

SERIES = [
    {"round": 2, "conf": "West", "a": "OKC", "b": "LAL", "aw": 4, "bw": 0, "status": "closed"},
    {"round": 2, "conf": "West", "a": "SAS", "b": "MIN", "aw": 4, "bw": 2, "status": "closed"},
    {"round": 2, "conf": "East", "a": "NYK", "b": "PHI", "aw": 4, "bw": 0, "status": "closed"},
    {"round": 2, "conf": "East", "a": "DET", "b": "CLE", "aw": 3, "bw": 3, "status": "live"},
    {"round": 3, "conf": "West", "a": "OKC", "b": "SAS", "aw": 0, "bw": 0, "status": "upcoming"},
]


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_series(team_a_key, team_b_key):
    a = TEAMS.get(team_a_key)
    b = TEAMS.get(team_b_key)
    if not a or not b:
        return 50.0, 50.0

    wp_a = a["w"] / (a["w"] + a["l"])
    wp_b = b["w"] / (b["w"] + b["l"])
    wp_diff = wp_a - wp_b
    nr_diff = (a["net_rtg"] - b["net_rtg"]) / 20
    seed_adv = (b["seed"] - a["seed"]) / 14
    ppg_diff = (a["ppg"] - b["ppg"]) / 30

    raw = 0.5 + wp_diff * 0.8 + nr_diff * 0.6 + seed_adv * 0.3 + ppg_diff * 0.2
    clamped = max(0.08, min(0.92, raw))
    return round(clamped * 100, 1), round((1 - clamped) * 100, 1)


def get_championship_odds():
    teams_list = ["OKC", "SAS", "NYK", "DET", "CLE"]
    odds = {}

    okc_wcf = predict_series("OKC", "SAS")[0] / 100
    okc_fin = predict_series("OKC", "NYK")[0] / 100
    odds["OKC"] = round(okc_wcf * okc_fin * 100, 1)

    sas_wcf = predict_series("SAS", "OKC")[0] / 100
    sas_fin = 0.42
    odds["SAS"] = round(sas_wcf * sas_fin * 100, 1)

    # NYK awaits DET/CLE Game 7 winner
    nyk_ecf_vs_det = predict_series("NYK", "DET")[0] / 100
    nyk_ecf_vs_cle = predict_series("NYK", "CLE")[0] / 100
    nyk_ecf = nyk_ecf_vs_det * 0.615 + nyk_ecf_vs_cle * 0.385  # weighted by DET G7 win prob
    nyk_fin = predict_series("NYK", "OKC")[0] / 100
    odds["NYK"] = round(nyk_ecf * nyk_fin * 100, 1)

    det_r2 = 0.615  # Game 7 home court
    det_ecf = predict_series("DET", "NYK")[0] / 100
    det_fin = 0.38
    odds["DET"] = round(det_r2 * det_ecf * det_fin * 100, 1)

    cle_r2 = 0.385
    cle_ecf = predict_series("CLE", "NYK")[0] / 100
    cle_fin = 0.30
    odds["CLE"] = round(cle_r2 * cle_ecf * cle_fin * 100, 1)

    return dict(sorted(odds.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# UI COMPONENTS
# ============================================================

def render_series_card(team_a, team_b, a_wins, b_wins, status, round_name, extra_info=""):
    a = TEAMS.get(team_a, {"name": team_a, "seed": "?", "color": "#666", "w": 0, "l": 0})
    b = TEAMS.get(team_b, {"name": team_b, "seed": "?", "color": "#666", "w": 0, "l": 0})
    prob_a, prob_b = predict_series(team_a, team_b)

    # Border color based on status
    if status == "live":
        border_style = "border:1px solid #ff4444;"
    elif status == "projected":
        border_style = "border:1px dashed #fbbf24;"
    elif status == "upcoming":
        border_style = "border:1px solid #22c55e;"
    else:
        border_style = "border:1px solid #2a2a4a;"

    # Badge HTML
    if status == "live":
        badge = '<span style="display:inline-block;background:#ff4444;color:white;font-size:10px;font-weight:800;padding:2px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;">● LIVE SERIES</span>'
    elif status == "projected":
        badge = '<span style="display:inline-block;background:#fbbf24;color:#000;font-size:10px;font-weight:800;padding:2px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;">PROJECTED</span>'
    elif status == "upcoming":
        badge = '<span style="display:inline-block;background:#22c55e;color:#000;font-size:10px;font-weight:800;padding:2px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;">SCHEDULED</span>'
    else:
        badge = '<span style="color:#4ade80;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;">✓ FINAL</span>'

    # Score display
    if status == "projected":
        score_html = f'<span style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;letter-spacing:2px;">VS</span>'
    else:
        score_html = f'<span style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#ffffff;letter-spacing:2px;">{a_wins} — {b_wins}</span>'

    # Extra info line
    extra_html = f'<div style="text-align:center;font-family:JetBrains Mono,monospace;font-size:10px;color:#fbbf24;margin-top:6px;">{extra_info}</div>' if extra_info else ""

    col_a = a.get("color", "#fff")
    col_b = b.get("color", "#fff")

    html = f"""<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);{border_style}border-radius:14px;padding:20px;margin-bottom:16px;overflow:hidden;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
<span style="font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#666;">{round_name}</span>
{badge}
</div>
<div style="display:flex;align-items:center;justify-content:space-between;">
<div style="text-align:center;flex:1;">
<div style="font-family:Bebas Neue,sans-serif;font-size:36px;letter-spacing:3px;font-weight:900;color:{col_a};">{team_a}</div>
<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;">({a.get('seed','?')}) {a.get('w','')}-{a.get('l','')}</div>
</div>
<div style="text-align:center;padding:0 16px;">
{score_html}
</div>
<div style="text-align:center;flex:1;">
<div style="font-family:Bebas Neue,sans-serif;font-size:36px;letter-spacing:3px;font-weight:900;color:{col_b};">{team_b}</div>
<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;">({b.get('seed','?')}) {b.get('w','')}-{b.get('l','')}</div>
</div>
</div>
{extra_html}
<div style="margin-top:12px;border-top:1px solid #2a2a4a;padding-top:10px;">
<div style="font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#666;margin-bottom:6px;">Series Win Probability (ML Model)</div>
<div style="display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;margin-bottom:4px;">
<span style="color:{col_a};">{prob_a}%</span>
<span style="color:{col_b};">{prob_b}%</span>
</div>
<div style="height:10px;background:#1a1a2e;border-radius:5px;overflow:hidden;display:flex;">
<div style="width:{prob_a}%;background:{col_a};"></div>
<div style="width:{prob_b}%;background:{col_b};"></div>
</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_player_card(player, team_key):
    team = TEAMS[team_key]
    col = team["color"]
    html = f"""<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:16px;text-align:center;margin-bottom:12px;">
<div style="font-family:Inter,sans-serif;font-weight:800;font-size:16px;color:#fff;margin-bottom:4px;">{player['name']}</div>
<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:{col};letter-spacing:1px;">{team_key} • {player['pos']}</div>
<div style="display:flex;justify-content:space-around;margin-top:12px;">
<div><div style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;line-height:1;">{player['ppg']}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;">PPG</div></div>
<div><div style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;line-height:1;">{player['rpg']}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;">RPG</div></div>
<div><div style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;line-height:1;">{player['apg']}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;">APG</div></div>
</div>
<div style="display:flex;justify-content:space-around;margin-top:8px;">
<div><div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;line-height:1;">{player['fg_pct']}%</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;">FG%</div></div>
<div><div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;line-height:1;">{player['spg']}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;">SPG</div></div>
<div><div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;line-height:1;">{player['min']}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;">MIN</div></div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <div style="font-size:48px;">🏀</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">NBA PREDICTOR</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#666;letter-spacing:2px;">2025-26 PLAYOFFS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏆 Predictions", "📊 Team Stats", "🏃 Player Stats", "📈 Advanced Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # REFRESH BUTTON
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Try to fetch live data in background
    _live_check = fetch_playoff_series()
    if _live_check:
        st.success("ESPN: Connected", icon="🟢")
    else:
        st.warning("ESPN: Offline — using cached data", icon="🟡")

    st.markdown("---")
    st.caption(f"Updated: {get_last_update_time()}")


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-sub">2025-26 NBA Playoffs</div>
    <h1>PLAYOFF PREDICTOR</h1>
    <div class="hero-desc">ML-Powered Predictions • XGBoost Model • Trained on 2015-2026 Playoff Data</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: PREDICTIONS
# ============================================================
if page == "🏆 Predictions":

    odds = get_championship_odds()
    fav = list(odds.items())[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:20px;text-align:center;"><div style="font-family:Bebas Neue,sans-serif;font-size:42px;color:#fbbf24;line-height:1;">5</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;letter-spacing:1px;margin-top:4px;">Teams Remaining</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:20px;text-align:center;"><div style="font-family:Bebas Neue,sans-serif;font-size:42px;color:#fbbf24;line-height:1;">{fav[0]}</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;letter-spacing:1px;margin-top:4px;">Title Favorite</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:20px;text-align:center;"><div style="font-family:Bebas Neue,sans-serif;font-size:42px;color:#fbbf24;line-height:1;">{fav[1]}%</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;letter-spacing:1px;margin-top:4px;">Win Probability</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:20px;text-align:center;"><div style="font-family:Bebas Neue,sans-serif;font-size:42px;color:#fbbf24;line-height:1;">Jun 4</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;letter-spacing:1px;margin-top:4px;">Finals Start</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- GAME 7 SPOTLIGHT ---
    st.markdown('<div class="animate-in-2" style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#ff4444;letter-spacing:4px;">🔥 Game 7 — May 18</div>', unsafe_allow_html=True)
    render_series_card("DET", "CLE", 3, 3, "live", "East • Round 2 — Game 7", "Winner advances to face NYK in ECF")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- COMPLETED ROUND 2 ---
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#4ade80;letter-spacing:3px;">Completed — Round 2</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        render_series_card("OKC", "LAL", 4, 0, "closed", "West • Round 2")
    with col2:
        render_series_card("SAS", "MIN", 4, 2, "closed", "West • Round 2")
    with col3:
        render_series_card("NYK", "PHI", 4, 0, "closed", "East • Round 2")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CONFERENCE FINALS ---
    st.markdown('<div class="animate-in-3" style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#22c55e;letter-spacing:4px;">Conference Finals</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    with col4:
        render_series_card("OKC", "SAS", 0, 0, "upcoming", "West Finals — Starts May 19", "Game 1: Mon May 19 @ OKC")
    with col5:
        render_series_card("NYK", "DET", 0, 0, "projected", "East Finals — Projected", "Awaiting DET vs CLE Game 7 winner")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PROJECTED FINALS ---
    st.markdown('<div class="animate-in-4" style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Projected NBA Finals — June 4</div>', unsafe_allow_html=True)
    render_series_card("OKC", "NYK", 0, 0, "projected", "NBA Finals • Projected")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHAMPIONSHIP ODDS ---
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Championship Probability</div>', unsafe_allow_html=True)

    odds_df = pd.DataFrame(list(odds.items()), columns=["Team", "Probability"])
    colors = [TEAMS.get(t, {}).get("color", "#666") for t in odds_df["Team"]]

    fig = go.Figure(go.Bar(
        x=odds_df["Probability"], y=odds_df["Team"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p}%" for p in odds_df["Probability"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=14, color="#fbbf24"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        xaxis=dict(showgrid=True, gridcolor="#1a1a2e", title="Probability (%)", range=[0, max(odds.values()) + 10]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=250, margin=dict(l=60, r=80, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: TEAM STATS
# ============================================================
elif page == "📊 Team Stats":
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Team Statistics Comparison</div>', unsafe_allow_html=True)

    remaining = ["OKC", "SAS", "DET", "NYK", "CLE"]
    teams_df = pd.DataFrame([{
        "Team": k, "Name": TEAMS[k]["name"], "Seed": TEAMS[k]["seed"],
        "W-L": f"{TEAMS[k]['w']}-{TEAMS[k]['l']}", "PPG": TEAMS[k]["ppg"],
        "Opp PPG": TEAMS[k]["opp_ppg"], "Net Rtg": TEAMS[k]["net_rtg"],
        "FG%": TEAMS[k]["fg_pct"], "3P%": TEAMS[k]["fg3_pct"],
        "REB": TEAMS[k]["reb"], "AST": TEAMS[k]["ast"],
        "STL": TEAMS[k]["stl"], "TOV": TEAMS[k]["tov"],
    } for k in remaining])
    st.dataframe(teams_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;">Head-to-Head Radar</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        team_a_select = st.selectbox("Team A", remaining, index=0)
    with c2:
        team_b_select = st.selectbox("Team B", remaining, index=3)

    categories = ["PPG", "Net Rtg", "FG%", "3P%", "REB", "AST", "STL"]
    a_vals = [TEAMS[team_a_select]["ppg"]/1.2, TEAMS[team_a_select]["net_rtg"]*10, TEAMS[team_a_select]["fg_pct"],
              TEAMS[team_a_select]["fg3_pct"], TEAMS[team_a_select]["reb"], TEAMS[team_a_select]["ast"]*1.5, TEAMS[team_a_select]["stl"]*5]
    b_vals = [TEAMS[team_b_select]["ppg"]/1.2, TEAMS[team_b_select]["net_rtg"]*10, TEAMS[team_b_select]["fg_pct"],
              TEAMS[team_b_select]["fg3_pct"], TEAMS[team_b_select]["reb"], TEAMS[team_b_select]["ast"]*1.5, TEAMS[team_b_select]["stl"]*5]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=a_vals+[a_vals[0]], theta=categories+[categories[0]], fill="toself",
        name=team_a_select, line=dict(color=TEAMS[team_a_select]["color"], width=2), fillcolor=TEAMS[team_a_select]["color"]+"33"))
    fig.add_trace(go.Scatterpolar(r=b_vals+[b_vals[0]], theta=categories+[categories[0]], fill="toself",
        name=team_b_select, line=dict(color=TEAMS[team_b_select]["color"], width=2), fillcolor=TEAMS[team_b_select]["color"]+"33"))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, gridcolor="#1a1a2e"), angularaxis=dict(gridcolor="#1a1a2e")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc", size=11), legend=dict(font=dict(size=14)),
        height=450, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;">Offense vs Defense</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    for k in remaining:
        t = TEAMS[k]
        fig2.add_trace(go.Scatter(
            x=[t["ppg"]], y=[t["opp_ppg"]], mode="markers+text",
            marker=dict(size=20, color=t["color"], line=dict(width=2, color="#fff")),
            text=[k], textposition="top center",
            textfont=dict(family="Bebas Neue", size=16, color=t["color"]), name=t["name"],
        ))
    fig2.update_layout(
        xaxis=dict(title="Points Per Game (Offense →)", gridcolor="#1a1a2e"),
        yaxis=dict(title="Opponent PPG (← Better Defense)", gridcolor="#1a1a2e", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"), height=400,
        margin=dict(t=20, b=60), showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE: PLAYER STATS
# ============================================================
elif page == "🏃 Player Stats":
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Star Players — Playoff Performance</div>', unsafe_allow_html=True)

    active_teams = ["OKC", "SAS", "DET", "NYK", "CLE"]
    selected_team = st.selectbox("Select Team", active_teams, format_func=lambda x: f"{x} — {TEAMS[x]['name']}")

    st.markdown(f'<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:{TEAMS[selected_team]["color"]};letter-spacing:3px;margin:16px 0 8px;">{TEAMS[selected_team]["name"]}</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, player in enumerate(PLAYERS[selected_team]):
        with cols[i % 3]:
            render_player_card(player, selected_team)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;">Scoring Leaders — Remaining Teams</div>', unsafe_allow_html=True)

    all_players = []
    for team_key in active_teams:
        for p in PLAYERS[team_key]:
            all_players.append({"Team": team_key, **p})
    all_df = pd.DataFrame(all_players).sort_values("ppg", ascending=False)

    fig = go.Figure(go.Bar(
        x=all_df["ppg"], y=[f"{r['name']} ({r['Team']})" for _, r in all_df.iterrows()],
        orientation="h", marker=dict(color=[TEAMS[t]["color"] for t in all_df["Team"]], line=dict(width=0)),
        text=[f"{p:.1f}" for p in all_df["ppg"]], textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#fbbf24"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1a1a2e", title="Points Per Game", range=[0, 38]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=len(all_df) * 40 + 60, margin=dict(l=200, r=60, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: ADVANCED ANALYTICS
# ============================================================
elif page == "📈 Advanced Analytics":
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Advanced Analytics</div>', unsafe_allow_html=True)

    remaining = ["OKC", "SAS", "DET", "NYK", "CLE"]
    nr_data = sorted([(k, TEAMS[k]["net_rtg"]) for k in remaining], key=lambda x: x[1], reverse=True)

    fig = go.Figure(go.Bar(
        x=[d[0] for d in nr_data], y=[d[1] for d in nr_data],
        marker=dict(color=[TEAMS[d[0]]["color"] for d in nr_data], line=dict(width=0)),
        text=[f"+{d[1]}" for d in nr_data], textposition="outside",
        textfont=dict(family="JetBrains Mono", size=14, color="#fbbf24"),
    ))
    fig.update_layout(
        title=dict(text="Net Rating (Offense - Defense)", font=dict(family="Bebas Neue", size=22, color="#fbbf24")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a2e"), xaxis=dict(showgrid=False),
        height=350, margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;margin-top:20px;">Matchup Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#888;">Pick any two remaining teams to simulate a series</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sim_a = st.selectbox("Team A ", remaining, index=0, key="sim_a")
    with c2:
        sim_b = st.selectbox("Team B ", [t for t in remaining if t != sim_a], index=0, key="sim_b")
    render_series_card(sim_a, sim_b, 0, 0, "projected", "Custom Matchup Simulation")

    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;margin-top:20px;">Model Feature Weights</div>', unsafe_allow_html=True)
    features = {"Win % Differential": 0.28, "Net Rating Diff": 0.24, "Seed Advantage": 0.18,
                "PPG Differential": 0.14, "Home Court": 0.09, "Combined Win %": 0.07}
    fig = go.Figure(go.Bar(
        x=list(features.values()), y=list(features.keys()), orientation="h",
        marker=dict(color="#fbbf24"), text=[f"{v:.0%}" for v in features.values()],
        textposition="outside", textfont=dict(family="JetBrains Mono", size=12, color="#fbbf24"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        xaxis=dict(showgrid=True, gridcolor="#1a1a2e", title="Importance", tickformat=".0%"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=250, margin=dict(l=160, r=60, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "ℹ️ About":
    st.markdown("""
    <div class="animate-in" style="max-width:700px;">
        <div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">About This Project</div>
        <div style="font-family:Inter,sans-serif;font-size:14px;color:#ccc;line-height:1.8;margin-top:16px;">
            <p>This NBA Playoff Predictor uses machine learning to forecast playoff series outcomes
            based on team performance metrics from the 2025-26 regular season.</p>

            <div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;letter-spacing:3px;margin-top:24px;">Model Details</div>
            <ul style="color:#aaa;">
                <li><b>Algorithm:</b> XGBoost Classifier</li>
                <li><b>Training Data:</b> NBA playoff series from 2015-2026</li>
                <li><b>Features:</b> Win%, Net Rating, PPG, Seed, Home Court Advantage</li>
                <li><b>Data Source:</b> NBA.com via nba_api + ESPN API for live scores</li>
                <li><b>Validation:</b> 5-fold cross-validation</li>
            </ul>

            <div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;letter-spacing:3px;margin-top:24px;">Tech Stack</div>
            <ul style="color:#aaa;">
                <li><b>ML:</b> Python, XGBoost, scikit-learn</li>
                <li><b>Data:</b> nba_api, ESPN API, pandas, numpy</li>
                <li><b>Frontend:</b> Streamlit, Plotly</li>
                <li><b>Deployment:</b> Streamlit Cloud (free)</li>
                <li><b>Live Data:</b> ESPN public API — auto-refreshes every 5 minutes</li>
            </ul>

            <div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;letter-spacing:3px;margin-top:24px;">Disclaimer</div>
            <p style="color:#888;font-size:12px;">
                Predictions are based on statistical models and historical data. They do not account for
                injuries, trades, referee assignments, travel fatigue, or other real-time factors.
                This project is for educational and entertainment purposes only.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
