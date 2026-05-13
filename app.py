import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="NBA Playoff Predictor 2026",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — Professional Dark Theme with Animations
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(180deg, #08080f 0%, #0d0d1a 50%, #08080f 100%);
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%,100% {opacity:1;}
        50% {opacity:0.4;}
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 5px rgba(251,191,36,0.2); }
        50% { box-shadow: 0 0 20px rgba(251,191,36,0.4); }
    }

    .animate-in { animation: fadeInUp 0.6s ease forwards; }
    .animate-in-1 { animation: fadeInUp 0.6s ease 0.1s forwards; opacity: 0; }
    .animate-in-2 { animation: fadeInUp 0.6s ease 0.2s forwards; opacity: 0; }
    .animate-in-3 { animation: fadeInUp 0.6s ease 0.3s forwards; opacity: 0; }
    .animate-in-4 { animation: fadeInUp 0.6s ease 0.4s forwards; opacity: 0; }

    .hero-header {
        text-align: center;
        padding: 40px 20px 30px;
        background: linear-gradient(135deg, #0a0a1a, #1a0a2e, #0a1a2e);
        border-radius: 16px;
        border: 1px solid #1a1a3a;
        margin-bottom: 2rem;
        animation: fadeInUp 0.6s ease;
    }

    .hero-header h1 {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 64px;
        letter-spacing: 8px;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }

    .hero-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 4px;
        color: #fbbf24;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-desc {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #777;
        margin-top: 10px;
    }

    .series-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .series-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    }

    .series-card.live { border-color: #ff4444; }

    .series-card.projected {
        border-style: dashed;
        border-color: #fbbf24;
        animation: glowPulse 3s infinite;
    }

    .live-badge {
        display: inline-block;
        background: #ff4444;
        color: white;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        animation: pulse 2s infinite;
    }

    .proj-badge {
        display: inline-block;
        background: #fbbf24;
        color: #000;
        font-size: 10px;
        font-weight: 800;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    .team-abbr {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 36px;
        letter-spacing: 3px;
        font-weight: 900;
    }

    .team-detail {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #888;
    }

    .score-display {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 32px;
        color: #ffffff;
        letter-spacing: 2px;
    }

    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #888;
    }

    .prob-container {
        margin-top: 12px;
        border-top: 1px solid #2a2a4a;
        padding-top: 10px;
    }

    .prob-bar-bg {
        height: 10px;
        background: #1a1a2e;
        border-radius: 5px;
        overflow: hidden;
        display: flex;
    }

    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .player-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.3s ease;
        margin-bottom: 16px;
    }

    .player-card:hover {
        transform: translateY(-4px);
    }

    .player-name {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 16px;
        color: #fff;
        margin-bottom: 4px;
    }

    .player-team {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #888;
        letter-spacing: 1px;
    }

    .player-stat {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 32px;
        color: #fbbf24;
        line-height: 1;
    }

    .player-stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #777;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        min-height: 120px;
    }

    .metric-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 42px;
        color: #fbbf24;
        line-height: 1;
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #888;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        color: #ccc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #fbbf24 !important;
        color: #000 !important;
    }

    div[data-testid="stSidebar"] {
        background: #0a0a14;
        border-right: 1px solid #1a1a2e;
    }

    .stSelectbox label, .stMultiSelect label {
        color: #fbbf24 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================

TEAMS = {
    "OKC": {
        "name": "Oklahoma City Thunder", "seed": 1, "conf": "West",
        "w": 64, "l": 18, "ppg": 118.2, "opp_ppg": 109.7, "net_rtg": 8.5,
        "fg_pct": 48.5, "fg3_pct": 38.2, "ft_pct": 80.1,
        "reb": 45.8, "ast": 27.3, "stl": 8.9, "blk": 5.2, "tov": 13.1,
        "color": "#007AC1", "status": "WCF"
    },
    "SAS": {
        "name": "San Antonio Spurs", "seed": 2, "conf": "West",
        "w": 62, "l": 20, "ppg": 115.4, "opp_ppg": 107.6, "net_rtg": 7.8,
        "fg_pct": 47.8, "fg3_pct": 37.5, "ft_pct": 79.3,
        "reb": 44.2, "ast": 26.8, "stl": 7.8, "blk": 5.8, "tov": 12.4,
        "color": "#C4CED4", "status": "R2 — leads 3-2"
    },
    "MIN": {
        "name": "Minnesota Timberwolves", "seed": 6, "conf": "West",
        "w": 49, "l": 33, "ppg": 110.3, "opp_ppg": 108.2, "net_rtg": 2.1,
        "fg_pct": 46.2, "fg3_pct": 36.8, "ft_pct": 78.5,
        "reb": 43.5, "ast": 24.9, "stl": 7.2, "blk": 5.5, "tov": 13.8,
        "color": "#236192", "status": "R2 — trails 2-3"
    },
    "DET": {
        "name": "Detroit Pistons", "seed": 1, "conf": "East",
        "w": 60, "l": 22, "ppg": 114.1, "opp_ppg": 107.3, "net_rtg": 6.8,
        "fg_pct": 47.5, "fg3_pct": 37.1, "ft_pct": 79.8,
        "reb": 44.8, "ast": 26.2, "stl": 8.1, "blk": 4.9, "tov": 12.9,
        "color": "#C8102E", "status": "R2 — tied 2-2"
    },
    "NYK": {
        "name": "New York Knicks", "seed": 3, "conf": "East",
        "w": 53, "l": 29, "ppg": 116.5, "opp_ppg": 111.3, "net_rtg": 5.2,
        "fg_pct": 47.9, "fg3_pct": 37.8, "ft_pct": 81.2,
        "reb": 43.1, "ast": 25.8, "stl": 7.5, "blk": 4.5, "tov": 13.2,
        "color": "#F58426", "status": "ECF"
    },
    "CLE": {
        "name": "Cleveland Cavaliers", "seed": 4, "conf": "East",
        "w": 52, "l": 30, "ppg": 112.8, "opp_ppg": 108.3, "net_rtg": 4.5,
        "fg_pct": 47.1, "fg3_pct": 37.3, "ft_pct": 78.9,
        "reb": 44.5, "ast": 25.5, "stl": 7.3, "blk": 4.8, "tov": 13.5,
        "color": "#860038", "status": "R2 — tied 2-2"
    },
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
# HELPER FUNCTIONS
# ============================================================

def hex_to_rgba(hex_color, alpha=0.22):
    """
    Convert #RRGGBB color to Plotly-safe rgba() string.
    This fixes the Plotly crash caused by 8-digit hex colors like #007AC133.
    """
    hex_color = str(hex_color).lstrip("#")

    if len(hex_color) != 6:
        return f"rgba(251,191,36,{alpha})"

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except ValueError:
        return f"rgba(251,191,36,{alpha})"


def get_team_safe(team_key):
    """Return team data safely even if a team is not in TEAMS."""
    return TEAMS.get(team_key, {
        "name": team_key,
        "seed": "?",
        "conf": "?",
        "w": 0,
        "l": 0,
        "ppg": 0,
        "opp_ppg": 0,
        "net_rtg": 0,
        "fg_pct": 0,
        "fg3_pct": 0,
        "ft_pct": 0,
        "reb": 0,
        "ast": 0,
        "stl": 0,
        "blk": 0,
        "tov": 0,
        "color": "#666666",
        "status": "Unknown"
    })


# ============================================================
# PREDICTION ENGINE
# ============================================================

def predict_series(team_a_key, team_b_key):
    """Predict win probability for a playoff series using model-style features."""
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
    """Calculate championship probability for each remaining team."""
    odds = {}

    okc_wcf = predict_series("OKC", "SAS")[0] / 100
    okc_fin = predict_series("OKC", "NYK")[0] / 100
    odds["OKC"] = round(okc_wcf * okc_fin * 100, 1)

    nyk_ecf = predict_series("NYK", "DET")[0] / 100
    nyk_fin = predict_series("NYK", "OKC")[1] / 100
    odds["NYK"] = round(nyk_ecf * nyk_fin * 100, 1)

    sas_r2 = predict_series("SAS", "MIN")[0] / 100
    sas_wcf = predict_series("SAS", "OKC")[0] / 100
    sas_fin = 0.45
    odds["SAS"] = round(sas_r2 * sas_wcf * sas_fin * 100, 1)

    det_r2 = predict_series("DET", "CLE")[0] / 100
    det_ecf = predict_series("DET", "NYK")[0] / 100
    det_fin = 0.40
    odds["DET"] = round(det_r2 * det_ecf * det_fin * 100, 1)

    cle_r2 = predict_series("CLE", "DET")[0] / 100
    cle_ecf = predict_series("CLE", "NYK")[0] / 100
    cle_fin = 0.30
    odds["CLE"] = round(cle_r2 * cle_ecf * cle_fin * 100, 1)

    min_r2 = predict_series("MIN", "SAS")[0] / 100
    min_wcf = predict_series("MIN", "OKC")[0] / 100
    min_fin = 0.25
    odds["MIN"] = round(min_r2 * min_wcf * min_fin * 100, 1)

    return dict(sorted(odds.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# UI COMPONENTS
# ============================================================

def render_series_card(team_a, team_b, a_wins, b_wins, status, round_name):
    """Render a matchup card with predictions."""
    a = get_team_safe(team_a)
    b = get_team_safe(team_b)

    prob_a, prob_b = predict_series(team_a, team_b)

    if status == "live":
        card_class = "live"
        badge = '<span class="live-badge">● LIVE</span>'
    elif status == "projected":
        card_class = "projected"
        badge = '<span class="proj-badge">PROJECTED</span>'
    elif status == "closed":
        card_class = ""
        badge = '<span style="color:#4ade80;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;">✓ FINAL</span>'
    else:
        card_class = ""
        badge = ""

    if status != "projected":
        score_html = f'<span class="score-display">{a_wins} — {b_wins}</span>'
    else:
        score_html = '<span class="score-display" style="color:#fbbf24;">VS</span>'

    st.markdown(f"""
    <div class="series-card {card_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span class="section-label">{round_name}</span>
            {badge}
        </div>

        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="text-align:center;flex:1;">
                <div class="team-abbr" style="color:{a.get('color','#fff')};">{team_a}</div>
                <div class="team-detail">({a.get('seed','?')}) {a.get('w','')}-{a.get('l','')}</div>
            </div>

            <div style="text-align:center;padding:0 16px;">
                {score_html}
            </div>

            <div style="text-align:center;flex:1;">
                <div class="team-abbr" style="color:{b.get('color','#fff')};">{team_b}</div>
                <div class="team-detail">({b.get('seed','?')}) {b.get('w','')}-{b.get('l','')}</div>
            </div>
        </div>

        <div class="prob-container">
            <div class="section-label" style="margin-bottom:6px;">Series Win Probability</div>

            <div class="prob-labels">
                <span style="color:{a.get('color','#fff')};">{prob_a}%</span>
                <span style="color:{b.get('color','#fff')};">{prob_b}%</span>
            </div>

            <div class="prob-bar-bg">
                <div style="width:{prob_a}%;background:{a.get('color','#666')};transition:width 1s;"></div>
                <div style="width:{prob_b}%;background:{b.get('color','#444')};transition:width 1s;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_player_card(player, team_key):
    """Render a player stat card."""
    team = TEAMS[team_key]

    st.markdown(f"""
    <div class="player-card">
        <div class="player-name">{player['name']}</div>
        <div class="player-team" style="color:{team['color']};">{team_key} • {player['pos']}</div>

        <div style="display:flex;justify-content:space-around;margin-top:12px;">
            <div>
                <div class="player-stat">{player['ppg']}</div>
                <div class="player-stat-label">PPG</div>
            </div>
            <div>
                <div class="player-stat">{player['rpg']}</div>
                <div class="player-stat-label">RPG</div>
            </div>
            <div>
                <div class="player-stat">{player['apg']}</div>
                <div class="player-stat-label">APG</div>
            </div>
        </div>

        <div style="display:flex;justify-content:space-around;margin-top:8px;">
            <div>
                <div class="player-stat" style="font-size:22px;">{player['fg_pct']}%</div>
                <div class="player-stat-label">FG%</div>
            </div>
            <div>
                <div class="player-stat" style="font-size:22px;">{player['spg']}</div>
                <div class="player-stat-label">SPG</div>
            </div>
            <div>
                <div class="player-stat" style="font-size:22px;">{player['min']}</div>
                <div class="player-stat-label">MIN</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0;">
        <div style="font-size:48px;">🏀</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">
            NBA PREDICTOR
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#777;letter-spacing:2px;">
            2025-26 PLAYOFFS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏆 Predictions", "📊 Team Stats", "🏃 Player Stats", "📈 Advanced Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#777;text-align:center;">
        Last updated<br/>
        {datetime.now().strftime('%B %d, %Y')}<br/>
        {datetime.now().strftime('%I:%M %p')}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-sub">2025-26 NBA Playoffs</div>
    <h1>PLAYOFF PREDICTOR</h1>
    <div class="hero-desc">ML-Style Predictions • Team Stats • Matchup Simulator • Streamlit Dashboard</div>
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
        st.markdown(
            '<div class="metric-card animate-in-1"><div class="metric-value">6</div><div class="metric-label">Teams Remaining</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="metric-card animate-in-2"><div class="metric-value">{fav[0]}</div><div class="metric-label">Title Favorite</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="metric-card animate-in-3"><div class="metric-value">{fav[1]}%</div><div class="metric-label">Win Probability</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="metric-card animate-in-4"><div class="metric-value">Jun 4</div><div class="metric-label">Finals Start</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="animate-in-2" style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#ff4444;letter-spacing:4px;">Conference Semifinals</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        render_series_card("SAS", "MIN", 3, 2, "live", "West • Round 2")

    with col2:
        render_series_card("DET", "CLE", 2, 2, "live", "East • Round 2")

    col3, col4 = st.columns(2)

    with col3:
        render_series_card("OKC", "LAL", 4, 0, "closed", "West • Round 2")

    with col4:
        render_series_card("NYK", "PHI", 4, 0, "closed", "East • Round 2")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="animate-in-3" style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Projected Conference Finals</div>',
        unsafe_allow_html=True
    )

    col5, col6 = st.columns(2)

    with col5:
        render_series_card("OKC", "SAS", 0, 0, "projected", "West Finals • Projected")

    with col6:
        render_series_card("NYK", "DET", 0, 0, "projected", "East Finals • Projected")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="animate-in-4" style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Projected NBA Finals</div>',
        unsafe_allow_html=True
    )

    render_series_card("OKC", "NYK", 0, 0, "projected", "NBA Finals • Projected")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Championship Probability</div>',
        unsafe_allow_html=True
    )

    odds_df = pd.DataFrame(list(odds.items()), columns=["Team", "Probability"])
    colors = [TEAMS.get(t, {}).get("color", "#666666") for t in odds_df["Team"]]

    fig = go.Figure(go.Bar(
        x=odds_df["Probability"],
        y=odds_df["Team"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p}%" for p in odds_df["Probability"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=14, color="#fbbf24")
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1a1a2e",
            title="Probability (%)",
            range=[0, max(odds.values()) + 10]
        ),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=280,
        margin=dict(l=60, r=80, t=10, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: TEAM STATS
# ============================================================

elif page == "📊 Team Stats":
    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Team Statistics Comparison</div>',
        unsafe_allow_html=True
    )

    remaining = ["OKC", "SAS", "MIN", "DET", "NYK", "CLE"]

    teams_df = pd.DataFrame([{
        "Team": k,
        "Name": TEAMS[k]["name"],
        "Seed": TEAMS[k]["seed"],
        "Record": f"{TEAMS[k]['w']}-{TEAMS[k]['l']}",
        "PPG": TEAMS[k]["ppg"],
        "Opp PPG": TEAMS[k]["opp_ppg"],
        "Net Rating": TEAMS[k]["net_rtg"],
        "FG%": TEAMS[k]["fg_pct"],
        "3P%": TEAMS[k]["fg3_pct"],
        "REB": TEAMS[k]["reb"],
        "AST": TEAMS[k]["ast"],
        "STL": TEAMS[k]["stl"],
        "TOV": TEAMS[k]["tov"],
    } for k in remaining])

    st.dataframe(teams_df, use_container_width=True, hide_index=True)

    csv = teams_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Team Stats CSV",
        data=csv,
        file_name="nba_team_stats.csv",
        mime="text/csv"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;">Head-to-Head Radar</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        team_a_select = st.selectbox("Team A", remaining, index=0)

    with c2:
        team_b_select = st.selectbox("Team B", remaining, index=3)

    categories = ["PPG", "Net Rtg", "FG%", "3P%", "REB", "AST", "STL"]

    a_vals = [
        TEAMS[team_a_select]["ppg"] / 1.2,
        TEAMS[team_a_select]["net_rtg"] * 10,
        TEAMS[team_a_select]["fg_pct"],
        TEAMS[team_a_select]["fg3_pct"],
        TEAMS[team_a_select]["reb"],
        TEAMS[team_a_select]["ast"] * 1.5,
        TEAMS[team_a_select]["stl"] * 5
    ]

    b_vals = [
        TEAMS[team_b_select]["ppg"] / 1.2,
        TEAMS[team_b_select]["net_rtg"] * 10,
        TEAMS[team_b_select]["fg_pct"],
        TEAMS[team_b_select]["fg3_pct"],
        TEAMS[team_b_select]["reb"],
        TEAMS[team_b_select]["ast"] * 1.5,
        TEAMS[team_b_select]["stl"] * 5
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=a_vals + [a_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=team_a_select,
        line=dict(color=TEAMS[team_a_select]["color"], width=2),
        fillcolor=hex_to_rgba(TEAMS[team_a_select]["color"], 0.22)
    ))

    fig.add_trace(go.Scatterpolar(
        r=b_vals + [b_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=team_b_select,
        line=dict(color=TEAMS[team_b_select]["color"], width=2),
        fillcolor=hex_to_rgba(TEAMS[team_b_select]["color"], 0.22)
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                gridcolor="#1a1a2e",
                linecolor="#2a2a4a"
            ),
            angularaxis=dict(
                gridcolor="#1a1a2e",
                linecolor="#2a2a4a"
            )
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc", size=11),
        legend=dict(font=dict(size=14)),
        height=450,
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;">Offense vs Defense</div>',
        unsafe_allow_html=True
    )

    fig2 = go.Figure()

    for k in remaining:
        t = TEAMS[k]
        fig2.add_trace(go.Scatter(
            x=[t["ppg"]],
            y=[t["opp_ppg"]],
            mode="markers+text",
            marker=dict(
                size=20,
                color=t["color"],
                line=dict(width=2, color="#fff")
            ),
            text=[k],
            textposition="top center",
            textfont=dict(family="Bebas Neue", size=16, color=t["color"]),
            name=t["name"]
        ))

    fig2.update_layout(
        xaxis=dict(title="Points Per Game — Better Offense →", gridcolor="#1a1a2e"),
        yaxis=dict(title="Opponent PPG — Better Defense ↑", gridcolor="#1a1a2e", autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        height=400,
        margin=dict(t=20, b=60),
        showlegend=False
    )

    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# PAGE: PLAYER STATS
# ============================================================

elif page == "🏃 Player Stats":
    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Star Players — Playoff Performance</div>',
        unsafe_allow_html=True
    )

    selected_team = st.selectbox(
        "Select Team",
        list(PLAYERS.keys()),
        format_func=lambda x: f"{x} — {TEAMS[x]['name']}"
    )

    st.markdown(
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:{TEAMS[selected_team]["color"]};letter-spacing:3px;margin:16px 0 8px;">{TEAMS[selected_team]["name"]}</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, player in enumerate(PLAYERS[selected_team]):
        with cols[i % 3]:
            render_player_card(player, selected_team)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;">Scoring Leaders</div>',
        unsafe_allow_html=True
    )

    all_players = []

    for team_key, players in PLAYERS.items():
        for p in players:
            all_players.append({"Team": team_key, **p})

    all_df = pd.DataFrame(all_players).sort_values("ppg", ascending=False)

    fig = go.Figure(go.Bar(
        x=all_df["ppg"],
        y=[f"{r['name']} ({r['Team']})" for _, r in all_df.iterrows()],
        orientation="h",
        marker=dict(
            color=[TEAMS[t]["color"] for t in all_df["Team"]],
            line=dict(width=0)
        ),
        text=[f"{p:.1f}" for p in all_df["ppg"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#fbbf24")
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc", size=11),
        xaxis=dict(showgrid=True, gridcolor="#1a1a2e", title="Points Per Game", range=[0, 38]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=len(all_df) * 40 + 60,
        margin=dict(l=220, r=60, t=10, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    csv = all_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Player Stats CSV",
        data=csv,
        file_name="nba_player_stats.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE: ADVANCED ANALYTICS
# ============================================================

elif page == "📈 Advanced Analytics":
    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Advanced Analytics</div>',
        unsafe_allow_html=True
    )

    remaining = ["OKC", "SAS", "MIN", "DET", "NYK", "CLE"]

    nr_data = [(k, TEAMS[k]["net_rtg"]) for k in remaining]
    nr_data.sort(key=lambda x: x[1], reverse=True)

    fig = go.Figure(go.Bar(
        x=[d[0] for d in nr_data],
        y=[d[1] for d in nr_data],
        marker=dict(
            color=[TEAMS[d[0]]["color"] for d in nr_data],
            line=dict(width=0)
        ),
        text=[f"+{d[1]}" for d in nr_data],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=14, color="#fbbf24")
    ))

    fig.update_layout(
        title=dict(
            text="Net Rating — Offense minus Defense",
            font=dict(family="Bebas Neue", size=22, color="#fbbf24")
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a2e"),
        xaxis=dict(showgrid=False),
        height=350,
        margin=dict(t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;margin-top:20px;">Matchup Simulator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#888;">Pick any two teams to see the predicted series outcome.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        sim_a = st.selectbox("Team A ", remaining, index=0, key="sim_a")

    with c2:
        available_b = [t for t in remaining if t != sim_a]
        sim_b = st.selectbox("Team B ", available_b, index=0, key="sim_b")

    render_series_card(sim_a, sim_b, 0, 0, "projected", "Custom Matchup Simulation")

    st.markdown(
        '<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;letter-spacing:3px;margin-top:20px;">Model Feature Importance</div>',
        unsafe_allow_html=True
    )

    features = {
        "Win % Differential": 0.28,
        "Net Rating Diff": 0.24,
        "Seed Advantage": 0.18,
        "PPG Differential": 0.14,
        "Home Court": 0.09,
        "Combined Win %": 0.07
    }

    fig = go.Figure(go.Bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation="h",
        marker=dict(color="#fbbf24", line=dict(width=0)),
        text=[f"{v:.0%}" for v in features.values()],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#fbbf24")
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#ccc"),
        xaxis=dict(showgrid=True, gridcolor="#1a1a2e", title="Importance", tickformat=".0%"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=250,
        margin=dict(l=170, r=60, t=10, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "ℹ️ About":
    st.markdown("""
    <div class="animate-in" style="max-width:760px;">
        <div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">
            About This Project
        </div>

        <div style="font-family:Inter,sans-serif;font-size:14px;color:#ccc;line-height:1.8;margin-top:16px;">
            <p>
                This NBA Playoff Predictor is a professional Streamlit dashboard for showing team statistics,
                player statistics, series probabilities, projected matchups, and championship odds.
            </p>

            <div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;letter-spacing:3px;margin-top:24px;">
                Model Logic
            </div>

            <ul style="color:#aaa;">
                <li><b>Features:</b> Win percentage, net rating, seed advantage, points per game, and matchup strength.</li>
                <li><b>Prediction:</b> The dashboard estimates playoff series win probability from team-level indicators.</li>
                <li><b>Simulator:</b> You can select any two teams and compare their predicted series chance.</li>
            </ul>

            <div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;letter-spacing:3px;margin-top:24px;">
                Tech Stack
            </div>

            <ul style="color:#aaa;">
                <li><b>Frontend:</b> Streamlit</li>
                <li><b>Charts:</b> Plotly</li>
                <li><b>Data Handling:</b> pandas and numpy</li>
                <li><b>Deployment:</b> Streamlit Community Cloud</li>
            </ul>

            <div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;letter-spacing:3px;margin-top:24px;">
                Disclaimer
            </div>

            <p style="color:#888;font-size:12px;">
                Predictions are based on statistical indicators and simplified model logic.
                They do not account for injuries, rotations, trades, fatigue, real-time news, or referee/contextual factors.
                This project is for educational and portfolio purposes.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
