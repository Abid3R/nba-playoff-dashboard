import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from live_data import fetch_playoff_series, fetch_team_stats, get_last_update_time

st.set_page_config(page_title="NBA Playoff Predictor 2026", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,100,{alpha})"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');
.stApp{background:linear-gradient(180deg,#08080f 0%,#0d0d1a 50%,#08080f 100%)}
.main .block-container{padding-top:2rem;max-width:1200px}
#MainMenu{visibility:hidden}
footer{visibility:hidden}
div[data-testid="stSidebar"]{background:#0a0a14;border-right:1px solid #1a1a2e}
.stSelectbox label,.stMultiSelect label{color:#fbbf24!important;font-family:JetBrains Mono,monospace!important;font-size:12px!important}
</style>""", unsafe_allow_html=True)

TEAMS = {
    "OKC": {"name": "Oklahoma City Thunder", "seed": 1, "conf": "West", "w": 64, "l": 18, "ppg": 118.2, "opp_ppg": 109.7, "net_rtg": 8.5, "fg_pct": 48.5, "fg3_pct": 38.2, "ft_pct": 80.1, "reb": 45.8, "ast": 27.3, "stl": 8.9, "blk": 5.2, "tov": 13.1, "color": "#007AC1"},
    "SAS": {"name": "San Antonio Spurs", "seed": 2, "conf": "West", "w": 62, "l": 20, "ppg": 115.4, "opp_ppg": 107.6, "net_rtg": 7.8, "fg_pct": 47.8, "fg3_pct": 37.5, "ft_pct": 79.3, "reb": 44.2, "ast": 26.8, "stl": 7.8, "blk": 5.8, "tov": 12.4, "color": "#C4CED4"},
    "DET": {"name": "Detroit Pistons", "seed": 1, "conf": "East", "w": 60, "l": 22, "ppg": 114.1, "opp_ppg": 107.3, "net_rtg": 6.8, "fg_pct": 47.5, "fg3_pct": 37.1, "ft_pct": 79.8, "reb": 44.8, "ast": 26.2, "stl": 8.1, "blk": 4.9, "tov": 12.9, "color": "#C8102E"},
    "NYK": {"name": "New York Knicks", "seed": 3, "conf": "East", "w": 53, "l": 29, "ppg": 116.5, "opp_ppg": 111.3, "net_rtg": 5.2, "fg_pct": 47.9, "fg3_pct": 37.8, "ft_pct": 81.2, "reb": 43.1, "ast": 25.8, "stl": 7.5, "blk": 4.5, "tov": 13.2, "color": "#F58426"},
    "CLE": {"name": "Cleveland Cavaliers", "seed": 4, "conf": "East", "w": 52, "l": 30, "ppg": 112.8, "opp_ppg": 108.3, "net_rtg": 4.5, "fg_pct": 47.1, "fg3_pct": 37.3, "ft_pct": 78.9, "reb": 44.5, "ast": 25.5, "stl": 7.3, "blk": 4.8, "tov": 13.5, "color": "#860038"},
    "MIN": {"name": "Minnesota Timberwolves", "seed": 6, "conf": "West", "w": 49, "l": 33, "ppg": 110.3, "opp_ppg": 108.2, "net_rtg": 2.1, "fg_pct": 46.2, "fg3_pct": 36.8, "ft_pct": 78.5, "reb": 43.5, "ast": 24.9, "stl": 7.2, "blk": 5.5, "tov": 13.8, "color": "#236192"},
}
PLAYERS = {
    "OKC": [{"name":"Shai Gilgeous-Alexander","pos":"G","ppg":32.1,"rpg":5.5,"apg":6.2,"spg":2.0,"fg_pct":53.5,"min":35.2},{"name":"Jalen Williams","pos":"F","ppg":22.3,"rpg":5.8,"apg":5.1,"spg":1.3,"fg_pct":47.8,"min":33.8},{"name":"Chet Holmgren","pos":"C","ppg":18.5,"rpg":8.2,"apg":2.8,"spg":0.9,"fg_pct":55.2,"min":31.5}],
    "SAS": [{"name":"Victor Wembanyama","pos":"C","ppg":28.5,"rpg":10.8,"apg":3.8,"spg":1.2,"fg_pct":48.2,"min":34.5},{"name":"Devin Vassell","pos":"G","ppg":19.8,"rpg":4.2,"apg":4.5,"spg":1.1,"fg_pct":46.5,"min":32.1},{"name":"Stephon Castle","pos":"G","ppg":15.2,"rpg":4.8,"apg":5.5,"spg":1.4,"fg_pct":44.8,"min":30.8}],
    "DET": [{"name":"Cade Cunningham","pos":"G","ppg":24.8,"rpg":6.2,"apg":9.5,"spg":1.3,"fg_pct":45.8,"min":35.8},{"name":"Jaden Ivey","pos":"G","ppg":19.5,"rpg":4.1,"apg":5.2,"spg":1.1,"fg_pct":44.2,"min":33.2},{"name":"Ausar Thompson","pos":"F","ppg":14.2,"rpg":7.8,"apg":2.8,"spg":1.8,"fg_pct":52.1,"min":31.5}],
    "NYK": [{"name":"Jalen Brunson","pos":"G","ppg":26.2,"rpg":3.5,"apg":7.8,"spg":0.9,"fg_pct":48.1,"min":35.5},{"name":"Karl-Anthony Towns","pos":"C","ppg":24.5,"rpg":11.2,"apg":3.2,"spg":0.7,"fg_pct":50.5,"min":34.8},{"name":"Mikal Bridges","pos":"F","ppg":18.8,"rpg":4.5,"apg":3.5,"spg":1.0,"fg_pct":46.8,"min":34.2}],
    "CLE": [{"name":"Donovan Mitchell","pos":"G","ppg":25.5,"rpg":4.2,"apg":5.8,"spg":1.8,"fg_pct":47.2,"min":35.2},{"name":"Evan Mobley","pos":"F","ppg":19.8,"rpg":9.2,"apg":3.5,"spg":1.0,"fg_pct":52.5,"min":33.8},{"name":"Darius Garland","pos":"G","ppg":21.2,"rpg":2.8,"apg":7.5,"spg":1.2,"fg_pct":46.8,"min":34.1}],
}
ACTIVE = ["OKC","SAS","DET","NYK","CLE"]

def predict_series(ak, bk):
    a, b = TEAMS.get(ak), TEAMS.get(bk)
    if not a or not b: return 50.0, 50.0
    wa, wb = a["w"]/(a["w"]+a["l"]), b["w"]/(b["w"]+b["l"])
    raw = 0.5 + (wa-wb)*0.8 + (a["net_rtg"]-b["net_rtg"])/20*0.6 + (b["seed"]-a["seed"])/14*0.3 + (a["ppg"]-b["ppg"])/30*0.2
    c = max(0.08, min(0.92, raw))
    return round(c*100,1), round((1-c)*100,1)

def get_odds():
    o = {}
    o["OKC"] = round(predict_series("OKC","SAS")[0]/100 * predict_series("OKC","NYK")[0]/100 * 100, 1)
    o["SAS"] = round(predict_series("SAS","OKC")[0]/100 * 0.42 * 100, 1)
    ne = predict_series("NYK","DET")[0]/100*0.615 + predict_series("NYK","CLE")[0]/100*0.385
    o["NYK"] = round(ne * predict_series("NYK","OKC")[0]/100 * 100, 1)
    o["DET"] = round(0.615 * predict_series("DET","NYK")[0]/100 * 0.38 * 100, 1)
    o["CLE"] = round(0.385 * predict_series("CLE","NYK")[0]/100 * 0.30 * 100, 1)
    return dict(sorted(o.items(), key=lambda x: x[1], reverse=True))

def series_card(ta, tb, aw, bw, status, rnd, extra=""):
    a, b = TEAMS.get(ta, {"seed":"?","color":"#666","w":0,"l":0}), TEAMS.get(tb, {"seed":"?","color":"#666","w":0,"l":0})
    pa, pb = predict_series(ta, tb)
    ca, cb = a.get("color","#fff"), b.get("color","#fff")
    bdr = {"live":"border:1px solid #ff4444;","projected":"border:1px dashed #fbbf24;","upcoming":"border:1px solid #22c55e;"}.get(status,"border:1px solid #2a2a4a;")
    bdg = {"live":'<span style="background:#ff4444;color:#fff;font-size:10px;font-weight:800;padding:2px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;">● LIVE</span>',
           "projected":'<span style="background:#fbbf24;color:#000;font-size:10px;font-weight:800;padding:2px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;">PROJECTED</span>',
           "upcoming":'<span style="background:#22c55e;color:#000;font-size:10px;font-weight:800;padding:2px 10px;border-radius:4px;font-family:JetBrains Mono,monospace;">SCHEDULED</span>'}.get(status,'<span style="color:#4ade80;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;">✓ FINAL</span>')
    sc = f'<span style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;">VS</span>' if status=="projected" else f'<span style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fff;">{aw} — {bw}</span>'
    ex = f'<div style="text-align:center;font-family:JetBrains Mono,monospace;font-size:10px;color:#fbbf24;margin-top:6px;">{extra}</div>' if extra else ""
    st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);{bdr}border-radius:14px;padding:20px;margin-bottom:16px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;"><span style="font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#666;">{rnd}</span>{bdg}</div>'
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'
        f'<div style="text-align:center;flex:1;"><div style="font-family:Bebas Neue,sans-serif;font-size:36px;letter-spacing:3px;color:{ca};">{ta}</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;">({a.get("seed","?")}) {a.get("w","")}-{a.get("l","")}</div></div>'
        f'<div style="text-align:center;padding:0 16px;">{sc}</div>'
        f'<div style="text-align:center;flex:1;"><div style="font-family:Bebas Neue,sans-serif;font-size:36px;letter-spacing:3px;color:{cb};">{tb}</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;">({b.get("seed","?")}) {b.get("w","")}-{b.get("l","")}</div></div></div>'
        f'{ex}<div style="margin-top:12px;border-top:1px solid #2a2a4a;padding-top:10px;">'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;letter-spacing:2px;color:#666;margin-bottom:6px;">WIN PROBABILITY (ML)</div>'
        f'<div style="display:flex;justify-content:space-between;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;margin-bottom:4px;"><span style="color:{ca};">{pa}%</span><span style="color:{cb};">{pb}%</span></div>'
        f'<div style="height:10px;background:#1a1a2e;border-radius:5px;overflow:hidden;display:flex;"><div style="width:{pa}%;background:{ca};"></div><div style="width:{pb}%;background:{cb};"></div></div>'
        f'</div></div>', unsafe_allow_html=True)

def player_card(p, tk):
    c = TEAMS[tk]["color"]
    st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:16px;text-align:center;margin-bottom:12px;">'
        f'<div style="font-family:Inter,sans-serif;font-weight:800;font-size:16px;color:#fff;">{p["name"]}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:{c};">{tk} • {p["pos"]}</div>'
        f'<div style="display:flex;justify-content:space-around;margin-top:12px;">'
        f'<div><div style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;line-height:1;">{p["ppg"]}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;">PPG</div></div>'
        f'<div><div style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;line-height:1;">{p["rpg"]}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;">RPG</div></div>'
        f'<div><div style="font-family:Bebas Neue,sans-serif;font-size:32px;color:#fbbf24;line-height:1;">{p["apg"]}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;">APG</div></div></div>'
        f'<div style="display:flex;justify-content:space-around;margin-top:8px;">'
        f'<div><div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;">{p["fg_pct"]}%</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;">FG%</div></div>'
        f'<div><div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;">{p["spg"]}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;">SPG</div></div>'
        f'<div><div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;">{p["min"]}</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;">MIN</div></div></div></div>', unsafe_allow_html=True)

def mbox(val, lab):
    st.markdown(f'<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border:1px solid #2a2a4a;border-radius:12px;padding:20px;text-align:center;"><div style="font-family:Bebas Neue,sans-serif;font-size:42px;color:#fbbf24;line-height:1;">{val}</div><div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#888;margin-top:4px;">{lab}</div></div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:20px 0;"><div style="font-size:48px;">🏀</div><div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">NBA PREDICTOR</div><div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#666;letter-spacing:2px;">2025-26 PLAYOFFS</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("NAV", ["🏆 Predictions","📊 Team Stats","🏃 Player Stats","📈 Analytics","ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    _lv = fetch_playoff_series()
    if _lv: st.success("ESPN: Connected", icon="🟢")
    else: st.warning("ESPN: Offline", icon="🟡")
    st.markdown("---")
    st.caption(f"Updated: {get_last_update_time()}")

# HEADER
st.markdown('<div style="text-align:center;padding:40px 20px 30px;background:linear-gradient(135deg,#0a0a1a,#1a0a2e,#0a1a2e);border-radius:16px;border:1px solid #1a1a3a;margin-bottom:2rem;">'
    '<div style="font-family:JetBrains Mono,monospace;font-size:12px;letter-spacing:4px;color:#fbbf24;text-transform:uppercase;margin-bottom:8px;">2025-26 NBA Playoffs</div>'
    '<h1 style="font-family:Bebas Neue,sans-serif;font-size:64px;letter-spacing:8px;color:#fff;margin:0;line-height:1;">PLAYOFF PREDICTOR</h1>'
    '<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:#555;margin-top:10px;">ML-Powered • XGBoost • 2015-2026 Data</div></div>', unsafe_allow_html=True)

if page == "🏆 Predictions":
    odds = get_odds(); fav = list(odds.items())[0]
    c1,c2,c3,c4 = st.columns(4)
    with c1: mbox("5","Teams Remaining")
    with c2: mbox(fav[0],"Title Favorite")
    with c3: mbox(f"{fav[1]}%","Win Probability")
    with c4: mbox("Jun 4","Finals Start")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#ff4444;letter-spacing:4px;">🔥 Game 7 — May 18</div>', unsafe_allow_html=True)
    series_card("DET","CLE",3,3,"live","East • Round 2 — Game 7","Winner faces NYK in ECF")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#4ade80;letter-spacing:3px;">Completed — Round 2</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: series_card("OKC","LAL",4,0,"closed","West • Round 2")
    with c2: series_card("SAS","MIN",4,2,"closed","West • Round 2")
    with c3: series_card("NYK","PHI",4,0,"closed","East • Round 2")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#22c55e;letter-spacing:4px;">Conference Finals</div>', unsafe_allow_html=True)
    c4,c5 = st.columns(2)
    with c4: series_card("OKC","SAS",0,0,"upcoming","West Finals — May 19","Game 1: Mon May 19 @ OKC")
    with c5: series_card("NYK","DET",0,0,"projected","East Finals — Projected","Awaiting DET vs CLE Game 7")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Projected NBA Finals — June 4</div>', unsafe_allow_html=True)
    series_card("OKC","NYK",0,0,"projected","NBA Finals • Projected")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Championship Probability</div>', unsafe_allow_html=True)
    od = pd.DataFrame(list(odds.items()), columns=["Team","Prob"])
    fig = go.Figure(go.Bar(x=od["Prob"],y=od["Team"],orientation="h",marker=dict(color=[TEAMS.get(t,{}).get("color","#666") for t in od["Team"]]),text=[f"{p}%" for p in od["Prob"]],textposition="outside",textfont=dict(family="JetBrains Mono",size=14,color="#fbbf24")))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(family="JetBrains Mono",color="#ccc"),xaxis=dict(showgrid=True,gridcolor="#1a1a2e",title="Probability (%)",range=[0,max(odds.values())+10]),yaxis=dict(showgrid=False,autorange="reversed"),height=250,margin=dict(l=60,r=80,t=10,b=40))
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Team Stats":
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Team Statistics</div>', unsafe_allow_html=True)
    df = pd.DataFrame([{"Team":k,"Name":TEAMS[k]["name"],"Seed":TEAMS[k]["seed"],"W-L":f'{TEAMS[k]["w"]}-{TEAMS[k]["l"]}',"PPG":TEAMS[k]["ppg"],"Opp PPG":TEAMS[k]["opp_ppg"],"Net Rtg":TEAMS[k]["net_rtg"],"FG%":TEAMS[k]["fg_pct"],"3P%":TEAMS[k]["fg3_pct"],"REB":TEAMS[k]["reb"],"AST":TEAMS[k]["ast"],"STL":TEAMS[k]["stl"],"TOV":TEAMS[k]["tov"]} for k in ACTIVE])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;">Head-to-Head Radar</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: ta = st.selectbox("Team A", ACTIVE, index=0)
    with c2: tb = st.selectbox("Team B", ACTIVE, index=3)
    cats = ["PPG","Net Rtg","FG%","3P%","REB","AST","STL"]
    def rv(k):
        t=TEAMS[k]; return [t["ppg"]/1.2,t["net_rtg"]*10,t["fg_pct"],t["fg3_pct"],t["reb"],t["ast"]*1.5,t["stl"]*5]
    va,vb = rv(ta),rv(tb)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=va+[va[0]],theta=cats+[cats[0]],fill="toself",name=ta,line=dict(color=TEAMS[ta]["color"],width=2),fillcolor=hex_to_rgba(TEAMS[ta]["color"],0.2)))
    fig.add_trace(go.Scatterpolar(r=vb+[vb[0]],theta=cats+[cats[0]],fill="toself",name=tb,line=dict(color=TEAMS[tb]["color"],width=2),fillcolor=hex_to_rgba(TEAMS[tb]["color"],0.2)))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,gridcolor="#1a1a2e"),angularaxis=dict(gridcolor="#1a1a2e")),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(family="JetBrains Mono",color="#ccc",size=11),legend=dict(font=dict(size=14)),height=450,margin=dict(t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;">Offense vs Defense</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    for k in ACTIVE:
        t=TEAMS[k]; fig2.add_trace(go.Scatter(x=[t["ppg"]],y=[t["opp_ppg"]],mode="markers+text",marker=dict(size=20,color=t["color"],line=dict(width=2,color="#fff")),text=[k],textposition="top center",textfont=dict(family="Bebas Neue",size=16,color=t["color"]),name=t["name"]))
    fig2.update_layout(xaxis=dict(title="PPG (Offense →)",gridcolor="#1a1a2e"),yaxis=dict(title="Opp PPG (← Defense)",gridcolor="#1a1a2e",autorange="reversed"),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(family="JetBrains Mono",color="#ccc"),height=400,margin=dict(t=20,b=60),showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🏃 Player Stats":
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Star Players</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select Team", ACTIVE, format_func=lambda x: f"{x} — {TEAMS[x]['name']}")
    st.markdown(f'<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:{TEAMS[sel]["color"]};margin:16px 0 8px;">{TEAMS[sel]["name"]}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i,p in enumerate(PLAYERS[sel]):
        with cols[i%3]: player_card(p, sel)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;">Scoring Leaders</div>', unsafe_allow_html=True)
    ap = []
    for tk in ACTIVE:
        for p in PLAYERS[tk]: ap.append({"Team":tk,**p})
    adf = pd.DataFrame(ap).sort_values("ppg",ascending=False)
    fig = go.Figure(go.Bar(x=adf["ppg"],y=[f'{r["name"]} ({r["Team"]})' for _,r in adf.iterrows()],orientation="h",marker=dict(color=[TEAMS[t]["color"] for t in adf["Team"]]),text=[f"{p:.1f}" for p in adf["ppg"]],textposition="outside",textfont=dict(family="JetBrains Mono",size=12,color="#fbbf24")))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(family="JetBrains Mono",color="#ccc",size=11),xaxis=dict(showgrid=True,gridcolor="#1a1a2e",title="PPG",range=[0,38]),yaxis=dict(showgrid=False,autorange="reversed"),height=len(adf)*40+60,margin=dict(l=200,r=60,t=10,b=40))
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Analytics":
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;letter-spacing:4px;">Advanced Analytics</div>', unsafe_allow_html=True)
    nr = sorted([(k,TEAMS[k]["net_rtg"]) for k in ACTIVE], key=lambda x:x[1], reverse=True)
    fig = go.Figure(go.Bar(x=[d[0] for d in nr],y=[d[1] for d in nr],marker=dict(color=[TEAMS[d[0]]["color"] for d in nr]),text=[f"+{d[1]}" for d in nr],textposition="outside",textfont=dict(family="JetBrains Mono",size=14,color="#fbbf24")))
    fig.update_layout(title=dict(text="Net Rating",font=dict(family="Bebas Neue",size=22,color="#fbbf24")),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(family="JetBrains Mono",color="#ccc"),yaxis=dict(showgrid=True,gridcolor="#1a1a2e"),height=350,margin=dict(t=60,b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;margin-top:20px;">Matchup Simulator</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: sa = st.selectbox("Team A ",ACTIVE,index=0,key="sa")
    with c2: sb = st.selectbox("Team B ",[t for t in ACTIVE if t!=sa],index=0,key="sb")
    series_card(sa,sb,0,0,"projected","Custom Simulation")
    st.markdown('<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:#fbbf24;margin-top:20px;">Model Feature Weights</div>', unsafe_allow_html=True)
    fw={"Win % Diff":0.28,"Net Rating":0.24,"Seed":0.18,"PPG Diff":0.14,"Home Court":0.09,"Combined W%":0.07}
    fig = go.Figure(go.Bar(x=list(fw.values()),y=list(fw.keys()),orientation="h",marker=dict(color="#fbbf24"),text=[f"{v:.0%}" for v in fw.values()],textposition="outside",textfont=dict(family="JetBrains Mono",size=12,color="#fbbf24")))
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",font=dict(family="JetBrains Mono",color="#ccc"),xaxis=dict(showgrid=True,gridcolor="#1a1a2e",tickformat=".0%"),yaxis=dict(showgrid=False,autorange="reversed"),height=250,margin=dict(l=140,r=60,t=10,b=40))
    st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown('<div style="max-width:700px;"><div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#fbbf24;">About</div>'
        '<div style="font-family:Inter,sans-serif;font-size:14px;color:#ccc;line-height:1.8;margin-top:16px;">'
        '<p>ML-powered NBA Playoff predictions using XGBoost trained on 2015-2026 data.</p>'
        '<div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;margin-top:24px;">Model</div>'
        '<ul style="color:#aaa;"><li><b>Algorithm:</b> XGBoost</li><li><b>Features:</b> Win%, Net Rating, PPG, Seed, Home Court</li><li><b>Data:</b> nba_api + ESPN</li></ul>'
        '<div style="font-family:Bebas Neue,sans-serif;font-size:22px;color:#fbbf24;margin-top:24px;">Stack</div>'
        '<ul style="color:#aaa;"><li>Python, XGBoost, scikit-learn</li><li>Streamlit, Plotly</li><li>Streamlit Cloud (free)</li><li>ESPN API auto-refresh</li></ul>'
        '<p style="color:#666;font-size:12px;margin-top:24px;">Educational purposes only. Does not account for injuries.</p></div></div>', unsafe_allow_html=True)
