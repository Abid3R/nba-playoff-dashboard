"""Data utilities for live NBA team stats.

The app tries to fetch current season stats from nba_api. If NBA.com blocks or
rate-limits the request, the app falls back to a local demo dataset so the UI
still works on Streamlit Cloud.
"""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd

EAST_TEAMS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
    "NYK", "ORL", "PHI", "TOR", "WAS"
}
WEST_TEAMS = {
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC",
    "PHX", "POR", "SAC", "SAS", "UTA"
}

TEAM_COLORS: Dict[str, str] = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}


def _conference_for_abbr(abbr: str) -> str:
    if abbr in EAST_TEAMS:
        return "East"
    if abbr in WEST_TEAMS:
        return "West"
    return "Unknown"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare columns used by the app and model."""
    df = df.copy()

    if "TEAM_ABBREVIATION" not in df.columns and "TEAM_ABBREV" in df.columns:
        df = df.rename(columns={"TEAM_ABBREV": "TEAM_ABBREVIATION"})

    if "TEAM_NAME" not in df.columns and "GROUP_VALUE" in df.columns:
        df = df.rename(columns={"GROUP_VALUE": "TEAM_NAME"})

    df["CONFERENCE"] = df["TEAM_ABBREVIATION"].map(_conference_for_abbr)
    df["TEAM_COLOR"] = df["TEAM_ABBREVIATION"].map(TEAM_COLORS).fillna("#334155")

    # Add seeds inside each conference using win percentage.
    df["W_PCT"] = pd.to_numeric(df.get("W_PCT", 0.0), errors="coerce").fillna(0.0)
    df["SEED"] = (
        df.groupby("CONFERENCE")["W_PCT"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    for col in ["PTS", "REB", "AST", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS", "NET_RATING", "OFF_RATING", "DEF_RATING", "PACE"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Friendly display columns.
    df["WIN_PCT_DISPLAY"] = (df["W_PCT"] * 100).round(1)
    df["FG_PCT_DISPLAY"] = (df["FG_PCT"] * 100).round(1)
    df["FG3_PCT_DISPLAY"] = (df["FG3_PCT"] * 100).round(1)
    df["FT_PCT_DISPLAY"] = (df["FT_PCT"] * 100).round(1)

    return df.sort_values(["CONFERENCE", "SEED"]).reset_index(drop=True)


def fetch_team_stats(season: str = "2025-26", season_type: str = "Regular Season") -> Tuple[pd.DataFrame, bool, str]:
    """Fetch live NBA team stats using nba_api.

    Returns: (dataframe, is_live, message)
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats

        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
            timeout=30,
        ).get_data_frames()[0]

        advanced = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            timeout=30,
        ).get_data_frames()[0]

        keep_adv = [c for c in ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"] if c in advanced.columns]
        merged = pd.merge(base, advanced[keep_adv], on="TEAM_ID", how="left")
        return _normalize_columns(merged), True, "Live NBA stats loaded successfully."
    except Exception as exc:
        return _normalize_columns(get_demo_team_stats()), False, f"Live NBA API failed, so demo data is shown. Reason: {exc}"


def get_demo_team_stats() -> pd.DataFrame:
    """Fallback demo dataset for all 30 teams.

    These values are only placeholders for UI demonstration when live data cannot
    be reached. The app clearly marks the source when this fallback is used.
    """
    rows = [
        ("BOS", "Boston Celtics", .780, 120.6, 46.3, 26.9, 12.2, .487, .389, .812, 11.2, 122.0, 110.8, 99.2),
        ("NYK", "New York Knicks", .610, 112.8, 45.2, 24.4, 12.4, .465, .369, .781, 5.5, 118.0, 112.5, 96.7),
        ("MIL", "Milwaukee Bucks", .600, 118.1, 44.2, 26.5, 13.0, .487, .374, .772, 1.2, 117.2, 116.0, 101.2),
        ("CLE", "Cleveland Cavaliers", .598, 109.8, 43.3, 27.4, 12.8, .479, .366, .765, 4.2, 115.0, 110.8, 97.1),
        ("ORL", "Orlando Magic", .573, 104.9, 42.3, 24.7, 14.7, .462, .352, .759, 3.0, 112.4, 109.4, 96.1),
        ("IND", "Indiana Pacers", .573, 123.3, 41.5, 30.8, 13.1, .507, .374, .782, 3.5, 121.5, 118.0, 102.4),
        ("PHI", "Philadelphia 76ers", .573, 107.8, 43.0, 24.9, 12.1, .465, .363, .826, 2.8, 115.1, 112.3, 96.4),
        ("MIA", "Miami Heat", .549, 108.1, 42.4, 25.8, 12.7, .466, .370, .818, 1.0, 113.3, 112.3, 95.5),
        ("ATL", "Atlanta Hawks", .524, 112.2, 44.8, 26.6, 13.7, .465, .364, .792, 0.5, 116.3, 115.8, 100.2),
        ("CHI", "Chicago Bulls", .476, 111.2, 43.8, 25.0, 12.2, .469, .358, .788, -1.2, 113.8, 115.0, 98.4),
        ("TOR", "Toronto Raptors", .430, 110.0, 42.8, 28.4, 13.8, .471, .351, .755, -3.1, 112.2, 115.3, 99.1),
        ("BKN", "Brooklyn Nets", .390, 108.6, 42.0, 25.4, 13.4, .459, .362, .762, -5.0, 110.5, 115.5, 98.0),
        ("CHA", "Charlotte Hornets", .256, 106.6, 40.3, 24.8, 14.4, .458, .353, .780, -8.9, 108.1, 117.0, 99.0),
        ("WAS", "Washington Wizards", .183, 113.7, 41.1, 27.9, 14.1, .470, .348, .764, -9.5, 111.0, 120.5, 101.0),
        ("DET", "Detroit Pistons", .171, 109.9, 43.3, 25.5, 15.2, .463, .349, .785, -9.9, 109.3, 119.2, 100.4),
        ("OKC", "Oklahoma City Thunder", .695, 118.3, 42.0, 27.1, 11.8, .490, .389, .825, 7.5, 120.1, 112.6, 99.8),
        ("DEN", "Denver Nuggets", .695, 112.3, 44.4, 29.5, 12.5, .496, .374, .762, 4.1, 118.5, 114.4, 96.3),
        ("MIN", "Minnesota Timberwolves", .683, 110.2, 43.6, 26.6, 13.4, .485, .387, .777, 5.8, 116.0, 110.2, 96.8),
        ("LAC", "LA Clippers", .622, 111.4, 43.0, 25.5, 12.7, .489, .381, .817, 4.0, 117.0, 113.0, 96.2),
        ("DAL", "Dallas Mavericks", .610, 117.9, 42.9, 25.7, 12.5, .481, .371, .758, 3.2, 118.2, 115.0, 98.6),
        ("PHX", "Phoenix Suns", .600, 108.6, 41.1, 26.8, 13.8, .493, .382, .808, 2.5, 116.1, 113.6, 97.6),
        ("NOP", "New Orleans Pelicans", .600, 110.4, 44.0, 27.0, 13.0, .486, .383, .771, 3.0, 116.8, 113.8, 97.9),
        ("LAL", "Los Angeles Lakers", .573, 109.9, 43.2, 28.5, 13.4, .499, .377, .782, 1.5, 115.2, 113.7, 99.1),
        ("SAC", "Sacramento Kings", .561, 116.6, 44.0, 28.3, 12.6, .477, .366, .743, 1.4, 116.4, 115.0, 100.1),
        ("GSW", "Golden State Warriors", .561, 117.8, 46.7, 29.3, 14.0, .477, .380, .781, 1.5, 116.1, 114.6, 101.8),
        ("HOU", "Houston Rockets", .500, 108.0, 45.5, 24.8, 13.2, .459, .352, .777, 0.2, 113.0, 112.8, 99.0),
        ("UTA", "Utah Jazz", .378, 115.7, 45.6, 27.2, 15.1, .467, .355, .829, -5.8, 112.0, 117.8, 100.5),
        ("MEM", "Memphis Grizzlies", .329, 105.8, 42.6, 24.7, 14.9, .439, .346, .766, -7.0, 108.9, 115.9, 98.9),
        ("POR", "Portland Trail Blazers", .256, 106.4, 42.7, 23.1, 14.3, .439, .345, .791, -8.5, 107.8, 116.3, 98.6),
        ("SAS", "San Antonio Spurs", .268, 112.1, 44.2, 29.9, 14.5, .462, .347, .782, -6.4, 110.3, 116.7, 101.1),
    ]
    cols = [
        "TEAM_ABBREVIATION", "TEAM_NAME", "W_PCT", "PTS", "REB", "AST", "TOV",
        "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS", "OFF_RATING", "DEF_RATING", "PACE"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["NET_RATING"] = df["OFF_RATING"] - df["DEF_RATING"]
    return df
