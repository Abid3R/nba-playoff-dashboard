# live_data.py — Fetches real-time NBA playoff data
# Uses ESPN's free public API (no key needed)

import requests
import streamlit as st
from datetime import datetime

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_STANDINGS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings"
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"

# Team abbreviation mapping (ESPN name → your abbreviation)
ESPN_ABBR_MAP = {
    "OKC": "OKC", "SA": "SAS", "SAS": "SAS", "MIN": "MIN",
    "DET": "DET", "NY": "NYK", "NYK": "NYK", "CLE": "CLE",
    "LAL": "LAL", "PHI": "PHI", "PHX": "PHX", "POR": "POR",
    "HOU": "HOU", "DEN": "DEN", "ORL": "ORL", "ATL": "ATL",
    "TOR": "TOR", "BOS": "BOS", "MIL": "MIL", "MIA": "MIA",
    "IND": "IND", "CHI": "CHI", "BKN": "BKN", "SAC": "SAC",
    "GS": "GSW", "GSW": "GSW", "LAC": "LAC", "DAL": "DAL",
    "MEM": "MEM", "NO": "NOP", "NOP": "NOP", "CHA": "CHA",
    "WAS": "WAS", "UTA": "UTA",
}

TEAM_COLORS = {
    "OKC": "#007AC1", "SAS": "#C4CED4", "MIN": "#236192",
    "DET": "#C8102E", "NYK": "#F58426", "CLE": "#860038",
    "LAL": "#552583", "PHI": "#006BB6", "PHX": "#E56020",
    "BOS": "#007A33", "MIL": "#00471B", "MIA": "#98002E",
}


@st.cache_data(ttl=300)  # Cache for 5 minutes then auto-refresh
def fetch_playoff_series():
    """
    Fetch current NBA playoff series data from ESPN.
    Returns a list of series dicts.
    """
    try:
        # ESPN doesn't have a direct "playoff bracket" endpoint,
        # so we'll fetch the scoreboard which shows current/recent games
        resp = requests.get(ESPN_SCOREBOARD, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        series_list = []
        seen = set()

        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            if len(competitors) == 2:
                notes = event.get("status", {}).get("type", {}).get("description", "")
                series_notes = ""
                for note in event.get("notes", []):
                    series_notes = note.get("headline", "")

                home = competitors[0]
                away = competitors[1]

                home_abbr = ESPN_ABBR_MAP.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
                away_abbr = ESPN_ABBR_MAP.get(away["team"]["abbreviation"], away["team"]["abbreviation"])

                matchup_key = tuple(sorted([home_abbr, away_abbr]))
                if matchup_key not in seen:
                    seen.add(matchup_key)
                    series_list.append({
                        "home": home_abbr,
                        "away": away_abbr,
                        "home_score": home.get("score", "0"),
                        "away_score": away.get("score", "0"),
                        "series_note": series_notes,
                        "status": notes,
                    })

        return series_list if series_list else None

    except Exception as e:
        return None


@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_team_stats():
    """Fetch current team stats from ESPN."""
    try:
        resp = requests.get(ESPN_STANDINGS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        teams = {}
        for group in data.get("children", []):
            conf = group.get("name", "")
            for standing in group.get("standings", {}).get("entries", []):
                team_info = standing.get("team", {})
                abbr = ESPN_ABBR_MAP.get(
                    team_info.get("abbreviation", ""),
                    team_info.get("abbreviation", "")
                )
                stats_dict = {}
                for stat in standing.get("stats", []):
                    stats_dict[stat["name"]] = stat.get("value", 0)

                teams[abbr] = {
                    "name": team_info.get("displayName", abbr),
                    "abbr": abbr,
                    "conf": "East" if "East" in conf else "West",
                    "w": int(stats_dict.get("wins", 0)),
                    "l": int(stats_dict.get("losses", 0)),
                    "ppg": round(stats_dict.get("pointsFor", 0) / max(stats_dict.get("gamesPlayed", 1), 1), 1),
                    "opp_ppg": round(stats_dict.get("pointsAgainst", 0) / max(stats_dict.get("gamesPlayed", 1), 1), 1),
                    "color": TEAM_COLORS.get(abbr, "#666"),
                }

        return teams if teams else None

    except Exception as e:
        return None


@st.cache_data(ttl=300)
def fetch_team_roster_stats(team_abbr):
    """Fetch player stats for a specific team from ESPN."""
    # ESPN team ID mapping
    ESPN_TEAM_IDS = {
        "OKC": "25", "SAS": "24", "MIN": "16", "DET": "8",
        "NYK": "18", "CLE": "5", "LAL": "13", "PHI": "20",
    }
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return None

    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        players = []
        for athlete in data.get("athletes", []):
            players.append({
                "name": athlete.get("displayName", "Unknown"),
                "pos": athlete.get("position", {}).get("abbreviation", "?"),
                "jersey": athlete.get("jersey", "?"),
                "age": athlete.get("age", "?"),
            })

        return players[:15] if players else None

    except Exception:
        return None


def get_last_update_time():
    """Return formatted timestamp for display."""
    return datetime.now().strftime("%B %d, %Y at %I:%M %p")
