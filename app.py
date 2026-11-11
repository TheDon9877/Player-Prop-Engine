import time
import sqlite3
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Tuple, Literal
from scipy.stats import nbinom, gamma, poisson
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats
except ImportError:
    playergamelogs = None
    leaguedashteamstats = None

try:
    import nfl_data_py as nfl
except ImportError:
    nfl = None

DB_PATH = "prop_engine.sqlite"
PropFamily = Literal["negative_binomial", "gamma", "poisson_td"]

FEATURE_COLS = [
    "roll_5_stat",
    "roll_15_stat",
    "roll_5_vol",
    "roll_5_share",
    "opp_def_rating",
    "vegas_total",
    "spread",
    "days_rest",
]

ODDS_API_MARKETS = {
    "NBA": {
        "Points": "player_points",
        "Rebounds": "player_rebounds",
        "Assists": "player_assists",
        "3PM": "player_threes",
    },
    "NFL": {
        "Anytime TD": "player_anytime_td",
        "Rushing Yards": "player_rush_yds",
        "Receiving Yards": "player_reception_yds",
        "Receptions": "player_receptions",
        "Passing TDs": "player_pass_tds",
        "Passing Yards": "player_pass_yds",
    },
}


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


@st.cache_resource
def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_game_logs (
            game_date TEXT,
            sport TEXT,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            minutes_or_snaps REAL,
            usage_or_share REAL,
            opp_def_rating REAL,
            vegas_total REAL,
            spread REAL,
            days_rest REAL,
            market TEXT,
            actual_stat REAL,
            book_line REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            closing_over_odds INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logged_bets_v2 (
            bet_id TEXT PRIMARY KEY,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            game_date TEXT,
            sport TEXT,
            player_name TEXT,
            market TEXT,
            line REAL,
            side TEXT,
            model_prob REAL,
            fair_prob REAL,
            book_odds INTEGER,
            edge REAL,
            kelly_pct REAL,
            status TEXT DEFAULT 'PENDING',
            actual_stat REAL,
            profit_units REAL DEFAULT 0.0
        )
    """)
    conn.commit()
    conn.close()
    return True


def save_logs(df: pd.DataFrame):
    cols = [
        "game_date", "sport", "player_name", "team", "opponent",
        "minutes_or_snaps", "usage_or_share", "opp_def_rating",
        "vegas_total", "spread", "days_rest", "market", "actual_stat",
        "book_line", "over_odds", "under_odds", "closing_over_odds"
    ]
    out = df[cols].copy()
    out["game_date"] = out["game_date"].astype(str)
    conn = get_conn()
    out.to_sql("player_game_logs", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def load_logs(sport: str, market: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM player_game_logs WHERE sport = ? AND market = ? ORDER BY game_date",
        conn,
        params=(sport, market),
    )
    conn.close()
    return df


def log_selected_bets(bets_df: pd.DataFrame, sport: str, game_date) -> int:
    conn = get_conn()
    cur = conn.cursor()
    inserted = 0
    g_date_str = str(game_date)
    for _, row in bets_df.iterrows():
        bet_id = f"{g_date_str}_{sport}_{row['player']}_{row['market']}_{row['side']}"
        exists = cur.execute("SELECT 1 FROM logged_bets_v2 WHERE bet_id = ?", (bet_id,)).fetchone()
        if not exists:
            cur.execute(
                """
                INSERT INTO logged_bets_v2 
                (bet_id, game_date, sport, player_name, market, line, side, 
                 model_prob, fair_prob, book_odds, edge, kelly_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet_id, g_date_str, sport, row["player"], row["market"],
                    float(row["line"]), row["side"], float(row["model_prob"]),
                    float(row["fair_market_prob"]), int(row["book_odds"]),
                    float(row["edge"]), float(row["kelly_fraction"]),
                ),
            )
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


def auto_grade_pending_bets() -> int:
    conn = get_conn()
    cur = conn.cursor()
    pending = pd.read_sql_query("SELECT * FROM logged_bets_v2 WHERE status = 'PENDING'", conn)
    graded_count = 0

    for _, bet in pending.iterrows():
        match = cur.execute(
            """
            SELECT actual_stat FROM player_game_logs
            WHERE game_date = ? AND sport = ? AND player_name = ? 
              AND market = ? AND actual_stat IS NOT NULL
            LIMIT 1
            """,
            (str(bet["game_date"]), bet["sport"], bet["player_name"], bet["market"]),
        ).fetchone()

        if match:
            actual = float(match[0])
            line = float(bet["line"])
            odds = int(bet["book_odds"])
            stake = float(bet["kelly_pct"]) * 100

            if actual == line:
                status, profit = "PUSH", 0.0
            else:
                won = (actual > line) if bet["side"] == "OVER" else (actual < line)
                dec_odds = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
                status = "WON" if won else "LOST"
                profit = round(stake * (dec_odds - 1), 3) if won else round(-stake, 3)

            cur.execute(
                "UPDATE logged_bets_v2 SET status = ?, actual_stat = ?, profit_units = ? WHERE bet_id = ?",
                (status, actual, profit, bet["bet_id"]),
            )
            graded_count += 1

    conn.commit()
    conn.close()
    return graded_count


def load_all_bets() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM logged_bets_v2 ORDER BY game_date DESC, logged_at DESC", conn)
    conn.close()
    return df


@dataclass
class BetOpportunity:
    player: str
    market: str
    line: float
    side: str
    model_prob: float
    fair_market_prob: float
    book_odds: int
    edge: float
    kelly_fraction: float


class DistributionalPropEngine:
    def __init__(self, family: PropFamily, hgb_params: Dict = None):
        self.family = family
        self.hgb_params = hgb_params or {
            "learning_rate": 0.04,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 15,
            "max_iter": 300,
            "random_state": 42,
        }
        self.mean_model = None
        self.var_model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)

        loss_type = "poisson" if self.family in ["negative_binomial", "poisson_td"] else "gamma"
        y_clean = np.clip(y_full.values.astype(float), 1e-3 if loss_type == "gamma" else 0.0, None)

        self.mean_model = HistGradientBoostingRegressor(loss=loss_type, **self.hgb_params)
        self.mean_model.fit(X_full, y_clean)

        if self.family in ["negative_binomial", "gamma"]:
            preds = np.clip(self.mean_model.predict(X_full), 1e-3, None)
            sq_err = np.clip((y_full.values.astype(float) - preds) ** 2, 1e-3, None)

            self.var_model = HistGradientBoostingRegressor(loss="gamma", **self.hgb_params)
            self.var_model.fit(X_full, sq_err)

    def predict_probabilities(self, X: pd.DataFrame, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.clip(self.mean_model.predict(X), 1e-3, None)

        if self.family == "poisson_td":
            prob_over = 1.0 - poisson.pmf(0, mu)
            return prob_over, 1.0 - prob_over

        var = np.clip(self.var_model.predict(X), mu + 1e-3, None)

        if self.family == "negative_binomial":
            r = (mu ** 2) / np.maximum(var - mu, 1e-3)
            p = r / (r + mu)
            prob_under = nbinom.cdf(np.floor(lines), r, p)
            return 1.0 - prob_under, prob_under

        alpha = (mu ** 2) / var
        theta = var / mu
        prob_under = gamma.cdf(lines, a=alpha, scale=theta)
        return 1.0 - prob_under, prob_under


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def power_devig(over_odds: int, under_odds: int) -> Tuple[float, float]:
    p_o = american_to_implied(over_odds)
    p_u = american_to_implied(under_odds)
    if p_o + p_u <= 1.0:
        return p_o, p_u
    k = brentq(lambda x: (p_o ** x) + (p_u ** x) - 1.0, 0.5, 5.0)
    return p_o ** k, p_u ** k


def calculate_kelly(model_prob: float, book_odds: int, fraction: float = 0.25) -> float:
    decimal_odds = (book_odds / 100) + 1 if book_odds > 0 else (100 / abs(book_odds)) + 1
    b = decimal_odds - 1.0
    q = 1.0 - model_prob
    full_kelly = ((b * model_prob) - q) / b
    return max(0.0, full_kelly * fraction)


def evaluate_slate(
    df_slate: pd.DataFrame,
    prob_over: np.ndarray,
    prob_under: np.ndarray,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> list[BetOpportunity]:
    opportunities = []
    for i, row in df_slate.iterrows():
        if pd.isna(row["book_line"]):
            continue
        fair_over, fair_under = power_devig(int(row["over_odds"]), int(row["under_odds"]))

        edge_over = prob_over[i] - fair_over
        if edge_over >= min_edge:
            opportunities.append(
                BetOpportunity(
                    player=row["player_name"],
                    market=row["market"],
                    line=float(row["book_line"]),
                    side="OVER",
                    model_prob=round(float(prob_over[i]), 4),
                    fair_market_prob=round(float(fair_over), 4),
                    book_odds=int(row["over_odds"]),
                    edge=round(float(edge_over), 4),
                    kelly_fraction=round(calculate_kelly(prob_over[i], int(row["over_odds"]), kelly_fraction), 4),
                )
            )

        edge_under = prob_under[i] - fair_under
        if edge_under >= min_edge:
            opportunities.append(
                BetOpportunity(
                    player=row["player_name"],
                    market=row["market"],
                    line=float(row["book_line"]),
                    side="UNDER",
                    model_prob=round(float(prob_under[i]), 4),
                    fair_market_prob=round(float(fair_under), 4),
                    book_odds=int(row["under_odds"]),
                    edge=round(float(edge_under), 4),
                    kelly_fraction=round(calculate_kelly(prob_under[i], int(row["under_odds"]), kelly_fraction), 4),
                )
            )

    return sorted(opportunities, key=lambda x: x.edge, reverse=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_name", "game_date"]).copy()
    if "days_rest" not in df.columns:
        df["days_rest"] = 2.0

    df["roll_5_stat"] = df.groupby("player_name")["actual_stat"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["roll_15_stat"] = df.groupby("player_name")["actual_stat"].transform(
        lambda x: x.shift(1).rolling(15, min_periods=1).mean()
    )
    df["roll_5_vol"] = df.groupby("player_name")["minutes_or_snaps"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["roll_5_share"] = df.groupby("player_name")["usage_or_share"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    return df.dropna(subset=["roll_5_stat"])


def monte_carlo_injury_sim(team_slate: pd.DataFrame, out_players: list[str], n_sims: int = 10000) -> pd.DataFrame:
    active = team_slate[~team_slate["player_name"].isin(out_players)].copy()
    vacated_share = team_slate[team_slate["player_name"].isin(out_players)]["roll_5_share"].sum()

    if vacated_share > 0 and len(active) > 0:
        weights = active["roll_5_share"] / np.maximum(active["roll_5_share"].sum(), 1e-3)
        active["sim_share"] = active["roll_5_share"] + (vacated_share * 0.75 * weights)
        scale_factor = active["sim_share"] / np.maximum(active["roll_5_share"], 1e-3)
    else:
        active["sim_share"] = active["roll_5_share"]
        scale_factor = 1.0

    rng = np.random.default_rng(42)
    adj_means = np.clip(active["roll_5_stat"].values * scale_factor, 0.1, None)
    shapes = np.clip(adj_means * 1.5, 1.0, None)
    scales = adj_means / shapes
    sims = rng.gamma(shape=shapes[:, None], scale=scales[:, None], size=(len(active), n_sims))

    active["mc_adj_mean"] = np.round(sims.mean(axis=1), 2)
    active["mc_adj_std"] = np.round(sims.std(axis=1), 2)
    lines = active["book_line"].fillna(active["roll_5_stat"]).values
    active["mc_prob_over"] = np.round((sims > lines[:, None]).mean(axis=1), 4)
    return active


def run_walk_forward_backtest(df: pd.DataFrame, family: PropFamily, min_edge: float = 0.03, kelly_frac: float = 0.25) -> pd.DataFrame:
    df = df.dropna(subset=["actual_stat", "book_line"]).copy()
    dates = sorted(df["game_date"].unique())
    if len(dates) < 6:
        return pd.DataFrame()

    split_idx = int(len(dates) * 0.6)
    results = []

    for i in range(split_idx, len(dates) - 1):
        train_cutoff = dates[i]
        test_date = dates[i + 1]

        train_df = df[df["game_date"] < train_cutoff]
        val_df = df[df["game_date"] == train_cutoff]
        test_df = df[df["game_date"] == test_date].copy()

        if len(train_df) < 30 or len(test_df) == 0:
            continue

        engine = DistributionalPropEngine(family=family)
        engine.fit(train_df[FEATURE_COLS], train_df["actual_stat"], val_df[FEATURE_COLS], val_df["actual_stat"])
        prob_over, prob_under = engine.predict_probabilities(test_df[FEATURE_COLS], test_df["book_line"].values)

        for idx, (_, row) in enumerate(test_df.iterrows()):
            fair_o, fair_u = power_devig(int(row["over_odds"]), int(row["under_odds"]))
            close_o, close_u = power_devig(int(row["closing_over_odds"]), -int(row["closing_over_odds"]))

            for side, p_mod, p_fair, p_close, odds in [
                ("OVER", prob_over[idx], fair_o, close_o, int(row["over_odds"])),
                ("UNDER", prob_under[idx], fair_u, close_u, int(row["under_odds"])),
            ]:
                edge = p_mod - p_fair
                if edge >= min_edge:
                    stake = calculate_kelly(p_mod, odds, kelly_frac)
                    won = (row["actual_stat"] > row["book_line"]) if side == "OVER" else (row["actual_stat"] < row["book_line"])
                    dec_odds = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
                    profit = stake * (dec_odds - 1) if won else -stake
                    clv = p_close - p_fair

                    results.append({
                        "date": test_date,
                        "player": row["player_name"],
                        "side": side,
                        "line": row["book_line"],
                        "actual": row["actual_stat"],
                        "edge": edge,
                        "clv": clv,
                        "stake": stake,
                        "profit": profit,
                        "won": won,
                    })

    return pd.DataFrame(results)


def seed_demo_data() -> int:
    rng = np.random.default_rng(42)
    players = [
        ("Luka Doncic", "DAL", "BOS", 36.0, 0.36, 32.5, 9.0, 9.5, 3.5),
        ("Kyrie Irving", "DAL", "BOS", 35.0, 0.28, 25.5, 5.0, 5.5, 3.0),
        ("Jayson Tatum", "BOS", "DAL", 36.5, 0.31, 27.5, 8.5, 4.5, 3.0),
        ("Jaylen Brown", "BOS", "DAL", 34.0, 0.27, 23.5, 5.5, 3.5, 2.0),
        ("Nikola Jokic", "DEN", "LAL", 35.0, 0.32, 26.5, 12.5, 9.0, 1.5),
        ("LeBron James", "LAL", "DEN", 35.0, 0.30, 25.5, 7.5, 8.0, 2.0),
    ]
    base_date = datetime.now().date() - timedelta(days=30)
    rows = []

    for day in range(31):
        g_date = base_date + timedelta(days=day)
        is_today = (day == 30)
        for name, team, opp, mins, usg, pts, reb, ast, threes in players:
            for mkt, mean_val in [("Points", pts), ("Rebounds", reb), ("Assists", ast), ("3PM", threes)]:
                actual = np.nan if is_today else float(max(0, rng.poisson(mean_val)))
                rows.append({
                    "game_date": str(g_date),
                    "sport": "NBA",
                    "player_name": name,
                    "team": team,
                    "opponent": opp,
                    "minutes_or_snaps": mins + rng.normal(0, 2),
                    "usage_or_share": usg,
                    "opp_def_rating": 113.5,
                    "vegas_total": 228.5,
                    "spread": 3.5,
                    "days_rest": 2.0,
                    "market": mkt,
                    "actual_stat": actual,
                    "book_line": mean_val,
                    "over_odds": -110,
                    "under_odds": -110,
                    "closing_over_odds": -115,
                })

    df = pd.DataFrame(rows)
    save_logs(df)
    return len(df)


def sync_nba_logs(season: str = "2025-26") -> int:
    if playergamelogs is None:
        raise RuntimeError("nba_api library not installed.")
    logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    time.sleep(0.6)

    team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season).get_data_frames()[0]
    def_map = dict(zip(team_stats["TEAM_ABBREVIATION"], team_stats["DEF_RATING"]))

    logs["game_date_dt"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.sort_values(["PLAYER_NAME", "game_date_dt"])
    logs["days_rest"] = logs.groupby("PLAYER_NAME")["game_date_dt"].diff().dt.days.fillna(3).clip(0, 5)

    mins = np.maximum(logs["MIN"], 1.0)
    usage_proxy = (logs["FGA"] + 0.44 * logs["FTA"] + logs["TOV"]) / mins
    market_map = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST", "3PM": "FG3M"}
    frames = []

    for market_name, col in market_map.items():
        opp_teams = logs["MATCHUP"].str[-3:]
        df = pd.DataFrame({
            "game_date": logs["game_date_dt"].dt.date.astype(str),
            "sport": "NBA",
            "player_name": logs["PLAYER_NAME"],
            "team": logs["TEAM_ABBREVIATION"],
            "opponent": opp_teams,
            "minutes_or_snaps": logs["MIN"].astype(float),
            "usage_or_share": usage_proxy.astype(float),
            "opp_def_rating": opp_teams.map(def_map).fillna(114.0).astype(float),
            "vegas_total": 226.5,
            "spread": 0.0,
            "days_rest": logs["days_rest"].astype(float),
            "market": market_name,
            "actual_stat": logs[col].astype(float),
            "book_line": np.nan,
            "over_odds": -110,
            "under_odds": -110,
            "closing_over_odds": -110,
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    save_logs(combined)
    return len(combined)


def sync_nfl_logs(years: list[int] = None) -> int:
    if years is None:
        years = [2024, 2025]
    if nfl is None:
        raise RuntimeError("nfl_data_py library not installed.")
    weekly = nfl.import_weekly_data(years=years)
    weekly["total_tds"] = weekly["rushing_tds"].fillna(0) + weekly["receiving_tds"].fillna(0)
    opp_def_rank = weekly.groupby("opponent_team")["passing_yards"].mean().rank(pct=True).to_dict()

    market_map = {
        "Anytime TD": ("total_tds", "target_share"),
        "Rushing Yards": ("rushing_yards", "carries"),
        "Receiving Yards": ("receiving_yards", "target_share"),
        "Receptions": ("receptions", "target_share"),
        "Passing TDs": ("passing_tds", "attempts"),
        "Passing Yards": ("passing_yards", "attempts"),
    }
    frames = []

    for market_name, (stat_col, share_col) in market_map.items():
        sub = weekly[weekly[stat_col].notna()].copy()
        approx_date = pd.to_datetime(sub["season"].astype(str) + "-09-01") + pd.to_timedelta((sub["week"] - 1) * 7, unit="D")
        df = pd.DataFrame({
            "game_date": approx_date.dt.date.astype(str),
            "sport": "NFL",
            "player_name": sub["player_display_name"],
            "team": sub["recent_team"],
            "opponent": sub["opponent_team"],
            "minutes_or_snaps": sub["attempts"].fillna(0) + sub["targets"].fillna(0) + sub["carries"].fillna(0),
            "usage_or_share": sub[share_col].fillna(0.15).astype(float),
            "opp_def_rating": sub["opponent_team"].map(opp_def_rank).fillna(0.50).astype(float) * 100,
            "vegas_total": 45.0,
            "spread": 0.0,
            "days_rest": 7.0,
            "market": market_name,
            "actual_stat": sub[stat_col].astype(float),
            "book_line": np.nan,
            "over_odds": -110,
            "under_odds": -110,
            "closing_over_odds": -110,
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    save_logs(combined)
    return len(combined)


def fetch_live_prop_odds(api_key: str, sport: str, market: str) -> pd.DataFrame:
    sport_key = "basketball_nba" if sport == "NBA" else "americanfootball_nfl"
    market_key = ODDS_API_MARKETS[sport][market]

    events_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    events = requests.get(events_url, params={"apiKey": api_key}, timeout=15).json()

    rows = []
    today = str(datetime.now().date())

    if isinstance(events, list):
        for ev in events:
            event_id = ev["id"]
            odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
            res = requests.get(
                odds_url,
                params={"apiKey": api_key, "regions": "us", "markets": market_key, "oddsFormat": "american"},
                timeout=15,
            ).json()

            for book in res.get("bookmakers", [])[:1]:
                for mkt in book.get("markets", []):
                    by_player = {}
                    for outcome in mkt.get("outcomes", []):
                        player = outcome.get("description")
                        side = outcome.get("name")
                        line = outcome.get("point", 0.5)
                        price = outcome.get("price", -110)

                        by_player.setdefault(player, {"line": line, "over": -110, "under": -110})
                        if side in ["Over", "Yes"]:
                            by_player[player]["over"] = price
                        elif side in ["Under", "No"]:
                            by_player[player]["under"] = price

                    for player, vals in by_player.items():
                        rows.append({
                            "game_date": today,
                            "sport": sport,
                            "player_name": player,
                            "team": ev.get("home_team", ""),
                            "opponent": ev.get("away_team", ""),
                            "minutes_or_snaps": np.nan,
                            "usage_or_share": np.nan,
                            "opp_def_rating": 114.0 if sport == "NBA" else 50.0,
                            "vegas_total": 226.5 if sport == "NBA" else 45.5,
                            "spread": 0.0,
                            "days_rest": 2.0,
                            "market": market,
                            "actual_stat": np.nan,
                            "book_line": float(vals["line"]),
                            "over_odds": int(vals["over"]),
                            "under_odds": int(vals["under"]),
                            "closing_over_odds": int(vals["over"]),
                        })

    live_df = pd.DataFrame(rows)
    if not live_df.empty:
        save_logs(live_df)
    return live_df


st.set_page_config(page_title="Quant Prop Engine", layout="wide")
init_db()

st.title("Unified Sports Prop ML & Simulation Engine")

sport = st.sidebar.selectbox("Select Sport", ["NBA", "NFL"])
markets = {
    "NBA": {
        "Points": "negative_binomial",
        "Rebounds": "negative_binomial",
        "Assists": "negative_binomial",
        "3PM": "negative_binomial",
    },
    "NFL": {
        "Anytime TD": "poisson_td",
        "Rushing Yards": "gamma",
        "Receiving Yards": "gamma",
        "Receptions": "negative_binomial",
        "Passing TDs": "poisson_td",
        "Passing Yards": "gamma",
    },
}
market = st.sidebar.selectbox("Select Prop Market", list(markets[sport].keys()))
family = markets[sport][market]
min_edge = st.sidebar.slider("Minimum +EV Edge (%)", 1.0, 10.0, 3.0) / 100.0
kelly_frac = st.sidebar.slider("Fractional Kelly Multiplier", 0.10, 0.50, 0.25)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Daily +EV Slate",
    "2. Monte Carlo Injury Simulator",
    "3. Walk-Forward Backtest & CLV",
    "4. Data & API Sync",
    "5. Live Bet Tracker & Auto-Grader",
])

raw_df = load_logs(sport, market)

with tab1:
    st.subheader(f"Today's +EV {sport} {market} Opportunities")
    if len(raw_df) > 60:
        feat_df = engineer_features(raw_df)
        latest_date = feat_df["game_date"].max()
        train = feat_df[(feat_df["game_date"] < latest_date) & (feat_df["actual_stat"].notna())]
        today = feat_df[feat_df["game_date"] == latest_date].reset_index(drop=True)

        if len(train) > 30 and not today.empty and today["book_line"].notna().any():
            engine = DistributionalPropEngine(family=family)
            split = int(len(train) * 0.85)
            engine.fit(
                train[FEATURE_COLS].iloc[:split], train["actual_stat"].iloc[:split],
                train[FEATURE_COLS].iloc[split:], train["actual_stat"].iloc[split:],
            )
            p_over, p_under = engine.predict_probabilities(today[FEATURE_COLS], today["book_line"].fillna(0).values)
            bets = evaluate_slate(today, p_over, p_under, min_edge=min_edge, kelly_fraction=kelly_frac)

            if bets:
                bets_df = pd.DataFrame([vars(b) for b in bets])
                st.dataframe(bets_df, use_container_width=True)
                if st.button("Log All +EV Picks to Tracker"):
                    added = log_selected_bets(bets_df, sport=sport, game_date=latest_date)
                    st.success(f"Logged {added} new bet(s) into your central tracker.")
            else:
                st.info("No props exceeded your minimum edge threshold today. Try lowering the Minimum +EV Edge slider on the left.")
        else:
            st.warning("Historical logs found, but no live sportsbook lines exist for today's slate. Pull live odds or load demo data in Tab 4.")
    else:
        st.warning("Database is empty for this market. Open Tab 4 to click 'Load Instant Demo Data' or sync live APIs.")

with tab2:
    st.subheader("Interactive Injury & Usage Redistribution Simulator")
    if len(raw_df) > 0:
        feat_df = engineer_features(raw_df)
        teams = sorted(feat_df["team"].dropna().unique())
        if teams:
            selected_team = st.selectbox("Select Team", teams)
            team_df = feat_df[feat_df["team"] == selected_team].sort_values("game_date").drop_duplicates("player_name", keep="last")
            out_players = st.multiselect("Mark Players OUT (Injury Report)", team_df["player_name"].tolist())

            if st.button("Run 10,000 Monte Carlo Simulations"):
                sim_results = monte_carlo_injury_sim(team_df, out_players)
                st.dataframe(
                    sim_results[["player_name", "roll_5_stat", "mc_adj_mean", "mc_adj_std", "book_line", "mc_prob_over"]],
                    use_container_width=True,
                )
    else:
        st.info("Load data in Tab 4 first to simulate team injury scenarios.")

with tab3:
    st.subheader("Walk-Forward Historical Validation & Closing Line Value")
    if len(raw_df) > 80:
        if st.button("Run Walk-Forward Backtest"):
            feat_df = engineer_features(raw_df)
            bt_df = run_walk_forward_backtest(feat_df, family=family, min_edge=min_edge, kelly_frac=kelly_frac)
            if not bt_df.empty:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Bets", len(bt_df))
                c2.metric("Win Rate", f"{bt_df['won'].mean():.1%}")
                c3.metric("Total Units Won", f"{bt_df['profit'].sum():+.2f}u")
                c4.metric("Avg Closing Line Value", f"{bt_df['clv'].mean():+.2%}")
                bt_df["cum_profit"] = bt_df["profit"].cumsum()
                st.line_chart(bt_df.set_index("date")["cum_profit"])
                st.dataframe(bt_df, use_container_width=True)
            else:
                st.info("Not enough paired historical lines and box scores to run a multi-day backtest yet.")
    else:
        st.info("Load data in Tab 4 first to run historical backtests.")

with tab4:
    st.subheader("Central Data Store & API Connectors")
    st.write(f"Current rows in database for **{sport} - {market}**: **{len(raw_df):,}**")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Quick Test (No API Key Needed)")
        if st.button("Load Instant Demo Data"):
            count = seed_demo_data()
            st.success(f"Loaded {count} sample NBA rows!")
            st.rerun()

    with c2:
        st.markdown("#### Live Historical Stats")
        if st.button("Pull NBA Historical Logs"):
            with st.spinner("Downloading NBA stats..."):
                count = sync_nba_logs("2025-26")
                st.success(f"Synced {count:,} NBA rows.")

        if st.button("Pull NFL Historical Logs"):
            with st.spinner("Downloading NFL stats..."):
                count = sync_nfl_logs()
                st.success(f"Synced {count:,} NFL rows.")

    with c3:
        st.markdown("#### Live Sportsbook Lines")
        api_key = st.text_input("The-Odds-API Key", type="password")
        if st.button(f"Fetch Live {sport} {market} Odds"):
            if not api_key:
                st.error("Enter your Odds API key first.")
            else:
                with st.spinner("Fetching live lines..."):
                    live_df = fetch_live_prop_odds(api_key, sport, market)
                    if not live_df.empty:
                        st.success(f"Loaded {len(live_df)} live lines.")
                        st.dataframe(live_df[["player_name", "book_line", "over_odds", "under_odds"]])
                    else:
                        st.warning("No live lines returned for this market right now.")

with tab5:
    st.subheader("Real-World Bet Performance & Auto-Grader")
    if st.button("Auto-Grade Pending Bets from Synced Box Scores"):
        graded = auto_grade_pending_bets()
        st.success(f"Graded {graded} completed bet(s).")

    history_df = load_all_bets()
    if not history_df.empty:
        settled = history_df[history_df["status"].isin(["WON", "LOST", "PUSH"])]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Logged Bets", len(history_df))
        c2.metric("Pending Bets", len(history_df[history_df["status"] == "PENDING"]))

        if not settled.empty:
            c3.metric("Settled Win Rate", f"{(settled['status'] == 'WON').mean():.1%}")
            c4.metric("Net Profit (Units)", f"{settled['profit_units'].sum():+.2f}u")
            chart_df = settled.sort_values("game_date").copy()
            chart_df["cum_units"] = chart_df["profit_units"].cumsum()
            st.line_chart(chart_df.set_index("game_date")["cum_units"])

        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No bets logged yet. Generate picks in Tab 1 and click 'Log All +EV Picks to Tracker'.")
