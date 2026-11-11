import os
import time
import sqlite3
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

DB_PATH = "prop_engine.db"
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

MARKETS_CONFIG = {
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


# ==============================================================================
# 1. THREAD-SAFE SQLITE DATABASE (WAL MODE)
# ==============================================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
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
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP,
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


def load_logs(sport: str, market: str = None) -> pd.DataFrame:
    conn = get_conn()
    if market:
        query = "SELECT * FROM player_game_logs WHERE sport = ? AND market = ? ORDER BY game_date"
        df = pd.read_sql_query(query, conn, params=(sport, market))
    else:
        query = "SELECT * FROM player_game_logs WHERE sport = ? ORDER BY game_date"
        df = pd.read_sql_query(query, conn, params=(sport,))
    conn.close()
    return df


def log_selected_bets(bets_df: pd.DataFrame, sport: str, game_date: str) -> int:
    conn = get_conn()
    inserted = 0
    for _, row in bets_df.iterrows():
        bet_id = f"{game_date}_{sport}_{row['player']}_{row['market']}_{row['side']}"
        cur = conn.execute("SELECT 1 FROM logged_bets_v2 WHERE bet_id = ?", (bet_id,))
        if cur.fetchone() is None:
            conn.execute(
                """
                INSERT INTO logged_bets_v2 
                (bet_id, game_date, sport, player_name, market, line, side, 
                 model_prob, fair_prob, book_odds, edge, kelly_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet_id, str(game_date), sport, row["player"], row["market"],
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
    pending = pd.read_sql_query("SELECT * FROM logged_bets_v2 WHERE status = 'PENDING'", conn)
    graded_count = 0

    for _, bet in pending.iterrows():
        cur = conn.execute(
            """
            SELECT actual_stat FROM player_game_logs
            WHERE game_date = ? AND sport = ? AND player_name = ? 
              AND market = ? AND actual_stat IS NOT NULL
            LIMIT 1
            """,
            (bet["game_date"], bet["sport"], bet["player_name"], bet["market"]),
        )
        match = cur.fetchone()
        if match is not None and match[0] is not None:
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

            conn.execute(
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


# ==============================================================================
# 2. DISTRIBUTIONAL ML ENGINE (SCIKIT-LEARN HISTGRADIENTBOOSTING)
# ==============================================================================
@dataclass
class BetOpportunity:
    player: str
    team: str
    market: str
    line: float
    side: str
    proj_mean: float
    model_prob: float
    fair_market_prob: float
    book_odds: int
    edge: float
    kelly_fraction: float


class DistributionalPropEngine:
    def __init__(self, family: PropFamily):
        self.family = family
        self.mean_model = None
        self.var_model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        loss_type = "poisson" if self.family in ["negative_binomial", "poisson_td"] else "gamma"
        y_clean = np.clip(y_train.values, 1e-3 if loss_type == "gamma" else 0.0, None)

        self.mean_model = HistGradientBoostingRegressor(
            loss=loss_type,
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
        )
        self.mean_model.fit(X_train, y_clean)

        if self.family in ["negative_binomial", "gamma"]:
            preds = np.clip(self.mean_model.predict(X_train), 1e-3, None)
            sq_err = np.clip((y_train.values - preds) ** 2, 1e-3, None)
            self.var_model = HistGradientBoostingRegressor(
                loss="gamma",
                learning_rate=0.04,
                max_iter=200,
                max_leaf_nodes=23,
                min_samples_leaf=20,
                random_state=42,
            )
            self.var_model.fit(X_train, sq_err)

    def predict_params(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.clip(self.mean_model.predict(X), 1e-3, None)
        if self.family == "poisson_td":
            return mu, mu
        var = np.clip(self.var_model.predict(X), mu + 1e-3, None)
        return mu, var

    def predict_probabilities(self, X: pd.DataFrame, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, var = self.predict_params(X)

        if self.family == "poisson_td":
            prob_over = 1.0 - poisson.pmf(0, mu)
            return prob_over, 1.0 - prob_over, mu

        if self.family == "negative_binomial":
            r = (mu ** 2) / np.maximum(var - mu, 1e-3)
            p = r / (r + mu)
            prob_under = nbinom.cdf(np.floor(lines), r, p)
            return 1.0 - prob_under, prob_under, mu

        alpha = (mu ** 2) / var
        theta = var / mu
        prob_under = gamma.cdf(lines, a=alpha, scale=theta)
        return 1.0 - prob_under, prob_under, mu


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def prob_to_american(prob: float) -> int:
    prob = np.clip(prob, 0.01, 0.99)
    if prob >= 0.5:
        return int(round(-100 * prob / (1.0 - prob)))
    return int(round(100 * (1.0 - prob) / prob))


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
    proj_mu: np.ndarray,
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
                    team=row.get("team", ""),
                    market=row["market"],
                    line=float(row["book_line"]),
                    side="OVER",
                    proj_mean=round(float(proj_mu[i]), 2),
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
                    team=row.get("team", ""),
                    market=row["market"],
                    line=float(row["book_line"]),
                    side="UNDER",
                    proj_mean=round(float(proj_mu[i]), 2),
                    model_prob=round(float(prob_under[i]), 4),
                    fair_market_prob=round(float(fair_under), 4),
                    book_odds=int(row["under_odds"]),
                    edge=round(float(edge_under), 4),
                    kelly_fraction=round(calculate_kelly(prob_under[i], int(row["under_odds"]), kelly_fraction), 4),
                )
            )

    return sorted(opportunities, key=lambda x: x.edge, reverse=True)


# ==============================================================================
# 3. FEATURE STORE, MULTI-MARKET SCANNER, MONTE CARLO & BACKTESTER
# ==============================================================================
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
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
    return df


def run_all_star_scanner(sport: str, min_edge: float, kelly_frac: float) -> Tuple[pd.DataFrame, str]:
    """Scans EVERY market for the selected sport simultaneously to build a master +EV board."""
    all_bets = []
    latest_date_str = ""

    for mkt_name, family in MARKETS_CONFIG[sport].items():
        raw_mkt = load_logs(sport, mkt_name)
        if len(raw_mkt) < 40:
            continue
        feat_df = engineer_features(raw_mkt)
        latest_date = feat_df["game_date"].max()
        latest_date_str = str(latest_date)

        train = feat_df[(feat_df["game_date"] < latest_date) & (feat_df["actual_stat"].notna())]
        today = feat_df[feat_df["game_date"] == latest_date].reset_index(drop=True)

        if len(train) >= 30 and not today.empty and today["book_line"].notna().any():
            engine = DistributionalPropEngine(family=family)
            engine.fit(train[FEATURE_COLS], train["actual_stat"])
            p_over, p_under, mu = engine.predict_probabilities(today[FEATURE_COLS], today["book_line"].fillna(0).values)
            bets = evaluate_slate(today, p_over, p_under, mu, min_edge=min_edge, kelly_fraction=kelly_frac)
            all_bets.extend([vars(b) for b in bets])

    if not all_bets:
        return pd.DataFrame(), latest_date_str
    master_df = pd.DataFrame(all_bets).sort_values("edge", ascending=False).reset_index(drop=True)
    return master_df, latest_date_str


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

    for i in range(split_idx, len(dates)):
        test_date = dates[i]
        train_df = df[df["game_date"] < test_date]
        test_df = df[df["game_date"] == test_date].copy()

        if len(train_df) < 30 or len(test_df) == 0:
            continue

        engine = DistributionalPropEngine(family=family)
        engine.fit(train_df[FEATURE_COLS], train_df["actual_stat"])
        prob_over, prob_under, _ = engine.predict_probabilities(test_df[FEATURE_COLS], test_df["book_line"].values)

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


# ==============================================================================
# 4. DATA SYNC & DEMO SEEDER (NBA + NFL)
# ==============================================================================
def seed_demo_data() -> int:
    rng = np.random.default_rng(42)
    nba_players = [
        ("Luka Doncic", "DAL", "BOS", 36.0, 0.36, 32.5, 9.0, 9.5, 3.5),
        ("Kyrie Irving", "DAL", "BOS", 35.0, 0.28, 25.5, 5.0, 5.5, 3.0),
        ("Jayson Tatum", "BOS", "DAL", 36.5, 0.31, 27.5, 8.5, 4.5, 3.0),
        ("Jaylen Brown", "BOS", "DAL", 34.0, 0.27, 23.5, 5.5, 3.5, 2.0),
        ("Nikola Jokic", "DEN", "LAL", 35.0, 0.32, 26.5, 12.5, 9.0, 1.5),
        ("LeBron James", "LAL", "DEN", 35.0, 0.30, 25.5, 7.5, 8.0, 2.0),
        ("Shai Gilgeous-Alexander", "OKC", "MIN", 34.5, 0.34, 31.0, 5.5, 6.5, 1.5),
        ("Anthony Edwards", "MIN", "OKC", 35.5, 0.32, 27.5, 5.5, 5.0, 3.0),
    ]
    nfl_players = [
        ("Josh Allen", "BUF", "KC", 36.0, 0.35, 0.35, 38.5, 0.0, 0.0, 2.1, 268.5),
        ("Patrick Mahomes", "KC", "BUF", 38.0, 0.38, 0.15, 22.5, 0.0, 0.0, 2.2, 275.5),
        ("Ja'Marr Chase", "CIN", "BAL", 9.5, 0.29, 0.72, 0.0, 92.5, 7.5, 0.0, 0.0),
        ("Derrick Henry", "BAL", "CIN", 21.0, 0.33, 0.85, 96.5, 12.5, 1.5, 0.0, 0.0),
        ("Saquon Barkley", "PHI", "DAL", 22.0, 0.34, 0.82, 102.5, 21.5, 2.5, 0.0, 0.0),
        ("CeeDee Lamb", "DAL", "PHI", 10.5, 0.31, 0.65, 0.0, 88.5, 7.5, 0.0, 0.0),
    ]

    base_date = datetime.now().date() - timedelta(days=25)
    rows = []

    for day in range(26):
        g_date = str(base_date + timedelta(days=day))
        is_today = (day == 25)
        for name, team, opp, mins, usg, pts, reb, ast, threes in nba_players:
            for mkt, mean_val in [("Points", pts), ("Rebounds", reb), ("Assists", ast), ("3PM", threes)]:
                actual = np.nan if is_today else float(max(0, rng.poisson(mean_val * rng.uniform(0.88, 1.15))))
                book_line = round(mean_val * rng.choice([0.92, 1.0, 1.08]) * 2) / 2
                rows.append({
                    "game_date": g_date, "sport": "NBA", "player_name": name,
                    "team": team, "opponent": opp, "minutes_or_snaps": mins + rng.normal(0, 2),
                    "usage_or_share": usg, "opp_def_rating": 113.5, "vegas_total": 228.5,
                    "spread": 3.5, "days_rest": 2.0, "market": mkt, "actual_stat": actual,
                    "book_line": max(0.5, book_line), "over_odds": -110, "under_odds": -110,
                    "closing_over_odds": -118,
                })

        for name, team, opp, vol, usg, td, rush, rec_y, rec, p_td, p_yds in nfl_players:
            nfl_mkts = [
                ("Anytime TD", td, 0.5), ("Rushing Yards", rush, rush),
                ("Receiving Yards", rec_y, rec_y), ("Receptions", rec, rec),
                ("Passing TDs", p_td, 1.5), ("Passing Yards", p_yds, p_yds),
            ]
            for mkt, mean_val, line_base in nfl_mkts:
                if mean_val == 0.0 and line_base == 0.0:
                    continue
                if mkt in ["Anytime TD", "Passing TDs", "Receptions"]:
                    actual = np.nan if is_today else float(rng.poisson(mean_val))
                else:
                    actual = np.nan if is_today else float(max(0, rng.normal(mean_val, mean_val * 0.28)))
                rows.append({
                    "game_date": g_date, "sport": "NFL", "player_name": name,
                    "team": team, "opponent": opp, "minutes_or_snaps": vol,
                    "usage_or_share": usg, "opp_def_rating": 52.0, "vegas_total": 47.5,
                    "spread": 2.5, "days_rest": 7.0, "market": mkt, "actual_stat": actual,
                    "book_line": max(0.5, round(line_base * 2) / 2), "over_odds": -110,
                    "under_odds": -110, "closing_over_odds": -116,
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


def fetch_live_prop_odds(api_key: str, sport: str, market: str, include_sharp: bool = True) -> pd.DataFrame:
    sport_key = "basketball_nba" if sport == "NBA" else "americanfootball_nfl"
    market_key = ODDS_API_MARKETS[sport][market]
    regions = "us,eu" if include_sharp else "us"

    games_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    games_res = requests.get(
        games_url,
        params={"apiKey": api_key, "regions": "us", "markets": "spreads,totals"},
        timeout=15,
    ).json()

    game_contexts = {}
    if isinstance(games_res, list):
        for g in games_res:
            gid = g["id"]
            total = 226.5 if sport == "NBA" else 45.5
            spread = 0.0
            for book in g.get("bookmakers", [])[:1]:
                for m in book.get("markets", []):
                    if m["key"] == "totals" and m.get("outcomes"):
                        total = m["outcomes"][0].get("point", total)
                    if m["key"] == "spreads" and m.get("outcomes"):
                        spread = abs(m["outcomes"][0].get("point", 0.0))
            game_contexts[gid] = {"total": total, "spread": spread}

    events_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    events = requests.get(events_url, params={"apiKey": api_key}, timeout=15).json()

    rows = []
    today = str(datetime.now().date())

    if isinstance(events, list):
        for ev in events:
            event_id = ev["id"]
            ctx = game_contexts.get(event_id, {"total": 226.5, "spread": 0.0})
            odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
            res = requests.get(
                odds_url,
                params={"apiKey": api_key, "regions": regions, "markets": market_key, "oddsFormat": "american"},
                timeout=15,
            ).json()

            bookmakers = res.get("bookmakers", [])
            if not bookmakers:
                continue

            # Prefer DraftKings/FanDuel for retail line, Pinnacle for closing benchmark
            retail_book = next((b for b in bookmakers if b["key"] in ["draftkings", "fanduel"]), bookmakers[0])
            sharp_book = next((b for b in bookmakers if b["key"] == "pinnacle"), retail_book)

            sharp_prices = {}
            for mkt in sharp_book.get("markets", []):
                for outcome in mkt.get("outcomes", []):
                    if outcome.get("name") in ["Over", "Yes"]:
                        sharp_prices[outcome.get("description")] = outcome.get("price", -110)

            for mkt in retail_book.get("markets", []):
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
                        "vegas_total": float(ctx["total"]),
                        "spread": float(ctx["spread"]),
                        "days_rest": 2.0,
                        "market": market,
                        "actual_stat": np.nan,
                        "book_line": float(vals["line"]),
                        "over_odds": int(vals["over"]),
                        "under_odds": int(vals["under"]),
                        "closing_over_odds": int(sharp_prices.get(player, vals["over"])),
                    })

    live_df = pd.DataFrame(rows)
    if not live_df.empty:
        save_logs(live_df)
    return live_df


# ==============================================================================
# 5. PRO SPORTS TERMINAL UI & STYLING
# ==============================================================================
st.set_page_config(page_title="APEX PROP | Quant Betting Terminal", layout="wide")
init_db()

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .terminal-header {
        background: linear-gradient(90deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 18px;
    }
    .terminal-title { font-size: 28px; font-weight: 800; color: #38bdf8; margin: 0; letter-spacing: 0.5px; }
    .terminal-sub { font-size: 14px; color: #94a3b8; margin-top: 4px; }
    .pick-card {
        background-color: #131c2e;
        border: 1px solid #26354d;
        border-left: 5px solid #10b981;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .pick-header { display: flex; justify-content: space-between; align-items: center; }
    .pick-player { font-size: 18px; font-weight: 700; color: #f8fafc; }
    .edge-badge {
        background-color: rgba(16, 185, 129, 0.16);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }
    .pick-line { font-size: 15px; color: #38bdf8; font-weight: 600; margin: 6px 0; }
    .pick-meta { font-size: 12px; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="terminal-header">
    <div class="terminal-title">APEX PROP // QUANTITATIVE SPORTS BETTING TERMINAL</div>
    <div class="terminal-sub">Distributional Gradient-Boosted Engine • Multi-Market Scanner • Monte Carlo Injury Simulator • Devigged Kelly Sizing</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("Engine Parameters")
sport = st.sidebar.selectbox("League", ["NBA", "NFL"])
market = st.sidebar.selectbox("Single-Market Focus", list(MARKETS_CONFIG[sport].keys()))
family = MARKETS_CONFIG[sport][market]

min_edge = st.sidebar.slider("Minimum +EV Edge (%)", 1.0, 12.0, 3.0, step=0.5) / 100.0
kelly_frac = st.sidebar.slider("Fractional Kelly Multiplier", 0.10, 0.50, 0.25, step=0.05)
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=100, max_value=1000000, value=5000, step=250)

# Run Master All-Star Scanner automatically for top KPIs
master_bets_df, scan_date = run_all_star_scanner(sport, min_edge=min_edge, kelly_frac=kelly_frac)

# Top KPI Strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("Active League", f"{sport} ({len(MARKETS_CONFIG[sport])} Markets)")
k2.metric("Total +EV Edges Today", len(master_bets_df))
top_edge_val = f"+{master_bets_df['edge'].max():.1%}" if not master_bets_df.empty else "0.0%"
k3.metric("Highest Market Edge", top_edge_val)
total_rec_stake = master_bets_df["kelly_fraction"].sum() * bankroll if not master_bets_df.empty else 0.0
k4.metric("Total Recommended Exposure", f"${total_rec_stake:,.0f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. All-Star Best Bets Board",
    "2. Single-Market Deep Dive",
    "3. Parlay +EV Builder",
    "4. Monte Carlo Injury Sim",
    "5. Walk-Forward Backtest",
    "6. Data Sync & Bet Tracker",
])

# ==============================================================================
# TAB 1: MULTI-MARKET ALL-STAR SCANNER & SLIP EXPORTER
# ==============================================================================
with tab1:
    st.subheader(f"{sport} All-Star Multi-Market Best Bets ({scan_date or 'Today'})")
    if not master_bets_df.empty:
        # Spotlight Top 3 Plays
        st.markdown("#### Top 3 Highest-Edge Plays Across All Markets")
        top3 = master_bets_df.head(3)
        cols = st.columns(len(top3))
        for idx, (_, row) in enumerate(top3.iterrows()):
            dollar_stake = row["kelly_fraction"] * bankroll
            odds_str = f"+{row['book_odds']}" if row["book_odds"] > 0 else str(row["book_odds"])
            with cols[idx]:
                st.markdown(f"""
                <div class="pick-card">
                    <div class="pick-header">
                        <span class="pick-player">{row['player']} ({row['team']})</span>
                        <span class="edge-badge">+{row['edge']*100:.1f}% EDGE</span>
                    </div>
                    <div class="pick-line">{row['side']} {row['line']} {row['market']} ({odds_str})</div>
                    <div class="pick-meta">
                        Model Proj: <b>{row['proj_mean']}</b> | Model Win%: <b>{row['model_prob']*100:.1f}%</b> vs Fair: {row['fair_market_prob']*100:.1f}%<br>
                        Kelly Stake: <b>{row['kelly_fraction']*100:.2f}% (${dollar_stake:,.0f})</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Complete Ranked +EV Board")
        display_df = master_bets_df.copy()
        display_df["rec_bet_$"] = (display_df["kelly_fraction"] * bankroll).round(0)
        st.dataframe(
            display_df,
            column_config={
                "edge": st.column_config.ProgressColumn("Model Edge", format="%.1f%%", min_value=0, max_value=0.20),
                "model_prob": st.column_config.NumberColumn("Model Win %", format="%.1f%%"),
                "fair_market_prob": st.column_config.NumberColumn("Fair Odds %", format="%.1f%%"),
                "kelly_fraction": st.column_config.NumberColumn("Kelly %", format="%.2f%%"),
                "rec_bet_$": st.column_config.NumberColumn("Rec Wager ($)", format="$%d"),
            },
            use_container_width=True,
        )

        # Action Bar: Log Picks + Slip Exporter + CSV Download
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Log All Top Plays to Bet Tracker", type="primary"):
                added = log_selected_bets(master_bets_df, sport=sport, game_date=scan_date)
                st.success(f"Saved {added} new bet(s) into your Bet Tracker!")
        with b2:
            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Board as CSV", data=csv_data, file_name=f"{sport}_top_props_{scan_date}.csv", mime="text/csv")
        with b3:
            with st.expander("Copy-Paste Bet Slip"):
                slip_lines = [f"APEX PROP {sport} CARD ({scan_date})"]
                for _, r in master_bets_df.head(8).iterrows():
                    o_str = f"+{r['book_odds']}" if r["book_odds"] > 0 else str(r["book_odds"])
                    slip_lines.append(f"• {r['player']} {r['side']} {r['line']} {r['market']} ({o_str}) | Edge: +{r['edge']*100:.1f}% | ${r['kelly_fraction']*bankroll:,.0f}")
                st.code("
".join(slip_lines), language="text")
    else:
        st.warning("No +EV plays found across markets yet. Open Tab 6 to click 'Load Instant Demo Data' or sync live odds.")


# ==============================================================================
# TAB 2: SINGLE-MARKET DEEP DIVE & PROBABILITY DISTRIBUTION CURVE
# ==============================================================================
raw_df = load_logs(sport, market)

with tab2:
    st.subheader(f"{sport} {market} — Distributional Probability Explorer")
    if len(raw_df) > 40:
        feat_df = engineer_features(raw_df)
        latest_date = feat_df["game_date"].max()
        train = feat_df[(feat_df["game_date"] < latest_date) & (feat_df["actual_stat"].notna())]
        today = feat_df[feat_df["game_date"] == latest_date].reset_index(drop=True)

        if len(train) >= 30 and not today.empty and today["book_line"].notna().any():
            engine = DistributionalPropEngine(family=family)
            engine.fit(train[FEATURE_COLS], train["actual_stat"])
            p_over, p_under, mu = engine.predict_probabilities(today[FEATURE_COLS], today["book_line"].fillna(0).values)
            _, var = engine.predict_params(today[FEATURE_COLS])

            today["Model_Proj"] = np.round(mu, 2)
            today["Prob_Over"] = np.round(p_over, 4)
            today["Prob_Under"] = np.round(p_under, 4)

            selected_player = st.selectbox("Select Player to Visualize Outcome Curve", today["player_name"].tolist())
            p_idx = today[today["player_name"] == selected_player].index[0]
            p_row = today.iloc[p_idx]
            p_mu = mu[p_idx]
            p_var = var[p_idx]
            line_val = float(p_row["book_line"])

            # Plot exact distribution curve
            x_max = max(int(p_mu * 2.5), int(line_val * 2.0), 10)
            if family in ["negative_binomial", "poisson_td"]:
                x_vals = np.arange(0, x_max + 1)
                if family == "poisson_td":
                    y_vals = poisson.pmf(x_vals, p_mu)
                else:
                    r = (p_mu ** 2) / max(p_var - p_mu, 1e-3)
                    p = r / (r + p_mu)
                    y_vals = nbinom.pmf(x_vals, r, p)
                colors = ["#ef4444" if x < line_val else "#10b981" for x in x_vals]
                fig = go.Figure(data=[go.Bar(x=x_vals, y=y_vals, marker_color=colors)])
            else:
                x_vals = np.linspace(0.1, x_max, 250)
                alpha = (p_mu ** 2) / p_var
                theta = p_var / p_mu
                y_vals = gamma.pdf(x_vals, a=alpha, scale=theta)
                fig = go.Figure()
                under_mask = x_vals <= line_val
                over_mask = x_vals >= line_val
                fig.add_trace(go.Scatter(x=x_vals[under_mask], y=y_vals[under_mask], fill="tozeroy", name="Under Zone", line_color="#ef4444"))
                fig.add_trace(go.Scatter(x=x_vals[over_mask], y=y_vals[over_mask], fill="tozeroy", name="Over Zone", line_color="#10b981"))

            fig.add_vline(x=line_val, line_width=3, line_dash="dash", line_color="#38bdf8", annotation_text=f"Book Line: {line_val}")
            fig.update_layout(
                title=f"{selected_player} — {market} Outcome Distribution (Over: {p_over[p_idx]*100:.1f}% | Under: {p_under[p_idx]*100:.1f}%)",
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                xaxis_title=market,
                yaxis_title="Probability Density",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(today[["player_name", "team", "book_line", "Model_Proj", "Prob_Over", "Prob_Under", "over_odds", "under_odds"]], use_container_width=True)
    else:
        st.info("Load data in Tab 6 first to inspect individual player probability distributions.")


# ==============================================================================
# TAB 3: PARLAY & CORRELATION EV BUILDER
# ==============================================================================
with tab3:
    st.subheader("Multi-Leg Parlay +EV Calculator")
    if not master_bets_df.empty and len(master_bets_df) >= 2:
        master_bets_df["leg_label"] = (
            master_bets_df["player"] + " — " + master_bets_df["side"] + " " +
            master_bets_df["line"].astype(str) + " " + master_bets_df["market"] +
            " (" + master_bets_df["book_odds"].astype(str) + ")"
        )
        selected_legs = st.multiselect(
            "Select 2 to 6 +EV Legs to Combine:",
            master_bets_df["leg_label"].tolist(),
            default=master_bets_df["leg_label"].tolist()[:2],
        )

        if len(selected_legs) >= 2:
            sub = master_bets_df[master_bets_df["leg_label"].isin(selected_legs)]
            combined_model_prob = float(np.prod(sub["model_prob"].values))

            # Calculate standard sportsbook parlay decimal odds
            dec_odds_list = [(o / 100) + 1 if o > 0 else (100 / abs(o)) + 1 for o in sub["book_odds"].values]
            parlay_dec_odds = float(np.prod(dec_odds_list))
            parlay_american = int(round((parlay_dec_odds - 1) * 100)) if parlay_dec_odds >= 2.0 else int(round(-100 / (parlay_dec_odds - 1)))
            book_implied_prob = 1.0 / parlay_dec_odds
            parlay_ev = (combined_model_prob * parlay_dec_odds) - 1.0

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Combined Parlay Odds", f"+{parlay_american}" if parlay_american > 0 else str(parlay_american))
            p2.metric("True Model Hit Prob", f"{combined_model_prob:.1%}")
            p3.metric("Sportsbook Implied Prob", f"{book_implied_prob:.1%}")
            p4.metric("Expected Value (ROI)", f"{parlay_ev:+.1%}")
    else:
        st.info("Need at least 2 +EV plays on today's All-Star board to build a parlay.")


# ==============================================================================
# TAB 4: MONTE CARLO INJURY SIMULATOR
# ==============================================================================
with tab4:
    st.subheader("Interactive Injury & Usage Redistribution Simulator (10,000 Sims)")
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
        st.info("Load data in Tab 6 first to run injury redistribution simulations.")


# ==============================================================================
# TAB 5: WALK-FORWARD BACKTEST & CLV
# ==============================================================================
with tab5:
    st.subheader(f"Walk-Forward Backtest & Closing Line Value ({sport} {market})")
    if len(raw_df) > 60:
        if st.button("Execute Walk-Forward Validation"):
            feat_df = engineer_features(raw_df)
            bt_df = run_walk_forward_backtest(feat_df, family=family, min_edge=min_edge, kelly_frac=kelly_frac)
            if not bt_df.empty:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Simulated Bets", len(bt_df))
                c2.metric("Win Rate", f"{bt_df['won'].mean():.1%}")
                c3.metric("Net Profit (Units)", f"{bt_df['profit'].sum():+.2f}u")
                c4.metric("Avg Closing Line Value", f"{bt_df['clv'].mean():+.2%}")

                bt_df["cum_profit"] = bt_df["profit"].cumsum()
                fig_bt = px.area(bt_df, x="date", y="cum_profit", title="Cumulative Bankroll Growth (Units)")
                fig_bt.update_layout(template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#111827")
                st.plotly_chart(fig_bt, use_container_width=True)
                st.dataframe(bt_df, use_container_width=True)
            else:
                st.info("Need more historical days with paired sportsbook lines to run a multi-day backtest.")
    else:
        st.info("Load data in Tab 6 first to run historical backtests.")


# ==============================================================================
# TAB 6: DATA SYNC & BET TRACKER
# ==============================================================================
with tab6:
    st.subheader("Central Data Pipeline & Real-World Bet Auto-Grader")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 1. Instant Demo Seeder")
        if st.button("Load NBA + NFL Demo Data"):
            cnt = seed_demo_data()
            st.success(f"Loaded {cnt} NBA & NFL prop records! Switch to Tab 1 now.")

    with c2:
        st.markdown("#### 2. Sync Official Box Scores")
        if st.button("Pull Live NBA Season Logs"):
            with st.spinner("Syncing NBA logs..."):
                cnt = sync_nba_logs("2025-26")
                st.success(f"Added {cnt:,} NBA rows.")
        if st.button("Pull Live NFL Season Logs"):
            with st.spinner("Syncing NFL logs..."):
                cnt = sync_nfl_logs()
                st.success(f"Added {cnt:,} NFL rows.")

    with c3:
        st.markdown("#### 3. Fetch Live Sportsbook Odds")
        api_key = st.text_input("The-Odds-API Key", type="password")
        if st.button(f"Sync All {sport} Markets Today"):
            if not api_key:
                st.error("Enter your free Odds API key first.")
            else:
                total_lines = 0
                with st.spinner(f"Pulling all {sport} markets + Pinnacle sharp lines..."):
                    for mkt_name in MARKETS_CONFIG[sport].keys():
                        res_df = fetch_live_prop_odds(api_key, sport, mkt_name, include_sharp=True)
                        total_lines += len(res_df)
                st.success(f"Synced {total_lines} live prop lines across all {sport} markets!")

    st.divider()
    st.subheader("Real-World Bet Tracker & Auto-Grader")
    if st.button("Auto-Grade Pending Bets"):
        graded = auto_grade_pending_bets()
        st.success(f"Graded {graded} completed bet(s).")

    history_df = load_all_bets()
    if not history_df.empty:
        settled = history_df[history_df["status"].isin(["WON", "LOST", "PUSH"])]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Logged Bets", len(history_df))
        m2.metric("Pending Bets", len(history_df[history_df["status"] == "PENDING"]))
        if not settled.empty:
            m3.metric("Settled Win Rate", f"{(settled['status'] == 'WON').mean():.1%}")
            m4.metric("Net Profit", f"{settled['profit_units'].sum():+.2f}u")
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No bets logged yet. Click 'Log All Top Plays to Bet Tracker' in Tab 1 to start tracking.")
