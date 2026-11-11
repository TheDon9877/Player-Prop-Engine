import time
import requests
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
            stake_dollars REAL DEFAULT 0.0,
            status TEXT DEFAULT 'PENDING',
            actual_stat REAL,
            profit_units REAL DEFAULT 0.0
        )
    """)
    conn.commit()
    return True


def save_logs(df: pd.DataFrame):
    cols = [
        "game_date", "sport", "player_name", "team", "opponent",
        "minutes_or_snaps", "usage_or_share", "opp_def_rating",
        "vegas_total", "spread", "days_rest", "market", "actual_stat",
        "book_line", "over_odds", "under_odds", "closing_over_odds"
    ]
    df = df[cols].copy()
    df["game_date"] = df["game_date"].astype(str)
    conn = get_conn()
    df.to_sql("player_game_logs", conn, if_exists="append", index=False)
    conn.commit()


def load_logs(sport: str, market: str) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM player_game_logs WHERE sport = ? AND market = ? ORDER BY game_date",
        conn,
        params=(sport, market),
    )
    return df


def log_selected_bets(bets_df: pd.DataFrame, sport: str, game_date: str) -> int:
    conn = get_conn()
    inserted = 0
    for _, row in bets_df.iterrows():
        bet_id = f"{game_date}_{sport}_{row['player']}_{row['market']}_{row['side']}"
        cur = conn.execute("SELECT 1 FROM logged_bets_v2 WHERE bet_id = ?", (bet_id,))
        if not cur.fetchone():
            conn.execute(
                """
                INSERT INTO logged_bets_v2 
                (bet_id, game_date, sport, player_name, market, line, side, 
                 model_prob, fair_prob, book_odds, edge, kelly_pct, stake_dollars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet_id, str(game_date), sport, row["player"], row["market"],
                    float(row["line"]), row["side"], float(row["model_prob"]),
                    float(row["fair_market_prob"]), int(row["book_odds"]),
                    float(row["edge"]), float(row["kelly_fraction"]),
                    float(row.get("stake_dollars", 0.0)),
                ),
            )
            inserted += 1
    conn.commit()
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

            conn.execute(
                "UPDATE logged_bets_v2 SET status = ?, actual_stat = ?, profit_units = ? WHERE bet_id = ?",
                (status, actual, profit, bet["bet_id"]),
            )
            graded_count += 1

    conn.commit()
    return graded_count


def load_all_bets() -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query("SELECT * FROM logged_bets_v2 ORDER BY game_date DESC, logged_at DESC", conn)


@dataclass
class BetOpportunity:
    player: str
    team: str
    market: str
    line: float
    side: str
    proj_mean: float
    proj_std: float
    model_prob: float
    fair_market_prob: float
    book_odds: int
    edge: float
    kelly_fraction: float
    stake_dollars: float


class DistributionalPropEngine:
    def __init__(self, family: PropFamily, hgb_params: Dict = None):
        self.family = family
        self.hgb_params = hgb_params or {
            "learning_rate": 0.04,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "max_iter": 300,
            "random_state": 42,
        }
        self.mean_model = None
        self.var_model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        loss = "poisson" if self.family in ["negative_binomial", "poisson_td"] else "gamma"
        y_clean = np.clip(y_train.values, 1e-3 if loss == "gamma" else 0.0, None)

        self.mean_model = HistGradientBoostingRegressor(loss=loss, **self.hgb_params)
        self.mean_model.fit(X_train, y_clean)

        if self.family in ["negative_binomial", "gamma"]:
            train_preds = np.clip(self.mean_model.predict(X_train), 1e-3, None)
            sq_err_train = np.clip((y_train.values - train_preds) ** 2, 1e-3, None)

            self.var_model = HistGradientBoostingRegressor(loss="gamma", **self.hgb_params)
            self.var_model.fit(X_train, sq_err_train)

    def predict_params(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.clip(self.mean_model.predict(X), 1e-3, None)
        if self.family == "poisson_td":
            return mu, mu
        var = np.clip(self.var_model.predict(X), mu + 1e-3, None)
        return mu, var

    def predict_probabilities(self, X: pd.DataFrame, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mu, var = self.predict_params(X)

        if self.family == "poisson_td":
            prob_over = 1.0 - poisson.pmf(0, mu)
            return prob_over, 1.0 - prob_over, mu, np.sqrt(var)

        if self.family == "negative_binomial":
            r = (mu ** 2) / np.maximum(var - mu, 1e-3)
            p = r / (r + mu)
            prob_under = nbinom.cdf(np.floor(lines), r, p)
            return 1.0 - prob_under, prob_under, mu, np.sqrt(var)

        alpha = (mu ** 2) / var
        theta = var / mu
        prob_under = gamma.cdf(lines, a=alpha, scale=theta)
        return 1.0 - prob_under, prob_under, mu, np.sqrt(var)


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def implied_to_american(prob: float) -> int:
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
    mu: np.ndarray,
    std: np.ndarray,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    bankroll: float = 1000.0,
) -> list[BetOpportunity]:
    opportunities = []
    for i, row in df_slate.iterrows():
        if pd.isna(row["book_line"]):
            continue
        fair_over, fair_under = power_devig(int(row["over_odds"]), int(row["under_odds"]))

        edge_over = prob_over[i] - fair_over
        if edge_over >= min_edge:
            k_frac = calculate_kelly(prob_over[i], int(row["over_odds"]), kelly_fraction)
            opportunities.append(
                BetOpportunity(
                    player=row["player_name"],
                    team=str(row.get("team", "")),
                    market=row["market"],
                    line=float(row["book_line"]),
                    side="OVER",
                    proj_mean=round(float(mu[i]), 2),
                    proj_std=round(float(std[i]), 2),
                    model_prob=round(float(prob_over[i]), 4),
                    fair_market_prob=round(float(fair_over), 4),
                    book_odds=int(row["over_odds"]),
                    edge=round(float(edge_over), 4),
                    kelly_fraction=round(k_frac, 4),
                    stake_dollars=round(k_frac * bankroll, 2),
                )
            )

        edge_under = prob_under[i] - fair_under
        if edge_under >= min_edge:
            k_frac = calculate_kelly(prob_under[i], int(row["under_odds"]), kelly_fraction)
            opportunities.append(
                BetOpportunity(
                    player=row["player_name"],
                    team=str(row.get("team", "")),
                    market=row["market"],
                    line=float(row["book_line"]),
                    side="UNDER",
                    proj_mean=round(float(mu[i]), 2),
                    proj_std=round(float(std[i]), 2),
                    model_prob=round(float(prob_under[i]), 4),
                    fair_market_prob=round(float(fair_under), 4),
                    book_odds=int(row["under_odds"]),
                    edge=round(float(edge_under), 4),
                    kelly_fraction=round(k_frac, 4),
                    stake_dollars=round(k_frac * bankroll, 2),
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


def scan_all_markets(sport: str, min_edge: float, kelly_frac: float, bankroll: float) -> Tuple[pd.DataFrame, str]:
    all_opportunities = []
    latest_date_found = ""

    for mkt_name, family in MARKETS_CONFIG[sport].items():
        raw_df = load_logs(sport, mkt_name)
        if len(raw_df) < 40:
            continue

        feat_df = engineer_features(raw_df)
        latest_date = feat_df["game_date"].max()
        if latest_date > latest_date_found:
            latest_date_found = latest_date

        train = feat_df[(feat_df["game_date"] < latest_date) & (feat_df["actual_stat"].notna())]
        today = feat_df[feat_df["game_date"] == latest_date].reset_index(drop=True)

        if len(train) >= 30 and not today.empty and today["book_line"].notna().any():
            engine = DistributionalPropEngine(family=family)
            engine.fit(train[FEATURE_COLS], train["actual_stat"])
            p_over, p_under, mu, std = engine.predict_probabilities(today[FEATURE_COLS], today["book_line"].fillna(0).values)
            bets = evaluate_slate(today, p_over, p_under, mu, std, min_edge=min_edge, kelly_fraction=kelly_frac, bankroll=bankroll)
            all_opportunities.extend([vars(b) for b in bets])

    if not all_opportunities:
        return pd.DataFrame(), latest_date_found

    master_df = pd.DataFrame(all_opportunities).sort_values("edge", ascending=False).reset_index(drop=True)
    return master_df, latest_date_found


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
        test_df = df[df["game_date"] == test_date].copy()

        if len(train_df) < 40 or len(test_df) == 0:
            continue

        engine = DistributionalPropEngine(family=family)
        engine.fit(train_df[FEATURE_COLS], train_df["actual_stat"])
        prob_over, prob_under, _, _ = engine.predict_probabilities(test_df[FEATURE_COLS], test_df["book_line"].values)

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
        ("Anthony Davis", "LAL", "DEN", 35.5, 0.29, 24.5, 12.0, 3.5, 0.5),
        ("Jamal Murray", "DEN", "LAL", 34.0, 0.26, 21.5, 4.0, 6.5, 2.5),
    ]
    base_date = datetime.now().date() - timedelta(days=30)
    rows = []

    for day in range(31):
        g_date = (base_date + timedelta(days=day)).isoformat()
        is_today = (day == 30)
        for name, team, opp, mins, usg, pts, reb, ast, threes in players:
            for mkt, mean_val in [("Points", pts), ("Rebounds", reb), ("Assists", ast), ("3PM", threes)]:
                actual = np.nan if is_today else float(max(0, rng.poisson(mean_val)))
                line_offset = rng.choice([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]) if is_today else 0.0
                rows.append({
                    "game_date": g_date,
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
                    "book_line": max(0.5, round(mean_val + line_offset, 1)),
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
    today = datetime.now().date().isoformat()

    if isinstance(events, list):
        for ev in events:
            event_id = ev["id"]
            odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
            res = requests.get(
                odds_url,
                params={"apiKey": api_key, "regions": "us,eu", "markets": market_key, "oddsFormat": "american"},
                timeout=15,
            ).json()

            bookmakers = res.get("bookmakers", [])
            if not bookmakers:
                continue

            us_book = next((b for b in bookmakers if b["key"] in ["draftkings", "fanduel", "betmgm"]), bookmakers[0])
            sharp_book = next((b for b in bookmakers if b["key"] == "pinnacle"), us_book)

            sharp_prices = {}
            for mkt in sharp_book.get("markets", []):
                for outcome in mkt.get("outcomes", []):
                    if outcome.get("name") in ["Over", "Yes"]:
                        sharp_prices[outcome.get("description")] = outcome.get("price", -110)

            for mkt in us_book.get("markets", []):
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
                        "closing_over_odds": int(sharp_prices.get(player, vals["over"])),
                    })

    live_df = pd.DataFrame(rows)
    if not live_df.empty:
        save_logs(live_df)
    return live_df


# ==============================================================================
# UI STYLING & DASHBOARD
# ==============================================================================
st.set_page_config(page_title="APEX PROP | Quant Betting Terminal", layout="wide")
init_db()

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131b2e, #0f1626);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetricValue"] { color: #38bdf8; font-weight: 700; }
    .pick-card {
        background: linear-gradient(135deg, #162033 0%, #0f172a 100%);
        border: 1px solid #26354d;
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }
    .pick-title { font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
    .pick-sub { font-size: 0.9rem; color: #94a3b8; margin-bottom: 10px; }
    .edge-badge {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34d399;
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 10px 0 20px 0;">
    <span style="background:#1e293b; color:#38bdf8; padding:4px 12px; border-radius:999px; font-size:0.8rem; font-weight:700; letter-spacing:1px;">
        APEX PROP ENGINE v3.0
    </span>
    <h1 style="margin-top:8px; margin-bottom:4px; font-size:2.2rem; color:#f8fafc;">
        Quantitative Sports Prop Terminal
    </h1>
    <p style="color:#94a3b8; font-size:1rem; margin:0;">
        Multi-Market Distributional Machine Learning • Sharp Power Devigging • Monte Carlo Simulation
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Engine Controls")
    sport = st.selectbox("League / Sport", ["NBA", "NFL"])
    market = st.selectbox("Deep-Dive Market (Tabs 2 & 5)", list(MARKETS_CONFIG[sport].keys()))
    family = MARKETS_CONFIG[sport][market]

    st.markdown("---")
    st.markdown("### Risk & Bankroll Sizing")
    bankroll = st.number_input("Total Bankroll ($)", min_value=100, max_value=1000000, value=1000, step=100)
    min_edge = st.slider("Minimum +EV Edge (%)", 1.0, 12.0, 3.0, 0.5) / 100.0
    kelly_frac = st.slider("Fractional Kelly Multiplier", 0.10, 0.50, 0.25, 0.05)

    st.markdown("---")
    st.caption(f"Deep-Dive Family: `{family}`")

master_df, scan_date = scan_all_markets(sport, min_edge, kelly_frac, bankroll)
raw_df = load_logs(sport, market)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active League", f"{sport} (All Markets)")
kpi2.metric("Total +EV Edges", len(master_df))
kpi3.metric("Top Edge Found", f"+{master_df['edge'].max():.1%}" if not master_df.empty else "0.0%")
kpi4.metric("Total Kelly Allocation", f"${master_df['stake_dollars'].sum():,.2f}" if not master_df.empty else "$0.00")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "All-Star +EV Board",
    "Distribution Curves",
    "Parlay +EV Builder",
    "Injury Simulator",
    "Backtest & CLV",
    "Data Sync & Auto-Grader",
])

with tab1:
    st.subheader(f"Master Multi-Market +EV Board ({sport})")
    if not master_df.empty:
        st.markdown("#### Highest-Conviction Plays Across All Markets")
        top_cols = st.columns(min(3, len(master_df)))
        for idx, (_, row) in enumerate(master_df.head(3).iterrows()):
            odds_str = f"+{row['book_odds']}" if row['book_odds'] > 0 else f"{row['book_odds']}"
            with top_cols[idx]:
                st.markdown(f"""
                <div class="pick-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="pick-title">{row['player']}</span>
                        <span class="edge-badge">+{row['edge']:.1%} EDGE</span>
                    </div>
                    <div class="pick-sub">{row['market']} • <b>{row['side']} {row['line']}</b> ({odds_str})</div>
                    <div style="font-size:0.85rem; color:#cbd5e1;">
                        Proj Mean: <b>{row['proj_mean']}</b> | Model Win: <b>{row['model_prob']:.1%}</b> vs Fair: <b>{row['fair_market_prob']:.1%}</b><br>
                        Recommended Bet: <b style="color:#38bdf8;">${row['stake_dollars']:.2f}</b> ({row['kelly_fraction']:.2%} Kelly)
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Complete Ranked Opportunity Board")
        disp_df = master_df.copy()
        disp_df["book_odds"] = disp_df["book_odds"].apply(lambda x: f"+{x}" if x > 0 else str(x))

        st.dataframe(
            disp_df[["player", "team", "market", "side", "line", "book_odds", "proj_mean", "model_prob", "fair_market_prob", "edge", "stake_dollars"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "player": "Player",
                "team": "Team",
                "market": "Market",
                "side": "Pick",
                "line": st.column_config.NumberColumn("Line", format="%.1f"),
                "book_odds": "Odds",
                "proj_mean": st.column_config.NumberColumn("Model Proj", format="%.2f"),
                "model_prob": st.column_config.NumberColumn("Model Prob", format="%.1%"),
                "fair_market_prob": st.column_config.NumberColumn("Fair Prob", format="%.1%"),
                "edge": st.column_config.ProgressColumn("+EV Edge", format="%.1%", min_value=0, max_value=0.20),
                "stake_dollars": st.column_config.NumberColumn("Kelly Bet ($)", format="$%.2f"),
            },
        )

        act1, act2, act3 = st.columns(3)
        with act1:
            if st.button("Log All +EV Picks to Tracker", type="primary", use_container_width=True):
                added = log_selected_bets(master_df, sport=sport, game_date=scan_date)
                st.success(f"Saved {added} new bet(s) into your central tracker.")
        with act2:
            csv_data = disp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Board as CSV",
                data=csv_data,
                file_name=f"apex_props_{sport}_{scan_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with act3:
            slip_lines = [
                f"{r['player']} - {r['market']} {r['side']} {r['line']} ({r['book_odds']}) | Edge: +{r['edge']:.1%} | Bet: ${r['stake_dollars']:.2f}"
                for _, r in disp_df.head(10).iterrows()
            ]
            with st.expander("Copy Text Bet Slip"):
                st.code("\n".join(slip_lines), language="text")
    else:
        st.info("No +EV edges found right now. Go to Tab 6 ('Data Sync & Auto-Grader') and click 'Load Instant Demo Data' or fetch live sportsbook odds.")

with tab2:
    st.subheader(f"Outcome Probability Distribution Explorer ({sport} • {market})")
    if len(raw_df) > 40:
        feat_df = engineer_features(raw_df)
        latest_date = feat_df["game_date"].max()
        train = feat_df[(feat_df["game_date"] < latest_date) & (feat_df["actual_stat"].notna())]
        today = feat_df[feat_df["game_date"] == latest_date].reset_index(drop=True)

        if len(train) >= 30 and not today.empty:
            engine = DistributionalPropEngine(family=family)
            engine.fit(train[FEATURE_COLS], train["actual_stat"])
            p_over, p_under, mu, std = engine.predict_probabilities(today[FEATURE_COLS], today["book_line"].fillna(0).values)

            selected_player = st.selectbox("Select Player to Inspect Bell Curve", today["player_name"].tolist())
            p_idx = today[today["player_name"] == selected_player].index[0]

            p_mu = float(mu[p_idx])
            p_sd = float(std[p_idx])
            p_line = float(today.loc[p_idx, "book_line"]) if pd.notna(today.loc[p_idx, "book_line"]) else round(p_mu, 1)

            x_vals = np.linspace(max(0, p_mu - 3.5 * p_sd), p_mu + 3.5 * p_sd, 200)
            y_vals = (1 / (p_sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - p_mu) / p_sd) ** 2)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals[x_vals <= p_line], y=y_vals[x_vals <= p_line],
                fill="tozeroy", name=f"Under {p_line} ({p_under[p_idx]:.1%})",
                line=dict(color="#ef4444"), fillcolor="rgba(239, 68, 68, 0.25)"
            ))
            fig.add_trace(go.Scatter(
                x=x_vals[x_vals >= p_line], y=y_vals[x_vals >= p_line],
                fill="tozeroy", name=f"Over {p_line} ({p_over[p_idx]:.1%})",
                line=dict(color="#10b981"), fillcolor="rgba(16, 185, 129, 0.3)"
            ))
            fig.add_vline(x=p_line, line_dash="dash", line_color="#38bdf8", annotation_text=f"Book Line: {p_line}")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                title=f"{selected_player} — Projected {market} Distribution (Mean: {p_mu:.2f}, Std Dev: {p_sd:.2f})",
                xaxis_title=market,
                yaxis_title="Probability Density",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load data in Tab 6 first to inspect player probability curves.")

with tab3:
    st.subheader("Multi-Leg Parlay +EV Calculator")
    if not master_df.empty and len(master_df) >= 2:
        master_df["leg_label"] = master_df.apply(
            lambda r: f"{r['player']} | {r['market']} {r['side']} {r['line']} ({r['book_odds']})", axis=1
        )
        selected_legs = st.multiselect(
            "Select 2 to 6 +EV Legs to Combine",
            master_df["leg_label"].tolist(),
            default=master_df["leg_label"].tolist()[:min(3, len(master_df))],
        )

        if len(selected_legs) >= 2:
            sub = master_df[master_df["leg_label"].isin(selected_legs)]
            combined_model_prob = float(np.prod(sub["model_prob"].values))
            dec_odds_list = [(o / 100) + 1 if o > 0 else (100 / abs(o)) + 1 for o in sub["book_odds"].values]
            combined_dec_odds = float(np.prod(dec_odds_list))
            combined_american = implied_to_american(1.0 / combined_dec_odds)
            parlay_ev = (combined_model_prob * combined_dec_odds) - 1.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Parlay Book Odds", f"+{combined_american}" if combined_american > 0 else str(combined_american))
            c2.metric("True Model Hit Prob", f"{combined_model_prob:.1%}")
            c3.metric("Book Implied Prob", f"{(1.0 / combined_dec_odds):.1%}")
            c4.metric("Parlay Expected ROI", f"{parlay_ev:+.1%}")
    else:
        st.info("Generate at least two +EV plays in Tab 1 to build and price parlays.")

with tab4:
    st.subheader("10,000-Iteration Monte Carlo Injury Simulator")
    if len(raw_df) > 0:
        feat_df = engineer_features(raw_df)
        teams = sorted(feat_df["team"].dropna().unique())
        if teams:
            c1, c2 = st.columns([1, 2])
            with c1:
                selected_team = st.selectbox("Select Team", teams)
                team_df = feat_df[feat_df["team"] == selected_team].sort_values("game_date").drop_duplicates("player_name", keep="last")
                out_players = st.multiselect("Mark Players OUT", team_df["player_name"].tolist())
                run_sim = st.button("Run 10,000 Simulations", type="primary", use_container_width=True)
            with c2:
                if run_sim:
                    sim_results = monte_carlo_injury_sim(team_df, out_players)
                    st.dataframe(
                        sim_results[["player_name", "roll_5_stat", "mc_adj_mean", "mc_adj_std", "book_line", "mc_prob_over"]],
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info("Load data in Tab 6 first to simulate injury scenarios.")

with tab5:
    st.subheader(f"Walk-Forward Historical Backtest & CLV ({sport} • {market})")
    if len(raw_df) > 60:
        if st.button("Run Walk-Forward Backtest", type="primary"):
            feat_df = engineer_features(raw_df)
            bt_df = run_walk_forward_backtest(feat_df, family=family, min_edge=min_edge, kelly_frac=kelly_frac)
            if not bt_df.empty:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Historical Bets", len(bt_df))
                c2.metric("Win Rate", f"{bt_df['won'].mean():.1%}")
                c3.metric("Net Profit", f"{bt_df['profit'].sum():+.2f}u")
                c4.metric("Avg Closing Line Value", f"{bt_df['clv'].mean():+.2%}")

                bt_df["cum_profit"] = bt_df["profit"].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bt_df["date"], y=bt_df["cum_profit"],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color="#38bdf8", width=3),
                    fillcolor="rgba(56, 189, 248, 0.15)",
                ))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#111827", title="Cumulative Bankroll Growth (Units)", height=340)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(bt_df, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough multi-day historical lines to backtest yet.")
    else:
        st.info("Load data in Tab 6 first to run backtests.")

with tab6:
    st.subheader("Data Connectors & Real-World Bet Auto-Grader")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 1. Instant Sandbox")
        if st.button("Load Instant Demo Data", use_container_width=True):
            count = seed_demo_data()
            st.success(f"Loaded {count} sample rows across all 4 NBA markets! Switch to Tab 1.")

    with c2:
        st.markdown("#### 2. Sync Official Box Scores")
        if st.button("Pull NBA Historical Logs", use_container_width=True):
            with st.spinner("Downloading NBA stats..."):
                count = sync_nba_logs("2025-26")
                st.success(f"Synced {count:,} NBA rows.")
        if st.button("Pull NFL Historical Logs", use_container_width=True):
            with st.spinner("Downloading NFL stats..."):
                count = sync_nfl_logs()
                st.success(f"Synced {count:,} NFL rows.")

    with c3:
        st.markdown("#### 3. Pull Live Sportsbook Odds")
        api_key = st.text_input("The-Odds-API Key", type="password")
        if st.button(f"Sync All {sport} Markets Live", use_container_width=True):
            if not api_key:
                st.error("Enter your Odds API key first.")
            else:
                total_lines = 0
                with st.spinner(f"Scanning all {sport} prop markets..."):
                    for mkt_name in MARKETS_CONFIG[sport].keys():
                        df_live = fetch_live_prop_odds(api_key, sport, mkt_name)
                        total_lines += len(df_live)
                st.success(f"Synced {total_lines} live lines across all {sport} markets!")

    st.markdown("---")
    st.markdown("#### Live Bet Performance & Auto-Grader")
    if st.button("Auto-Grade Pending Bets Now"):
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
        st.dataframe(history_df, use_container_width=True, hide_index=True)
