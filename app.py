import time
import sqlite3
import requests
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

# ==============================================================================
# 1. PAGE CONFIG & PRO ANALYTICS TERMINAL STYLING
# ==============================================================================
st.set_page_config(
    page_title="APEX PROP | Quant Sports Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Dark Theme & Typography */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    /* Custom Header Banner */
    .terminal-header {
        background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 26px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .terminal-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }
    .terminal-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    .live-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* Top KPI Metric Cards */
    .kpi-card {
        background-color: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .kpi-accent-green { color: #10B981; }
    .kpi-accent-blue { color: #38BDF8; }

    /* Featured Spotlight Pick Cards */
    .pick-card {
        background: linear-gradient(180deg, #162033 0%, #111827 100%);
        border: 1px solid #2A3950;
        border-top: 3px solid #10B981;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.4);
    }
    .pick-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .pick-player {
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .edge-pill {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .pick-line-box {
        background-color: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 10px 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #38BDF8;
        text-align: center;
    }
    .pick-meta-grid {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: #94A3B8;
        margin-top: 8px;
    }
    .pick-meta-val {
        color: #E2E8F0;
        font-weight: 700;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATABASE LAYER (THREAD-SAFE SQLITE WAL MODE)
# ==============================================================================
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
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con


@st.cache_resource
def init_db():
    con = get_conn()
    con.execute("""
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
    con.execute("""
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
    con.commit()
    con.close()
    return True


def save_logs(df: pd.DataFrame):
    cols = [
        "game_date", "sport", "player_name", "team", "opponent",
        "minutes_or_snaps", "usage_or_share", "opp_def_rating",
        "vegas_total", "spread", "days_rest", "market", "actual_stat",
        "book_line", "over_odds", "under_odds", "closing_over_odds"
    ]
    clean_df = df[cols].copy()
    clean_df["game_date"] = clean_df["game_date"].astype(str)
    con = get_conn()
    clean_df.to_sql("player_game_logs", con, if_exists="append", index=False)
    con.commit()
    con.close()


def load_logs(sport: str, market: str) -> pd.DataFrame:
    con = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM player_game_logs WHERE sport = ? AND market = ? ORDER BY game_date",
        con,
        params=(sport, market),
    )
    con.close()
    return df


def log_selected_bets(bets_df: pd.DataFrame, sport: str, game_date) -> int:
    con = get_conn()
    cur = con.cursor()
    inserted = 0
    g_str = str(game_date)
    for _, row in bets_df.iterrows():
        bet_id = f"{g_str}_{sport}_{row['player']}_{row['market']}_{row['side']}"
        cur.execute("SELECT 1 FROM logged_bets_v2 WHERE bet_id = ?", (bet_id,))
        if not cur.fetchone():
            cur.execute(
                """
                INSERT INTO logged_bets_v2 
                (bet_id, game_date, sport, player_name, market, line, side, 
                 model_prob, fair_prob, book_odds, edge, kelly_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bet_id, g_str, sport, row["player"], row["market"],
                    float(row["line"]), row["side"], float(row["model_prob"]),
                    float(row["fair_market_prob"]), int(row["book_odds"]),
                    float(row["edge"]), float(row["kelly_fraction"]),
                ),
            )
            inserted += 1
    con.commit()
    con.close()
    return inserted


def auto_grade_pending_bets() -> int:
    con = get_conn()
    cur = con.cursor()
    pending = pd.read_sql_query("SELECT * FROM logged_bets_v2 WHERE status = 'PENDING'", con)
    graded_count = 0

    for _, bet in pending.iterrows():
        cur.execute(
            """
            SELECT actual_stat FROM player_game_logs
            WHERE game_date = ? AND sport = ? AND player_name = ? 
              AND market = ? AND actual_stat IS NOT NULL
            LIMIT 1
            """,
            (str(bet["game_date"]), bet["sport"], bet["player_name"], bet["market"]),
        )
        match = cur.fetchone()

        if match and match[0] is not None:
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

    con.commit()
    con.close()
    return graded_count


def load_all_bets() -> pd.DataFrame:
    con = get_conn()
    df = pd.read_sql_query("SELECT * FROM logged_bets_v2 ORDER BY game_date DESC, logged_at DESC", con)
    con.close()
    return df


# ==============================================================================
# 3. DISTRIBUTIONAL ML ENGINE (SCIKIT-LEARN HISTGRADIENTBOOSTING)
# ==============================================================================
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
    proj_mean: float
    proj_std: float


class DistributionalPropEngine:
    def __init__(self, family: PropFamily):
        self.family = family
        self.mean_model = None
        self.var_model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        X_full = pd.concat([X_train, X_val]) if X_val is not None and not X_val.empty else X_train
        y_full = pd.concat([y_train, y_val]) if y_val is not None and not y_val.empty else y_train

        loss_type = "poisson" if self.family in ["negative_binomial", "poisson_td"] else "gamma"
        y_clean = np.clip(y_full.values.astype(float), 1e-3 if loss_type == "gamma" else 0.0, None)

        self.mean_model = HistGradientBoostingRegressor(
            loss=loss_type,
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=25,
            min_samples_leaf=15,
            random_state=42,
        )
        self.mean_model.fit(X_full, y_clean)

        if self.family in ["negative_binomial", "gamma"]:
            preds = np.clip(self.mean_model.predict(X_full), 1e-3, None)
            sq_err = np.clip((y_full.values.astype(float) - preds) ** 2, 1e-3, None)

            self.var_model = HistGradientBoostingRegressor(
                loss="gamma",
                learning_rate=0.04,
                max_iter=200,
                max_leaf_nodes=20,
                min_samples_leaf=15,
                random_state=42,
            )
            self.var_model.fit(X_full, sq_err)

    def predict_params(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.clip(self.mean_model.predict(X), 1e-3, None)
        if self.family == "poisson_td":
            return mu, mu
        var = np.clip(self.var_model.predict(X), mu + 1e-3, None)
        return mu, var

    def predict_probabilities(self, X: pd.DataFrame, lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, var = self.predict_params(X)

        if self.family == "poisson_td":
            prob_over = 1.0 - poisson.pmf(0, mu)
            return prob_over, 1.0 - prob_over

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
    mu: np.ndarray,
    var: np.ndarray,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> list[BetOpportunity]:
    opportunities = []
    for i, row in df_slate.iterrows():
        if pd.isna(row["book_line"]):
            continue
        fair_over, fair_under = power_devig(int(row["over_odds"]), int(row["under_odds"]))
        std_val = round(float(np.sqrt(var[i])), 2)
        mean_val = round(float(mu[i]), 2)

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
                    proj_mean=mean_val,
                    proj_std=std_val,
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
                    proj_mean=mean_val,
                    proj_std=std_val,
                )
            )

    return sorted(opportunities, key=lambda x: x.edge, reverse=True)


# ==============================================================================
# 4. FEATURE ENGINEERING, MONTE CARLO SIM & BACKTESTER
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


# ==============================================================================
# 5. DATA SYNC & DEMO GENERATOR
# ==============================================================================
def seed_demo_data() -> int:
    rng = np.random.default_rng(42)
    players = [
        ("Luka Doncic", "DAL", "BOS", 36.0, 0.36, 32.5, 9.0, 9.5, 3.5),
        ("Kyrie Irving", "DAL", "BOS", 35.0, 0.28, 25.5, 5.0, 5.5, 3.0),
        ("Jayson Tatum", "BOS", "DAL", 36.5, 0.31, 27.5, 8.5, 4.5, 3.0),
        ("Jaylen Brown", "BOS", "DAL", 34.0, 0.27, 23.5, 5.5, 3.5, 2.0),
        ("Nikola Jokic", "DEN", "LAL", 35.0, 0.32, 26.5, 12.5, 9.0, 1.5),
        ("LeBron James", "LAL", "DEN", 35.0, 0.30, 25.5, 7.5, 8.0, 2.0),
        ("Anthony Davis", "LAL", "DEN", 35.5, 0.29, 24.5, 11.5, 3.5, 0.5),
        ("Jamal Murray", "DEN", "LAL", 33.5, 0.26, 21.5, 4.0, 6.5, 2.5),
    ]
    base_date = datetime.now().date() - timedelta(days=30)
    rows = []

    for day in range(31):
        g_date = str(base_date + timedelta(days=day))
        is_today = (day == 30)
        for name, team, opp, mins, usg, pts, reb, ast, threes in players:
            for mkt, mean_val in [("Points", pts), ("Rebounds", reb), ("Assists", ast), ("3PM", threes)]:
                actual = np.nan if is_today else float(max(0, rng.poisson(mean_val)))
                # Add slight realistic edge offset to today's lines
                line_offset = -1.5 if (is_today and name in ["Luka Doncic", "Nikola Jokic", "Jayson Tatum"]) else 0.0
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
                    "book_line": max(0.5, mean_val + line_offset),
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


# ==============================================================================
# 6. PLOTLY VISUALIZATION HELPERS
# ==============================================================================
def plot_player_distribution(player_name: str, market: str, family: PropFamily, mu: float, var: float, line: float):
    fig = go.Figure()

    if family == "poisson_td":
        x = np.arange(0, 6)
        pmf = poisson.pmf(x, mu)
        colors = ["#EF4444" if val <= line else "#10B981" for val in x]
        fig.add_trace(go.Bar(x=x, y=pmf, marker_color=colors, name="Probability"))
    elif family == "negative_binomial":
        r = (mu ** 2) / max(var - mu, 1e-3)
        p = r / (r + mu)
        max_x = int(max(line * 2.2, mu + 4 * np.sqrt(var), 10))
        x = np.arange(0, max_x)
        pmf = nbinom.pmf(x, r, p)
        colors = ["#EF4444" if val <= line else "#10B981" for val in x]
        fig.add_trace(go.Bar(x=x, y=pmf, marker_color=colors, name="Probability"))
    else:
        alpha = (mu ** 2) / var
        theta = var / mu
        max_x = max(line * 2.2, mu + 4 * np.sqrt(var), 20)
        x = np.linspace(0.01, max_x, 300)
        pdf = gamma.pdf(x, a=alpha, scale=theta)

        x_under = x[x <= line]
        y_under = pdf[x <= line]
        x_over = x[x >= line]
        y_over = pdf[x >= line]

        fig.add_trace(go.Scatter(x=x_under, y=y_under, fill="tozeroy", mode="lines", line=dict(color="#EF4444"), name="Under Zone"))
        fig.add_trace(go.Scatter(x=x_over, y=y_over, fill="tozeroy", mode="lines", line=dict(color="#10B981"), name="Over Zone"))

    fig.add_vline(
        x=line,
        line_width=2.5,
        line_dash="dash",
        line_color="#38BDF8",
        annotation_text=f"Book Line: {line}",
        annotation_position="top right",
        annotation_font_color="#38BDF8",
    )

    fig.update_layout(
        title=f"{player_name} — Modeled {market} Probability Distribution (Mean: {mu:.1f})",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0"),
        xaxis_title=market,
        yaxis_title="Probability Density",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=True,
    )
    return fig


def plot_equity_curve(df: pd.DataFrame, date_col: str, profit_col: str, title: str):
    chart_df = df.sort_values(date_col).copy()
    chart_df["cum_units"] = chart_df[profit_col].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df[date_col],
        y=chart_df["cum_units"],
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#10B981", width=3),
        fillcolor="rgba(16, 185, 129, 0.15)",
        name="Cumulative Units",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0"),
        xaxis_title="Date",
        yaxis_title="Net Profit (Units)",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


# ==============================================================================
# 7. MAIN APPLICATION EXECUTION & DASHBOARD LAYOUT
# ==============================================================================
init_db()

# Header Banner
st.markdown("""
<div class="terminal-header">
    <div>
        <div class="terminal-title">APEX PROP // QUANTITATIVE BETTING TERMINAL</div>
        <div class="terminal-subtitle">Distributional Gradient Boosting • Power Devigging • Monte Carlo Injury Engine</div>
    </div>
    <div class="live-badge">● ENGINE ONLINE</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Strategy Parameters")
    sport = st.selectbox("League / Sport", ["NBA", "NFL"])
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
    market = st.selectbox("Prop Market", list(markets[sport].keys()))
    family = markets[sport][market]

    st.divider()
    min_edge = st.slider("Min +EV Edge Threshold (%)", 1.0, 12.0, 2.5, step=0.5) / 100.0
    kelly_frac = st.slider("Fractional Kelly Sizing", 0.10, 0.50, 0.25, step=0.05)
    st.caption("Quarter-Kelly (0.25) is recommended to minimize bankroll volatility.")

raw_df = load_logs(sport, market)

# Run Slate Model Early to Populate Top KPI Strip
bets = []
today = pd.DataFrame()
mu_today = np.array([])
var_today = np.array([])
latest_date = None

if len(raw_df) > 20:
    feat_df = engineer_features(raw_df)
    if not feat_df.empty:
        latest_date = feat_df["game_date"].max()
        train = feat_df[(feat_df["game_date"] < latest_date) & (feat_df["actual_stat"].notna())]
        today = feat_df[feat_df["game_date"] == latest_date].reset_index(drop=True)

        if len(train) > 15 and not today.empty and today["book_line"].notna().any():
            engine = DistributionalPropEngine(family=family)
            split = int(len(train) * 0.85)
            engine.fit(
                train[FEATURE_COLS].iloc[:split], train["actual_stat"].iloc[:split],
                train[FEATURE_COLS].iloc[split:], train["actual_stat"].iloc[split:],
            )
            lines_arr = today["book_line"].fillna(0).values
            p_over, p_under = engine.predict_probabilities(today[FEATURE_COLS], lines_arr)
            mu_today, var_today = engine.predict_params(today[FEATURE_COLS])
            bets = evaluate_slate(today, p_over, p_under, mu_today, var_today, min_edge=min_edge, kelly_fraction=kelly_frac)

# Top-of-Page KPI Summary Strip
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Active Market</div>
        <div class="kpi-value kpi-accent-blue">{sport} • {market}</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">+EV Edges Detected</div>
        <div class="kpi-value">{len(bets)} Plays</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    top_edge_str = f"+{bets[0].edge * 100:.1f}%" if bets else "0.0%"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Highest Slate Edge</div>
        <div class="kpi-value kpi-accent-green">{top_edge_str}</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    total_units = sum(b.kelly_fraction * 100 for b in bets)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Recommended Exposure</div>
        <div class="kpi-value">{total_units:.2f}u</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Top Plays & +EV Slate",
    "🎲 Monte Carlo Injury Sim",
    "📊 Walk-Forward Backtest",
    "🔌 Data & Odds Sync",
    "💰 Bet Tracker & Auto-Grader",
])

with tab1:
    if bets:
        st.markdown("### 🏆 Featured Top +EV Plays")
        top_picks = bets[:3]
        cols = st.columns(len(top_picks))
        for idx, pick in enumerate(top_picks):
            odds_fmt = f"+{pick.book_odds}" if pick.book_odds > 0 else f"{pick.book_odds}"
            stake_u = pick.kelly_fraction * 100
            with cols[idx]:
                st.markdown(f"""
                <div class="pick-card">
                    <div class="pick-header">
                        <span class="pick-player">{pick.player}</span>
                        <span class="edge-pill">+{pick.edge*100:.1f}% EDGE</span>
                    </div>
                    <div class="pick-line-box">
                        {pick.side} {pick.line} {pick.market} ({odds_fmt})
                    </div>
                    <div class="pick-meta-grid">
                        <span>Model Win %: <span class="pick-meta-val">{pick.model_prob*100:.1f}%</span></span>
                        <span>Fair Market: <span class="pick-meta-val">{pick.fair_market_prob*100:.1f}%</span></span>
                    </div>
                    <div class="pick-meta-grid">
                        <span>Proj. Mean: <span class="pick-meta-val">{pick.proj_mean}</span></span>
                        <span>Kelly Stake: <span class="pick-meta-val" style="color:#34D399;">{stake_u:.2f} Units</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        col_table, col_chart = st.columns([1.3, 1])
        with col_table:
            st.markdown("### 📋 Full Actionable Slate")
            bets_df = pd.DataFrame([vars(b) for b in bets])
            display_df = bets_df.copy()
            display_df["model_prob"] = (display_df["model_prob"] * 100).round(1)
            display_df["fair_market_prob"] = (display_df["fair_market_prob"] * 100).round(1)
            display_df["edge"] = (display_df["edge"] * 100).round(1)
            display_df["stake_units"] = (display_df["kelly_fraction"] * 100).round(2)

            st.dataframe(
                display_df[["player", "side", "line", "book_odds", "model_prob", "fair_market_prob", "edge", "stake_units"]],
                column_config={
                    "player": "Player",
                    "side": "Side",
                    "line": st.column_config.NumberColumn("Line", format="%.1f"),
                    "book_odds": st.column_config.NumberColumn("Odds", format="%d"),
                    "model_prob": st.column_config.NumberColumn("Model %", format="%.1f%%"),
                    "fair_market_prob": st.column_config.NumberColumn("Fair %", format="%.1f%%"),
                    "edge": st.column_config.ProgressColumn("Edge %", min_value=0, max_value=15, format="%.1f%%"),
                    "stake_units": st.column_config.NumberColumn("Stake (u)", format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )

            if st.button("📥 Log All +EV Picks to Bet Tracker", type="primary"):
                added = log_selected_bets(bets_df, sport=sport, game_date=latest_date)
                st.success(f"Logged {added} new bet(s) into your central tracker!")

        with col_chart:
            st.markdown("### 📈 Probability Curve Inspector")
            player_choices = [b.player for b in bets]
            selected_player = st.selectbox("Inspect Player Distribution", player_choices)
            p_idx = today[today["player_name"] == selected_player].index[0]
            fig_dist = plot_player_distribution(
                selected_player,
                market,
                family,
                float(mu_today[p_idx]),
                float(var_today[p_idx]),
                float(today.loc[p_idx, "book_line"]),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    elif len(raw_df) > 20:
        st.info("No props exceeded your minimum edge threshold today. Try lowering the **Min +EV Edge Threshold** slider in the left sidebar or sync today's live odds in Tab 4.")
    else:
        st.warning("👋 Welcome! Your database is currently empty. Open **Tab 4: Data & Odds Sync** and click **Load Instant Demo Data** to populate the dashboard in 1 second.")

with tab2:
    st.subheader("🎲 Interactive Injury & Usage Redistribution Simulator")
    st.caption("Select players ruled OUT on the injury report to dynamically redistribute their vacated usage/target share and simulate 10,000 game outcomes.")
    if len(raw_df) > 0:
        feat_df = engineer_features(raw_df)
        teams = sorted(feat_df["team"].dropna().unique())
        if teams:
            c_team, c_out = st.columns([1, 2])
            with c_team:
                selected_team = st.selectbox("Select Team", teams)
            team_df = feat_df[feat_df["team"] == selected_team].sort_values("game_date").drop_duplicates("player_name", keep="last")
            with c_out:
                out_players = st.multiselect("Mark Players OUT (Injury Report)", team_df["player_name"].tolist())

            if st.button("⚡ Run 10,000 Monte Carlo Simulations", type="primary"):
                sim_results = monte_carlo_injury_sim(team_df, out_players)
                sim_results["mc_prob_over_pct"] = (sim_results["mc_prob_over"] * 100).round(1)
                st.dataframe(
                    sim_results[["player_name", "roll_5_stat", "mc_adj_mean", "mc_adj_std", "book_line", "mc_prob_over_pct"]],
                    column_config={
                        "player_name": "Active Player",
                        "roll_5_stat": st.column_config.NumberColumn("Base L5 Avg", format="%.1f"),
                        "mc_adj_mean": st.column_config.NumberColumn("Sim Mean", format="%.2f"),
                        "mc_adj_std": st.column_config.NumberColumn("Sim Volatility", format="%.2f"),
                        "book_line": st.column_config.NumberColumn("Book Line", format="%.1f"),
                        "mc_prob_over_pct": st.column_config.ProgressColumn("Sim Over %", min_value=0, max_value=100, format="%.1f%%"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Load data in Tab 4 first to simulate team injury scenarios.")

with tab3:
    st.subheader("📊 Walk-Forward Historical Validation & Closing Line Value")
    st.caption("Trains strictly on past dates (t < T) and predicts future dates (t >= T) to prevent look-ahead bias.")
    if len(raw_df) > 40:
        if st.button("🚀 Run Walk-Forward Backtest", type="primary"):
            feat_df = engineer_features(raw_df)
            bt_df = run_walk_forward_backtest(feat_df, family=family, min_edge=min_edge, kelly_frac=kelly_frac)
            if not bt_df.empty:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Backtested Bets", len(bt_df))
                m2.metric("Win Rate", f"{bt_df['won'].mean():.1%}")
                m3.metric("Net Profit (Units)", f"{bt_df['profit'].sum():+.2f}u")
                m4.metric("Avg Closing Line Value", f"{bt_df['clv'].mean():+.2%}")

                st.plotly_chart(plot_equity_curve(bt_df, "date", "profit", "Walk-Forward Backtest Cumulative Profit (Units)"), use_container_width=True)
                st.dataframe(bt_df, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough paired historical lines and box scores to run a multi-day backtest yet.")
    else:
        st.info("Load data in Tab 4 first to run historical backtests.")

with tab4:
    st.subheader("🔌 Central Data Store & Live API Connectors")
    st.write(f"Current records stored for **{sport} — {market}**: **{len(raw_df):,} rows**")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 1. Instant Sandbox Mode")
        st.caption("Generates 30 days of historical box scores + today's live slate so you can test every feature immediately.")
        if st.button("⚡ Load Instant Demo Data", type="primary"):
            count = seed_demo_data()
            st.success(f"Loaded {count:,} sample rows! Click **Tab 1** to view your Top Plays.")
            st.rerun()

    with c2:
        st.markdown("#### 2. Official League Box Scores")
        st.caption("Sync real historical game logs to train the gradient-boosted models.")
        if st.button("🏀 Pull NBA Historical Logs"):
            with st.spinner("Downloading NBA stats..."):
                count = sync_nba_logs("2025-26")
                st.success(f"Synced {count:,} NBA rows.")

        if st.button("🏈 Pull NFL Historical Logs"):
            with st.spinner("Downloading NFL stats..."):
                count = sync_nfl_logs()
                st.success(f"Synced {count:,} NFL rows.")

    with c3:
        st.markdown("#### 3. Live Sportsbook Odds")
        st.caption("Stream today's player prop lines from US sportsbooks via The-Odds-API.")
        api_key = st.text_input("The-Odds-API Key", type="password")
        if st.button(f"📡 Fetch Live {sport} {market} Odds"):
            if not api_key:
                st.error("Enter your Odds API key first.")
            else:
                with st.spinner("Fetching live lines..."):
                    live_df = fetch_live_prop_odds(api_key, sport, market)
                    if not live_df.empty:
                        st.success(f"Loaded {len(live_df)} live lines.")
                        st.dataframe(live_df[["player_name", "book_line", "over_odds", "under_odds"]], hide_index=True)
                    else:
                        st.warning("No live lines returned for this market right now.")

with tab5:
    st.subheader("💰 Real-World Bet Performance & Auto-Grader")
    if st.button("🔄 Auto-Grade Pending Bets from Synced Box Scores", type="primary"):
        graded = auto_grade_pending_bets()
        st.success(f"Graded {graded} completed bet(s).")

    history_df = load_all_bets()
    if not history_df.empty:
        settled = history_df[history_df["status"].isin(["WON", "LOST", "PUSH"])]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Logged Bets", len(history_df))
        b2.metric("Pending Action", len(history_df[history_df["status"] == "PENDING"]))

        if not settled.empty:
            b3.metric("Settled Win Rate", f"{(settled['status'] == 'WON').mean():.1%}")
            b4.metric("Net Profit (Units)", f"{settled['profit_units'].sum():+.2f}u")
            st.plotly_chart(plot_equity_curve(settled, "game_date", "profit_units", "Live Tracked Bankroll Curve (Units)"), use_container_width=True)

        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No bets logged yet. Generate picks in Tab 1 and click **Log All +EV Picks to Bet Tracker**.")
