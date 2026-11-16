"""Configuration: defaults here, overrides from config.yaml, secrets from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Universe:
    underlying_types: list = field(default_factory=lambda: ["COIN"])
    exclude: list = field(default_factory=lambda: ["USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "USD1USDT"])
    require_weex: bool = True            # only signal symbols tradable on Weex
    min_qvol_24h: float = 10_000_000     # data-source (Binance) 24h quote volume floor, USD
    min_weex_qvol_24h: float = 300_000   # Weex-side tradability floor, USD
    weex_only_min_qvol: float = 5_000_000  # Weex-only coins need this much Weex volume
    min_listing_days: float = 3.0


@dataclass
class Trigger:
    z_1h: float = 2.0
    z_4h: float = 2.0
    r_24h: float = 0.10
    rvol_1h: float = 1.3
    rvol_24h: float = 2.0
    min_pos_1h: float = 0.5
    # Below ~0.5% ATR the stop would be under 1%, where fees + slippage eat >25% of the risk.
    min_atr_pct: float = 0.0050
    max_atr_pct: float = 0.0200
    cooldown_bars: int = 6
    min_history_bars: int = 600          # ~2 days before a symbol can trigger


@dataclass
class Trade:
    sl_atr_mult: float = 2.0
    rr: float = 2.0
    horizon_bars: int = 96               # 8h time stop
    min_sl_pct: float = 0.010
    max_sl_pct: float = 0.025
    target_margin_return: float = 0.10   # +10% on margin at TP
    max_leverage: int = 10
    taker_fee: float = 0.0008            # Weex VIP0
    maker_fee: float = 0.0002
    slippage: float = 0.0004             # per market fill
    entry_valid_minutes: int = 10
    # grid recorded for research (labels for alternatives to the primary config)
    grid_sl_mult: list = field(default_factory=lambda: [1.5, 2.0, 3.0])
    grid_rr: list = field(default_factory=lambda: [1.5, 2.0, 3.0])
    grid_horizon: list = field(default_factory=lambda: [48, 96])


@dataclass
class Grind:
    """Steady-trend ("grind") setup: a coin that has moved steadily one way for hours, without one
    big candle doing the work. Thresholds come from the 30-day study in docs/DESIGN.md section 12."""
    enabled: bool = True
    min_r6: float = 0.04                 # >= 4% move over the last 6h
    min_r2: float = 0.60                 # straightness of the 6h path (R^2 of 15m closes)
    min_er: float = 0.25                 # 15m efficiency ratio over 6h
    min_hours_with: int = 4              # at least 4 of the last 6 hours moved with the trend
    max_big1h: float = 0.45              # no single hour did more than 45% of the 6h move
    require_market_aligned: bool = True  # only trade grinds going the same way as the median altcoin (24h)
    min_rank: float = 0.0                # rank = R^2 x efficiency x min(6h move, 15%). The 31-day replay showed
                                         # higher ranks did NOT do better, so no cut-off: rank only orders them
    live: bool = False                   # paper-only until the grind track has a proven forward record
    signals_from_5m: bool = False        # grind paper trades come from the grind lab (hourly); 5m grinds are logged only
    sl_atr1h_mult: float = 2.5           # stop = 2.5 x hourly ATR
    rr: float = 1.0                      # target = 1 x stop
    horizon_bars: int = 288              # hold up to 24h
    min_sl_pct: float = 0.015
    max_sl_pct: float = 0.08
    cooldown_bars: int = 72              # one grind candidate per coin and side per 6h
    max_open: int = 2
    max_per_day: int = 3
    min_gap_hours: float = 4.0           # spread the day's grind signals out
    grid_sl_mult: list = field(default_factory=lambda: [1.5, 2.5])
    grid_rr: list = field(default_factory=lambda: [1.0, 2.0])


@dataclass
class GrindLab:
    """Nightly study of what successful steady grinders share + hourly detection and paper trades."""
    enabled: bool = True
    days: float = 120.0                  # history studied (hourly bars)
    nightly_utc_hour: int = 1
    nightly_utc_minute: int = 10
    min_score: int = 8                   # checklist traits (of 12) needed for a paper trade
    stop: float = 0.03                   # paper-trade plan = the study's success definition
    target: float = 0.06
    horizon_hours: int = 24


@dataclass
class Risk:
    risk_per_trade: float = 0.01         # fraction of account lost if the stop is hit
    max_open: int = 2
    max_per_day: int = 3
    daily_stop_r: float = -2.0
    max_consecutive_losses: int = 2
    safe_mode_7d_r: float = -4.0
    symbol_cooldown_hours: float = 4.0
    max_spread_frac_of_sl: float = 0.12  # Weex spread must be < 12% of the stop distance
    max_spread_bps: float = 15.0
    btc_shock_15m: float = 0.012         # veto if BTC moved this much against the trade in 15m
    max_aligned_funding: float = 0.0010  # veto: paying >0.10% per interval to hold the trade
    max_weex_dev_bps: float = 40.0       # Weex vs Binance price deviation


@dataclass
class Signal:
    target_per_day: float = 1.0
    fallback_after_utc_hour: int = 14    # grade-B "best of day" allowed after this hour
    min_ev_r: float = 0.10               # minimum expected value (in R) for any signal
    rules_min_score: float = 70.0        # used only when no model exists


@dataclass
class Learn:
    retrain_utc_hour: int = 0
    retrain_utc_minute: int = 40
    min_samples: int = 1500
    n_folds: int = 5
    holdout_days: float = 5.0
    time_decay_halflife_days: float = 45.0
    live_weight: float = 2.0
    min_auc: float = 0.52
    min_xsec_corr: float = 0.03
    # promotion to real-money signals needs statistical evidence, not a lucky backtest
    min_oos_trades: int = 40
    min_oos_t: float = 2.0
    min_paper_signals: int = 20          # closed forward (paper) signals required, with avg R > 0
    demote_avg_r: float = -0.10          # a live champion this bad over the recent signals goes back to paper
    max_iter: int = 200
    learning_rate: float = 0.04
    max_depth: int = 3
    min_samples_leaf: int = 150
    l2: float = 5.0
    # Market-wide and calendar features have one independent observation per day, so they are
    # kept out of the model until the labelled history spans at least this many days.
    market_features_min_days: float = 120.0
    # "xsec": learn which candidate is best at a given moment (label minus the 4h-block mean),
    # which cannot memorise what kind of day it was. "label": plain P(target first).
    objective: str = "xsec"


@dataclass
class Storage:
    db_path: str = "data/lts.sqlite"
    model_dir: str = "data/models"
    log_dir: str = "logs"
    keep_klines_days: int = 120
    feature_window_bars: int = 2200      # bars loaded for the live scan (7d lookbacks + margin)


@dataclass
class Telegram:
    enabled: bool = True
    report_utc_hour: int = 18            # 23:30 IST
    report_utc_minute: int = 0
    account_size_usdt: float = 0.0       # optional: lets the bot quote margin in USDT
    local_utc_offset_minutes: int = 330  # times are shown in UTC and in this local zone
    local_tz_label: str = "IST"


@dataclass
class Config:
    universe: Universe = field(default_factory=Universe)
    trigger: Trigger = field(default_factory=Trigger)
    trade: Trade = field(default_factory=Trade)
    grind: Grind = field(default_factory=Grind)
    grindlab: GrindLab = field(default_factory=GrindLab)
    risk: Risk = field(default_factory=Risk)
    signal: Signal = field(default_factory=Signal)
    learn: Learn = field(default_factory=Learn)
    storage: Storage = field(default_factory=Storage)
    telegram: Telegram = field(default_factory=Telegram)

    # secrets (environment only)
    telegram_token: str = ""
    telegram_chat_id: str = ""

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


def _merge(obj, data: dict, where: str = "config") -> None:
    names = {f.name: f for f in fields(obj)}
    for key, value in (data or {}).items():
        if key not in names:
            raise ValueError(f"Unknown setting '{key}' in {where}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"'{where}.{key}' must be a mapping")
            _merge(current, value, f"{where}.{key}")
        else:
            setattr(obj, key, value)


def load_env_file(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    candidates = [Path(path)] if path else [ROOT / "config.yaml", ROOT / "config.local.yaml"]
    for p in candidates:
        if p.exists():
            _merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {}, p.name)
    load_env_file(ROOT / ".env")
    cfg.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    cfg.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    # environment overrides (how the Docker deployment is configured)
    env = os.environ
    if env.get("LTS_DB_PATH"):
        cfg.storage.db_path = env["LTS_DB_PATH"]
    if env.get("LTS_MODEL_DIR"):
        cfg.storage.model_dir = env["LTS_MODEL_DIR"]
    if env.get("LTS_LOG_DIR"):
        cfg.storage.log_dir = env["LTS_LOG_DIR"]
    if env.get("ACCOUNT_SIZE_USDT"):
        cfg.telegram.account_size_usdt = float(env["ACCOUNT_SIZE_USDT"])
    if env.get("RISK_PER_TRADE"):
        risk = float(env["RISK_PER_TRADE"])
        if not 0 < risk <= 0.05:
            raise ValueError("RISK_PER_TRADE must be a fraction between 0 and 0.05 (0.01 = 1% of the account)")
        cfg.risk.risk_per_trade = risk
    if env.get("MAX_SIGNALS_PER_DAY"):
        cfg.risk.max_per_day = int(env["MAX_SIGNALS_PER_DAY"])
    return cfg
