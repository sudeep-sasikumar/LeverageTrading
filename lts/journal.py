"""The trader's own trades (logged from the dashboard, or by Claude from exchange screenshots).

Kept in its own SQLite file (data/my_trades.sqlite) so the dashboard can write it while the trading
database stays read-only to the dashboard.

Money fields are in USDT as the exchange shows them:
  realized_pnl  price PnL of the close, before fees (Weex "Realized PnL" on the close row)
  open_fee, close_fee  positive numbers (fees paid)
  funding       signed: + received, - paid
Net = realized_pnl - open_fee - close_fee + funding. When realized_pnl is missing it is worked out
from the prices: side x (exit / entry - 1) x margin x leverage.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS my_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts INTEGER NOT NULL, updated_ts INTEGER NOT NULL,
    symbol TEXT NOT NULL, side INTEGER NOT NULL, leverage REAL, margin REAL,
    entry_price REAL, exit_price REAL, tp_price REAL, sl_price REAL,
    open_ts INTEGER, close_ts INTEGER,
    open_fee REAL, close_fee REAL, funding REAL, realized_pnl REAL,
    status TEXT NOT NULL DEFAULT 'open',          -- open / closed
    exit_reason TEXT,                             -- tp / sl / manual / liquidated
    exchange TEXT DEFAULT 'Weex', source TEXT DEFAULT 'manual',   -- manual (form) / screenshot (logged by Claude)
    order_id TEXT, notes TEXT
);
"""
NUM = ("leverage", "margin", "entry_price", "exit_price", "tp_price", "sl_price", "open_fee", "close_fee", "funding",
       "realized_pnl")
INT = ("open_ts", "close_ts")
TEXT = {"exit_reason": 20, "exchange": 20, "source": 20, "order_id": 60, "notes": 1000}
FIELDS = ("symbol", "side", "status") + NUM + INT + tuple(TEXT)


def path_for(db_path: Path) -> Path:
    return db_path.parent / "my_trades.sqlite"


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10)
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def clean(data: dict, partial: bool = False) -> dict:
    """Validate a trade from the form/API. Raises ValueError with a plain message."""
    out: dict = {}
    if "symbol" in data or not partial:
        sym = str(data.get("symbol") or "").strip().upper().replace("-", "").replace("/", "").replace(" ", "")
        if sym and not sym.endswith("USDT"):
            sym += "USDT"
        if not (4 < len(sym) <= 30 and sym.replace("USDT", "").isalnum()):
            raise ValueError("coin is required, e.g. MEGAUSDT")
        out["symbol"] = sym
    if "side" in data or not partial:
        side = str(data.get("side", "")).lower()
        if side not in ("1", "-1", "long", "short"):
            raise ValueError("side must be long or short")
        out["side"] = 1 if side in ("1", "long") else -1
    for k in NUM:
        if k in data:
            v = data[k]
            if v in (None, ""):
                out[k] = None
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{k.replace('_', ' ')} must be a number")
            if k not in ("funding", "realized_pnl") and v < 0:
                raise ValueError(f"{k.replace('_', ' ')} cannot be negative")
            out[k] = v
    for k in INT:
        if k in data:
            v = data[k]
            out[k] = None if v in (None, "") else int(float(v))
    for k, n in TEXT.items():
        if k in data:
            out[k] = None if data[k] in (None, "") else str(data[k])[:n]
    if "status" in data or not partial:
        st = str(data.get("status") or ("closed" if out.get("exit_price") else "open")).lower()
        if st not in ("open", "closed"):
            raise ValueError("status must be open or closed")
        out["status"] = st
    if not partial and not out.get("entry_price"):
        raise ValueError("entry price is required")
    return out


def add(path: Path, data: dict) -> int:
    t = clean(data)
    now = int(time.time() * 1000)
    t.setdefault("open_ts", now)
    conn = connect(path)
    try:
        cols = ["created_ts", "updated_ts"] + list(t)
        cur = conn.execute(f"INSERT INTO my_trades({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                           [now, now] + list(t.values()))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def update(path: Path, trade_id: int, data: dict) -> bool:
    t = clean(data, partial=True)
    if not t:
        return False
    conn = connect(path)
    try:
        sets = ", ".join(f"{k}=?" for k in t) + ", updated_ts=?"
        cur = conn.execute(f"UPDATE my_trades SET {sets} WHERE id=?", list(t.values()) + [int(time.time() * 1000), trade_id])
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def delete(path: Path, trade_id: int) -> bool:
    conn = connect(path)
    try:
        cur = conn.execute("DELETE FROM my_trades WHERE id=?", (trade_id,))
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def derived(r: dict) -> dict:
    """Gross, fees, net and % on margin for one trade."""
    side, lev, margin = r.get("side") or 1, r.get("leverage"), r.get("margin")
    gross = r.get("realized_pnl")
    if gross is None and r.get("exit_price") and r.get("entry_price") and lev and margin:
        gross = side * (r["exit_price"] / r["entry_price"] - 1) * margin * lev
    fees = (r.get("open_fee") or 0.0) + (r.get("close_fee") or 0.0)
    net = None if gross is None else gross - fees + (r.get("funding") or 0.0)
    move = None
    if r.get("exit_price") and r.get("entry_price"):
        move = side * (r["exit_price"] / r["entry_price"] - 1)
    return {"gross": gross, "fees": fees, "net": net, "move": move,
            "net_pct": (net / margin) if (net is not None and margin) else None}


def read(path: Path, signals: list[dict] | None = None) -> dict:
    """All trades (newest first) with derived numbers, the signal each came from, and totals."""
    if not path.exists():
        return {"trades": [], "totals": {"n": 0}}
    conn = connect(path, readonly=True)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM my_trades ORDER BY COALESCE(open_ts, created_ts) DESC")]
    finally:
        conn.close()
    for r in rows:
        r.update(derived(r))
        r["signal"] = match_signal(r, signals or [])
    closed = [r for r in rows if r["status"] == "closed" and r["net"] is not None]
    wins = [r for r in closed if r["net"] > 0]
    totals = {"n": len(rows), "open": sum(1 for r in rows if r["status"] == "open"), "closed": len(closed),
              "wins": len(wins), "net": sum(r["net"] for r in closed), "fees": sum(r["fees"] for r in rows),
              "margin_pct_avg": (sum(r["net_pct"] for r in closed if r["net_pct"] is not None) / len(closed)) if closed else None}
    return {"trades": rows, "totals": totals}


def match_signal(trade: dict, signals: list[dict]) -> dict | None:
    """The system signal for the same coin and side sent up to 6h before the trade was opened."""
    t = trade.get("open_ts")
    if not t:
        return None
    best = None
    for s in signals:
        if s["symbol"] == trade["symbol"] and s["side"] == trade["side"] and s["ts"] - 15 * 60_000 <= t <= s["ts"] + 6 * 3_600_000:
            if best is None or abs(t - s["ts"]) < abs(t - best["ts"]):
                best = s
    if not best:
        return None
    return {"id": best.get("id"), "ts": best["ts"], "kind": best.get("kind"), "entry": best.get("entry"),
            "status": best.get("status"), "leverage": best.get("leverage")}
