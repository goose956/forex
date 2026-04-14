"""
tracker/database.py -- Database connection and schema management.

Connects to Railway PostgreSQL if DATABASE_URL is set.
Falls back to local SQLite at tracker/data/signals.db for offline testing.

Usage:
    from tracker.database import get_engine, get_session, test_connection, create_tables
    test_connection()
    create_tables()
"""

import os
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import (
    create_engine, Column, Integer, Text, Boolean,
    Numeric, Date, DateTime, ForeignKey, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy.pool import NullPool

# ---- Logging -----------------------------------------------------------------
LOG_PATH = Path(__file__).parent / "logs" / "db.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("db")

# ---- Engine ------------------------------------------------------------------
_engine = None
_SessionFactory = None
_using_sqlite = False


def _build_engine():
    """Create the SQLAlchemy engine. Tries Railway first, falls back to SQLite."""
    global _engine, _SessionFactory, _using_sqlite

    db_url = os.getenv("DATABASE_URL", "").strip()

    if db_url:
        # Railway PostgreSQL -- psycopg2 expects postgresql:// not postgres://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        log.info("Connecting to Railway PostgreSQL...")
        _engine = create_engine(db_url, poolclass=NullPool, echo=False)
        _using_sqlite = False
    else:
        sqlite_path = Path(__file__).parent / "data" / "signals.db"
        sqlite_path.parent.mkdir(exist_ok=True)
        log.info(f"DATABASE_URL not set -- using local SQLite: {sqlite_path}")
        _engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)
        _using_sqlite = True

    _SessionFactory = sessionmaker(bind=_engine)
    return _engine


def get_engine():
    global _engine
    if _engine is None:
        _build_engine()
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _build_engine()
    return _SessionFactory()


def is_using_sqlite() -> bool:
    get_engine()
    return _using_sqlite


# ---- ORM Base ----------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---- Table: signals ----------------------------------------------------------
class Signal(Base):
    __tablename__ = "signals"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    analysis_date       = Column(Date)
    pair                = Column(Text)
    signal              = Column(Text)           # BUY / SELL / HOLD
    confidence          = Column(Integer)        # 1-10
    entry_price         = Column(Numeric(12, 5))
    stop_loss           = Column(Numeric(12, 5))
    take_profit         = Column(Numeric(12, 5))
    risk_reward         = Column(Numeric(6, 2))
    primary_reason      = Column(Text)
    invalidation        = Column(Text)
    full_reasoning      = Column(Text)
    technical_summary   = Column(Text)
    fundamental_summary = Column(Text)
    news_summary        = Column(Text)
    sentiment_summary   = Column(Text)
    bull_argument       = Column(Text)
    bear_argument       = Column(Text)
    risk_assessment     = Column(Text)
    claude_signal       = Column(Text)
    claude_confidence   = Column(Integer)
    gpt_signal          = Column(Text)
    gpt_confidence      = Column(Integer)
    providers_agree     = Column(Boolean)
    llm_provider        = Column(Text)
    model_used          = Column(Text)
    tokens_used         = Column(Integer)
    estimated_cost_gbp  = Column(Numeric(8, 4))
    market_conditions   = Column(Text)           # trending / ranging / volatile
    trend_direction     = Column(Text)           # up / down / sideways
    above_200ma         = Column(Boolean)
    session             = Column(Text)           # london / newyork / overlap


# ---- Table: outcomes ---------------------------------------------------------
class Outcome(Base):
    __tablename__ = "outcomes"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    signal_id             = Column(Integer, ForeignKey("signals.id"))
    outcome_recorded_at   = Column(DateTime, default=datetime.utcnow)
    outcome_date          = Column(Date)
    actual_close_price    = Column(Numeric(12, 5))
    pips_moved            = Column(Numeric(8, 1))
    signal_correct        = Column(Boolean)
    directionally_correct = Column(Boolean)
    would_have_hit_tp     = Column(Boolean)
    would_have_hit_sl     = Column(Boolean)
    max_favorable_pips    = Column(Numeric(8, 1))
    max_adverse_pips      = Column(Numeric(8, 1))
    notes                 = Column(Text)


# ---- Table: market_snapshots -------------------------------------------------
class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date      = Column(Date)
    pair               = Column(Text)
    open_price         = Column(Numeric(12, 5))
    high_price         = Column(Numeric(12, 5))
    low_price          = Column(Numeric(12, 5))
    close_price        = Column(Numeric(12, 5))
    ma_50              = Column(Numeric(12, 5))
    ma_200             = Column(Numeric(12, 5))
    rsi_14             = Column(Numeric(6, 2))
    macd_value         = Column(Numeric(10, 5))
    macd_signal_line   = Column(Numeric(10, 5))
    atr_14             = Column(Numeric(10, 5))
    trend_direction    = Column(Text)
    above_200ma        = Column(Boolean)
    nearest_support    = Column(Numeric(12, 5))
    nearest_resistance = Column(Numeric(12, 5))


# ---- Table: weekly_reports ---------------------------------------------------
class WeeklyReport(Base):
    __tablename__ = "weekly_reports"

    id                       = Column(Integer, primary_key=True, autoincrement=True)
    week_ending              = Column(Date)
    total_signals            = Column(Integer)
    buy_signals              = Column(Integer)
    sell_signals             = Column(Integer)
    hold_signals             = Column(Integer)
    outcomes_recorded        = Column(Integer)
    win_rate                 = Column(Numeric(5, 2))
    avg_confidence_winners   = Column(Numeric(5, 2))
    avg_confidence_losers    = Column(Numeric(5, 2))
    total_paper_pips         = Column(Numeric(10, 1))
    agreement_win_rate       = Column(Numeric(5, 2))
    disagreement_win_rate    = Column(Numeric(5, 2))
    best_signal_id           = Column(Integer)
    worst_signal_id          = Column(Integer)
    ai_performance_analysis  = Column(Text)
    patterns_identified      = Column(Text)
    recommendations          = Column(Text)


# ---- Table: costs ------------------------------------------------------------
class Cost(Base):
    __tablename__ = "costs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_date      = Column(Date)
    provider      = Column(Text)
    model         = Column(Text)
    tokens_input  = Column(Integer)
    tokens_output = Column(Integer)
    cost_usd      = Column(Numeric(8, 4))
    cost_gbp      = Column(Numeric(8, 4))
    run_type      = Column(Text)   # daily / backtest / manual


# ---- Table: virtual_trades ---------------------------------------------------
class VirtualTrade(Base):
    __tablename__ = "virtual_trades"

    id                  = Column(Integer, primary_key=True)
    signal_id           = Column(Integer, ForeignKey("signals.id"), nullable=False, unique=True)
    opened_at           = Column(Date, nullable=False)
    closed_at           = Column(Date, nullable=True)

    direction           = Column(Text)           # BUY / SELL / HOLD
    entry_price         = Column(Numeric(10, 5))
    stop_loss           = Column(Numeric(10, 5))
    take_profit         = Column(Numeric(10, 5))
    sl_pips             = Column(Numeric(8, 1))
    tp_pips             = Column(Numeric(8, 1))

    opening_balance     = Column(Numeric(12, 2))  # account balance when trade opened
    risk_pct            = Column(Numeric(5, 2))   # e.g. 1.0
    risk_gbp            = Column(Numeric(10, 4))  # e.g. 10.00
    value_per_pip       = Column(Numeric(10, 6))  # risk_gbp / sl_pips
    spread_pips         = Column(Numeric(5, 2), default=1.5)
    spread_cost_gbp     = Column(Numeric(10, 4))

    status              = Column(Text, default="open")  # open / won / lost / expired / skipped
    outcome_type        = Column(Text, nullable=True)   # tp_hit / sl_hit / expired
    pips_result         = Column(Numeric(8, 1), nullable=True)
    gross_pnl_gbp       = Column(Numeric(10, 4), nullable=True)
    net_pnl_gbp         = Column(Numeric(10, 4), nullable=True)  # after spread
    closing_balance     = Column(Numeric(12, 2), nullable=True)

    confluence_grade    = Column(Text, nullable=True)
    ai_confidence       = Column(Integer, nullable=True)
    providers_agree     = Column(Boolean, nullable=True)


# ---- Table: model_votes ------------------------------------------------------
class ModelVote(Base):
    __tablename__ = "model_votes"

    id            = Column(Integer, primary_key=True)
    signal_id     = Column(Integer, ForeignKey("signals.id"), nullable=False)
    analysis_date = Column(Date, nullable=False)
    model_name    = Column(Text, nullable=False)   # e.g. "google/gemini-flash-1.5"
    provider      = Column(Text, nullable=False)   # "openrouter", "anthropic", "openai"
    signal        = Column(Text)                   # BUY / SELL / HOLD
    confidence    = Column(Integer)               # 1-10
    reasoning     = Column(Text)
    cost_usd      = Column(Numeric(10, 6))
    cost_gbp      = Column(Numeric(10, 6))
    latency_ms    = Column(Integer)
    # Outcome tracking (filled in by update_outcomes.py)
    was_correct   = Column(Boolean, nullable=True)
    pips_result   = Column(Numeric(8, 1), nullable=True)


# ---- Table: virtual_account --------------------------------------------------
class VirtualAccount(Base):
    __tablename__ = "virtual_account"

    id              = Column(Integer, primary_key=True)
    updated_at      = Column(Date, nullable=False)
    balance         = Column(Numeric(12, 2), nullable=False)
    total_trades    = Column(Integer, default=0)
    open_trades     = Column(Integer, default=0)
    total_pnl_gbp   = Column(Numeric(12, 4), default=0)
    note            = Column(Text, nullable=True)


# ---- Public functions --------------------------------------------------------

def initialise_virtual_account(session):
    """Create the virtual account row with 1000 GBP if it does not exist yet."""
    from datetime import date as _date
    existing = session.query(VirtualAccount).first()
    if not existing:
        session.add(VirtualAccount(
            updated_at    = _date.today(),
            balance       = 1000.00,
            total_trades  = 0,
            open_trades   = 0,
            total_pnl_gbp = 0,
            note          = "Account opened",
        ))
        session.commit()
        log.info("Virtual account initialised with GBP 1000.00")


def get_virtual_balance(session) -> float:
    """Return the current virtual account balance as a float."""
    row = session.query(VirtualAccount).order_by(VirtualAccount.id.desc()).first()
    if row is None:
        return 1000.00
    return float(row.balance)


def update_virtual_balance(session, new_balance: float, note: str = ""):
    """Update the virtual account balance row."""
    from datetime import date as _date
    row = session.query(VirtualAccount).order_by(VirtualAccount.id.desc()).first()
    if row is None:
        session.add(VirtualAccount(
            updated_at    = _date.today(),
            balance       = new_balance,
            note          = note,
        ))
    else:
        row.balance    = new_balance
        row.updated_at = _date.today()
        row.note       = note
    session.commit()
    log.info(f"Virtual account balance updated: GBP {new_balance:.2f} ({note})")


def add_confluence_columns():
    """
    Add confluence tracking columns to the signals table if they don't exist.
    Uses ALTER TABLE with graceful fallback for both PostgreSQL and SQLite.
    All columns are nullable -- safe to add to existing tables.
    """
    engine = get_engine()

    confluence_cols = [
        ("current_price",                "DECIMAL(10,5)"),
        ("price_50ma",                   "DECIMAL(10,5)"),
        ("price_200ma",                  "DECIMAL(10,5)"),
        ("above_200ma_conf",             "BOOLEAN"),
        ("ma_alignment",                 "TEXT"),
        ("trend_direction_conf",         "TEXT"),
        ("adx_value",                    "DECIMAL(6,2)"),
        ("trend_strength",               "TEXT"),
        ("rsi_value",                    "DECIMAL(6,2)"),
        ("rsi_condition",                "TEXT"),
        ("rsi_divergence",               "TEXT"),
        ("nearest_support",              "DECIMAL(10,5)"),
        ("nearest_resistance",           "DECIMAL(10,5)"),
        ("at_key_level",                 "BOOLEAN"),
        ("key_level_type",               "TEXT"),
        ("key_level_price",              "DECIMAL(10,5)"),
        ("dxy_trend",                    "TEXT"),
        ("dxy_1day_change_pct",          "DECIMAL(6,3)"),
        ("us_10yr_yield",                "DECIMAL(6,3)"),
        ("uk_10yr_yield",                "DECIMAL(6,3)"),
        ("yield_spread",                 "DECIMAL(6,3)"),
        ("yield_spread_direction",       "TEXT"),
        ("confluence_raw_score",         "INTEGER"),
        ("confluence_max_possible",      "INTEGER"),
        ("confluence_pct",               "DECIMAL(5,2)"),
        ("confluence_grade",             "TEXT"),
        ("recommended_position_pct",     "DECIMAL(4,2)"),
        ("confluence_summary",           "TEXT"),
        ("confluence_factors",           "TEXT"),
        ("confluence_data_completeness_pct", "DECIMAL(5,2)"),
    ]

    with engine.connect() as conn:
        for col_name, col_type in confluence_cols:
            try:
                conn.execute(text(
                    f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}"
                ))
                conn.commit()
                log.info("Added column: signals.%s", col_name)
            except Exception:
                # Column likely already exists -- silently skip
                try:
                    conn.rollback()
                except Exception:
                    pass


def add_ensemble_columns():
    """
    Add ensemble tracking columns to the signals table if they don't exist.
    All columns are nullable -- safe to add to existing tables.
    """
    engine = get_engine()
    ensemble_cols = [
        ("ensemble_vote_count",    "INTEGER"),
        ("ensemble_agreement_pct", "DECIMAL(5,2)"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in ensemble_cols:
            try:
                conn.execute(text(
                    f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}"
                ))
                conn.commit()
                log.info("Added column: signals.%s", col_name)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass


def create_tables():
    """Create all tables if they don't already exist, then add confluence and ensemble columns."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("All tables created (or already exist).")
    add_confluence_columns()
    add_ensemble_columns()
    print("Database tables ready.")


def test_connection():
    """
    Test the database connection.
    Prints a clear success or failure message.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_type = "SQLite (local)" if is_using_sqlite() else "Railway PostgreSQL"
        print(f"Database connection OK -- {db_type}")
        log.info(f"Connection test passed: {db_type}")
        return True
    except Exception as e:
        print(f"Database connection FAILED: {e}")
        log.error(f"Connection test failed: {e}")
        return False


def test_insert_delete():
    """Run a test insert and delete on the signals table to verify write access."""
    from datetime import date
    session = get_session()
    try:
        test_row = Signal(
            analysis_date=date.today(),
            pair="TEST",
            signal="HOLD",
            confidence=5,
            llm_provider="test",
            model_used="test",
        )
        session.add(test_row)
        session.commit()
        row_id = test_row.id
        session.delete(test_row)
        session.commit()
        print(f"Test insert/delete OK (row id {row_id} created and removed).")
        log.info(f"Test insert/delete passed (id={row_id})")
        return True
    except Exception as e:
        session.rollback()
        print(f"Test insert/delete FAILED: {e}")
        log.error(f"Test insert/delete failed: {e}")
        return False
    finally:
        session.close()


if __name__ == "__main__":
    print("\n-- Database Setup --")
    test_connection()
    create_tables()
    test_insert_delete()
    db_type = "SQLite (local)" if is_using_sqlite() else "Railway PostgreSQL"
    print(f"\nUsing: {db_type}")
