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


# ---- Public functions --------------------------------------------------------

def create_tables():
    """Create all tables if they don't already exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("All tables created (or already exist).")
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
