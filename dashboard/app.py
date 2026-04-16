"""
dashboard/app.py -- Forex Signal Tracker Streamlit Dashboard.

Run with:
    streamlit run dashboard/app.py

Pages:
    1. Today      -- current signal + recent performance + price chart
    2. History    -- full signal table with filters and expandable details
    3. Analytics  -- win rates, patterns, AI analysis (10+ outcomes required)
    4. Costs      -- spend tracking by provider and day
    5. Settings   -- API status, manual controls, database info
"""

import os
import sys
import json
from pathlib import Path
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="Forex Signal Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Light theme CSS ---------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .signal-buy  { color: #1a7f37; font-size: 2rem; font-weight: bold; }
    .signal-sell { color: #cf222e; font-size: 2rem; font-weight: bold; }
    .signal-hold { color: #6e7781; font-size: 2rem; font-weight: bold; }
    .agree-yes   { color: #1a7f37; }
    .agree-no    { color: #9a6700; }
    .win-row  td { color: #1a7f37 !important; }
    .loss-row td { color: #cf222e !important; }
</style>
""", unsafe_allow_html=True)


# ---- DB helpers --------------------------------------------------------------

@st.cache_resource
def get_db():
    from tracker.database import get_engine, create_tables
    engine = get_engine()
    create_tables()
    return engine


def db_ok():
    try:
        from tracker.database import test_connection
        return test_connection()
    except Exception:
        return False


def is_sqlite():
    try:
        from tracker.database import is_using_sqlite
        return is_using_sqlite()
    except Exception:
        return True


@st.cache_data(ttl=60)
def load_signals(days: int = 90):
    try:
        from tracker.database import get_session, Signal
        session = get_session()
        cutoff  = date.today() - timedelta(days=days)
        rows    = session.query(Signal).filter(Signal.analysis_date >= cutoff)\
                         .order_by(Signal.analysis_date.desc()).all()
        session.close()
        return rows
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_outcomes():
    try:
        from tracker.database import get_session, Outcome
        session = get_session()
        rows    = session.query(Outcome).all()
        session.close()
        return {r.signal_id: r for r in rows}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_costs(days: int = 90):
    try:
        from tracker.database import get_session, Cost
        session = get_session()
        cutoff  = date.today() - timedelta(days=days)
        rows    = session.query(Cost).filter(Cost.run_date >= cutoff)\
                         .order_by(Cost.run_date.desc()).all()
        session.close()
        return rows
    except Exception:
        return []


def signals_to_df(signals, outcomes_map):
    data = []
    for s in signals:
        o = outcomes_map.get(s.id)
        data.append({
            "id":          s.id,
            "Date":        s.analysis_date,
            "Signal":      s.signal or "",
            "Conf":        s.confidence or 0,
            "Entry":       float(s.entry_price  or 0),
            "SL":          float(s.stop_loss    or 0),
            "TP":          float(s.take_profit  or 0),
            "R:R":         float(s.risk_reward  or 0),
            "Claude":      s.claude_signal or "",
            "GPT":         s.gpt_signal   or "",
            "Agree":       "Yes" if s.providers_agree else "No",
            "Outcome":     (("Win" if o.signal_correct else ("Expired" if o.signal_correct is None else "Loss")) if o else "Pending"),
            "Pips":        float(o.pips_moved or 0) if o else None,
            "Trend":       s.trend_direction or "",
            "Above200MA":  "Yes" if s.above_200ma else "No",
            "full_reasoning":     s.full_reasoning or "",
            "technical_summary":  s.technical_summary or "",
            "fundamental_summary": s.fundamental_summary or "",
            "news_summary":       s.news_summary or "",
            "bull_argument":      s.bull_argument or "",
            "bear_argument":      s.bear_argument or "",
            "risk_assessment":    s.risk_assessment or "",
            "MTF Bias":    getattr(s, "mtf_bias",     None) or "",
            "MTF Aligned": getattr(s, "mtf_aligned",  None),
            "Weekly":      getattr(s, "weekly_trend",  None) or "",
            "4H":          getattr(s, "h4_trend",      None) or "",
            "MTF Notes":   getattr(s, "mtf_notes",     None) or "",
            "News Risk":    getattr(s, "news_risk_level",    None) or "",
            "News Events":  getattr(s, "news_event_names",   None) or "",
            "News Blocked": getattr(s, "news_trade_blocked", None),
            "VIX":          float(getattr(s, "vix_current", None) or 0) or None,
            "VIX Level":    getattr(s, "vix_level",   None) or "",
            "VIX Signal":   getattr(s, "vix_signal",  None) or "",
            "EURUSD Trend": getattr(s, "eurusd_trend", None) or "",
        })
    return pd.DataFrame(data)


# ---- Sidebar -----------------------------------------------------------------

def sidebar():
    st.sidebar.title("Forex Signal Tracker")
    st.sidebar.caption("GBP/USD AI Analysis System")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", ["Today", "Signal History", "Analytics", "Confluence", "Account", "Costs", "News & Calendar", "Settings"])

    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    days = st.sidebar.slider("Show last N days", 7, 180, 30)
    min_conf = st.sidebar.slider("Min confidence", 1, 10, 1)

    st.sidebar.divider()
    db_status = "🟢 Railway Connected" if not is_sqlite() else "🟡 Using Local SQLite"
    st.sidebar.caption(db_status)
    st.sidebar.caption(f"UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return page, days, min_conf


# ---- Page 1: Today -----------------------------------------------------------

def page_today(days, min_conf):
    st.title("Today's Signal")

    # ---- Live news risk banner (always shown, regardless of signal) ----
    try:
        from tracker.news_calendar import assess_news_risk
        news_risk = assess_news_risk()
        rl = news_risk.get("risk_level", "clear")

        if rl in ("binary", "high"):
            events = news_risk.get("high_impact_today", [])
            event_names = " | ".join(e.get("title", "Unknown event") for e in events[:4])
            icon = "BINARY EVENT" if rl == "binary" else "HIGH IMPACT NEWS"
            st.markdown(f"""
<div style="background:#9a0000;color:#fff;padding:22px 24px;border-radius:10px;
            text-align:center;margin-bottom:18px;border:2px solid #ff0000">
  <div style="font-size:1.6rem;font-weight:bold;letter-spacing:0.05em">
    🚫 DO NOT TRADE TODAY — {icon}
  </div>
  <div style="font-size:1.05rem;margin-top:8px;opacity:0.92">{event_names}</div>
  <div style="font-size:0.9rem;margin-top:8px;opacity:0.75">
    Signal tracked for data purposes only. Paper trade blocked automatically.
  </div>
</div>
""", unsafe_allow_html=True)

        elif rl == "medium":
            events = news_risk.get("high_impact_today", [])
            event_names = " | ".join(e.get("title", "") for e in events[:3])
            st.markdown(f"""
<div style="background:#7a4800;color:#fff;padding:16px 20px;border-radius:8px;
            text-align:center;margin-bottom:14px">
  <div style="font-size:1.2rem;font-weight:bold">
    ⚠️ CAUTION — Medium News Risk Today
  </div>
  <div style="font-size:0.95rem;margin-top:6px;opacity:0.9">{event_names}</div>
  <div style="font-size:0.85rem;margin-top:6px;opacity:0.75">
    Trade at reduced size if taking this signal.
  </div>
</div>
""", unsafe_allow_html=True)

        elif rl == "low":
            tomorrow_events = news_risk.get("high_impact_tomorrow", [])
            t_names = " | ".join(e.get("title", "") for e in tomorrow_events[:3])
            st.info(f"Advisory: High-impact news tomorrow — {t_names}. Consider closing any open positions today.")

    except Exception:
        pass  # Don't block the page if calendar is unavailable

    signals      = load_signals(days=1)
    all_signals  = load_signals(days=days)
    outcomes_map = load_outcomes()

    today_signals = [s for s in signals if s.analysis_date == date.today()]

    if today_signals:
        sig = today_signals[0]

        # ---- Row 1: Signal | Trade Levels | Providers ----
        col1, col2, col3 = st.columns(3)

        with col1:
            css_class = f"signal-{(sig.signal or 'hold').lower()}"
            st.markdown(f'<div class="{css_class}" style="font-size:2.5rem;font-weight:bold">{sig.signal or "HOLD"}</div>', unsafe_allow_html=True)
            # Show news-blocked badge if applicable
            try:
                _news_blocked = getattr(sig, "news_trade_blocked", None)
                _news_rl      = getattr(sig, "news_risk_level", None) or ""
                if _news_blocked:
                    st.markdown('<span style="background:#9a0000;color:#fff;padding:3px 10px;border-radius:4px;font-size:0.85rem;font-weight:bold">NO TRADE — NEWS</span>', unsafe_allow_html=True)
                elif _news_rl == "medium":
                    st.markdown('<span style="background:#7a4800;color:#fff;padding:3px 10px;border-radius:4px;font-size:0.85rem">CAUTION — NEWS</span>', unsafe_allow_html=True)
            except Exception:
                pass
            st.metric("Confidence", f"{sig.confidence or 0}/10")
            # providers_agree = ensemble-wide majority (>=60% of all models)
            try:
                vc = sig.ensemble_vote_count
                ap = float(sig.ensemble_agreement_pct or 0)
                if vc and vc > 2:
                    st.metric("Model Agreement", f"{ap:.0f}%", f"{vc} models")
                else:
                    agree_txt = "Yes" if sig.providers_agree else "No"
                    st.metric("Claude+GPT Agree", agree_txt)
            except Exception:
                agree_txt = "Yes" if sig.providers_agree else "No"
                st.metric("Claude+GPT Agree", agree_txt)
            # Confluence grade if available
            try:
                grade = sig.confluence_grade
                pct   = float(sig.confluence_pct or 0)
                if grade:
                    grade_colors = {"A+": "#1a7f37", "A": "#1a7f37", "B": "#9a6700", "C": "#cf222e", "D": "#cf222e"}
                    gc = grade_colors.get(grade, "#1a1a1a")
                    st.markdown(f'<span style="font-size:1.4rem;font-weight:bold;color:{gc}">Grade {grade}</span> <span style="color:#666">({pct:.0f}%)</span>', unsafe_allow_html=True)
            except Exception:
                pass

        with col2:
            st.markdown("**Trade Levels**")
            # Show order type + smart entry price if available
            try:
                order_type   = getattr(sig, "order_type",        None) or "market"
                smart_entry  = getattr(sig, "smart_entry_price", None)
                entry_display = float(smart_entry) if smart_entry else float(sig.entry_price or 0)
                entry_label  = f"Entry ({order_type.upper()})"
            except Exception:
                entry_display = float(sig.entry_price or 0)
                entry_label   = "Entry"
            st.metric(entry_label,   f"{entry_display:.5f}")
            st.metric("Stop Loss",   f"{float(sig.stop_loss    or 0):.5f}")
            st.metric("Take Profit", f"{float(sig.take_profit  or 0):.5f}")
            st.metric("Risk/Reward", f"1:{float(sig.risk_reward or 0):.1f}")

        with col3:
            st.markdown("**AI Votes**")
            st.metric("Claude",  sig.claude_signal or "N/A", f"{sig.claude_confidence or 0}/10")
            st.metric("GPT-4o",  sig.gpt_signal    or "N/A", f"{sig.gpt_confidence    or 0}/10")
            # Check if ensemble overrode Claude+GPT
            try:
                c_sig = sig.claude_signal or ""
                g_sig = sig.gpt_signal    or ""
                final = sig.signal        or ""
                vc    = sig.ensemble_vote_count
                ap    = float(sig.ensemble_agreement_pct or 0)
                if vc and vc > 2:
                    st.metric("Ensemble", f"{ap:.0f}% agree", f"{vc} models")
                    # Flag if ensemble overrode both main providers
                    if c_sig == g_sig and c_sig != final and c_sig != "":
                        st.warning(f"Ensemble overrode Claude+GPT ({c_sig} → {final})")
            except Exception:
                pass

        # ---- Ensemble vote breakdown ----
        try:
            from sqlalchemy import text as sa_text
            from tracker.ensemble import OPENROUTER_WEIGHTS, MAIN_MODEL_WEIGHTS
            eng = get_db()
            with eng.connect() as _conn:
                votes_df = pd.read_sql(sa_text(
                    "SELECT model_name, provider, signal, confidence FROM model_votes "
                    "WHERE signal_id = :sid ORDER BY confidence DESC"
                ), _conn, params={"sid": sig.id})

            if not votes_df.empty:
                st.markdown("**Ensemble Votes**")

                # Build full vote list including Claude + GPT from signal record
                all_votes = []
                all_votes.append({
                    "model": "claude-sonnet-4-6", "signal": sig.claude_signal or "HOLD",
                    "confidence": sig.claude_confidence or 5,
                    "weight": MAIN_MODEL_WEIGHTS.get("claude-sonnet-4-6", 3.0),
                })
                all_votes.append({
                    "model": "gpt-4o", "signal": sig.gpt_signal or "HOLD",
                    "confidence": sig.gpt_confidence or 5,
                    "weight": MAIN_MODEL_WEIGHTS.get("gpt-4o", 2.0),
                })
                for _, vr in votes_df.iterrows():
                    all_votes.append({
                        "model": vr["model_name"],
                        "signal": vr["signal"],
                        "confidence": vr["confidence"],
                        "weight": OPENROUTER_WEIGHTS.get(vr["model_name"], 1.0),
                    })

                # Weighted totals per direction
                totals = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
                total_w = 0.0
                for v in all_votes:
                    s = v["signal"] or "HOLD"
                    if s in totals:
                        totals[s] += v["weight"]
                        total_w   += v["weight"]

                # Horizontal weighted bar chart
                fig_votes = go.Figure()
                colors = {"BUY": "#1a7f37", "SELL": "#cf222e", "HOLD": "#6e7781"}
                for direction in ["BUY", "SELL", "HOLD"]:
                    w = totals[direction]
                    pct = round(w / total_w * 100, 1) if total_w > 0 else 0
                    fig_votes.add_trace(go.Bar(
                        name=direction,
                        x=[w],
                        y=["Votes"],
                        orientation="h",
                        marker_color=colors[direction],
                        text=f"{direction} {pct:.0f}%",
                        textposition="inside",
                        insidetextanchor="middle",
                        hovertemplate=f"{direction}: {w:.1f} weighted votes ({pct}%)<extra></extra>",
                    ))
                fig_votes.update_layout(
                    barmode="stack",
                    height=70,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_votes, use_container_width=True, config={"displayModeBar": False})

                # Detail table
                detail = pd.DataFrame(all_votes)
                detail["model"] = detail["model"].str.split("/").str[-1].str[:28]
                detail = detail.rename(columns={"model": "Model", "signal": "Vote", "confidence": "Conf", "weight": "Weight"})
                st.dataframe(detail[["Model","Vote","Conf","Weight"]],
                             use_container_width=True, hide_index=True)
        except Exception:
            pass

        # ---- Multi-timeframe alignment ----
        try:
            mtf_bias    = getattr(sig, "mtf_bias",    None)
            mtf_aligned = getattr(sig, "mtf_aligned", None)
            weekly_trend = getattr(sig, "weekly_trend", None)
            h4_trend     = getattr(sig, "h4_trend",    None)
            mtf_notes    = getattr(sig, "mtf_notes",   None)
            if mtf_bias:
                st.markdown("**Multi-Timeframe Analysis**")
                mcols = st.columns(4)
                mcols[0].metric("Weekly Trend", (weekly_trend or "N/A").upper())
                mcols[1].metric("4H Trend",     (h4_trend    or "N/A").upper())
                mcols[2].metric("MTF Bias",     mtf_bias or "N/A")
                if mtf_aligned is not None:
                    aligned_label = "Aligned" if mtf_aligned else "Conflict"
                    aligned_color = "#1a7f37" if mtf_aligned else "#cf222e"
                    mcols[3].markdown(
                        f'<span style="font-size:1.1rem;font-weight:bold;color:{aligned_color}">'
                        f'{aligned_label}</span>', unsafe_allow_html=True
                    )
                    if not mtf_aligned:
                        st.warning(
                            f"Daily signal ({sig.signal}) conflicts with MTF bias ({mtf_bias}). "
                            f"Paper trade skipped — logged for analysis only."
                        )
                if mtf_notes:
                    st.caption(f"MTF: {mtf_notes}")
        except Exception:
            pass

        # ---- Risk Environment (VIX + EURUSD) ----
        try:
            vix_current = getattr(sig, "vix_current", None)
            vix_level   = getattr(sig, "vix_level",   None)
            vix_signal  = getattr(sig, "vix_signal",  None)
            eurusd_trend = getattr(sig, "eurusd_trend", None)
            if vix_current is not None or eurusd_trend is not None:
                st.markdown("**Risk Environment**")
                rc1, rc2 = st.columns(2)
                if vix_current is not None:
                    vix_val = float(vix_current)
                    vix_color = "#1a7f37" if vix_signal == "clear" else ("#9a6700" if vix_signal == "caution" else "#cf222e")
                    rc1.markdown(
                        f'<span style="font-size:1.1rem;font-weight:bold;color:{vix_color}">'
                        f'VIX {vix_val:.1f}</span> <span style="color:#666">({vix_level})</span>',
                        unsafe_allow_html=True
                    )
                    if vix_signal == "avoid":
                        st.warning(f"VIX {vix_val:.1f} — high market fear. Paper trade may be affected.")
                    elif vix_signal == "caution":
                        st.info(f"VIX {vix_val:.1f} — elevated volatility. Trade with caution.")
                if eurusd_trend is not None:
                    eur_color = "#1a7f37" if eurusd_trend == "up" else ("#cf222e" if eurusd_trend == "down" else "#6e7781")
                    rc2.markdown(
                        f'<span style="font-size:1.1rem;font-weight:bold;color:{eur_color}">'
                        f'EURUSD {eurusd_trend.upper()}</span>',
                        unsafe_allow_html=True
                    )
        except Exception:
            pass

        # Primary reason expander
        if sig.primary_reason:
            with st.expander("View reasoning"):
                st.write(sig.primary_reason)

    else:
        st.info("No signal for today yet. Expected: 07:00 UTC via GitHub Actions.")
        if st.button("Run Manual Analysis Now"):
            st.info("Run in terminal:  python scripts/run_daily.py")

    st.divider()

    # ---- Row 2: Key metrics (compact) ----
    df = signals_to_df(all_signals, outcomes_map)
    if not df.empty:
        resolved   = df[df["Outcome"].isin(["Win", "Loss"])]
        wins       = resolved[resolved["Outcome"] == "Win"]
        win_rate   = round(len(wins) / len(resolved) * 100, 1) if len(resolved) > 0 else 0
        total_pips = resolved["Pips"].sum() if "Pips" in resolved.columns else 0
        costs      = load_costs(days=30)
        mtd        = sum(float(c.cost_gbp or 0) for c in costs)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", len(df))
        m2.metric("Win Rate",      f"{win_rate}%")
        m3.metric("Paper Pips",    f"{total_pips:+.0f}")
        m4.metric("Month Cost",    f"GBP {mtd:.2f}")

    # ---- Row 3: Recent table + chart side by side ----
    t_col, c_col = st.columns([1, 2])

    with t_col:
        st.subheader("Recent (14 days)")
        recent = load_signals(days=14)
        if recent:
            rdf = signals_to_df(recent, outcomes_map)
            show = [c for c in ["Date","Signal","Conf","Outcome","Pips"] if c in rdf.columns]
            st.dataframe(rdf[show], use_container_width=True, hide_index=True, height=320)

    with c_col:
        st.subheader("GBP/USD — Last 60 Days")
        _render_price_chart(all_signals, outcomes_map)


def _render_price_chart(signals, outcomes_map):
    try:
        import yfinance as yf
        df = yf.download("GBPUSD=X", period="90d", progress=False, auto_adjust=True)
        if df.empty:
            st.caption("Price chart unavailable.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = pd.to_datetime(df.index)

        ma50  = df["Close"].rolling(50).mean()
        ma200 = df["Close"].rolling(200).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="GBPUSD",
            increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        ))
        fig.add_trace(go.Scatter(x=df.index, y=ma50,  name="50 MA",  line=dict(color="#2196F3", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=ma200, name="200 MA", line=dict(color="#FF9800", width=1)))

        # Signal markers
        for sig in signals:
            if not sig.analysis_date or not sig.entry_price:
                continue
            o = outcomes_map.get(sig.id)
            colour = "#00c853" if sig.signal == "BUY" else ("#ff1744" if sig.signal == "SELL" else "#9e9e9e")
            if o and not o.signal_correct:
                colour = colour.replace("ff", "88")
            symbol = "triangle-up" if sig.signal == "BUY" else ("triangle-down" if sig.signal == "SELL" else "circle")
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(sig.analysis_date)],
                y=[float(sig.entry_price)],
                mode="markers",
                marker=dict(symbol=symbol, size=12, color=colour),
                name=f"{sig.signal} {sig.analysis_date}",
                showlegend=False,
            ))

        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"), height=400,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")


# ---- Page 2: Signal History -------------------------------------------------

def page_history(days, min_conf):
    st.title("Signal History")

    signals      = load_signals(days=days)
    outcomes_map = load_outcomes()
    df           = signals_to_df(signals, outcomes_map)

    if df.empty:
        st.info("No signals recorded yet.")
        return

    # Filters
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        sig_filter = st.selectbox("Signal", ["All", "BUY", "SELL", "HOLD"])
    with f2:
        out_filter = st.selectbox("Outcome", ["All", "Win", "Loss", "Expired", "Pending"])
    with f3:
        prov_filter = st.selectbox("Agreement", ["All", "Yes", "No"])
    with f4:
        mtf_filter = st.selectbox("MTF Alignment", ["All", "Aligned", "Conflict", "No data"])
    with f5:
        conf_filter = st.slider("Min confidence", 1, 10, min_conf, key="hist_conf")

    filtered = df.copy()
    if sig_filter != "All":
        filtered = filtered[filtered["Signal"] == sig_filter]
    if out_filter != "All":
        filtered = filtered[filtered["Outcome"] == out_filter]
    if prov_filter != "All":
        filtered = filtered[filtered["Agree"] == prov_filter]
    if mtf_filter == "Aligned":
        filtered = filtered[filtered["MTF Aligned"] == True]
    elif mtf_filter == "Conflict":
        filtered = filtered[filtered["MTF Aligned"] == False]
    elif mtf_filter == "No data":
        filtered = filtered[filtered["MTF Bias"] == ""]
    filtered = filtered[filtered["Conf"] >= conf_filter]

    st.caption(f"Showing {len(filtered)} of {len(df)} signals")

    # Export
    csv = filtered.drop(columns=["full_reasoning","technical_summary","fundamental_summary",
                                  "news_summary","bull_argument","bear_argument","risk_assessment"],
                        errors="ignore").to_csv(index=False)
    st.download_button("Download CSV", csv, "signals_export.csv", "text/csv")

    # Table with expandable rows
    display_cols = ["Date","Signal","Conf","Entry","SL","TP","R:R","Claude","GPT","Agree","Outcome","Pips"]
    available    = [c for c in display_cols if c in filtered.columns]

    total_filtered = len(filtered)
    pg_col1, pg_col2 = st.columns([3, 1])
    with pg_col1:
        st.caption(f"{total_filtered} signal{'s' if total_filtered != 1 else ''} match filters")
    with pg_col2:
        page_size = st.selectbox("Rows", [25, 50, 100, "All"], index=1, key="hist_page_size")
    paged = filtered if page_size == "All" else filtered.head(int(page_size))
    if total_filtered > (0 if page_size == "All" else int(page_size)):
        shown = total_filtered if page_size == "All" else int(page_size)
        if shown < total_filtered:
            st.caption(f"Showing {shown} of {total_filtered} — increase rows to see more")

    for _, row in paged.iterrows():
        outcome_icon = "✅" if row["Outcome"] == "Win" else ("❌" if row["Outcome"] == "Loss" else ("⌛" if row["Outcome"] == "Expired" else "⏳"))
        label = (f"{row['Date']}  |  {row['Signal']}  |  Conf: {row['Conf']}/10  "
                 f"|  Agree: {row['Agree']}  |  {outcome_icon} {row['Outcome']}")
        if row.get("Pips") is not None:
            label += f"  ({row['Pips']:+.0f} pips)"
        if row.get("MTF Bias"):
            mtf_tag = "MTF:Aligned" if row.get("MTF Aligned") else "MTF:Conflict"
            label += f"  |  {mtf_tag}"
        if row.get("News Risk") in ("binary", "high"):
            label += "  |  NEWS:Blocked"
        elif row.get("News Risk") == "medium":
            label += "  |  NEWS:Caution"
        with st.expander(label):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Entry:** {row['Entry']:.5f}")
                st.write(f"**Stop Loss:** {row['SL']:.5f}")
                st.write(f"**Take Profit:** {row['TP']:.5f}")
                st.write(f"**R:R:** 1:{row['R:R']:.1f}")
            with c2:
                st.write(f"**Claude:** {row['Claude']}")
                st.write(f"**GPT-4o:** {row['GPT']}")
                st.write(f"**Agree:** {row['Agree']}")
                st.write(f"**Trend:** {row['Trend']}  Above 200MA: {row['Above200MA']}")
                if row.get("MTF Bias"):
                    aligned_txt = "Aligned" if row.get("MTF Aligned") else "Conflict"
                    st.write(f"**MTF:** Weekly={row['Weekly'].upper() or 'N/A'}  4H={row['4H'].upper() or 'N/A'}  Bias={row['MTF Bias']}  ({aligned_txt})")
                if row.get("News Risk") and row.get("News Risk") != "clear":
                    blocked_txt = "YES — trade blocked" if row.get("News Blocked") else "No"
                    st.write(f"**News Risk:** {row['News Risk'].upper()}  |  Blocked: {blocked_txt}")
                    if row.get("News Events"):
                        st.caption(f"Events: {row['News Events']}")

            if row.get("technical_summary"):
                st.write("**Technical:**")
                st.caption(row["technical_summary"][:600])
            if row.get("bull_argument"):
                st.write("**Bull argument:**")
                st.caption(row["bull_argument"][:400])
            if row.get("bear_argument"):
                st.write("**Bear argument:**")
                st.caption(row["bear_argument"][:400])
            if row.get("risk_assessment"):
                st.write("**Risk assessment:**")
                st.caption(row["risk_assessment"][:400])
            if row.get("full_reasoning"):
                st.write("**Final decision:**")
                st.caption(row["full_reasoning"][:600])


# ---- Page 3: Analytics -------------------------------------------------------

def page_analytics(days, min_conf):
    st.title("Analytics")

    signals      = load_signals(days=days)
    outcomes_map = load_outcomes()
    df           = signals_to_df(signals, outcomes_map)
    resolved     = df[df["Outcome"].isin(["Win", "Loss"])] if not df.empty else pd.DataFrame()

    if len(resolved) < 10:
        remaining = 10 - len(resolved)
        st.info(f"Analytics will appear once 10+ signals have recorded outcomes.\n"
                f"You need {remaining} more. Keep going!")
        return

    # Row 1: Win rate by confidence + by day of week
    st.subheader("Reliability Analysis")
    r1a, r1b = st.columns(2)

    with r1a:
        conf_wr = resolved.groupby("Conf")["Outcome"].apply(
            lambda x: round((x == "Win").sum() / len(x) * 100, 1)
        ).reset_index()
        conf_wr.columns = ["Confidence", "Win Rate %"]
        fig = px.bar(conf_wr, x="Confidence", y="Win Rate %",
                     title="Win Rate by Confidence Level",
                     color="Win Rate %", color_continuous_scale=["#ff1744","#FF9800","#00c853"])
        fig.add_hline(y=60, line_dash="dash", line_color="#9e9e9e", annotation_text="60% target")
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
        st.plotly_chart(fig, use_container_width=True)

    with r1b:
        resolved_copy = resolved.copy()
        resolved_copy["DayOfWeek"] = pd.to_datetime(resolved_copy["Date"]).dt.day_name()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        dow_wr = resolved_copy.groupby("DayOfWeek")["Outcome"].apply(
            lambda x: round((x == "Win").sum() / len(x) * 100, 1)
        ).reindex(day_order).reset_index()
        dow_wr.columns = ["Day", "Win Rate %"]
        fig2 = px.bar(dow_wr, x="Day", y="Win Rate %",
                      title="Win Rate by Day of Week",
                      color="Win Rate %", color_continuous_scale=["#ff1744","#FF9800","#00c853"])
        fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Signal distribution + agreement analysis
    st.subheader("Signal & Agreement Analysis")
    r2a, r2b = st.columns(2)

    with r2a:
        dist = df["Signal"].value_counts().reset_index()
        dist.columns = ["Signal", "Count"]
        fig3 = px.pie(dist, names="Signal", values="Count",
                      title="Signal Distribution",
                      color="Signal",
                      color_discrete_map={"BUY":"#00c853","SELL":"#ff1744","HOLD":"#9e9e9e"})
        fig3.update_layout(paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
        st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        agree_wr    = resolved[resolved["Agree"] == "Yes"]["Outcome"].apply(lambda x: x == "Win").mean() * 100 if len(resolved[resolved["Agree"] == "Yes"]) > 0 else 0
        disagree_wr = resolved[resolved["Agree"] == "No"]["Outcome"].apply(lambda x: x == "Win").mean()  * 100 if len(resolved[resolved["Agree"] == "No"])  > 0 else 0
        fig4 = go.Figure(go.Bar(
            x=["Both Agree", "Disagree"],
            y=[round(agree_wr, 1), round(disagree_wr, 1)],
            marker_color=["#00c853","#ff9800"],
        ))
        fig4.update_layout(
            title="Win Rate: Agreement vs Disagreement",
            yaxis_title="Win Rate %",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Cumulative P&L
    st.subheader("Cumulative Paper P&L")
    if "Pips" in resolved.columns:
        cum = resolved.sort_values("Date").copy()
        cum["CumPips"] = cum["Pips"].cumsum()
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=cum["Date"], y=cum["CumPips"], fill="tozeroy",
                                   fillcolor="rgba(0,200,83,0.1)", line=dict(color="#00c853"),
                                   name="Cumulative Pips"))
        fig5.add_hline(y=0, line_color="#9e9e9e")
        fig5.update_layout(
            yaxis_title="Pips",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"), height=300,
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Row 4: Market conditions
    st.subheader("Win Rate by Market Conditions")
    r4a, r4b = st.columns(2)
    with r4a:
        if "Trend" in resolved.columns:
            trend_wr = resolved.groupby("Trend")["Outcome"].apply(
                lambda x: round((x=="Win").sum()/len(x)*100, 1)
            ).reset_index()
            trend_wr.columns = ["Trend", "Win Rate %"]
            fig6 = px.bar(trend_wr, x="Trend", y="Win Rate %", title="Win Rate by Trend",
                          color="Win Rate %", color_continuous_scale=["#ff1744","#00c853"])
            fig6.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
            st.plotly_chart(fig6, use_container_width=True)
    with r4b:
        if "Above200MA" in resolved.columns:
            ma_wr = resolved.groupby("Above200MA")["Outcome"].apply(
                lambda x: round((x=="Win").sum()/len(x)*100, 1)
            ).reset_index()
            ma_wr.columns = ["Above 200MA", "Win Rate %"]
            fig7 = px.bar(ma_wr, x="Above 200MA", y="Win Rate %", title="Win Rate: Above vs Below 200MA",
                          color="Win Rate %", color_continuous_scale=["#ff1744","#00c853"])
            fig7.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
            st.plotly_chart(fig7, use_container_width=True)

    # Row 5: Provider accuracy breakdown
    st.subheader("Provider Accuracy")
    st.caption("How accurate is each AI provider individually?")

    try:
        engine = get_db()
        with engine.connect() as conn:
            from sqlalchemy import text as sqlt
            df_prov = pd.read_sql(sqlt("""
                SELECT s.claude_signal, s.claude_confidence,
                       s.gpt_signal, s.gpt_confidence,
                       s.providers_agree, s.signal AS combined_signal,
                       o.signal_correct, o.pips_moved
                FROM signals s
                JOIN outcomes o ON s.id = o.signal_id
                WHERE s.claude_signal IS NOT NULL
                  AND s.gpt_signal IS NOT NULL
            """), conn)

        if df_prov.empty:
            st.info("Provider accuracy data will appear once outcomes are recorded.")
        else:
            # Calculate per-provider accuracy
            # Claude: was claude_signal correct direction vs pips_moved?
            def provider_correct(row, provider):
                sig = row[f'{provider}_signal']
                pips = row['pips_moved'] if row['pips_moved'] is not None else 0
                if sig == 'BUY':
                    return pips > 0
                elif sig == 'SELL':
                    return pips < 0
                return None  # HOLD -- exclude

            df_prov['claude_correct'] = df_prov.apply(lambda r: provider_correct(r, 'claude'), axis=1)
            df_prov['gpt_correct']    = df_prov.apply(lambda r: provider_correct(r, 'gpt'),    axis=1)

            claude_data  = df_prov[df_prov['claude_correct'].notna()]
            gpt_data     = df_prov[df_prov['gpt_correct'].notna()]
            agree_data   = df_prov[df_prov['providers_agree'] == True]
            disagree_data = df_prov[df_prov['providers_agree'] == False]

            claude_wr  = round(claude_data['claude_correct'].mean() * 100, 1) if len(claude_data) > 0 else 0
            gpt_wr     = round(gpt_data['gpt_correct'].mean() * 100, 1)       if len(gpt_data) > 0 else 0
            agree_wr2  = round((agree_data['signal_correct'] == True).mean() * 100, 1)   if len(agree_data) > 0 else 0
            disagree_wr2 = round((disagree_data['signal_correct'] == True).mean() * 100, 1) if len(disagree_data) > 0 else 0

            # Summary metrics row
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Claude Win Rate",    f"{claude_wr}%",   f"{len(claude_data)} signals")
            p2.metric("GPT-4o Win Rate",    f"{gpt_wr}%",      f"{len(gpt_data)} signals")
            p3.metric("When Both Agree",    f"{agree_wr2}%",   f"{len(agree_data)} signals")
            p4.metric("When Disagree",      f"{disagree_wr2}%", f"{len(disagree_data)} signals")

            # Bar chart comparison
            prov_df = pd.DataFrame({
                'Provider': ['Claude', 'GPT-4o', 'Both Agree', 'Disagree'],
                'Win Rate %': [claude_wr, gpt_wr, agree_wr2, disagree_wr2],
                'Signals': [len(claude_data), len(gpt_data), len(agree_data), len(disagree_data)],
            })
            fig_prov = px.bar(
                prov_df, x='Provider', y='Win Rate %',
                title='Win Rate by Provider',
                color='Win Rate %',
                color_continuous_scale=['#cf222e', '#FF9800', '#1a7f37'],
                text='Signals',
            )
            fig_prov.add_hline(y=50, line_dash="dash", line_color="#9e9e9e", annotation_text="50% baseline")
            fig_prov.update_traces(texttemplate='%{text} signals', textposition='outside')
            fig_prov.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8f9fa",
                                   font=dict(color="#1a1a1a"), showlegend=False)
            st.plotly_chart(fig_prov, use_container_width=True)

            # Verdict
            if claude_wr > gpt_wr + 10:
                verdict = f"Claude is outperforming GPT-4o by {claude_wr - gpt_wr:.0f}% -- Claude signals carry more weight."
            elif gpt_wr > claude_wr + 10:
                verdict = f"GPT-4o is outperforming Claude by {gpt_wr - claude_wr:.0f}% -- GPT signals carry more weight."
            else:
                verdict = "Claude and GPT-4o are performing similarly -- neither has a clear edge yet."

            if agree_wr2 > disagree_wr2 + 10:
                verdict += f" Agreement signals win {agree_wr2}% vs {disagree_wr2}% when they disagree -- filter to agreement-only signals for best results."
            elif len(agree_data) < 5:
                verdict += " Not enough data yet to draw conclusions on agreement value."

            st.info(verdict)

    except Exception as e:
        st.warning(f"Provider accuracy unavailable: {e}")

    # Row 6: Model Leaderboard
    st.subheader("Model Leaderboard")
    st.caption("Accuracy of each AI model over time. Updates as outcomes are recorded.")

    try:
        engine = get_db()
        with engine.connect() as conn:
            from sqlalchemy import text as sqlt

            # Get all OpenRouter model votes with outcomes
            df_votes = pd.read_sql(sqlt("""
                SELECT mv.model_name, mv.provider, mv.signal, mv.confidence,
                       mv.was_correct, mv.pips_result, mv.analysis_date,
                       mv.cost_gbp, mv.latency_ms
                FROM model_votes mv
                WHERE mv.was_correct IS NOT NULL
            """), conn)

            # Also include Claude and GPT from signals table
            df_main = pd.read_sql(sqlt("""
                SELECT
                    'claude-sonnet-4-6' as model_name, 'anthropic' as provider,
                    s.claude_signal as signal, s.claude_confidence as confidence,
                    CASE WHEN s.claude_signal = o.outcome_signal THEN true ELSE false END as was_correct,
                    o.pips_moved as pips_result, s.analysis_date,
                    NULL::numeric as cost_gbp, NULL::integer as latency_ms
                FROM signals s JOIN outcomes o ON s.id = o.signal_id
                WHERE s.claude_signal IS NOT NULL
                UNION ALL
                SELECT
                    'gpt-4o-mini' as model_name, 'openai' as provider,
                    s.gpt_signal as signal, s.gpt_confidence as confidence,
                    CASE WHEN s.gpt_signal = o.outcome_signal THEN true ELSE false END as was_correct,
                    o.pips_moved as pips_result, s.analysis_date,
                    NULL::numeric as cost_gbp, NULL::integer as latency_ms
                FROM signals s JOIN outcomes o ON s.id = o.signal_id
                WHERE s.gpt_signal IS NOT NULL
            """), conn)

        # Combine
        df_all = pd.concat([df_main, df_votes], ignore_index=True) if not df_votes.empty else df_main

        if df_all.empty:
            st.info("Model leaderboard will appear once outcomes are recorded.")
        else:
            # Calculate stats per model
            stats = df_all.groupby("model_name").apply(lambda g: pd.Series({
                "Provider":    g["provider"].iloc[0],
                "Signals":     len(g),
                "Win Rate %":  round((g["was_correct"] == True).mean() * 100, 1),
                "Avg Conf":    round(g["confidence"].mean(), 1),
                "Avg Pips":    round(g["pips_result"].astype(float).mean(), 1),
                "Rank Score":  round((g["was_correct"] == True).mean() * 100 + g["confidence"].mean(), 1),
            })).reset_index().rename(columns={"model_name": "Model"})

            # Sort by win rate
            stats = stats.sort_values("Win Rate %", ascending=False).reset_index(drop=True)

            # Add rank
            stats.insert(0, "Rank", range(1, len(stats) + 1))

            # Colour top model green, bottom red
            def highlight_rank(row):
                if row["Rank"] == 1:
                    return ["background-color: #d4edda"] * len(row)
                elif row["Rank"] == len(stats):
                    return ["background-color: #f8d7da"] * len(row)
                return [""] * len(row)

            st.dataframe(
                stats[["Rank", "Model", "Provider", "Signals", "Win Rate %", "Avg Conf", "Avg Pips"]].style.apply(highlight_rank, axis=1),
                use_container_width=True, hide_index=True
            )

            if len(stats) >= 2:
                best      = stats.iloc[0]["Model"]
                worst     = stats.iloc[-1]["Model"]
                best_wr   = stats.iloc[0]["Win Rate %"]
                worst_wr  = stats.iloc[-1]["Win Rate %"]
                if best_wr - worst_wr > 10:
                    st.info(f"{best} is the top performer at {best_wr}% win rate. "
                            f"Consider weighting its signals more heavily. "
                            f"{worst} is weakest at {worst_wr}%.")
                else:
                    st.info("All models performing similarly. More data needed to identify the best performers.")

    except Exception as e:
        st.caption(f"Model leaderboard unavailable: {e}")

    # Row 7: AI pattern analysis
    st.subheader("AI Pattern Analysis")
    st.caption("Ask Claude to analyse your signal history and identify patterns.")
    if st.button("Run Pattern Analysis"):
        with st.spinner("Asking Claude to analyse your data..."):
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                data_text = resolved[["Date","Signal","Conf","Agree","Outcome","Pips","Trend","Above200MA"]].to_string()

                prompt = f"""You are analysing the historical performance of an AI forex signal system trading GBP/USD.
Here is the signal history with outcomes:

{data_text}

Please analyse this data and provide:
1. Overall assessment: does this system show positive expectancy?
2. At what confidence level do signals become reliably profitable?
3. Are there patterns in timing, market conditions, or provider agreement?
4. What specific conditions should I require before acting on a signal?
5. What is working well and what isn't?
6. Specific recommendations to improve signal quality.

Be direct and honest. If the data shows the system isn't working, say so clearly."""

                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    messages=[{"role":"user","content":prompt}],
                )
                analysis = msg.content[0].text
                st.write(analysis)
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ---- Page 4: Costs -----------------------------------------------------------

def page_costs():
    st.title("Cost Tracking")

    costs = load_costs(days=90)
    if not costs:
        st.info("No cost data recorded yet.")
        return

    df = pd.DataFrame([{
        "Date":     c.run_date,
        "Provider": c.provider or "",
        "Model":    c.model    or "",
        "In Tokens": c.tokens_input  or 0,
        "Out Tokens": c.tokens_output or 0,
        "Cost USD": float(c.cost_usd or 0),
        "Cost GBP": float(c.cost_gbp or 0),
        "Run Type": c.run_type or "",
    } for c in costs])

    today  = date.today()
    today_spend = df[df["Date"] == today]["Cost GBP"].sum()
    week_spend  = df[df["Date"] >= today - timedelta(days=7)]["Cost GBP"].sum()
    month_spend = df[df["Date"] >= date(today.year, today.month, 1)]["Cost GBP"].sum()
    total_spend = df["Cost GBP"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Today",     f"GBP {today_spend:.4f}")
    c2.metric("This week", f"GBP {week_spend:.4f}")
    c3.metric("This month",f"GBP {month_spend:.4f}")
    c4.metric("All time",  f"GBP {total_spend:.4f}")

    # Daily spend chart
    st.subheader("Daily Spend (last 30 days)")
    daily = df.groupby(["Date","Provider"])["Cost GBP"].sum().reset_index()
    fig = px.bar(daily, x="Date", y="Cost GBP", color="Provider",
                 color_discrete_map={"anthropic":"#7c4dff","openai":"#00acc1","both":"#00bfa5"},
                 title="Daily Spend by Provider (GBP)")
    fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Metrics")
    n_signals  = len(df) // 2 if len(df) > 0 else 0  # 2 runs per signal (claude+gpt)
    cost_per_s = round(total_spend / n_signals, 4) if n_signals > 0 else 0
    st.write(f"Total runs: {len(df)}  |  Est. signals: {n_signals}  |  Avg cost/signal: GBP {cost_per_s:.4f}")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    # Provider breakdown
    st.subheader("Provider Breakdown")
    prov = df.groupby("Provider")["Cost GBP"].sum().reset_index()
    figp = px.pie(prov, names="Provider", values="Cost GBP", title="Spend by Provider (GBP)",
                  color="Provider",
                  color_discrete_map={"anthropic":"#7c4dff","openai":"#00acc1","both":"#00bfa5"})
    figp.update_layout(paper_bgcolor="#0e1117", font=dict(color="#e0e0e0"))
    st.plotly_chart(figp, use_container_width=True)


# ---- Page 5: Settings --------------------------------------------------------

def page_settings():
    st.title("Settings & Status")

    # API status
    st.subheader("API Status")

    def test_api(name, fn):
        try:
            ok = fn()
            st.success(f"🟢 {name}: Connected") if ok else st.error(f"🔴 {name}: Error")
        except Exception as e:
            st.error(f"🔴 {name}: {e}")

    col1, col2 = st.columns(2)
    with col1:
        db_connected = db_ok()
        if db_connected:
            db_label = "Railway PostgreSQL" if not is_sqlite() else "Local SQLite"
            st.success(f"🟢 Database: {db_label}")
        else:
            st.error("🔴 Database: Not connected")

        if os.getenv("ANTHROPIC_API_KEY"):
            st.success("🟢 Anthropic API key: Set")
        else:
            st.error("🔴 Anthropic API key: Missing")

    with col2:
        if os.getenv("OPENAI_API_KEY"):
            st.success("🟢 OpenAI API key: Set")
        else:
            st.error("🔴 OpenAI API key: Missing")

        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            st.success("🟢 Alpha Vantage key: Set")
        else:
            st.warning("🟡 Alpha Vantage key: Missing (yfinance is default)")

    if st.button("Test Live API Connections"):
        with st.spinner("Testing..."):
            try:
                import anthropic
                c = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=5,
                                  messages=[{"role":"user","content":"OK"}])
                st.success("Anthropic: OK")
            except Exception as e:
                st.error(f"Anthropic: {e}")
            try:
                import openai
                o = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                o.chat.completions.create(model="gpt-4o-mini", max_tokens=5,
                                          messages=[{"role":"user","content":"OK"}])
                st.success("OpenAI: OK")
            except Exception as e:
                st.error(f"OpenAI: {e}")

    st.divider()

    # Manual controls
    st.subheader("Manual Controls")
    st.info("To run an analysis, use the batch files in the scheduler/ folder, or run from terminal:\n"
            "```\npython scripts/run_daily.py\n```")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Export Full Data (Excel)"):
            try:
                import io
                import openpyxl
                from sqlalchemy import text as sqlt

                engine = get_db()
                buf = io.BytesIO()

                with pd.ExcelWriter(buf, engine="openpyxl") as writer:

                    # Sheet 1: Signals + confluence
                    with engine.connect() as conn:
                        df_sig = pd.read_sql(sqlt("SELECT * FROM signals ORDER BY analysis_date DESC"), conn)
                    if not df_sig.empty:
                        # Drop large JSON column to keep file readable
                        if "confluence_factors" in df_sig.columns:
                            df_sig = df_sig.drop(columns=["confluence_factors"])
                        df_sig.to_excel(writer, sheet_name="Signals", index=False)

                    # Sheet 2: Outcomes
                    with engine.connect() as conn:
                        df_out = pd.read_sql(sqlt("""
                            SELECT s.analysis_date, s.signal, s.ai_confidence,
                                   s.claude_signal, s.claude_confidence,
                                   s.gpt_signal, s.gpt_confidence,
                                   s.providers_agree, s.confluence_grade,
                                   s.confluence_pct, s.entry_price,
                                   s.stop_loss, s.take_profit,
                                   o.outcome_date, o.pips_moved, o.signal_correct,
                                   o.would_have_hit_tp, o.would_have_hit_sl,
                                   o.max_favorable_pips, o.max_adverse_pips
                            FROM signals s
                            JOIN outcomes o ON s.id = o.signal_id
                            ORDER BY s.analysis_date DESC
                        """), conn)
                    if not df_out.empty:
                        df_out.to_excel(writer, sheet_name="Outcomes", index=False)

                    # Sheet 3: Paper trades
                    with engine.connect() as conn:
                        df_vt = pd.read_sql(sqlt("""
                            SELECT vt.opened_at, vt.direction, vt.confluence_grade,
                                   vt.entry_price, vt.sl_pips, vt.tp_pips,
                                   vt.risk_pct, vt.risk_gbp, vt.spread_cost_gbp,
                                   vt.status, vt.outcome_type, vt.pips_result,
                                   vt.gross_pnl_gbp, vt.net_pnl_gbp,
                                   vt.opening_balance, vt.closing_balance,
                                   vt.ai_confidence, vt.providers_agree
                            FROM virtual_trades vt
                            ORDER BY vt.opened_at DESC
                        """), conn)
                    if not df_vt.empty:
                        df_vt.to_excel(writer, sheet_name="Paper Trades", index=False)

                    # Sheet 4: Costs
                    with engine.connect() as conn:
                        df_cost = pd.read_sql(sqlt("SELECT * FROM costs ORDER BY run_date DESC"), conn)
                    if not df_cost.empty:
                        df_cost.to_excel(writer, sheet_name="API Costs", index=False)

                    # Sheet 5: Summary for LLM context
                    today = date.today()
                    summary_rows = [
                        ["Export date", str(today)],
                        ["System", "GBP/USD Daily Forex Signal Tracker"],
                        ["AI providers", "Claude (Anthropic) + GPT-4o-mini (OpenAI)"],
                        ["Signal frequency", "Once per weekday at 07:00 UTC"],
                        ["Trade resolution", "5 trading days (TP hit / SL hit / expiry)"],
                        ["Starting paper balance", "GBP 1000"],
                        ["Spread used", "1.5 pips"],
                        ["", ""],
                        ["--- CONFLUENCE SCORING ---", ""],
                        ["Grade A+", "85-100% -- 1.5% risk"],
                        ["Grade A",  "70-84%  -- 1.0% risk"],
                        ["Grade B",  "55-69%  -- 0.5% risk (minimum real trade)"],
                        ["Grade C",  "40-54%  -- shadow trade only (not in paper account)"],
                        ["Grade D",  "below 40% -- skipped"],
                        ["", ""],
                        ["--- 7 CONFLUENCE FACTORS ---", ""],
                        ["1. Trend alignment", "Price vs 200MA + 50MA (max 3pts)"],
                        ["2. ADX strength", "Trend strength indicator (max 2pts)"],
                        ["3. RSI condition", "Overbought/oversold (max 2pts)"],
                        ["4. Key level", "Support/resistance proximity (max 2pts)"],
                        ["5. DXY direction", "Dollar index trend (max 2pts)"],
                        ["6. Yield spread", "UK vs US 10yr yields (max 3pts)"],
                        ["7. RSI divergence", "Price/RSI divergence (max 2pts)"],
                        ["AI confidence", "Combined AI score (max 3pts)"],
                        ["Provider agreement", "Both providers agree (max 2pts)"],
                        ["", ""],
                        ["--- SUGGESTED LLM ANALYSIS QUESTIONS ---", ""],
                        ["1", "Which confluence grade performs best? Is the scoring adding value?"],
                        ["2", "Which individual factors correlate most with winning trades?"],
                        ["3", "Is there a pattern in the days of week or market conditions for wins?"],
                        ["4", "Do Claude and GPT agree more on winning trades than losing ones?"],
                        ["5", "Is the paper account profitable? What is the expectancy per trade?"],
                        ["6", "Which factors should be removed or weighted higher?"],
                        ["7", "What minimum conditions would have filtered out the most losses?"],
                    ]
                    df_summary = pd.DataFrame(summary_rows, columns=["Field", "Value"])
                    df_summary.to_excel(writer, sheet_name="LLM Context", index=False)

                buf.seek(0)
                fname = f"forex_tracker_export_{today}.xlsx"
                st.download_button(
                    label="Download Excel Export",
                    data=buf,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.success(f"Ready: {fname} — 5 sheets: Signals, Outcomes, Paper Trades, API Costs, LLM Context")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col_b:
        backtest_date = st.date_input("Backtest date", value=date.today() - timedelta(days=7))
        if st.button("Run Backtest"):
            st.info(f"To backtest, run in terminal:\n```\npython scripts/run_daily.py --date {backtest_date}\n```")

    st.divider()

    # Database info
    st.subheader("Database Info")
    try:
        from tracker.database import get_session, Signal, Outcome, Cost
        session  = get_session()
        n_sigs   = session.query(Signal).count()
        n_outs   = session.query(Outcome).count()
        n_costs  = session.query(Cost).count()
        first    = session.query(Signal).order_by(Signal.analysis_date).first()
        session.close()

        st.write(f"Total signals: {n_sigs}")
        st.write(f"Total outcomes: {n_outs}")
        st.write(f"Total cost records: {n_costs}")
        st.write(f"Oldest signal: {first.analysis_date if first else 'None'}")
        db_type = "Local SQLite" if is_sqlite() else "Railway PostgreSQL"
        st.write(f"Connection type: {db_type}")
    except Exception as e:
        st.error(f"Could not query database: {e}")


# ---- Page 6: Confluence -------------------------------------------------------

def page_confluence():
    st.title("Confluence Analysis")

    engine = get_db()

    # Fetch today's signal
    today = date.today().isoformat()
    row = None
    with engine.connect() as conn:
        from sqlalchemy import text
        row = conn.execute(text(
            "SELECT * FROM signals WHERE analysis_date = :d ORDER BY id DESC LIMIT 1"
        ), {'d': today}).fetchone()

    # If today has no confluence data, try yesterday
    _showing_stale = False
    if not row or not (row._mapping.get('confluence_grade')):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with engine.connect() as conn:
            from sqlalchemy import text
            row = conn.execute(text(
                "SELECT * FROM signals WHERE analysis_date = :d ORDER BY id DESC LIMIT 1"
            ), {'d': yesterday}).fetchone()
        _showing_stale = True

    if not row or not (row._mapping.get('confluence_grade')):
        st.info("Today's confluence data will appear after the morning analysis runs.")
    else:
        if _showing_stale:
            st.warning("No signal for today yet — showing yesterday's confluence data.")
        mapping = row._mapping
        grade = mapping.get('confluence_grade') or 'N/A'
        pct = mapping.get('confluence_pct') or 0
        signal = mapping.get('signal') or 'N/A'
        conf = mapping.get('confidence') or 0
        agrees = mapping.get('providers_agree') or False
        pos = mapping.get('recommended_position_pct') or 0
        summary = mapping.get('confluence_summary') or ''
        factors_json = mapping.get('confluence_factors') or '{}'
        analysis_date = mapping.get('analysis_date') or 'N/A'

        try:
            factors = json.loads(factors_json) if factors_json else {}
        except Exception:
            factors = {}

        # Grade colours
        grade_colors = {'A+': '#1a7f37', 'A': '#1a7f37', 'B': '#9a6700', 'C': '#cf222e', 'D': '#cf222e'}
        grade_color = grade_colors.get(grade, '#1a1a1a')

        st.caption(f"Signal date: {analysis_date}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:3rem;font-weight:bold;color:{grade_color}'>{grade}</span>"
                f"<br><span style='font-size:1.2rem'>{pct}%</span></div>",
                unsafe_allow_html=True
            )
        with col2:
            sig_color = '#1a7f37' if signal == 'BUY' else '#cf222e' if signal == 'SELL' else '#6e7781'
            agree_icon = 'Yes' if agrees else 'No'
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:2rem;font-weight:bold;color:{sig_color}'>{signal}</span>"
                f"<br>AI Confidence: {conf}/10<br>Providers agree: {agree_icon}</div>",
                unsafe_allow_html=True
            )
        with col3:
            pos_float = float(pos) if pos else 0.0
            pos_label = {1.5: 'Aggressive', 1.0: 'Full Size', 0.5: 'Half Size',
                         0.25: 'Minimum', 0.0: 'Avoid'}.get(pos_float, 'Unknown')
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:1.5rem;font-weight:bold'>{pos_label}</span>"
                f"<br>{pos}% risk</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Factor table
        if factors:
            st.subheader("Factor Breakdown")
            rows = []
            for fname, fdata in factors.items():
                if not isinstance(fdata, dict):
                    continue
                score = fdata.get('score')
                max_pts = fdata.get('max', 0)
                value = fdata.get('value', 'N/A')
                label = fdata.get('label', '')
                if score is None:
                    status = 'N/A'
                elif score == max_pts:
                    status = 'Full'
                elif score == 0:
                    status = 'Zero'
                else:
                    status = 'Partial'
                rows.append({
                    'Factor': fname.replace('_', ' ').title(),
                    'Value': str(value) if value else 'N/A',
                    'Score': f"{score}/{max_pts}" if score is not None else 'N/A',
                    'Status': status,
                    'Label': label,
                })

            df_factors = pd.DataFrame(rows)

            def highlight_row(row):
                if row['Status'] == 'Full':
                    return ['background-color: #d4edda'] * len(row)
                elif row['Status'] == 'Zero':
                    return ['background-color: #f8d7da'] * len(row)
                elif row['Status'] == 'Partial':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return ['background-color: #e9ecef'] * len(row)

            st.dataframe(
                df_factors[['Factor', 'Value', 'Score', 'Status', 'Label']].style.apply(highlight_row, axis=1),
                use_container_width=True
            )

        if summary:
            st.subheader("Analysis Summary")
            st.info(summary)

    st.markdown("---")

    # Historical performance section
    st.subheader("Historical Performance")
    with engine.connect() as conn:
        from sqlalchemy import text
        outcome_count = conn.execute(text("SELECT COUNT(*) FROM outcomes")).scalar()

    if outcome_count < 15:
        st.info(
            f"Historical analysis will appear once 15+ signal outcomes have been recorded.\n"
            f"Currently: {outcome_count} outcomes recorded.\n"
            f"Keep running the system daily -- this section unlocks automatically."
        )
    else:
        with engine.connect() as conn:
            from sqlalchemy import text
            df_hist = pd.read_sql(text("""
                SELECT s.confluence_grade, o.signal_correct as outcome
                FROM signals s
                JOIN outcomes o ON s.id = o.signal_id
                WHERE s.confluence_grade IS NOT NULL
            """), conn)

        if not df_hist.empty:
            grade_stats = df_hist.groupby('confluence_grade').apply(
                lambda x: pd.Series({
                    'win_rate': x['outcome'].astype(bool).mean() * 100,
                    'count': len(x)
                })
            ).reset_index()
            fig = px.bar(
                grade_stats, x='confluence_grade', y='win_rate',
                title='Win Rate by Confluence Grade',
                color='confluence_grade',
                color_discrete_map={'A+': '#1a7f37', 'A': '#2ea04f', 'B': '#9a6700',
                                    'C': '#cf222e', 'D': '#8b0000'},
                text='count',
            )
            fig.update_layout(yaxis_title='Win Rate %', xaxis_title='Grade',
                              plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font=dict(color="#e0e0e0"))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Signal history with confluence
    st.subheader("Signal History with Confluence")
    with engine.connect() as conn:
        from sqlalchemy import text
        df_signals = pd.read_sql(text("""
            SELECT s.analysis_date as date, s.signal, s.confidence as ai_confidence,
                   s.confluence_grade, s.confluence_pct,
                   o.signal_correct as outcome, o.pips_moved as pips_result
            FROM signals s
            LEFT JOIN outcomes o ON s.id = o.signal_id
            ORDER BY s.analysis_date DESC
            LIMIT 30
        """), conn)

    if not df_signals.empty:
        st.dataframe(df_signals, use_container_width=True)
    else:
        st.info("No signals recorded yet.")


# ---- Page 7: Account (Paper Trading) -----------------------------------------

def page_account():
    st.title("Paper Trading Account")
    st.caption("Virtual GBP 1000 account -- 1.5 pip spread -- all trades in GBP")

    engine = get_db()

    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            df_trades = pd.read_sql(text("""
                SELECT vt.*, s.analysis_date, s.signal AS sig_signal,
                       s.confluence_grade AS sig_grade
                FROM virtual_trades vt
                JOIN signals s ON vt.signal_id = s.id
                ORDER BY vt.opened_at DESC
            """), conn)

            account_row = conn.execute(text(
                "SELECT * FROM virtual_account ORDER BY id DESC LIMIT 1"
            )).fetchone()
    except Exception as e:
        st.warning(f"Could not load account data: {e}")
        st.info("No paper trades recorded yet. The first trade will appear after tomorrow's morning analysis.")
        return

    if df_trades.empty or account_row is None:
        st.info("No paper trades recorded yet. The first trade will appear after tomorrow's morning analysis.")
        return

    # ---- Top metrics row ------------------------------------------------
    STARTING = 1000.0
    balance   = float(account_row._mapping["balance"])
    delta_gbp = balance - STARTING
    ret_pct   = round((balance - STARTING) / STARTING * 100, 2)

    # Split real trades (B+) from shadow trades (C grade)
    real_trades    = df_trades[~df_trades["status"].str.startswith("shadow", na=False) & (~df_trades["status"].isin(["skipped", "skipped_conflict"]))]
    shadow_trades  = df_trades[df_trades["status"].str.startswith("shadow", na=False)]
    conflict_skips = df_trades[df_trades["status"] == "skipped_conflict"]
    pending_trades = real_trades[real_trades["status"] == "pending_entry"]
    cancelled_trades = real_trades[real_trades["status"] == "cancelled"]
    closed_trades  = real_trades[real_trades["status"].isin(["won", "lost", "expired"])]
    open_trades    = real_trades[real_trades["status"] == "open"]

    # Max drawdown
    max_dd = 0.0
    if not closed_trades.empty and "closing_balance" in closed_trades.columns:
        sorted_closed = closed_trades.sort_values("opened_at").copy()
        balances = [STARTING] + [float(b) for b in sorted_closed["closing_balance"] if b is not None]
        peak = STARTING
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Current Balance",
        f"GBP {balance:,.2f}",
        delta=f"{'+' if delta_gbp >= 0 else ''}{delta_gbp:.2f} GBP",
    )
    m2.metric("Total Return", f"{'+' if ret_pct >= 0 else ''}{ret_pct:.2f}%")
    m3.metric("Open Trades",  len(open_trades),    delta=f"{len(pending_trades)} awaiting fill" if len(pending_trades) else None)
    m4.metric("Max Drawdown", f"{max_dd:.1f}%")

    # Fill rate banner
    total_limit_orders = len(open_trades) + len(closed_trades) + len(cancelled_trades) + len(pending_trades)
    total_filled = len(open_trades) + len(closed_trades)
    if total_limit_orders > 0:
        fill_rate = round(total_filled / total_limit_orders * 100, 0)
        st.caption(f"Limit order fill rate: {total_filled}/{total_limit_orders} ({fill_rate:.0f}%) — {len(cancelled_trades)} cancelled unfilled")

    st.divider()

    # ---- Equity curve ---------------------------------------------------
    st.subheader("Equity Curve")
    if not closed_trades.empty:
        eq = closed_trades.sort_values("opened_at")[
            ["opened_at", "closing_balance", "status"]
        ].copy()
        eq["closing_balance"] = eq["closing_balance"].astype(float)
        eq["opened_at"] = pd.to_datetime(eq["opened_at"])

        # Build equity series starting from 1000
        dates    = [pd.Timestamp(eq["opened_at"].iloc[0]) - pd.Timedelta(days=1)] + list(eq["opened_at"])
        balances = [STARTING] + list(eq["closing_balance"])

        fig_eq = go.Figure()

        # Fill above starting line (green)
        fig_eq.add_trace(go.Scatter(
            x=dates, y=balances,
            fill="tozeroy",
            fillcolor="rgba(0,200,83,0.08)",
            line=dict(color="#00c853", width=2),
            name="Balance",
            mode="lines",
        ))

        # Starting line
        fig_eq.add_hline(
            y=STARTING,
            line_dash="dash",
            line_color="#9e9e9e",
            annotation_text="GBP 1000 start",
        )

        # Trade close dots
        won_rows  = eq[eq["status"] == "won"]
        lost_rows = eq[eq["status"].isin(["lost", "expired"])]

        if not won_rows.empty:
            fig_eq.add_trace(go.Scatter(
                x=won_rows["opened_at"],
                y=won_rows["closing_balance"].astype(float),
                mode="markers",
                marker=dict(color="#00c853", size=8, symbol="circle"),
                name="Won",
                showlegend=True,
            ))

        if not lost_rows.empty:
            fig_eq.add_trace(go.Scatter(
                x=lost_rows["opened_at"],
                y=lost_rows["closing_balance"].astype(float),
                mode="markers",
                marker=dict(color="#cf222e", size=8, symbol="circle"),
                name="Lost / Expired",
                showlegend=True,
            ))

        fig_eq.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            height=350,
            yaxis_title="Balance (GBP)",
            xaxis_title="Date",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.caption("Equity curve will appear once the first trade closes.")

    st.divider()

    # ---- Trade statistics -----------------------------------------------
    st.subheader("Trade Statistics")
    c1, c2 = st.columns(2)

    resolved = real_trades[real_trades["status"].isin(["won", "lost", "expired"])]
    won      = resolved[resolved["status"] == "won"]
    lost_exp = resolved[resolved["status"].isin(["lost", "expired"])]

    n_total   = len(real_trades)
    n_won     = len(won)
    n_lost    = len(resolved) - n_won
    n_expired = len(resolved[resolved["outcome_type"] == "expired"])
    win_rate  = round(n_won / len(resolved) * 100, 1) if len(resolved) > 0 else 0.0
    avg_win   = round(won["net_pnl_gbp"].astype(float).mean(), 2) if not won.empty else 0.0
    avg_loss  = round(lost_exp["net_pnl_gbp"].astype(float).mean(), 2) if not lost_exp.empty else 0.0

    with c1:
        st.write(f"**Total trades:** {n_total}")
        st.write(f"**Won / Lost / Expired:** {n_won} / {n_lost - n_expired} / {n_expired}")
        st.write(f"**Win rate:** {win_rate}%")
        st.write(f"**Average win:** GBP {avg_win:+.2f}")
        st.write(f"**Average loss:** GBP {avg_loss:+.2f}")

    total_gross = resolved["gross_pnl_gbp"].astype(float).sum() if not resolved.empty else 0.0
    total_spread = resolved["spread_cost_gbp"].astype(float).sum() if not resolved.empty else 0.0
    total_net   = resolved["net_pnl_gbp"].astype(float).sum() if not resolved.empty else 0.0
    best_trade  = resolved["net_pnl_gbp"].astype(float).max() if not resolved.empty else 0.0
    worst_trade = resolved["net_pnl_gbp"].astype(float).min() if not resolved.empty else 0.0

    # Profit factor
    gross_wins   = won["net_pnl_gbp"].astype(float).sum()   if not won.empty      else 0.0
    gross_losses = abs(lost_exp["net_pnl_gbp"].astype(float).sum()) if not lost_exp.empty else 0.0
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf")

    # Consecutive win/loss streak
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    if not resolved.empty:
        for status in resolved.sort_values("opened_at")["status"]:
            if status == "won":
                cur_win  += 1; cur_loss = 0
            else:
                cur_loss += 1; cur_win  = 0
            max_win_streak  = max(max_win_streak,  cur_win)
            max_loss_streak = max(max_loss_streak, cur_loss)

    with c2:
        st.write(f"**Total gross P&L:** GBP {total_gross:+.2f}")
        st.write(f"**Total spread costs paid:** GBP {total_spread:.2f}")
        st.write(f"**Total net P&L:** GBP {total_net:+.2f}")
        st.write(f"**Best trade:** GBP {best_trade:+.2f}")
        st.write(f"**Worst trade:** GBP {worst_trade:+.2f}")
        pf_str = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
        st.write(f"**Profit factor:** {pf_str}  *(>1.5 = good)*")
        st.write(f"**Max win streak:** {max_win_streak}  |  **Max loss streak:** {max_loss_streak}")

    st.divider()

    # ---- Recent trades table --------------------------------------------
    st.subheader("Recent Trades (last 20 — Grade B+ only)")

    display = real_trades.head(20).copy()

    # Build display columns
    cols_show = []
    for col, label in [
        ("opened_at",       "Date"),
        ("direction",       "Direction"),
        ("order_type",      "Order"),
        ("confluence_grade","Grade"),
        ("entry_price",     "Entry"),
        ("fill_price",      "Fill"),
        ("sl_pips",         "SL Pips"),
        ("tp_pips",         "TP Pips"),
        ("risk_gbp",        "Risk GBP"),
        ("status",          "Result"),
        ("net_pnl_gbp",     "P&L"),
        ("closing_balance", "Balance"),
    ]:
        if col in display.columns:
            display = display.rename(columns={col: label})
            cols_show.append(label)

    for c in ["P&L", "Risk GBP", "Entry"]:
        if c in display.columns:
            display[c] = pd.to_numeric(display[c], errors="coerce").round(2)
    for c in ["SL Pips", "TP Pips"]:
        if c in display.columns:
            display[c] = pd.to_numeric(display[c], errors="coerce").round(1)

    available_cols = [c for c in cols_show if c in display.columns]

    def row_style(row):
        status = str(row.get("Result", "")).lower()
        if status == "won":
            return ["background-color: #d4edda"] * len(row)
        if status == "lost":
            return ["background-color: #f8d7da"] * len(row)
        if status == "open":
            return ["background-color: #fff3cd"] * len(row)
        if status == "pending_entry":
            return ["background-color: #cce5ff"] * len(row)   # blue -- awaiting fill
        if status == "cancelled":
            return ["background-color: #f5f5f5"] * len(row)   # grey -- unfilled
        return ["background-color: #e9ecef"] * len(row)

    try:
        styled = display[available_cols].style.apply(row_style, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(display[available_cols], use_container_width=True, hide_index=True)

    st.divider()

    # ---- Shadow trades (Grade C) ----------------------------------------
    st.divider()
    st.subheader("Grade C Shadow Trades")
    st.caption("These trades were NOT placed — tracked for comparison only. Would Grade C signals have been profitable?")

    if shadow_trades.empty:
        st.info("No Grade C shadow trades recorded yet.")
    else:
        shadow_resolved = shadow_trades[shadow_trades["status"].isin(["shadow_won", "shadow_lost", "shadow_expired"])]
        shadow_open     = shadow_trades[shadow_trades["status"] == "shadow"]

        if not shadow_resolved.empty:
            s_won  = len(shadow_resolved[shadow_resolved["status"] == "shadow_won"])
            s_lost = len(shadow_resolved) - s_won
            s_wr   = round(s_won / len(shadow_resolved) * 100, 1)
            s_pnl  = shadow_resolved["net_pnl_gbp"].astype(float).sum()
            s_avg  = shadow_resolved["net_pnl_gbp"].astype(float).mean()

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Shadow Trades", len(shadow_resolved))
            sc2.metric("Shadow Win Rate", f"{s_wr}%")
            sc3.metric("Total Shadow P&L", f"GBP {s_pnl:+.2f}")
            sc4.metric("Avg per Trade", f"GBP {s_avg:+.2f}")

            verdict = ""
            if len(shadow_resolved) >= 5:
                if s_wr >= 55 and s_pnl > 0:
                    verdict = "Grade C signals are profitable in shadow tracking. Consider lowering the minimum grade filter to C."
                elif s_wr < 45 or s_pnl < 0:
                    verdict = "Grade C signals are losing money in shadow tracking. The B+ filter is working correctly."
                else:
                    verdict = "Grade C signals are borderline. More data needed before drawing conclusions."
                st.info(verdict)

        # Shadow trade table
        shadow_display = shadow_trades.head(20).copy()
        for col, label in [("opened_at","Date"),("direction","Direction"),
                           ("confluence_grade","Grade"),("sl_pips","SL Pips"),
                           ("tp_pips","TP Pips"),("status","Result"),
                           ("net_pnl_gbp","Shadow P&L")]:
            if col in shadow_display.columns:
                shadow_display = shadow_display.rename(columns={col: label})

        show_cols = [c for c in ["Date","Direction","Grade","SL Pips","TP Pips","Result","Shadow P&L"]
                     if c in shadow_display.columns]
        if "Shadow P&L" in shadow_display.columns:
            shadow_display["Shadow P&L"] = pd.to_numeric(shadow_display["Shadow P&L"], errors="coerce").round(2)

        st.dataframe(shadow_display[show_cols], use_container_width=True, hide_index=True)

    # ---- Conflict skips -------------------------------------------------
    st.divider()
    st.subheader("Skipped — Conflicting Signal")
    st.caption("New signal was the opposite direction to an open trade. No trade placed to avoid holding opposing positions.")

    if conflict_skips.empty:
        st.info("No conflict skips recorded yet.")
    else:
        skip_display = conflict_skips.head(20).copy()
        for col, label in [("opened_at","Date"),("direction","Would-be Direction"),
                           ("confluence_grade","Grade"),("sl_pips","SL Pips"),("tp_pips","TP Pips")]:
            if col in skip_display.columns:
                skip_display = skip_display.rename(columns={col: label})
        show_cols = [c for c in ["Date","Would-be Direction","Grade","SL Pips","TP Pips"]
                     if c in skip_display.columns]
        st.dataframe(skip_display[show_cols], use_container_width=True, hide_index=True)

    st.divider()

    # ---- Grade performance table ----------------------------------------
    st.subheader("Performance by Confluence Grade (Real Trades)")

    grade_col = "confluence_grade"
    if grade_col in df_trades.columns and not resolved.empty:
        resolved_copy = resolved.copy()
        resolved_copy["net_pnl_gbp"] = pd.to_numeric(resolved_copy["net_pnl_gbp"], errors="coerce")

        grade_stats = resolved_copy.groupby(grade_col).apply(
            lambda g: pd.Series({
                "Trades":      len(g),
                "Win Rate %":  round((g["status"] == "won").mean() * 100, 1),
                "Avg P&L GBP": round(g["net_pnl_gbp"].mean(), 2),
                "Total P&L GBP": round(g["net_pnl_gbp"].sum(), 2),
            })
        ).reset_index().rename(columns={grade_col: "Grade"})

        st.dataframe(grade_stats, use_container_width=True, hide_index=True)
    else:
        st.caption("Grade performance will appear once trades with confluence grades have closed.")


# ---- Page: News & Calendar ---------------------------------------------------

def page_news():
    st.title("News & Calendar")
    st.caption("High-impact GBP and USD events this week -- source: Forex Factory")

    try:
        from tracker.news_calendar import assess_news_risk, get_week_events

        risk = assess_news_risk()
        rl = risk["risk_level"]

        # Risk banner
        if rl == "binary":
            st.error(f"BINARY EVENT TODAY -- DO NOT TRADE\n{risk['warning_message']}")
        elif rl == "high":
            st.error(f"HIGH NEWS RISK -- {risk['warning_message']}")
        elif rl == "medium":
            st.warning(f"MEDIUM RISK -- {risk['warning_message']}")
        elif rl == "low":
            st.info(f"Low risk. {risk['warning_message']}")
        else:
            st.success("CLEAR -- No high-impact events today. Safe to trade.")

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Today's Risk", rl.upper())
        c2.metric("High Impact Today", len(risk["high_impact_today"]))
        c3.metric("High Impact Tomorrow", len(risk["high_impact_tomorrow"]))

        st.divider()

        # Today's events
        st.subheader("Today's Events")
        if risk["all_today"]:
            rows = []
            for e in risk["all_today"]:
                rows.append({
                    "Time": e.get("date", "")[:16].replace("T", " "),
                    "Currency": e.get("country", ""),
                    "Event": e.get("title", ""),
                    "Impact": e.get("impact", ""),
                    "Previous": e.get("previous", "") or "-",
                    "Forecast": e.get("estimate", "") or "-",
                    "Actual": e.get("actual", "") or "Pending",
                })
            df_today = pd.DataFrame(rows)

            def colour_impact(row):
                if row.get("Impact") == "High":
                    return ["background-color: #f8d7da"] * len(row)
                elif row.get("Impact") == "Medium":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            try:
                st.dataframe(df_today.style.apply(colour_impact, axis=1),
                             use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(df_today, use_container_width=True, hide_index=True)
        else:
            st.info("No GBP or USD events today.")

        st.divider()

        # This week's high-impact events
        st.subheader("This Week -- High Impact Only")
        week_events = get_week_events()
        all_high = []
        for date_str, evts in sorted(week_events.items()):
            for e in evts:
                if e.get("impact") == "High":
                    all_high.append({
                        "Date": date_str,
                        "Time": e.get("date", "")[:16].replace("T", " "),
                        "Currency": e.get("country", ""),
                        "Event": e.get("title", ""),
                        "Previous": e.get("previous", "") or "-",
                        "Forecast": e.get("estimate", "") or "-",
                        "Actual": e.get("actual", "") or "Pending",
                    })

        if all_high:
            st.dataframe(pd.DataFrame(all_high), use_container_width=True, hide_index=True)
        else:
            st.info("No high-impact GBP or USD events this week.")

        st.caption("Data refreshed each page load from Forex Factory")

    except Exception as e:
        st.error(f"Could not load calendar: {e}")


# ---- Main router -------------------------------------------------------------

def main():
    get_db()  # ensure tables exist
    page, days, min_conf = sidebar()

    if page == "Today":
        page_today(days, min_conf)
    elif page == "Signal History":
        page_history(days, min_conf)
    elif page == "Analytics":
        page_analytics(days, min_conf)
    elif page == "Confluence":
        page_confluence()
    elif page == "Account":
        page_account()
    elif page == "Costs":
        page_costs()
    elif page == "News & Calendar":
        page_news()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
