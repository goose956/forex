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
            "Outcome":     ("Win" if o.signal_correct else "Loss") if o else "Pending",
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
        })
    return pd.DataFrame(data)


# ---- Sidebar -----------------------------------------------------------------

def sidebar():
    st.sidebar.title("Forex Signal Tracker")
    st.sidebar.caption("GBP/USD AI Analysis System")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", ["Today", "Signal History", "Analytics", "Confluence", "Account", "Costs", "Settings"])

    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    days = st.sidebar.slider("Show last N days", 7, 180, 30)
    min_conf = st.sidebar.slider("Min confidence", 1, 10, 1)

    st.sidebar.divider()
    db_status = "🟢 Railway Connected" if not is_sqlite() else "🟡 Using Local SQLite"
    st.sidebar.caption(db_status)
    st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return page, days, min_conf


# ---- Page 1: Today -----------------------------------------------------------

def page_today(days, min_conf):
    st.title("Today's Signal")

    signals      = load_signals(days=1)
    all_signals  = load_signals(days=days)
    outcomes_map = load_outcomes()

    # Today's signal
    today_signals = [s for s in signals if s.analysis_date == date.today()]

    if today_signals:
        sig = today_signals[0]
        o   = outcomes_map.get(sig.id)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            css_class = f"signal-{(sig.signal or 'hold').lower()}"
            st.markdown(f'<div class="{css_class}">{sig.signal or "HOLD"}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{sig.confidence or 0}/10")
            st.metric("Entry",       f"{float(sig.entry_price  or 0):.5f}")
            st.metric("Stop Loss",   f"{float(sig.stop_loss    or 0):.5f}")
            st.metric("Take Profit", f"{float(sig.take_profit  or 0):.5f}")
            st.metric("Risk/Reward", f"1:{float(sig.risk_reward or 0):.1f}")

        st.subheader("Provider Comparison")
        ca, ga = st.columns(2)
        with ca:
            st.metric("Claude", sig.claude_signal or "N/A", f"{sig.claude_confidence or 0}/10")
        with ga:
            st.metric("GPT-4o", sig.gpt_signal or "N/A", f"{sig.gpt_confidence or 0}/10")

        if sig.providers_agree:
            st.success("Both providers agree.")
        else:
            st.warning("Providers disagree -- confidence penalised.")

        if sig.primary_reason:
            with st.expander("Primary reason"):
                st.write(sig.primary_reason)

    else:
        st.info("No signal for today yet.")
        st.caption("Expected: 07:30 London time (run via GitHub Actions or manually).")
        if st.button("Run Manual Analysis Now"):
            st.info("To run manually, double-click:  scheduler/run_manual_analysis.bat")

    st.divider()

    # Key metrics
    st.subheader("Key Metrics")
    df = signals_to_df(all_signals, outcomes_map)

    if not df.empty:
        resolved = df[df["Outcome"] != "Pending"]
        wins     = resolved[resolved["Outcome"] == "Win"]
        win_rate = round(len(wins) / len(resolved) * 100, 1) if len(resolved) > 0 else 0
        total_pips = resolved["Pips"].sum() if "Pips" in resolved.columns else 0

        costs = load_costs(days=30)
        mtd   = sum(float(c.cost_gbp or 0) for c in costs)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", len(df))
        m2.metric("Win Rate",      f"{win_rate}%")
        m3.metric("Paper Pips",    f"{total_pips:+.0f}")
        m4.metric("Month Cost",    f"GBP {mtd:.2f}")

    # Recent performance table
    st.subheader("Recent Performance (14 days)")
    recent = load_signals(days=14)
    if recent:
        rdf = signals_to_df(recent, outcomes_map)[
            ["Date","Signal","Conf","Entry","SL","TP","Outcome","Pips","Agree"]
        ]
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    # Price chart
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
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sig_filter = st.selectbox("Signal", ["All", "BUY", "SELL", "HOLD"])
    with f2:
        out_filter = st.selectbox("Outcome", ["All", "Win", "Loss", "Pending"])
    with f3:
        prov_filter = st.selectbox("Agreement", ["All", "Yes", "No"])
    with f4:
        conf_filter = st.slider("Min confidence", 1, 10, min_conf, key="hist_conf")

    filtered = df.copy()
    if sig_filter != "All":
        filtered = filtered[filtered["Signal"] == sig_filter]
    if out_filter != "All":
        filtered = filtered[filtered["Outcome"] == out_filter]
    if prov_filter != "All":
        filtered = filtered[filtered["Agree"] == prov_filter]
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

    for _, row in filtered.head(50).iterrows():
        outcome_icon = "✅" if row["Outcome"] == "Win" else ("❌" if row["Outcome"] == "Loss" else "⏳")
        label = (f"{row['Date']}  |  {row['Signal']}  |  Conf: {row['Conf']}/10  "
                 f"|  Agree: {row['Agree']}  |  {outcome_icon} {row['Outcome']}")
        if row.get("Pips") is not None:
            label += f"  ({row['Pips']:+.0f} pips)"
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
    resolved     = df[df["Outcome"] != "Pending"] if not df.empty else pd.DataFrame()

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
            fig_prov.update_layout(paper_bgcolor="white", plot_bgcolor="#f8f9fa",
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
    if not row or not (row._mapping.get('confluence_grade')):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with engine.connect() as conn:
            from sqlalchemy import text
            row = conn.execute(text(
                "SELECT * FROM signals WHERE analysis_date = :d ORDER BY id DESC LIMIT 1"
            ), {'d': yesterday}).fetchone()

    if not row or not (row._mapping.get('confluence_grade')):
        st.info("Today's confluence data will appear after the morning analysis runs.")
    else:
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
    real_trades   = df_trades[~df_trades["status"].str.startswith("shadow", na=False) & (df_trades["status"] != "skipped")]
    shadow_trades = df_trades[df_trades["status"].str.startswith("shadow", na=False)]
    closed_trades = real_trades[real_trades["status"].isin(["won", "lost", "expired"])]
    open_trades   = real_trades[real_trades["status"] == "open"]

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
    m2.metric(
        "Total Return",
        f"{'+' if ret_pct >= 0 else ''}{ret_pct:.2f}%",
    )
    m3.metric("Open Trades", len(open_trades))
    m4.metric("Max Drawdown", f"{max_dd:.1f}%")

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

    with c2:
        st.write(f"**Total gross P&L:** GBP {total_gross:+.2f}")
        st.write(f"**Total spread costs paid:** GBP {total_spread:.2f}")
        st.write(f"**Total net P&L:** GBP {total_net:+.2f}")
        st.write(f"**Best trade:** GBP {best_trade:+.2f}")
        st.write(f"**Worst trade:** GBP {worst_trade:+.2f}")

    st.divider()

    # ---- Recent trades table --------------------------------------------
    st.subheader("Recent Trades (last 20 — Grade B+ only)")

    display = real_trades.head(20).copy()

    # Build display columns
    cols_show = []
    for col, label in [
        ("opened_at",       "Date"),
        ("direction",       "Direction"),
        ("confluence_grade","Grade"),
        ("entry_price",     "Entry"),
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
        if status in ("lost",):
            return ["background-color: #f8d7da"] * len(row)
        if status == "open":
            return ["background-color: #fff3cd"] * len(row)
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
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
