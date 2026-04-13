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

    page = st.sidebar.radio("Navigation", ["Today", "Signal History", "Analytics", "Costs", "Settings"])

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

    # Row 5: AI pattern analysis
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
        if st.button("Export Full Database"):
            signals      = load_signals(days=365)
            outcomes_map = load_outcomes()
            df = signals_to_df(signals, outcomes_map)
            csv = df.to_csv(index=False)
            st.download_button("Download signals.csv", csv, "signals_full.csv", "text/csv")

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
    elif page == "Costs":
        page_costs()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
