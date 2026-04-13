# Forex Signal Tracker

AI-powered GBP/USD signal system built on TradingAgents.
Generates daily BUY/SELL/HOLD signals using Claude + GPT-4o.
Tracks outcomes, measures performance, and improves over time.

---

## How It Works

```
GitHub Actions (cloud, daily at 07:00 UTC, weekdays)
    |
    +--> run_daily.py
          |
          +--> DataCollector      (fetches price data + news)
          +--> ForexTradingAgents (Claude analysis)
          +--> ForexTradingAgents (GPT-4o analysis)
          +--> compare + combine signals
          +--> calculate entry / SL / TP
          +--> save to Railway PostgreSQL
          +--> write daily report file
          |
          +--> update_outcomes.py (resolves signals from 5 days ago)

Your PC (any time you want to view results)
    |
    +--> streamlit run dashboard/app.py --> http://localhost:8501
```

**Your PC is not required for the daily signal to run.**
GitHub Actions handles everything automatically.

---

## Quick Start

### 1. Activate environment
```cmd
cd "C:\Users\User\Desktop\mission contol\projects\forex-signal-tracker"
venv\Scripts\activate
```

### 2. Open the dashboard
```cmd
scheduler\start_dashboard.bat
```
Or manually:
```cmd
streamlit run dashboard\app.py
```

### 3. Run a manual analysis
```cmd
scheduler\run_manual_analysis.bat
```
Or from terminal:
```cmd
python scripts\run_daily.py
```

### 4. Run a backtest
```cmd
scheduler\run_backtest.bat
```
Or from terminal:
```cmd
python scripts\run_daily.py --date 2024-06-15
```

---

## Set Up GitHub Actions (Automated Daily Runs)

This makes the signal run every weekday morning automatically,
whether your PC is on or off.

**Step 1 — Create a free GitHub account** (if you don't have one)
- Go to github.com → Sign up

**Step 2 — Create a new PRIVATE repository**
- github.com → New repository
- Name: `forex-signal-tracker`
- Set to **Private** (keeps your strategy private)
- Do NOT add README or .gitignore (we already have them)

**Step 3 — Push this project to GitHub**
```cmd
cd "C:\Users\User\Desktop\mission contol\projects\forex-signal-tracker"
git init
git add .
git commit -m "Initial setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/forex-signal-tracker.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your actual GitHub username.

**Step 4 — Add your API keys as GitHub Secrets**
1. Go to your repository on GitHub
2. Click **Settings** tab
3. Left sidebar: **Secrets and variables** → **Actions**
4. Click **New repository secret** for each key:

| Secret name | Value |
|-------------|-------|
| `ANTHROPIC_API_KEY` | Your Anthropic key |
| `OPENAI_API_KEY` | Your OpenAI key |
| `ALPHA_VANTAGE_API_KEY` | Your Alpha Vantage key |
| `DATABASE_URL` | Your Railway PostgreSQL URL (see below) |

**Step 5 — Enable Actions**
- Go to the **Actions** tab in your repository
- Click "I understand my workflows, go ahead and enable them"

**Step 6 — Monitor runs**
- Actions tab → "Daily Forex Signal"
- Each weekday morning you'll see a new run
- GitHub emails you if any run fails

**Step 7 — Trigger manually**
- Actions tab → "Daily Forex Signal" → "Run workflow" button
- Useful for testing before the first scheduled run

---

## Set Up Railway Database

Railway is free-tier PostgreSQL that stores all your signals permanently.

**Step 1 — Create Railway account**
- Go to railway.app → Sign in with GitHub

**Step 2 — Create a new project**
- New Project → Provision PostgreSQL
- Wait ~30 seconds for it to start

**Step 3 — Get your DATABASE_URL**
- Click the PostgreSQL service
- Go to **Connect** tab
- Copy the **Database URL** (starts with `postgresql://`)
- Add it to your `.env` file:
  ```
  DATABASE_URL=postgresql://...your url...
  ```
- Also add it as `DATABASE_URL` in GitHub Secrets (Step 4 above)

**Step 4 — Test the connection**
```cmd
venv\Scripts\activate
python -c "from tracker.database import test_connection; test_connection()"
```
Should show: `Database connection OK -- Railway PostgreSQL`

---

## Deploy Dashboard to Railway (Optional)

Makes the dashboard accessible from your phone or any device.

1. Go to railway.app → Your project
2. Click **New Service** → **GitHub Repo**
3. Select `forex-signal-tracker`
4. Railway will detect the `Procfile` and deploy automatically
5. Add the same environment variables in Railway's Variables tab
6. Your dashboard will be live at `your-project.railway.app`

---

## File Map

| File / Folder | Purpose |
|---------------|---------|
| `scripts/run_daily.py` | Master daily analysis script |
| `scripts/update_outcomes.py` | Auto-closes signals from 5 days ago |
| `scripts/weekly_report.py` | Weekly AI performance analysis |
| `tracker/database.py` | DB connection, ORM models, all tables |
| `tracker/data_collector.py` | Price data, indicators, news, calendar |
| `tracker/agents/forex_agents.py` | Forex-optimised TradingAgents wrapper |
| `dashboard/app.py` | Streamlit 5-page dashboard |
| `scheduler/*.bat` | Windows shortcut launchers |
| `.github/workflows/` | GitHub Actions automation |
| `tracker/reports/daily/` | Daily report text files |
| `tracker/reports/weekly/` | Weekly report text files |
| `tracker/logs/` | Log files (daily, outcomes, db, weekly) |
| `.env` | Your API keys (never commit) |

---

## Estimated Monthly Costs

Assuming 22 weekday runs per month, two providers per run:

| Provider | Model | Cost per run | Monthly (22 runs) |
|----------|-------|-------------|-------------------|
| Claude | claude-sonnet-4-6 | ~GBP 0.20 | ~GBP 4.40 |
| GPT-4o | gpt-4o | ~GBP 0.30 | ~GBP 6.60 |
| **Total** | | **~GBP 0.50/run** | **~GBP 11.00** |

Costs are tracked in the dashboard under "Costs" page.

Railway free tier: 500 hours/month (enough for the database; dashboard may need paid tier if always-on).

---

## Security

- `.env` is in `.gitignore` — never committed to GitHub
- API keys only in `.env` locally and GitHub Secrets in the cloud
- Never paste keys into code or terminal output
- Keep your GitHub repository **Private**
