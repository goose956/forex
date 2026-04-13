# Quick Start — Forex Signal Tracker

## Every day (automatic)
GitHub Actions runs at 07:00 UTC weekdays. Nothing to do.

## View today's signal
Double-click: `scheduler\start_dashboard.bat`
Then open: http://localhost:8501

## Run manually (if needed)
Double-click: `scheduler\run_manual_analysis.bat`

## Run a backtest
Double-click: `scheduler\run_backtest.bat`
Enter a past date when prompted.

## Activate environment (for terminal use)
```cmd
cd "C:\Users\User\Desktop\mission contol\projects\forex-signal-tracker"
venv\Scripts\activate
```

## Interpret the signal

| Field | Meaning |
|-------|---------|
| Signal | BUY / SELL / HOLD |
| Confidence | 1-10. Higher = stronger conviction |
| Entry | Current price — where you'd enter |
| Stop Loss | Where to exit if wrong |
| Take Profit | Target — minimum 2x stop distance |
| R:R | Risk/Reward ratio (aim for 1:2 or better) |
| Agreement | YES = both Claude and GPT agree |

**Do not act on HOLD signals.**
**Treat low confidence (<6) and disagreement signals with extra caution.**

## Costs
Dashboard → Costs page shows all spending.
Roughly GBP 0.50 per run, ~GBP 11/month.

## Update the project
```cmd
git pull
venv\Scripts\activate
pip install -r requirements.txt
```

## Security reminder
Never share your `.env` file. Never paste API keys in chat.
