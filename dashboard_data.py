"""Data helpers for the dashboard. No Streamlit imports here, so this file can be tested on its own."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("EGX_DATA_DIR") or Path(__file__).resolve().parent / "data")

PRED_CSV = DATA_DIR / "predictions.csv"
HISTORY_CSV = DATA_DIR / "dashboard_history.csv"
SCORES_CSV = DATA_DIR / "scores_history.csv"
REPORT_JSON = DATA_DIR / "model_report.json"
STATUS_JSON = DATA_DIR / "update_status.json"
BACKTEST_CSV = DATA_DIR / "backtest.csv"

RATINGS = ["Very Strong", "Strong", "Average", "Weak", "Very Weak"]
RATING_EMOJI = {"Very Strong": "🟢", "Strong": "🔵", "Average": "⚪", "Weak": "🟠", "Very Weak": "🔴"}


def files_signature():
    """Modification times of the data files. Used as the cache key: when a file changes, the data reloads."""
    return tuple(p.stat().st_mtime_ns if p.exists() else 0
                 for p in (PRED_CSV, HISTORY_CSV, SCORES_CSV, REPORT_JSON, STATUS_JSON, BACKTEST_CSV))


def _read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_all():
    missing = [p.name for p in (PRED_CSV, HISTORY_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    pred = pd.read_csv(PRED_CSV, parse_dates=["Date"])
    hist = pd.read_csv(HISTORY_CSV, parse_dates=["Date"])
    if SCORES_CSV.exists():
        scores = pd.read_csv(SCORES_CSV, parse_dates=["Date"])
    else:
        scores = pd.DataFrame(columns=["Date", "Ticker", "Score", "Rank", "Rating"])
    backtest = pd.read_csv(BACKTEST_CSV, parse_dates=["Date"]) if BACKTEST_CSV.exists() else pd.DataFrame()
    return {
        "predictions": pred.sort_values("Rank").reset_index(drop=True),
        "history": hist,
        "scores": scores,
        "report": _read_json(REPORT_JSON),
        "status": _read_json(STATUS_JSON),
        "backtest": backtest,
    }


def ranking_view(pred, query="", ratings=None):
    """The table shown on the main page: sorted by Rank (best first), optionally filtered."""
    df = pred.sort_values("Rank")
    if ratings:
        df = df[df["Rating"].isin(ratings)]
    q = (query or "").strip().lower()
    if q:
        in_ticker = df["Ticker"].astype(str).str.lower().str.contains(q, regex=False)
        in_name = df["Company_Name"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        df = df[in_ticker | in_name]
    out = pd.DataFrame({
        "Rank": df["Rank"],
        "Ticker": df["Ticker"],
        "Company": df["Company_Name"],
        "Rating": [RATING_EMOJI.get(r, "") + " " + str(r) for r in df["Rating"]],
        "Score": df["Score"],
        "Close": df["Close"],
        "1d %": df["Daily_Return_%"],
        "5d %": df["Return_5d_%"],
        "20d %": df["Return_20d_%"],
        "RSI": df["RSI_14"],
    })
    return out.reset_index(drop=True)


def ticker_history(hist, ticker, days):
    h = hist[hist["Ticker"] == ticker].sort_values("Date")
    return h.tail(days).reset_index(drop=True)


def ticker_rank_history(scores, ticker):
    s = scores[scores["Ticker"] == ticker].sort_values("Date")
    return s.reset_index(drop=True)


def data_age_days(asof):
    today = pd.Timestamp.now(tz="Africa/Cairo").tz_localize(None).normalize()
    return int((today - pd.Timestamp(asof).normalize()).days)


def build_warnings(status, report, asof):
    """Plain-language warnings shown at the top of the dashboard."""
    msgs = []
    age = data_age_days(asof)
    if age >= 5:
        msgs.append(f"The latest data is {age} days old. The daily update may have failed - check the GitHub Actions tab.")
    if status and status.get("news_ok") is False:
        msgs.append("News data could not be fetched in the last update. Sentiment values for the newest days are "
                    "carried forward from the last available day, so rankings may be slightly less accurate.")
    stale = (status or {}).get("stale_tickers") or {}
    if stale:
        msgs.append(f"{len(stale)} stock(s) have no recent price data: {', '.join(list(stale)[:8])}.")
    if report.get("market_score_above_test_range"):
        msgs.append("Today's average score is above the range the model saw during testing. "
                    "Treat the scores with extra caution.")
    return msgs


# ----------------------------------------------------------------------------- track record
TOP = ["Very Strong", "Strong"]
BOTTOM = ["Weak", "Very Weak"]
HORIZON = 10


def call_result(df):
    """'right' when a top-40% stock beat the market or a bottom-40% stock lagged it, 'wrong' otherwise, '-' for Average."""
    top, bottom = df["Rating"].isin(TOP), df["Rating"].isin(BOTTOM)
    right = (top & (df["Excess"] > 0)) | (bottom & (df["Excess"] < 0))
    return np.select([right, top | bottom], ["right", "wrong"], default="-")


def hit_summary(df):
    if df is None or df.empty:
        return None
    res = call_result(df)
    calls = int((res != "-").sum())
    if calls == 0:
        return None
    right = int((res == "right").sum())
    return {"calls": calls, "right": right, "hit_rate": right / calls * 100}


def rating_table(df):
    g = df.groupby("Rating")
    out = pd.DataFrame({"n": g.size(), "avg_return": g["Future_Return_10d_%"].mean(),
                        "avg_excess": g["Excess"].mean()}).reindex(RATINGS).rename_axis("Rating")
    out["n"] = out["n"].fillna(0).astype(int)
    return out


def stock_track_record(bt, ticker):
    """Test-period record (backtest.csv) for one stock."""
    if bt is None or bt.empty:
        return None
    s = bt[bt["Ticker"] == ticker].sort_values("Date")
    if s.empty:
        return None
    return {"rows": s, "hit": hit_summary(s), "market_hit": hit_summary(bt),
            "by_rating": rating_table(s), "days": int(len(s))}


def live_results(scores, hist, horizon=HORIZON):
    """Past daily scores joined with what really happened over the next `horizon` trading days."""
    if scores is None or scores.empty:
        return pd.DataFrame()
    px = hist[["Date", "Ticker", "Close"]].sort_values(["Ticker", "Date"]).copy()
    px["Future_Return_10d_%"] = (px.groupby("Ticker")["Close"].shift(-horizon) / px["Close"] - 1) * 100
    m = scores.merge(px[["Date", "Ticker", "Future_Return_10d_%"]], on=["Date", "Ticker"], how="left")
    m = m.dropna(subset=["Future_Return_10d_%"]).copy()
    if m.empty:
        return m
    m["Excess"] = m["Future_Return_10d_%"] - m.groupby("Date")["Future_Return_10d_%"].transform("mean")
    return m


def calls_table(df, n=30):
    t = df.sort_values("Date", ascending=False).head(n)
    return pd.DataFrame({
        "Date": t["Date"].dt.date,
        "Rating": [RATING_EMOJI.get(r, "") + " " + str(r) for r in t["Rating"]],
        "Score": t["Score"].round(3),
        "Actual 10d %": t["Future_Return_10d_%"].round(2),
        "vs market %": t["Excess"].round(2),
        "Result": [{"right": "\u2705 right", "wrong": "\u274c wrong"}.get(x, "-") for x in call_result(t)],
    }).reset_index(drop=True)
