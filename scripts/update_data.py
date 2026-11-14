"""Step 4 - تحديث الداتا (محوّل من 04_update_data.ipynb). بيطلّع data/egx_raw.csv و data/update_status.json"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from gdeltdoc import GdeltDoc, Filters

pd.set_option("display.width", 200)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # <repo>/data
DATA_DIR.mkdir(parents=True, exist_ok=True)
OLD_CSV  = DATA_DIR / "EGX_Merged_News_Econ_Dataset.csv"   # الداتا الأصلية (Milestone 2)
RAW_CSV  = DATA_DIR / "egx_raw.csv"                    # الملف اللي بيتحدث كل يوم
STATUS_JSON = DATA_DIR / "update_status.json"

EXCLUDE_TICKERS = ["EGBE.CA"]      # سهم داتاه بايظة (زي النوت)
OVERLAP_DAYS    = 30               # بننزّل آخر 30 يوم تاني عشان نتأكد إن الأسعار القديمة ما اتغيرتش
FULL_START      = "2020-01-01"     # لو سهم اتغيرت أسعاره نزّله من الأول
ADJ_TOLERANCE   = 0.005            # لو الفرق في السعر القديم أكتر من 0.5% نعتبره اتعدّل
CAIRO_CLOSE_HOUR = 15.5            # البورصة بتقفل 2:30 م، بعد 3:30 م بنعتبر شمعة النهارده كاملة

RAW_COLS = ["Date", "Ticker", "Company_Name", "Open", "High", "Low", "Close", "Volume",
            "USD_EGP", "Brent_Oil", "News_Volume", "Sentiment_Score"]

# نفس مواضيع الأخبار اللي في Milestone 2
NEWS_TOPICS = {
    "egypt_economy": "Egypt economy",
    "egyptian_pound": "Egyptian pound",
    "global_events": ["oil prices", "Suez Canal", "Egypt inflation", "Egypt interest rate"],
}

now_cairo = pd.Timestamp.now(tz="Africa/Cairo")
today = now_cairo.tz_localize(None).normalize()
market_closed = (now_cairo.hour + now_cairo.minute / 60) >= CAIRO_CLOSE_HOUR
print("Cairo time:", now_cairo.strftime("%Y-%m-%d %H:%M"), "| market closed:", market_closed)

src = RAW_CSV if RAW_CSV.exists() else OLD_CSV
print("Reading:", src)

old = pd.read_csv(src, usecols=lambda c: c in RAW_COLS)[RAW_COLS]
old["Date"] = pd.to_datetime(old["Date"])
old = old[~old["Ticker"].isin(EXCLUDE_TICKERS)]

tickers = sorted(old["Ticker"].unique())
names = old.drop_duplicates("Ticker").set_index("Ticker")["Company_Name"].to_dict()
ticker_last = old.groupby("Ticker")["Date"].max()
last_date = old["Date"].max()

print(f"{len(tickers)} tickers | {old['Date'].min().date()} -> {last_date.date()} | rows: {len(old):,}")

def download_prices(tickers, start, end):
    # بينزّل OHLCV لقايمة أسهم ويرجّعها في جدول واحد (Date, Ticker, Open, High, Low, Close, Volume)
    raw = yf.download(tickers, start=start, end=end, progress=False,
                      auto_adjust=True, group_by="ticker", threads=True)
    parts = []
    for t in tickers:
        try:
            d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        d = d.dropna(how="all")
        if d.empty:
            continue
        d = d.copy()
        d.columns.name = None
        d = d.reset_index()
        d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
        d["Ticker"] = t
        parts.append(d[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
    if not parts:
        return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
    return pd.concat(parts, ignore_index=True)


start = (last_date - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")      # end في yfinance مش شامل
print(f"Downloading {len(tickers)} tickers from {start} to {end} ...")

new_prices = download_prices(tickers, start, end)
# حماية: لو Yahoo رجّع صفر صفوف رغم إننا بنطلب آخر 30 يوم، يبقى فيه مشكلة (حظر/عطل). نوقف قبل ما نكتب أي ملف.
if new_prices.empty:
    sys.exit("ERROR: Yahoo Finance returned no prices at all - aborting, existing data left untouched.")
if not market_closed:      # قبل الإغلاق: شمعة النهارده ناقصة، نشيلها
    new_prices = new_prices[new_prices["Date"] < today]

print("Downloaded rows:", len(new_prices), "| tickers returned:", new_prices["Ticker"].nunique())
missing = sorted(set(tickers) - set(new_prices["Ticker"]))
if missing:
    print("No data returned (probably suspended or Yahoo issue):", missing)

ov = old.merge(new_prices, on=["Date", "Ticker"], suffixes=("_old", "_new"))
ov["rel_diff"] = (ov["Close_new"] / ov["Close_old"] - 1).abs()
median_diff = ov.groupby("Ticker")["rel_diff"].median()
adjusted = median_diff[median_diff > ADJ_TOLERANCE].index.tolist()

print(f"Overlapping rows compared: {len(ov):,}")
print("Tickers with changed history:", adjusted if adjusted else "none")

full_refresh = pd.DataFrame()
if adjusted:
    full_refresh = download_prices(adjusted, FULL_START, end)
    if not market_closed:
        full_refresh = full_refresh[full_refresh["Date"] < today]
    print("Re-downloaded full history for:", adjusted, "| rows:", len(full_refresh))

def download_eco(start, end):
    series = []
    for tick, col in {"EGP=X": "USD_EGP", "BZ=F": "Brent_Oil"}.items():
        d = yf.download(tick, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        if not d.empty:
            series.append(d[["Close"]].rename(columns={"Close": col}))
    if not series:
        return pd.DataFrame(columns=["Date", "USD_EGP", "Brent_Oil"])
    eco = pd.concat(series, axis=1).reset_index()
    eco["Date"] = pd.to_datetime(eco["Date"]).dt.tz_localize(None).dt.normalize()
    return eco


eco_new = download_eco(start, end)
print("Eco rows:", len(eco_new), "| last date:", eco_new["Date"].max() if len(eco_new) else None)

gd = GdeltDoc()


def fetch_range(kw, start, end, mode, tries=2):
    # Try the period; if it fails, split it in half until it is about a week long
    for _ in range(tries):
        try:
            f = Filters(keyword=kw, start_date=start, end_date=end)
            df = gd.timeline_search(mode, f)
            time.sleep(5)       # GDELT rate limit
            if df is None or df.empty or df.shape[1] < 2:     # GDELT sometimes returns an empty table
                raise ValueError("empty response")
            return [df]
        except Exception as e:
            print(f"   retry {mode} {start}->{end}: {str(e)[:70]}")
            time.sleep(10)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if (e - s).days <= 6:
        return None
    mid = (s + (e - s) / 2).normalize()
    left = fetch_range(kw, start, mid.strftime("%Y-%m-%d"), mode, tries)
    right = fetch_range(kw, (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), end, mode, tries)
    if left is None or right is None:
        return None
    return left + right


def to_daily(frames):
    frames = [f for f in frames if f is not None and not f.empty and f.shape[1] >= 2]
    if not frames:
        return None
    df = pd.concat([f.iloc[:, :2].set_axis(["Date", "val"], axis=1) for f in frames], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df.groupby("Date")["val"].mean()


news_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
news_end = today.strftime("%Y-%m-%d")
vol_series, tone_series, news_failed = {}, {}, []

if pd.Timestamp(news_start) > pd.Timestamp(news_end):
    print("No new days to fetch news for.")
else:
    print(f"Fetching news {news_start} -> {news_end}")
    for topic, kw in NEWS_TOPICS.items():
        for mode, store, prefix in [("timelinevol", vol_series, "Vol_"), ("timelinetone", tone_series, "Tone_")]:
            frames = fetch_range(kw, news_start, news_end, mode)
            daily = to_daily(frames) if frames is not None else None
            if daily is None:
                news_failed.append(f"{prefix}{topic}")
                print(f"   FAILED {prefix}{topic}")
            else:
                store[f"{prefix}{topic}"] = daily
                print(f"   ok {prefix}{topic}")

news_ok = bool(vol_series) and bool(tone_series)
print("News OK:", news_ok, "| failed:", news_failed)

def roll_weekend_to_sunday(d):
    if d.weekday() == 4:      # Friday
        return d + pd.Timedelta(days=2)
    if d.weekday() == 5:      # Saturday
        return d + pd.Timedelta(days=1)
    return d


daily_news_new = pd.DataFrame(columns=["Date", "News_Volume", "Sentiment_Score"])
if news_ok:
    nd = pd.concat({**vol_series, **tone_series}, axis=1)
    nd.index.name = "Date"
    nd = nd.reset_index()
    vol_cols = [c for c in nd.columns if str(c).startswith("Vol_")]
    tone_cols = [c for c in nd.columns if str(c).startswith("Tone_")]
    nd["News_Volume"] = nd[vol_cols].sum(axis=1)
    nd["Sentiment_Score"] = nd[tone_cols].mean(axis=1)
    nd["Trading_Date"] = nd["Date"].apply(roll_weekend_to_sunday)
    daily_news_new = (nd.groupby("Trading_Date")
                        .agg(News_Volume=("News_Volume", "sum"), Sentiment_Score=("Sentiment_Score", "mean"))
                        .reset_index().rename(columns={"Trading_Date": "Date"}))
    daily_news_new = daily_news_new[daily_news_new["Date"] > last_date]

print(daily_news_new.tail())

# ---- الأسعار ----
keep_old = old[~old["Ticker"].isin(adjusted)][["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]

# الأيام الجديدة فقط لكل سهم (بعد آخر يوم عنده)
np_ = new_prices[~new_prices["Ticker"].isin(adjusted)].copy()
np_ = np_[np_["Date"] > np_["Ticker"].map(ticker_last)]

prices = pd.concat([keep_old, np_, full_refresh], ignore_index=True)
prices = prices[prices["Date"] >= old["Date"].min()]               # نفس بداية الداتا القديمة
prices = prices.dropna(subset=["Close"]).drop_duplicates(["Ticker", "Date"], keep="last")

# ---- جدول التواريخ: الدولار وبرنت والأخبار ----
date_old = old.drop_duplicates("Date")[["Date", "USD_EGP", "Brent_Oil", "News_Volume", "Sentiment_Score"]]
all_dates = pd.DatetimeIndex(sorted(prices["Date"].unique()))

eco_part = eco_new[eco_new["Date"] > last_date][["Date", "USD_EGP", "Brent_Oil"]]
eco_all = (pd.concat([date_old[["Date", "USD_EGP", "Brent_Oil"]], eco_part])
             .drop_duplicates("Date", keep="first").set_index("Date").sort_index())
eco_all = eco_all.reindex(eco_all.index.union(all_dates)).ffill()

news_all = (pd.concat([date_old[["Date", "News_Volume", "Sentiment_Score"]], daily_news_new])
              .drop_duplicates("Date", keep="first").set_index("Date").sort_index())
news_all = news_all.reindex(news_all.index.union(all_dates)).ffill()

date_tbl = eco_all.join(news_all).reset_index().rename(columns={"index": "Date"})

final = prices.merge(date_tbl, on="Date", how="left")
final["Company_Name"] = final["Ticker"].map(names)
final = final[RAW_COLS].sort_values(["Ticker", "Date"]).reset_index(drop=True)

print("Final shape:", final.shape, "| last date:", final["Date"].max().date())
print("NaNs per column:", final.isna().sum()[final.isna().sum() > 0].to_dict() or "none")
final.tail()

if RAW_CSV.exists():
    shutil.copy(RAW_CSV, DATA_DIR / "egx_raw_backup.csv")      # نسخة احتياطية قبل ما نكتب فوقه

final.to_csv(RAW_CSV, index=False)

latest = final["Date"].max()
per_ticker_last = final.groupby("Ticker")["Date"].max()
stale = per_ticker_last[per_ticker_last < latest - pd.Timedelta(days=7)]

status = {
    "updated_at_cairo": now_cairo.strftime("%Y-%m-%d %H:%M"),
    "data_last_date": str(latest.date()),
    "rows": int(len(final)),
    "tickers": int(final["Ticker"].nunique()),
    "new_rows_added": int(len(final) - len(old)),
    "tickers_history_refreshed": adjusted,
    "tickers_without_new_data": missing,
    "stale_tickers": {t: str(d.date()) for t, d in stale.items()},
    "news_ok": bool(news_ok) or pd.Timestamp(news_start) > pd.Timestamp(news_end),
    "news_failed_parts": news_failed,
}
STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(status, ensure_ascii=False, indent=2))
