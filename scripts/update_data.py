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

from news_utils import (aggregate_news, detect_stale_dates, load_imputed, merge_news, save_imputed)

pd.set_option("display.width", 200)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # <repo>/data
DATA_DIR.mkdir(parents=True, exist_ok=True)
OLD_CSV  = DATA_DIR / "EGX_Merged_News_Econ_Dataset.csv"   # الداتا الأصلية (Milestone 2)
RAW_CSV  = DATA_DIR / "egx_raw.csv"                    # الملف اللي بيتحدث كل يوم
STATUS_JSON = DATA_DIR / "update_status.json"
IMPUTED_JSON = DATA_DIR / "news_imputed_dates.json"   # أيام أخبارها منسوخة (GDELT فشل) - بنحاول نصلحها كل تشغيل

EXCLUDE_TICKERS = ["EGBE.CA"]      # سهم داتاه بايظة (زي النوت)
OVERLAP_DAYS    = 30               # بننزّل آخر 30 يوم تاني عشان نتأكد إن الأسعار القديمة ما اتغيرتش
FULL_START      = "2020-01-01"     # لو سهم اتغيرت أسعاره نزّله من الأول
ADJ_TOLERANCE   = 0.005            # لو الفرق في السعر القديم أكتر من 0.5% نعتبره اتعدّل
NEWS_AUTO_DETECT_SINCE = "2026-09-30"   # آخر يوم في الداتا الأصلية (Milestone 2): ما بنلمسهاش، بنصلّح بس اللي الـ pipeline ضافه
NEWS_RETRY_WINDOW_DAYS = 120    # بنحاول نصلّح الأخبار المنسوخة لحد 120 يوم ورا
NEWS_TIME_BUDGET_SEC   = 40 * 60  # أقصى وقت للأخبار كلها، عشان GDELT البطيء ما يعطّلش الـ workflow
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


news_deadline = time.time() + NEWS_TIME_BUDGET_SEC


def fetch_range(kw, start, end, mode, tries=3):
    # Try the period (with growing waits); if it fails, split it in half until it is about a week long
    for attempt in range(tries):
        if time.time() > news_deadline:
            print("   news time budget finished - skipping")
            return None
        try:
            f = Filters(keyword=kw, start_date=start, end_date=end)
            df = gd.timeline_search(mode, f)
            time.sleep(5)       # GDELT rate limit
            if df is None or df.empty or df.shape[1] < 2:     # GDELT sometimes returns an empty table
                raise ValueError("empty response")
            return [df]
        except Exception as e:
            print(f"   retry {mode} {start}->{end}: {str(e)[:70]}")
            time.sleep(10 * (3 ** attempt))      # 10s ثم 30s ثم 90s
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


# ---- الأخبار: نجيب الأيام الجديدة + نصلّح الأيام اللي كانت منسوخة في تشغيلات فاشلة قبل كده ----
EXPECTED_VOL = [f"Vol_{t}" for t in NEWS_TOPICS]
EXPECTED_TONE = [f"Tone_{t}" for t in NEWS_TOPICS]

date_old_news = old.drop_duplicates("Date")[["Date", "News_Volume", "Sentiment_Score"]]
retry_cutoff = last_date - pd.Timedelta(days=NEWS_RETRY_WINDOW_DAYS)
if IMPUTED_JSON.exists():
    prev_imputed = load_imputed(IMPUTED_JSON)
else:       # أول مرة: نكتشف الأيام المنسوخة من الداتا نفسها (من بعد الداتا الأصلية)
    prev_imputed = detect_stale_dates(date_old_news, max(retry_cutoff, pd.Timestamp(NEWS_AUTO_DETECT_SINCE)))
    print("No news_imputed_dates.json yet - detected", len(prev_imputed), "carried-forward days in the data")
known_days = set(date_old_news["Date"])
prev_imputed = [d for d in prev_imputed if d in known_days]
repair_dates = [d for d in prev_imputed if d >= retry_cutoff]
too_old_imputed = [d for d in prev_imputed if d < retry_cutoff]      # بنسيبهم معلّمين من غير ما نحاول تاني

# يوم الأحد بيتجمّع من الجمعة والسبت قبله، فنبدأ قبل أقدم يوم محتاج إصلاح بيومين
news_start_ts = last_date + pd.Timedelta(days=1)
if repair_dates:
    news_start_ts = min(news_start_ts, min(repair_dates) - pd.Timedelta(days=2))
news_start = news_start_ts.strftime("%Y-%m-%d")
news_end = today.strftime("%Y-%m-%d")
vol_series, tone_series, news_failed = {}, {}, []


def fetch_news_part(topic, kw, mode, store, prefix):
    frames = fetch_range(kw, news_start, news_end, mode)
    daily = to_daily(frames) if frames is not None else None
    name = f"{prefix}{topic}"
    if daily is None:
        print(f"   FAILED {name}")
        return False
    store[name] = daily
    print(f"   ok {name}")
    return True


nothing_to_fetch = pd.Timestamp(news_start) > pd.Timestamp(news_end)
if nothing_to_fetch:
    print("No new days to fetch news for.")
else:
    print(f"Fetching news {news_start} -> {news_end} | days to repair: {len(repair_dates)}")
    parts = [(t, kw, m, st, pf) for t, kw in NEWS_TOPICS.items()
             for m, st, pf in [("timelinevol", vol_series, "Vol_"), ("timelinetone", tone_series, "Tone_")]]
    failed_parts = [p_ for p_ in parts if not fetch_news_part(*p_)]
    if failed_parts and time.time() < news_deadline:      # لفّة تانية بعد استراحة، GDELT غالبًا بيفك لوحده
        print(f"Retrying {len(failed_parts)} failed news part(s) after a pause ...")
        time.sleep(60)
        failed_parts = [p_ for p_ in failed_parts if not fetch_news_part(*p_)]
    news_failed = [f"{p_[4]}{p_[0]}" for p_ in failed_parts]

# الأخبار بتتقبل كلها أو ولا حاجة: مجموع/متوسط على مواضيع ناقصة مقياسه مختلف عن اللي الموديل اتدرّب عليه
fresh_news = None if (nothing_to_fetch or news_failed) else aggregate_news(vol_series, tone_series, EXPECTED_VOL, EXPECTED_TONE)
news_ok = nothing_to_fetch or not news_failed
print("News OK:", news_ok, "| failed:", news_failed)
if fresh_news is not None:
    print(fresh_news.tail())

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

news_all, imputed_now = merge_news(date_old_news, fresh_news, all_dates, repair_dates)
news_imputed = sorted(too_old_imputed + imputed_now)
repaired_now = [d for d in repair_dates if d not in set(imputed_now)]
print(f"News repaired: {len(repaired_now)} day(s) | still carried forward: {len(news_imputed)} day(s)")

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
    "news_ok": bool(news_ok),
    "news_failed_parts": news_failed,
    "news_days_repaired": len(repaired_now),
    "news_days_carried_forward": len(news_imputed),
    "news_latest_carried_forward": bool(len(news_imputed)) and pd.Timestamp(latest) in set(news_imputed),
}
STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
save_imputed(IMPUTED_JSON, news_imputed)

print(json.dumps(status, ensure_ascii=False, indent=2))
