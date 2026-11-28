"""منطق الأخبار (من غير yfinance ولا gdeltdoc) عشان يتختبر لوحده.

الفكرة: لو GDELT فشل في يوم، الأرقام بتتنسخ من آخر يوم (ffill) وبنسجّل التاريخ ده في
data/news_imputed_dates.json. في التشغيل اللي بعده update_data.py بيحاول يجيب الأيام دي تاني
ويبدّل القيم المنسوخة بقيم حقيقية. الداشبورد بيقرأ نفس الملف عشان يعلّم Sentiment إنه منسوخ.
"""
import json
from pathlib import Path

import pandas as pd

NEWS_COLS = ["News_Volume", "Sentiment_Score"]


# ----------------------------------------------------------------------------- الملف
def load_imputed(path):
    """قايمة التواريخ اللي أخبارها منسوخة (فاضية لو الملف مش موجود أو بايظ)."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return sorted(pd.to_datetime(data.get("dates", [])).normalize())
    except Exception:
        return []


def save_imputed(path, dates):
    dates = sorted(pd.to_datetime(list(dates)).normalize())
    payload = {
        "note": "Trading days whose News_Volume/Sentiment_Score were carried forward because GDELT failed. "
                "update_data.py retries these days on the next run.",
        "dates": [str(d.date()) for d in dates],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- كشف الأيام المنسوخة
def detect_stale_dates(date_news, since):
    """أيام بعد `since` رقم الأخبار فيها زي اليوم اللي قبله بالظبط (يعني اتنسخ).
    date_news: جدول فيه Date و News_Volume و Sentiment_Score (صف لكل يوم)."""
    d = date_news.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    same = (d["News_Volume"].diff() == 0) & (d["Sentiment_Score"].diff() == 0)
    return sorted(d.loc[same & (d["Date"] > pd.Timestamp(since)), "Date"])


# ----------------------------------------------------------------------------- تجميع
def roll_weekend_to_sunday(d):
    if d.weekday() == 4:      # الجمعة
        return d + pd.Timedelta(days=2)
    if d.weekday() == 5:      # السبت
        return d + pd.Timedelta(days=1)
    return d


def aggregate_news(vol_series, tone_series, expected_vol, expected_tone):
    """يحوّل السلاسل اليومية لجدول (Date, News_Volume, Sentiment_Score) بتاريخ يوم التداول.
    بيرجّع None لو أي جزء ناقص (وجدول فاضي لو مفيش يوم كامل لسه): المجموع/المتوسط على مواضيع ناقصة مقياسه مختلف عن اللي الموديل اتدرّب عليه،
    فالأحسن ننسخ ونعلّم اليوم بدل ما ندخّل رقم غلط.
    الأيام اللي ناقصة في أي موضوع بتتشال (بتفضل منسوخة)."""
    if set(vol_series) != set(expected_vol) or set(tone_series) != set(expected_tone):
        return None
    empty = pd.DataFrame(columns=["Date"] + NEWS_COLS)
    nd = pd.concat({**vol_series, **tone_series}, axis=1).dropna()
    if nd.empty:
        return empty
    nd.index.name = "Date"
    nd = nd.reset_index()
    nd["News_Volume"] = nd[list(expected_vol)].sum(axis=1)
    nd["Sentiment_Score"] = nd[list(expected_tone)].mean(axis=1)
    nd["Trading_Date"] = nd["Date"].apply(roll_weekend_to_sunday)
    # يوم الأحد بيتجمّع من الجمعة + السبت + الأحد (3 أيام)، وباقي الأيام يوم واحد. لو يوم ناقص يبقى الرقم أقل
    # من اللي الموديل اتدرّب عليه، فاليوم ده بيتشال ويفضل منسوخ.
    need = nd["Trading_Date"].dt.weekday.eq(6).map({True: 3, False: 1})
    have = nd.groupby("Trading_Date")["Date"].transform("count")
    nd = nd[have >= need]
    if nd.empty:
        return empty      # الأجزاء كلها جت، بس مفيش يوم تداول كامل لسه (مثلًا جمعة وسبت بس): مش فشل
    return (nd.groupby("Trading_Date")
              .agg(News_Volume=("News_Volume", "sum"), Sentiment_Score=("Sentiment_Score", "mean"))
              .reset_index().rename(columns={"Trading_Date": "Date"}))


# ----------------------------------------------------------------------------- دمج
def merge_news(old_news, fresh, all_dates, repair_dates):
    """old_news: Date/News_Volume/Sentiment_Score من الداتا الحالية (صف لكل يوم).
    fresh: ناتج aggregate_news أو None.
    all_dates: كل أيام التداول في الداتا النهائية.
    repair_dates: أيام كانت منسوخة وعايزين نصلحها.
    يرجّع (news_all بفهرس Date, imputed_dates): الأيام اللي لسه منسوخة بعد الدمج."""
    base = old_news.drop_duplicates("Date").set_index("Date").sort_index()[NEWS_COLS]
    old_last = base.index.max()
    repair = pd.DatetimeIndex(repair_dates).normalize()
    all_dates = pd.DatetimeIndex(all_dates)

    used = pd.DatetimeIndex([])
    if fresh is not None and len(fresh):
        f = fresh.drop_duplicates("Date").set_index("Date").sort_index()[NEWS_COLS]
        used = f.index[(f.index > old_last) | f.index.isin(repair)]
        base = pd.concat([base.drop(index=base.index.intersection(used)), f.loc[used]]).sort_index()

    candidates = set(repair) | set(all_dates[all_dates > old_last])
    imputed = sorted(d for d in candidates if d not in set(used) and d in set(all_dates))
    news_all = base.reindex(base.index.union(all_dates)).ffill()
    return news_all, imputed
