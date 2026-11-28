"""Step 5 - الـ Features + الموديل + التوقعات (محوّل من 05_train_and_predict.ipynb).
بيطلّع في data/: predictions.csv, dashboard_history.csv, scores_history.csv, model_report.json, الموديل .pkl"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # <repo>/data
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV  = DATA_DIR / "egx_raw.csv"                              # من 04_update_data
OLD_CSV  = DATA_DIR / "EGX_Merged_News_Econ_Dataset.csv"         # احتياطي لو 04 لسه ما اتشغلش
STATUS_JSON = DATA_DIR / "update_status.json"

MODEL_PATH   = DATA_DIR / "ionmonks_egx_ensemble_v3.pkl"
PRED_CSV     = DATA_DIR / "predictions.csv"
HISTORY_CSV  = DATA_DIR / "dashboard_history.csv"                # شارت كل سهم (آخر 260 يوم)
SCORES_CSV   = DATA_DIR / "scores_history.csv"                   # بيتراكم كل يوم
REPORT_JSON  = DATA_DIR / "model_report.json"
BACKTEST_CSV = DATA_DIR / "backtest.csv"                          # توقعات فترة الاختبار vs اللي حصل فعلًا

TARGET = "Target_10d_Trend"
TRAIN_WINDOW_DAYS = 750          # آخر 750 يوم تداول (زي النوت)
GAP_DAYS = 10
RETRAIN_EVERY_DAYS = 30
FORCE_RETRAIN = False            # خليها True لو عايز تدرّب من الأول
RUN_EVALUATION = True            # تقييم صادق على فترة الاختبار (بياخد ~دقيقة، سيبه True عند كل تدريب)
HISTORY_DAYS = 260

from features import (BEST_FEATURES, EXCLUDED_TICKERS, MIN_PRICE, RATINGS, RAW_COLS,
                      RECENT_VOLUME_DAYS, build_features)

src = RAW_CSV if RAW_CSV.exists() else OLD_CSV
print("Reading:", src)
raw = pd.read_csv(src, usecols=lambda c: c in RAW_COLS)[RAW_COLS]
raw["Date"] = pd.to_datetime(raw["Date"])

asof = raw["Date"].max()
print(f"{raw['Ticker'].nunique()} tickers | {raw['Date'].min().date()} -> {asof.date()} | rows: {len(raw):,}")

feats = build_features(raw)
print("Features table:", feats.shape)

def make_model():
    cat = CatBoostClassifier(
        iterations=800, learning_rate=0.01185, depth=5, l2_leaf_reg=2.6789,
        random_strength=1.8014, auto_class_weights="Balanced", random_state=42, verbose=0)
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.02, max_depth=4, scale_pos_weight=1.2,
        random_state=42, eval_metric="logloss")
    return VotingClassifier(estimators=[("CatBoost", cat), ("XGBoost", xgb)], voting="soft", weights=[0.6, 0.4])


def rating_from_pct(p):
    # p = ترتيب السهم اليومي من 0 لـ 1 (1 = أعلى Score)
    return np.select([p > 0.8, p > 0.6, p > 0.4, p > 0.2], RATINGS[:4], default=RATINGS[4])


def add_target(df):
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Future_Return_10d_%"] = (df.groupby("Ticker")["Close"].shift(-10) - df["Close"]) / df["Close"] * 100
    fr = df["Future_Return_10d_%"]
    df[TARGET] = np.select([fr >= 3.5, fr <= -3.0], [1, 0], default=-1).astype(float)
    df.loc[fr.isna(), TARGET] = np.nan
    return df


labeled = add_target(feats).dropna(subset=BEST_FEATURES + ["Future_Return_10d_%"])   # كل الصفوف اللي نعرف نتيجتها
train_pool = labeled[labeled[TARGET] != -1].copy()                                  # الحركات القوية بس للتدريب
train_pool[TARGET] = train_pool[TARGET].astype(int)
dates = np.sort(train_pool["Date"].unique())
print(f"Rows with known outcome: {len(labeled):,} | strong-move training pool: {len(train_pool):,} | trading days: {len(dates)}")

# ---- هل نحتاج نعيد التدريب؟ ----
status = json.loads(STATUS_JSON.read_text(encoding="utf-8")) if STATUS_JSON.exists() else {}
need_retrain, reason = FORCE_RETRAIN, "FORCE_RETRAIN"
artifact = None
if not FORCE_RETRAIN:
    if not MODEL_PATH.exists():
        need_retrain, reason = True, "no saved model"
    else:
        artifact = joblib.load(MODEL_PATH)
        age = (asof - pd.Timestamp(artifact["trained_until"])).days
        refreshed_after = (status.get("tickers_history_refreshed")
                           and pd.Timestamp(status.get("updated_at_cairo", "1970-01-01")) > pd.Timestamp(artifact.get("trained_at", "1970-01-01")))
        if artifact["features"] != BEST_FEATURES:
            need_retrain, reason = True, "feature list changed"
        elif age > RETRAIN_EVERY_DAYS + GAP_DAYS:
            need_retrain, reason = True, f"model is {age} days behind the data"
        elif refreshed_after:
            need_retrain, reason = True, "some stock prices were re-adjusted"
        elif RUN_EVALUATION and not BACKTEST_CSV.exists():
            need_retrain, reason = True, "backtest file missing"
print("Retrain:", need_retrain, "|", reason if need_retrain else f"model trained until {artifact['trained_until']}")

report = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}

if need_retrain and RUN_EVALUATION:
    split = int(len(dates) * 0.80)
    tr = train_pool[(train_pool["Date"] >= dates[max(0, split - TRAIN_WINDOW_DAYS)]) & (train_pool["Date"] <= dates[split])]
    te = labeled[labeled["Date"] >= dates[split + GAP_DAYS]].copy()          # كل الصفوف بما فيهم الحركات العرضية

    m_eval = make_model().fit(tr[BEST_FEATURES], tr[TARGET])
    te["Score"] = m_eval.predict_proba(te[BEST_FEATURES])[:, 1]

    strong = te[te[TARGET] != -1]
    auc = roc_auc_score(strong[TARGET].astype(int), strong["Score"])

    g = te.groupby("Date")
    te["Rating"] = rating_from_pct(g["Score"].rank(pct=True).values)
    te["Excess"] = te["Future_Return_10d_%"] - g["Future_Return_10d_%"].transform("mean")   # العائد بالنسبة لمتوسط السوق في نفس اليوم
    rows = []
    for name in RATINGS:
        x = te[te["Rating"] == name]
        rows.append({"rating": name, "n": int(len(x)),
                     "avg_10d_return": round(float(x["Future_Return_10d_%"].mean()), 2),
                     "excess_vs_market": round(float(x["Excess"].mean()), 2),
                     "pct_up": round(float((x["Future_Return_10d_%"] > 0).mean() * 100), 1)})
    top10 = g.apply(lambda x: x.nlargest(10, "Score")["Future_Return_10d_%"].mean()).mean()
    bot10 = g.apply(lambda x: x.nsmallest(10, "Score")["Future_Return_10d_%"].mean()).mean()
    ic = g.apply(lambda x: x["Score"].corr(x["Future_Return_10d_%"], method="spearman")).mean()
    day_mean = g["Score"].mean()

    report = {
        "test_period": [str(te["Date"].min().date()), str(te["Date"].max().date())],
        "rows": int(len(te)), "auc_strong_moves": round(float(auc), 4),
        "market_avg_10d_return": round(float(te["Future_Return_10d_%"].mean()), 2),
        "ratings": rows,
        "top10_daily_avg_return": round(float(top10), 2), "bottom10_daily_avg_return": round(float(bot10), 2),
        "rank_ic": round(float(ic), 3),
        "daily_avg_score_range_in_test": [round(float(day_mean.min()), 3), round(float(day_mean.max()), 3)],
    }
    print(f"Test {report['test_period'][0]} -> {report['test_period'][1]} | rows {len(te):,} | AUC (strong moves only) {auc:.3f}")
    print(f"Market average 10-day return: {report['market_avg_10d_return']:+.2f}%\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTop-10 by score each day: {top10:+.2f}% | Bottom-10: {bot10:+.2f}% | Rank IC: {ic:.3f}")

    # كل توقع في فترة الاختبار + اللي حصل فعلًا، عشان الداشبورد تعرض سجل كل سهم
    bt = te[["Date", "Ticker", "Score", "Rating", "Future_Return_10d_%", "Excess"]].copy()
    bt["Rank"] = te.groupby("Date")["Score"].rank(ascending=False, method="first").astype(int)
    bt.sort_values(["Ticker", "Date"]).round(4).to_csv(BACKTEST_CSV, index=False)
    print("Saved backtest ->", BACKTEST_CSV, f"({len(bt):,} rows)")

if need_retrain:
    final_train = train_pool[train_pool["Date"] >= dates[max(0, len(dates) - TRAIN_WINDOW_DAYS)]]
    print(f"Training on {final_train['Date'].min().date()} -> {final_train['Date'].max().date()} ({len(final_train):,} rows) ...")
    model = make_model().fit(final_train[BEST_FEATURES], final_train[TARGET])
    artifact = {
        "ensemble_model": model, "features": BEST_FEATURES, "horizon_days": 10, "ratings": RATINGS,
        "trained_until": str(final_train["Date"].max().date()),
        "trained_at": pd.Timestamp.now(tz="Africa/Cairo").strftime("%Y-%m-%d %H:%M"),
    }
    joblib.dump(artifact, MODEL_PATH)
    print("Saved ->", MODEL_PATH)
else:
    model = artifact["ensemble_model"]
    print("Using saved model (trained until", artifact["trained_until"] + ")")

cand = feats.dropna(subset=BEST_FEATURES)
last_rows = cand.sort_values("Date").groupby("Ticker").tail(1)
active = last_rows[last_rows["Date"] == asof].copy()

# مفيش تداول فعلي = حجم صفر في آخر 10 أيام (سعر ثابت، مؤشراته مش بتتحسب صح)
recent_vol = raw.sort_values("Date").groupby("Ticker").tail(RECENT_VOLUME_DAYS).groupby("Ticker")["Volume"].sum()
not_trading = set(recent_vol[recent_vol <= 0].index)
active = active[~active["Ticker"].isin(not_trading)].copy()

excluded = {}
for t in sorted(set(raw["Ticker"]) - set(active["Ticker"])):
    last_raw = raw.loc[raw["Ticker"] == t, "Date"].max()
    if t in EXCLUDED_TICKERS:        excluded[t] = "excluded (bad Yahoo data)"
    elif t in not_trading:           excluded[t] = f"no trading volume in the last {RECENT_VOLUME_DAYS} days (price is flat)"
    elif last_raw < asof:            excluded[t] = f"stopped trading (last row {last_raw.date()})"
    else:                            excluded[t] = "indicators undefined on the last day"
print(f"Scoring {len(active)} stocks as of {asof.date()} | excluded {len(excluded)}:")
for t, why in excluded.items():
    print(f"   {t}: {why}")

active["Score"] = model.predict_proba(active[BEST_FEATURES])[:, 1]
active["Percentile"] = active["Score"].rank(pct=True)
active["Rating"] = rating_from_pct(active["Percentile"].values)
active = active.sort_values("Score", ascending=False).reset_index(drop=True)
active["Rank"] = np.arange(1, len(active) + 1)

pred_cols = ["Rank", "Ticker", "Company_Name", "Score", "Rating", "Date", "Close", "Daily_Return_%",
             "Return_5d_%", "Return_20d_%", "RSI_14", "MACD_Hist_%", "Price_to_MA20_%", "Price_to_MA50_%",
             "Price_to_MA200_%", "BB_Position", "Relative_Volume", "Volatility_30d",
             "Sentiment_Score", "News_Volume", "USD_EGP", "Brent_Oil", "Market_Above_MA20_%"]
predictions = active[pred_cols].copy()
predictions["Score"] = predictions["Score"].round(4)
predictions = predictions.round({c: 3 for c in pred_cols if c not in ("Rank", "Score")
                                 and predictions[c].dtype.kind == "f"})
predictions.to_csv(PRED_CSV, index=False)

market_avg_score = float(active["Score"].mean())
print(f"\nAverage score of all stocks today: {market_avg_score:.3f}")
print(predictions["Rating"].value_counts().reindex(RATINGS).to_string())
predictions.head(15)[["Rank", "Ticker", "Company_Name", "Score", "Rating", "Close", "Return_5d_%", "RSI_14"]]

# تاريخ الأسعار والمؤشرات لكل سهم (للشارت في صفحة التفاصيل)
hist_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "MA_20", "MA_50", "MA_200",
             "BB_Upper", "BB_Lower", "RSI_14", "MACD_%", "MACD_Signal_%", "MACD_Hist_%"]
cutoff = asof - pd.Timedelta(days=int(HISTORY_DAYS * 1.5))
history = feats.loc[feats["Ticker"].isin(active["Ticker"]) & (feats["Date"] >= cutoff), hist_cols]
history = history.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(HISTORY_DAYS).round(4)
history.to_csv(HISTORY_CSV, index=False)

# تاريخ الـ Score: بيتراكم كل يوم تشغّل فيه النوت بوك (عشان نعرض تغيّر ترتيب السهم مع الوقت)
today_scores = active[["Date", "Ticker", "Score", "Rank", "Rating"]].copy()
if SCORES_CSV.exists():
    prev = pd.read_csv(SCORES_CSV, parse_dates=["Date"])
    today_scores = pd.concat([prev, today_scores]).drop_duplicates(["Date", "Ticker"], keep="last")
today_scores.sort_values(["Date", "Rank"]).to_csv(SCORES_CSV, index=False)

rng = report.get("daily_avg_score_range_in_test")
report.update({"asof": str(asof.date()), "stocks_scored": int(len(active)), "excluded": excluded,
               "market_avg_score_today": round(market_avg_score, 3),
               "market_score_above_test_range": bool(rng and market_avg_score > rng[1]),
               "model_trained_until": artifact["trained_until"],
               "generated_at": pd.Timestamp.now(tz="Africa/Cairo").strftime("%Y-%m-%d %H:%M")})
REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print("predictions.csv      :", len(predictions), "rows")
print("dashboard_history.csv:", len(history), "rows")
print("scores_history.csv   :", today_scores["Date"].nunique(), "day(s) accumulated")
print("model_report.json    : saved")
