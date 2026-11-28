"""Step 6 - ليه الموديل رفع السهم ده؟ (تفصيل مساهمة كل feature في الـ Score).

بيشتغل بعد train_and_predict.py، وبيقرأ الموديل المحفوظ (data/*.pkl) + data/egx_raw.csv،
وبيكتب data/explanations.csv: لكل سهم، أهم العوامل اللي زوّدت أو قللت الـ Score بتاعه النهارده،
مع قيمة العامل في السهم ومتوسط السوق في نفس اليوم (عشان يبان السهم مختلف في إيه بالظبط).

الطريقة: SHAP values من الموديلين نفسهم (CatBoost و XGBoost)، بنفس أوزان الـ ensemble
(0.6 كاتبوست + 0.4 إكسجيبوست). الأرقام بوحدات log-odds: موجب = رفع الـ Score، سالب = نزّله.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from xgboost import DMatrix

from features import BEST_FEATURES, RAW_COLS, build_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # <repo>/data
RAW_CSV = DATA_DIR / "egx_raw.csv"
OLD_CSV = DATA_DIR / "EGX_Merged_News_Economy.csv"
MODEL_PATH = DATA_DIR / "ionmonks_egx_ensemble_v3.pkl"
PRED_CSV = DATA_DIR / "predictions.csv"
EXPLAIN_CSV = DATA_DIR / "explanations.csv"

WEIGHTS = {"CatBoost": 0.6, "XGBoost": 0.4}      # نفس أوزان VotingClassifier في train_and_predict.py
RECENT_VOLUME_DAYS = 10

src = RAW_CSV if RAW_CSV.exists() else OLD_CSV
if not src.exists():
    raise SystemExit("مفيش egx_raw.csv ولا الداتا الأصلية - شغّل update_data.py الأول.")
if not MODEL_PATH.exists():
    raise SystemExit("مفيش موديل محفوظ - شغّل train_and_predict.py الأول.")
if not PRED_CSV.exists():
    raise SystemExit("مفيش predictions.csv - شغّل train_and_predict.py الأول.")

pred = pd.read_csv(PRED_CSV, parse_dates=["Date"])
asof = pred["Date"].max()
active = set(pred["Ticker"])

print("Reading:", src)
raw = pd.read_csv(src, usecols=lambda c: c in RAW_COLS)[RAW_COLS]
raw["Date"] = pd.to_datetime(raw["Date"])

feats = build_features(raw)
cand = feats.dropna(subset=BEST_FEATURES)
last_rows = cand.sort_values("Date").groupby("Ticker").tail(1)
today = last_rows[(last_rows["Date"] == asof) & (last_rows["Ticker"].isin(active))].copy()

# نفس استبعاد train_and_predict.py: سهم من غير تداول في آخر 10 أيام
recent_vol = raw.sort_values("Date").groupby("Ticker").tail(RECENT_VOLUME_DAYS).groupby("Ticker")["Volume"].sum()
today = today[~today["Ticker"].isin(set(recent_vol[recent_vol <= 0].index))]

missing = active - set(today["Ticker"])
if missing:
    print("تحذير: أسهم في predictions.csv مش موجودة في الـ features:", ", ".join(sorted(missing)))

artifact = joblib.load(MODEL_PATH)
feat_list = artifact["features"]
if feat_list != BEST_FEATURES:
    raise SystemExit("قايمة الـ features في الموديل المحفوظ مختلفة عن اللي في features.py - "
                     "لازم retrain (FORCE_RETRAIN = True) قبل الشرح.")
missing_cols = [c for c in feat_list if c not in today.columns]
if missing_cols:
    raise SystemExit("أعمدة ناقصة في الـ features: " + ", ".join(missing_cols))

today = today.sort_values("Ticker").reset_index(drop=True)
X = today[feat_list].astype(float)
ensemble = artifact["ensemble_model"]
estimators = dict(ensemble.named_estimators_)   # الموديلين المدرّبين (ensemble.estimators مجرد نسخ فاضية)

# ---- SHAP من الموديلين (كل واحد في مساحة الـ log-odds بتاعته) ----
cat_shap = np.asarray(estimators["CatBoost"].get_feature_importance(type="ShapValues", data=Pool(X)))
xgb_contribs = estimators["XGBoost"].get_booster().predict(DMatrix(X), pred_contribs=True)
if cat_shap.shape != (len(X), len(feat_list) + 1) or xgb_contribs.shape != (len(X), len(feat_list) + 1):
    raise SystemExit(f"شكل SHAP مش متوقع: cat={cat_shap.shape} xgb={xgb_contribs.shape}")
contrib = WEIGHTS["CatBoost"] * cat_shap[:, :-1] + WEIGHTS["XGBoost"] * xgb_contribs[:, :-1]

# ---- تأكيد إن الـ scores اللي حسبناها هي هي اللي في predictions.csv ----
recomputed = ensemble.predict_proba(X)[:, 1]
check = pd.DataFrame({"Ticker": today["Ticker"], "Score": recomputed}).merge(
    pred[["Ticker", "Score"]], on="Ticker", suffixes=("_recomputed", "_saved"))
diff = (check["Score_recomputed"] - check["Score_saved"]).abs().max() if len(check) else np.nan
print(f"Score check: {len(check)} stock(s) | max |diff| vs predictions.csv = {diff:.4f}")
if not np.isfinite(diff) or diff > 0.005:
    print("تحذير: الـ Score اللي طلع مش مطابق لـ predictions.csv - الشرح ممكن يكون لسهم غلط.")

# ---- متوسط السوق لكل feature (نفس اليوم) عشان نقارن السهم بالسوق ----
market_median = X.median()
pct_rank = X.rank(pct=True)

rows = []
for i, ticker in enumerate(today["Ticker"]):
    for j, feat in enumerate(feat_list):
        rows.append({
            "Date": asof.strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Feature": feat,
            "Contribution": round(float(contrib[i, j]), 5),
            "Value": round(float(X.iat[i, j]), 4),
            "Market_Median": round(float(market_median[feat]), 4),
            "Value_Pct_Rank": round(float(pct_rank.iat[i, j]), 3),
        })
expl = pd.DataFrame(rows).sort_values(["Ticker", "Contribution"], ascending=[True, False])
expl.to_csv(EXPLAIN_CSV, index=False)
print("Saved ->", EXPLAIN_CSV, f"({len(expl):,} rows, {expl['Ticker'].nunique()} stocks, {len(feat_list)} features)")

# ---- ملخص في الكونسول: أهم 5 عوامل لأعلى 5 أسهم ----
top = pred.sort_values("Rank").head(5)
for _, r in top.iterrows():
    mine = expl[expl["Ticker"] == r["Ticker"]].sort_values("Contribution", ascending=False)
    print(f"\n#{int(r['Rank'])} {r['Ticker']} ({r['Company_Name']}) | Score {r['Score']:.3f} | {r['Rating']}")
    for _, x in mine.head(5).iterrows():
        print(f"   + {x['Feature']:28s} value {x['Value']:>10.3f} (market {x['Market_Median']:>10.3f})  {x['Contribution']:+.4f}")
    for _, x in mine.tail(3).iloc[::-1].iterrows():
        print(f"   - {x['Feature']:28s} value {x['Value']:>10.3f} (market {x['Market_Median']:>10.3f})  {x['Contribution']:+.4f}")
