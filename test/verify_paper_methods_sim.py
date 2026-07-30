# -*- coding: utf-8 -*-
"""紙上計測中の買い方5種の一括シミュレーション(2025-12〜2026-05・月次再学習方式)

    py -X utf8 test/verify_paper_methods_sim.py

■ 対象(2026-07-29/30の判断会で紙上並走に回した5種+基準)
  基準   : v2.1(超混戦=案1 / 本命=保険複入り・5場20-30%cap6)
  変形ハ  : 超混戦の配分をA複200/B複100/AトリオTop1単100/BトリオTop1単100/E200/F200/G100
  伸び盛り増額: 超混戦で上位4艇内に伸び盛り(gap>+10pt)あり→案1を2倍(2,000円)
  KR旗増額 : 超混戦でレース内KR1位がB級→案1を2倍
  旗統合増額: 上記どちらかの旗→案1を2倍(9月の「アンカー古さスコア」の素形)
  外枠人気除外: 本命から人気=4-6号艇のレースを除外(買わない)

■ 注意(正直に)
この期間は各旗を発見したデータと重なるイン標本の整合確認。
独立検証は8月の前向き紙上成績(9/1判定)が本番。
学習は本番同様「各月をその前月末までの全データで学習」。
"""
import datetime as dt
import importlib.util
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
KONSEN_MAX, HONMEI_MAX, CAP = 0.20, 0.30, 6

# ---- 旗の材料(いずれもWF安全) ----------------------------------------------
print("KR指数(Elo)を構築中...", flush=True)
spec = importlib.util.spec_from_file_location(
    "abl", r"Y:\マイドライブ\boat\test\verify_v2_features_ablation.py")
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)
extra = abl.build_extra_features()
kr = {(r.race_id, int(r.lane)): float(r.kr_rating)
      for r in extra.itertuples() if r.kr_rating == r.kr_rating}

raw = sqlite3.connect(DB_PATH)
klass = {(rid, lane): kl for rid, lane, kl in raw.execute(
    "SELECT race_id, lane, racer_class FROM entries")}
races_date = dict(raw.execute("SELECT race_id, date FROM races"))
lane_racer, printed = {}, {}
for rid, lane, reg, n2 in raw.execute(
        "SELECT race_id, lane, reg_no, national_2rate FROM entries"):
    lane_racer[(rid, lane)] = reg
    printed[(rid, lane)] = n2
hist = defaultdict(list)
rows = []
for rid, lane, ao in raw.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    d = races_date.get(rid)
    reg = lane_racer.get((rid, lane))
    if d and reg:
        rows.append((d, reg, ao))
rows.sort()
for d, reg, ao in rows:
    hist[reg].append((d, 1 if ao <= 2 else 0))
hist_dates = {reg: [x[0] for x in v] for reg, v in hist.items()}
raw.close()


def gap_of(rid, lane):
    reg = lane_racer.get((rid, lane))
    n2 = printed.get((rid, lane))
    d = races_date.get(rid)
    if reg is None or n2 is None or d is None or reg not in hist:
        return None
    dates = hist_dates[reg]
    hi = bisect_left(dates, d)
    d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=90)).isoformat()
    lo = bisect_left(dates, d0)
    seg = hist[reg][lo:hi]
    if len(seg) < 12:
        return None
    return sum(t for _, t in seg) / len(seg) - n2 / 100.0


def rising_flag(rid, lanes):
    return any((g := gap_of(rid, l)) is not None and g > 0.10 for l in lanes[:4])


def kr_flag(rid, lanes):
    vals = [(kr.get((rid, l)), l) for l in lanes]
    if any(v is None for v, _ in vals):
        return False
    return (klass.get((rid, max(vals)[1])) or "").startswith("B")


# ---- プラン ----------------------------------------------------------------
def an1_plan(ranked):
    plan = P.ken_portfolio("荒れ注意", ranked, [],
                           P.picks_katsu(P.normalize_probs(ranked)), konsen=True)
    return [(bt, comb, y) for bt, comb, y, _s in plan]


def henkei_ha_plan(ranked):
    lanes = [r["lane"] for r in ranked]
    r1, r2, r3, r4, r5 = lanes[:5]
    probs = P.normalize_probs(ranked)
    tri = P.trifecta_probs(probs)

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"

    def top1(members):
        cands = [(o, p) for o, p in tri.items() if set(o) == set(members)]
        a, b, c = max(cands, key=lambda x: x[1])[0]
        return f"{a}-{b}-{c}"
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 100),
            ("3連単", top1((r1, r2, r3)), 100), ("3連単", top1((r1, r2, r4)), 100),
            ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200),
            ("3連複", trio(r3, r4, r5), 100)]


def honmei_plan(ranked):
    plan = P.ken_portfolio("荒れ注意", ranked, [],
                           P.picks_katsu(P.normalize_probs(ranked)))
    return [(bt, comb, y) for bt, comb, y, _s in plan]


# ---- シミュレーション -------------------------------------------------------
conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

ARMS = ("基準v2.1", "変形ハ", "伸び盛り増額", "KR旗増額", "旗統合増額", "外枠人気除外")
agg = defaultdict(lambda: [0, 0, 0, 0])   # {(arm, month): [st, rt, n, hit]}

for m in MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])

    ctx_by_day = defaultdict(list)
    for rid, g in month_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        ctx_by_day[g["date"].iloc[0]].append({
            "rid": rid, "ranked": ranked, "top": ranked[0]["prob"],
            "in5": int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES,
        })

    for d, cs in ctx_by_day.items():
        konsen = {c["rid"] for c in cs if c["top"] < KONSEN_MAX}
        pool = sorted((c for c in cs if c["in5"] and c["top"] < HONMEI_MAX),
                      key=lambda c: c["top"])
        honmei = {c["rid"] for c in pool[:CAP]}
        for c in cs:
            rid = c["rid"]
            lanes = [r["lane"] for r in c["ranked"]]
            pay = payout_map[rid]
            if rid in honmei:
                base = honmei_plan(c["ranked"])
                fav = lanes[0]
                for arm in ARMS:
                    if arm == "外枠人気除外" and fav >= 4:
                        continue          # 買わない
                    plan = base
                    st = sum(y for _, _, y in plan)
                    rt = sum(pay.get((bt, comb), 0) * y // 100
                             for bt, comb, y in plan)
                    a = agg[(arm, m)]
                    a[0] += st; a[1] += rt; a[2] += 1; a[3] += 1 if rt else 0
            elif rid in konsen:
                a1 = an1_plan(c["ranked"])
                if not a1:
                    continue
                ha = henkei_ha_plan(c["ranked"])
                rf = rising_flag(rid, lanes)
                kf = kr_flag(rid, lanes)
                for arm in ARMS:
                    plan = ha if arm == "変形ハ" else a1
                    mult = 1
                    if (arm == "伸び盛り増額" and rf) or \
                       (arm == "KR旗増額" and kf) or \
                       (arm == "旗統合増額" and (rf or kf)):
                        mult = 2
                    st = sum(y for _, _, y in plan) * mult
                    rt = sum(pay.get((bt, comb), 0) * y * mult // 100
                             for bt, comb, y in plan)
                    a = agg[(arm, m)]
                    a[0] += st; a[1] += rt; a[2] += 1; a[3] += 1 if rt else 0

print(f"\n===== 月別回収率 =====")
print("月       " + "".join(f"{a:<12}" for a in ARMS))
for m in MONTHS:
    row = f"{m}  "
    for arm in ARMS:
        st, rt, _n, _h = agg[(arm, m)]
        row += f"{(rt/st if st else 0):>8.1%}    "
    print(row)

print(f"\n===== 合計(2025-12〜2026-05) =====")
print(f"{'アーム':<10}{'R数':>6}{'投資':>12}{'的中率':>8}{'回収率':>8}{'損益':>12}{'基準比損益':>11}")
base_pnl = None
for arm in ARMS:
    st = sum(agg[(arm, m)][0] for m in MONTHS)
    rt = sum(agg[(arm, m)][1] for m in MONTHS)
    n = sum(agg[(arm, m)][2] for m in MONTHS)
    h = sum(agg[(arm, m)][3] for m in MONTHS)
    pnl = rt - st
    if arm == "基準v2.1":
        base_pnl = pnl
    d = f"{pnl-base_pnl:+,}" if arm != "基準v2.1" else "—"
    print(f"{arm:<10}{n:>6,}{st:>11,}円{h/n:>8.1%}{rt/st:>8.1%}{pnl:>+11,}円{d:>11}")
