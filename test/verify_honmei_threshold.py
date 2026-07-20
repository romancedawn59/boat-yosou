# -*- coding: utf-8 -*-
"""本命閾値グラデーションの検証CLI(2026-07-19発見 → 2026-07-20採用判断の根拠)

    py -X utf8 test/verify_honmei_threshold.py

v2選別(本命=5場×荒れ注意×上位6 + 超混戦=全場×20%未満)のうち、本命の判定閾値
(従来35%)を 35/30/28/25/22% と振って比較する。超混戦20%・cap6・買い目構成
(検証済みV2構成6点)は固定。ウォークフォワードはbacktest.py同一fold。

発見(2026-07-19): 28〜35%帯は利益貢献ゼロの詰め物(損益は35/30/28でほぼ横ばい)。
30%は利益を保ったまま投資-25%・DD-30%になる。帯別層別(25〜30%帯=回収率97%)・
他19場の独立再現(20〜35%相当は除き96.3%)とも整合。
→ 2026-07-20 ユーザー決定: 閾値30%を採用(config.HONMEI_PROB_MAX)。
  25%以下は利益を削る(効率最大化)ため今回は不採用。8月末の実弾+要注目観測で再判断。

結果は test/backtest_report_honmei_threshold.html に保存する。
注意: DDはfoldモデルの揺らぎに敏感。点推定を過信しない。
"""
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

KONSEN_MAX = 0.20
CAP = 6
THRESHOLDS = (0.35, 0.30, 0.28, 0.25, 0.22)

conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]
n_days = len(dates)

ctxs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if 1 not in actual[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                     "venue": int(g["venue_code"].iloc[0]),
                     "top": ranked[0]["prob"], "ranked": ranked, "probs": probs})
ctxs.sort(key=lambda c: (c["date"], c["rid"]))

graded = {}
def grade(c):
    if c["rid"] in graded:
        return graded[c["rid"]]
    plan = P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(c["probs"]))
    pay = payout_map[c["rid"]]
    stake = sum(y for _, _, y, _ in plan)
    ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
    res = actual[c["rid"]]
    santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
    graded[c["rid"]] = {"stake": stake, "ret": ret,
                        "out": CH.classify_outcome(stake, ret, santan)}
    return graded[c["rid"]]

def select_day(day_ctxs, honmei_max):
    """v2選別(本命閾値=honmei_max・cap6+全場超混戦20%)の購入レース集合"""
    hon = sorted((c for c in day_ctxs
                  if c["venue"] in TARGET_VENUE_CODES and c["top"] < honmei_max),
                 key=lambda c: c["top"])
    kon = [c for c in day_ctxs if c["top"] < KONSEN_MAX]
    return {c["rid"] for c in hon[:CAP]} | {c["rid"] for c in kon}

rows = []
print(f"\n=== 本命閾値グラデーション(walk-forward {n_days}日・超混戦20%固定・cap6) ===")
print(f"{'本命閾値':<8}{'R数':>6}{'R/日':>6}{'日予算平均':>9}{'的中率':>8}{'回収率':>8}"
      f"{'最大1発除き':>10}{'ガミ率':>7}{'損益':>12}{'最大DD':>10}{'最長連敗':>7}")
for th in THRESHOLDS:
    daily_pnl = {}
    tot = {"n": 0, "hits": 0, "junto": 0, "stake": 0, "ret": 0, "max_ret": 0}
    day_stakes = []
    for d, grp in groupby(ctxs, key=lambda c: c["date"]):
        day_list = list(grp)
        sel = select_day(day_list, th)
        pnl = 0.0
        dstake = 0
        for c in day_list:
            if c["rid"] not in sel:
                continue
            g = grade(c)
            tot["n"] += 1
            tot["stake"] += g["stake"]
            tot["ret"] += g["ret"]
            tot["max_ret"] = max(tot["max_ret"], g["ret"])
            dstake += g["stake"]
            pnl += g["ret"] - g["stake"]
            if g["ret"]:
                tot["hits"] += 1
                if g["out"] == "順当":
                    tot["junto"] += 1
        daily_pnl[d] = pnl
        day_stakes.append(dstake)
    series = [daily_pnl[d] for d in sorted(daily_pnl)]
    roi = tot["ret"] / tot["stake"]
    roi_ex = (tot["ret"] - tot["max_ret"]) / tot["stake"]
    junto = tot["junto"] / tot["hits"] if tot["hits"] else 0
    avg_budget = sum(day_stakes) / len(day_stakes)
    row = {"th": th, "n": tot["n"], "r_day": tot["n"] / n_days, "budget": avg_budget,
           "hit": tot["hits"] / tot["n"], "roi": roi, "roi_ex": roi_ex, "junto": junto,
           "pnl": tot["ret"] - tot["stake"], "dd": CH.max_drawdown(series),
           "streak": CH.longest_losing_streak(series)}
    rows.append(row)
    print(f"{th:<8.0%}{row['n']:>6}{row['r_day']:>6.1f}{row['budget']:>8,.0f}円"
          f"{row['hit']:>8.1%}{roi:>8.1%}{roi_ex:>10.1%}{junto:>7.1%}"
          f"{row['pnl']:>+11,.0f}円{row['dd']:>9,.0f}円{row['streak']:>6}日")

# HTMLレポート(不採用の根拠も残す文化)
report = Path(__file__).with_name("backtest_report_honmei_threshold.html")
trs = "\n".join(
    f"<tr{' class=chosen' if r['th'] == 0.30 else ''}><td>{r['th']:.0%}</td>"
    f"<td>{r['n']}</td><td>{r['r_day']:.1f}</td><td>{r['budget']:,.0f}円</td>"
    f"<td>{r['hit']:.1%}</td><td>{r['roi']:.1%}</td><td>{r['roi_ex']:.1%}</td>"
    f"<td>{r['junto']:.1%}</td><td>{r['pnl']:+,.0f}円</td><td>{r['dd']:,.0f}円</td>"
    f"<td>{r['streak']}日</td></tr>" for r in rows)
report.write_text(f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>検証: 本命閾値グラデーション(walk-forward {n_days}日)</title>
<style>
body{{font-family:sans-serif;max-width:960px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse}}td,th{{border:1px solid #999;padding:4px 10px;text-align:right}}
th{{background:#eee}}tr.chosen{{background:#e6f4e6;font-weight:bold}}
</style></head><body>
<h1>検証: 本命の判定閾値グラデーション</h1>
<p>v2選別(本命=5場×荒れ注意×上位{CAP}、超混戦=全場×{KONSEN_MAX:.0%}未満固定)の
本命閾値を振って比較。walk-forward {n_days}日・backtest.py同一fold・
買い目は全ルール共通の検証済みV2構成6点。再実行:
<code>py -X utf8 test/verify_honmei_threshold.py</code></p>
<table>
<tr><th>本命閾値</th><th>R数</th><th>R/日</th><th>日予算平均</th><th>的中率</th>
<th>回収率</th><th>最大1発除き</th><th>ガミ率</th><th>損益</th><th>最大DD</th><th>最長連敗</th></tr>
{trs}
</table>
<h2>結論(2026-07-20 ユーザー決定)</h2>
<ul>
<li><b>30%を採用</b>(config.HONMEI_PROB_MAX)。28〜35%帯は利益貢献ゼロの詰め物で、
30%は利益を保ったまま投資・DDを減らす。帯別層別(25〜30%帯=回収率97%)・
他19場の独立再現(20〜35%相当は最大1発除き96.3%)と整合し、
「利益を追わずリスクだけ捨てる」変更のため後知恵最適化の懸念が小さい</li>
<li><b>25%・22%は不採用</b>: 効率(回収率)は上がるが損益を削る(25〜28帯は利益貢献あり)。
8月末の実弾データ(ken_hon/ken_konsen)+要注目の前向き観測で再判断する</li>
<li>切った30〜35%帯は「要注目」観測枠で毎晩紙上採点を継続
(採用日を起点にした事前登録の前向き検証)</li>
<li>注意: DDはfoldモデルの揺らぎに敏感。点推定を過信しない</li>
</ul>
</body></html>
""", encoding="utf-8")
print(f"\nレポート: {report}")
