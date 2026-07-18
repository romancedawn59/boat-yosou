# -*- coding: utf-8 -*-
"""ケンさんv2案(現行cap6+全場×20%未満)の2026年5〜6月固定シミュレーションCLI

    py -X utf8 test/verify_ken_v2_202605_06.py

検証⑦⑧⑨と同じ枠組み: 2026-05-01より前の全データで1回だけ学習し、
2026-05-01〜06-30を評価(ケンさん案は全24場が対象なので評価も全場)。
walk-forward版(test/verify_ken_v2.py・229日)の補完で、既存の5〜6月系列
(検証⑧の本命152.6%等)と同じ土俵の数字を出す。月別内訳・最大DD・最長連敗つき。

出力: test/backtest_report_202605-06_ken_v2.html
"""
import sys
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import challengers as CH
import db
import predictors as P
from backtest import train_fold
from config import DB_PATH, JST, PROJECT_DIR, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

EVAL_START, EVAL_END = "2026-05-01", "2026-06-30"
TEST_DIR = PROJECT_DIR / "test"

RULES = ("現行cap10", "現行cap6", "全場20%のみ", "ケンさん案")


def build_contexts() -> tuple[list[dict], dict, dict, int]:
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)
    actual = defaultdict(dict)
    for rid, lane, order in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date BETWEEN ? AND ? AND res.arrival_order IS NOT NULL",
        (EVAL_START, EVAL_END)):
        actual[rid][order] = lane
    payout_map = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id WHERE r.date BETWEEN ? AND ?",
        (EVAL_START, EVAL_END)):
        payout_map[rid][(bt, comb)] = amt or 0
    conn.close()

    train_df = df[df["date"] < EVAL_START]
    eval_df = df[(df["date"] >= EVAL_START) & (df["date"] <= EVAL_END)].copy()  # 全場
    print(f"学習 {len(train_df):,}行 / 評価 {len(eval_df):,}行(全24場)")
    booster = train_fold(train_df)
    eval_df["pred"] = booster.predict(eval_df[FEATURE_COLUMNS])

    ctxs = []
    for rid, g in eval_df.groupby("race_id"):
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
    return ctxs, actual, payout_map, len(train_df)


def select_day(day_ctxs: list[dict], rule: str) -> set[str]:
    """1日分の購入レース集合(verify_ken_v2.pyと同一定義)"""
    hon = sorted((c for c in day_ctxs
                  if c["venue"] in TARGET_VENUE_CODES and c["top"] < 0.35),
                 key=lambda c: c["top"])
    kon = [c for c in day_ctxs if c["top"] < 0.20]
    if rule == "現行cap10":
        return {c["rid"] for c in hon[:10]}
    if rule == "現行cap6":
        return {c["rid"] for c in hon[:6]}
    if rule == "全場20%のみ":
        return {c["rid"] for c in kon}
    if rule == "ケンさん案":
        return {c["rid"] for c in hon[:6]} | {c["rid"] for c in kon}
    raise ValueError(rule)


def run() -> dict:
    ctxs, actual, payout_map, train_rows = build_contexts()
    n_days = len({c["date"] for c in ctxs})

    graded: dict[str, dict] = {}

    def grade(c):
        if c["rid"] not in graded:
            plan = P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(c["probs"]))
            pay = payout_map[c["rid"]]
            stake = sum(y for _, _, y, _ in plan)
            ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
            res = actual[c["rid"]]
            santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
            graded[c["rid"]] = {"stake": stake, "ret": ret,
                                "out": CH.classify_outcome(stake, ret, santan)}
        return graded[c["rid"]]

    results = {"train_rows": train_rows, "n_days": n_days, "rules": {}, "monthly": {}}
    for rule in RULES:
        daily_pnl = {}
        tot = defaultdict(float)
        monthly = {m: defaultdict(float) for m in ("2026-05", "2026-06")}
        for d, grp in groupby(ctxs, key=lambda c: c["date"]):
            day_list = list(grp)
            sel = select_day(day_list, rule)
            pnl = 0.0
            for c in day_list:
                if c["rid"] not in sel:
                    continue
                g = grade(c)
                for t in (tot, monthly[d[:7]]):
                    t["n"] += 1
                    t["stake"] += g["stake"]
                    t["ret"] += g["ret"]
                    t["max_ret"] = max(t["max_ret"], g["ret"])
                    if g["ret"]:
                        t["hits"] += 1
                        if g["out"] == "順当":
                            t["junto"] += 1
                pnl += g["ret"] - g["stake"]
            daily_pnl[d] = pnl
        series = [daily_pnl[d] for d in sorted(daily_pnl)]
        tot["dd"] = CH.max_drawdown(series)
        tot["streak"] = CH.longest_losing_streak(series)
        results["rules"][rule] = dict(tot)
        results["monthly"][rule] = {m: dict(v) for m, v in monthly.items()}
        roi = tot["ret"] / tot["stake"]
        print(f"{rule}: {tot['n']:.0f}R 回収率{roi:.1%} "
              f"(除き{(tot['ret']-tot['max_ret'])/tot['stake']:.1%}) "
              f"損益{tot['ret']-tot['stake']:+,.0f}円 DD{tot['dd']:,.0f}円 連敗{tot['streak']:.0f}日")
    return results


def render(r: dict) -> str:
    def row(label, s, n_days, adopt=False):
        if not s.get("stake"):
            return f"<tr><td>{label}</td><td colspan='9'>対象なし</td></tr>"
        roi = s["ret"] / s["stake"]
        roi_ex = (s["ret"] - s["max_ret"]) / s["stake"]
        junto = s["junto"] / s["hits"] if s.get("hits") else 0
        profit = s["ret"] - s["stake"]
        return (f"<tr{' class=adopt' if adopt else ''}><td>{label}</td>"
                f"<td class='num'>{s['n']:.0f}</td>"
                f"<td class='num'>{s['n']/n_days:.1f}</td>"
                f"<td class='num'>{s['hits']/s['n']:.1%}</td>"
                f"<td class='num {'pos' if roi>=1 else 'neg'}'>{roi:.1%}</td>"
                f"<td class='num {'pos' if roi_ex>=1 else 'neg'}'>{roi_ex:.1%}</td>"
                f"<td class='num'>{junto:.1%}</td>"
                f"<td class='num {'pos' if profit>=0 else 'neg'}'>{profit:+,.0f}円</td>"
                f"<td class='num'>{s['dd']:,.0f}円</td>"
                f"<td class='num'>{s['streak']:.0f}日</td></tr>")

    rule_rows = "".join(row(rule, r["rules"][rule], r["n_days"], adopt=(rule == "ケンさん案"))
                        for rule in RULES)
    month_rows = []
    for rule in RULES:
        for m in ("2026-05", "2026-06"):
            s = r["monthly"][rule][m]
            if not s.get("stake"):
                continue
            roi = s["ret"] / s["stake"]
            profit = s["ret"] - s["stake"]
            month_rows.append(
                f"<tr><td>{rule}</td><td>{m}</td><td class='num'>{s['n']:.0f}</td>"
                f"<td class='num'>{s['stake']:,.0f}円</td>"
                f"<td class='num {'pos' if roi>=1 else 'neg'}'>{roi:.1%}</td>"
                f"<td class='num {'pos' if profit>=0 else 'neg'}'>{profit:+,.0f}円</td></tr>")

    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ケンさんv2案 2026年5〜6月シミュレーション</title>
<style>
  body {{ font-family: sans-serif; margin: 0 auto; padding: 12px; background: #f6f8fa; max-width: 900px; }}
  h1 {{ font-size: 1.2rem; margin: 10px 4px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin-bottom: 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .83rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 6px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .pos {{ color: #1a7f37; font-weight: bold; }}
  .neg {{ color: #cf222e; }}
  .adopt {{ background: #d6efff55; }}
  .note {{ font-size: .78rem; color: #57606a; margin: 6px 4px; }}
</style>
</head>
<body>
<h1>ケンさんv2案(現行MAX6+全場×20%未満)の2026年5〜6月シミュレーション</h1>
<p class="note">実施: {updated} / 検証⑦⑧⑨と同じ枠組み(2026-05-01より前の{r['train_rows']:,}行で1回学習→
5〜6月{r['n_days']}日を評価。ケンさん案は全24場が対象のため評価も全場)。
買い目は全ルール共通の検証済みV2構成+C勝万舟。walk-forward版(229日)は test/verify_ken_v2.py 参照。</p>

<div class="card">
  <h2 style="margin-top:0">ルール別比較(2026-05-01〜06-30)</h2>
  <table>
    <tr><th>ルール</th><th class="num">R数</th><th class="num">R/日</th><th class="num">的中率</th>
        <th class="num">回収率</th><th class="num">最大1発除き</th><th class="num">ガミ率</th>
        <th class="num">損益</th><th class="num">最大DD</th><th class="num">最長連敗</th></tr>
    {rule_rows}
  </table>
  <p class="note">ルール定義: 現行=5場×1位勝率35%未満(優先度は1位勝率が低い順・日上限10または6) /
  全場20%=全24場×1位勝率20%未満(上限なし) / ケンさん案=現行cap6と全場20%の和集合
  (重複レースは1回だけ購入)。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">月別内訳</h2>
  <table>
    <tr><th>ルール</th><th>月</th><th class="num">R数</th><th class="num">投資</th>
        <th class="num">回収率</th><th class="num">損益</th></tr>
    {month_rows}
  </table>
</div>

<p class="note">⚠ 単一期間(5〜6月)・単一モデルでの評価。帯境界20%は5場の負け分析由来の後知恵で、
他19場での独立再現(walk-forward・除き200%)が主な頑健性の根拠。
7/30-31のv2判断会では本レポートとwalk-forward版の両方を見て決めること。
再実行: py -X utf8 test/verify_ken_v2_202605_06.py</p>
</body>
</html>
"""


if __name__ == "__main__":
    results = run()
    out = TEST_DIR / "backtest_report_202605-06_ken_v2.html"
    out.write_text(render(results), encoding="utf-8")
    print(f"レポート出力: {out}")
