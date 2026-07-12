# -*- coding: utf-8 -*-
"""検証⑪-B: 買い目構成チャレンジャーの遡及バックテスト(紙上専用)

    py -X utf8 test/verify_compositions.py

検証⑪-Aの双対: 選別をチャンピオン(現行=荒れ注意、1日上限10)に固定し、
買い目構成4案(現行/攻撃型/守備型/軸分散型。定義はchallengers.build_composition)を
同一fold・同一レース集合で比較する。ウォークフォワードはverify_challengers.pyと共通。

回収率・的中率・決着分布に加え、日次損益系列から最大ドローダウンと最長連敗を必ず
併記する(攻撃型が回収率で勝っても「谷の深さに耐えられるか」は数字ではなく
ケンさんの意思決定の領分。その判断材料を出すのが本レポートの主目的)。

出力: test/backtest_report_compositions.html / test/verify_compositions_results.json
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, TEST_START
from config import DB_PATH, JST, PROJECT_DIR
from verify_challengers import SMALL_SAMPLE_NOTE, select_with_cap, walk_forward_contexts

TEST_DIR = PROJECT_DIR / "test"


def grade_composition(name: str, ctx: dict, actual: dict, payout_map: dict) -> dict:
    plan = CH.build_composition(name, ctx["ranked"], P.picks_katsu(ctx["probs"]))
    pay = payout_map[ctx["race_id"]]
    stake = sum(y for _, _, y, _ in plan)
    ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
    res = actual[ctx["race_id"]]
    santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
    return {"date": ctx["date"], "stake": stake, "ret": ret,
            "outcome": CH.classify_outcome(stake, ret, santan)}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [r for r in rows if r["ret"]]
    stake = sum(r["stake"] for r in rows)
    ret = sum(r["ret"] for r in rows)
    max_ret = max((r["ret"] for r in rows), default=0)
    dist = {k: sum(1 for r in hits if r["outcome"] == k) for k in ("順当", "中波乱", "万舟")}

    # 日次損益系列(攻撃型の谷の深さを見るための主役)
    daily = defaultdict(float)
    for r in rows:
        daily[r["date"]] += r["ret"] - r["stake"]
    series = [daily[d] for d in sorted(daily)]
    return {
        "n": n, "hits": len(hits), "hit_rate": len(hits) / n if n else 0.0,
        "stake": stake, "ret": ret, "roi": ret / stake if stake else 0.0,
        "roi_excl_max": (ret - max_ret) / stake if stake else 0.0,
        "dist": dist, "profit": ret - stake,
        "max_drawdown": CH.max_drawdown(series),
        "longest_losing_days": CH.longest_losing_streak(series),
        "n_days": len(series),
    }


def main():
    conn = db.connect(DB_PATH)
    folds, actual, payout_map = walk_forward_contexts(conn)
    conn.close()

    # 選別はチャンピオン固定(1日上限10)
    selected_ctx: list[dict] = []
    for fold in folds:
        sel, _ = select_with_cap(fold, lambda c: CH.champion_score(c["ranked"]))
        selected_ctx.extend(c for c in fold if c["race_id"] in sel)
    selected_ctx.sort(key=lambda c: (c["date"], c["race_id"]))
    print(f"チャンピオン選別: {len(selected_ctx)}レース")

    results = {"n_selected": len(selected_ctx), "compositions": {}, "folds": {}}
    for name in CH.COMPOSITION_NAMES:
        rows = [grade_composition(name, c, actual, payout_map) for c in selected_ctx]
        results["compositions"][name] = summarize(rows)
        # fold別サマリ(rowsとfoldsの対応はctx順で一致)
        per_fold = []
        for i in range(1, N_FOLDS + 1):
            fr = [r for r, c in zip(rows, selected_ctx) if c["fold"] == i]
            per_fold.append(summarize(fr))
        results["folds"][name] = per_fold
        s = results["compositions"][name]
        print(f"{name}: 的中{s['hit_rate']:.1%} 回収率{s['roi']:.1%} "
              f"(最大1発除き{s['roi_excl_max']:.1%}) 損益{s['profit']:+,}円 "
              f"最大DD{s['max_drawdown']:,.0f}円 最長連敗{s['longest_losing_days']}日")

    results["updated"] = datetime.now(JST).isoformat(timespec="seconds")
    json_path = TEST_DIR / "verify_compositions_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (TEST_DIR / "backtest_report_compositions.html").write_text(
        render_report(results), encoding="utf-8")
    print(f"出力: {json_path}")
    print(f"出力: {TEST_DIR / 'backtest_report_compositions.html'}")


def render_report(r: dict) -> str:
    rows = []
    for name, s in r["compositions"].items():
        roi_cls = "pos" if s["roi"] >= 1 else "neg"
        ex_cls = "pos" if s["roi_excl_max"] >= 1 else "neg"
        pr_cls = "pos" if s["profit"] >= 0 else "neg"
        d = s["dist"]
        rows.append(
            f"<tr{' class=adopt' if name == '現行' else ''}><td>{name}</td>"
            f"<td class='num'>{s['hit_rate']:.1%}</td>"
            f"<td class='num {roi_cls}'>{s['roi']:.1%}</td>"
            f"<td class='num {ex_cls}'>{s['roi_excl_max']:.1%}</td>"
            f"<td class='num {pr_cls}'>{s['profit']:+,}円</td>"
            f"<td class='num neg'><b>{s['max_drawdown']:,.0f}円</b></td>"
            f"<td class='num'><b>{s['longest_losing_days']}日</b></td>"
            f"<td class='num'>{d['順当']}/{d['中波乱']}/{d['万舟']}</td></tr>")

    fold_rows = []
    for name in r["folds"]:
        for i, s in enumerate(r["folds"][name], 1):
            if s["n"] == 0:
                continue
            roi_cls = "pos" if s["roi"] >= 1 else "neg"
            fold_rows.append(
                f"<tr><td>fold{i}</td><td>{name}</td><td class='num'>{s['n']}</td>"
                f"<td class='num {roi_cls}'>{s['roi']:.1%}</td>"
                f"<td class='num'>{s['max_drawdown']:,.0f}円</td>"
                f"<td class='num'>{s['longest_losing_days']}日</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>検証⑪-B 買い目構成チャレンジャー(紙上)</title>
<style>
  body {{ font-family: sans-serif; margin: 0 auto; padding: 12px; background: #f6f8fa; max-width: 900px; }}
  h1 {{ font-size: 1.2rem; margin: 10px 4px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin-bottom: 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 6px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #1a7f37; font-weight: bold; }}
  .neg {{ color: #cf222e; }}
  .adopt {{ background: #d6efff55; }}
  .note {{ font-size: .78rem; color: #57606a; margin: 6px 4px; }}
  .warn {{ background: #fff8c5; border: 1px solid #d4a72c66; border-radius: 8px;
           padding: 8px 12px; font-size: .8rem; margin-bottom: 14px; }}
</style>
</head>
<body>
<h1>検証⑪-B 買い目構成チャレンジャー(すべて紙上・本番不変)</h1>
<p class="note">生成: {r['updated']} / 選別はチャンピオン(現行の荒れ注意・1日上限10)に固定した
{r['n_selected']}レースで、構成4案を同一fold・同一レース集合で比較。
ウォークフォワードは検証⑪-Aと共通({TEST_START}以降{N_FOLDS}分割)。
導入判断は行わない(2026-09-01のv2判断材料)。</p>
<div class="warn">{SMALL_SAMPLE_NOTE}<br>
回収率だけでなく<b>最大ドローダウンと最長連敗</b>を必ず見ること。
攻撃型が回収率で勝っても、谷に耐えられるかは資金と気持ちの問題であり、
その判断は数字ではなくケンさんの領分。</div>

<div class="card">
  <h2 style="margin-top:0">構成4案の比較(全fold合計・{r['n_selected']}レース)</h2>
  <table>
    <tr><th>構成</th><th class="num">的中率</th><th class="num">回収率</th>
        <th class="num">最大1発除き</th><th class="num">損益</th>
        <th class="num">最大DD</th><th class="num">最長連敗</th>
        <th class="num">順当/中波乱/万舟</th></tr>
    {''.join(rows)}
  </table>
  <p class="note">構成の定義: 現行=3連複200/200/100+3連単200/200+C100(検証済みV2) /
  攻撃型=3連複300/200+3連単100×4+C100(原案の250×2は舟券の100円単位制約で購入不可のため、
  結果を見る前に1=2=3を厚くする形へ事前調整) /
  守備型=3連複300/300/200/100+C100(3連単なし。仮置き配分がそのまま制約を満たすため無調整で採用) /
  軸分散型=現行の3連複3点目(1=3=4)を予測1位不在の2=3=4に置換。
  すべて合計1,000円・C勝万舟100円共通(C候補0点時は900円)。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">fold別(構成×期間)</h2>
  <table>
    <tr><th>fold</th><th>構成</th><th class="num">レース数</th>
        <th class="num">回収率</th><th class="num">最大DD</th><th class="num">最長連敗</th></tr>
    {''.join(fold_rows)}
  </table>
</div>

<p class="note">再実行: py -X utf8 test/verify_compositions.py(冪等・全期間を毎回再計算)</p>
</body>
</html>
"""


if __name__ == "__main__":
    main()
