# -*- coding: utf-8 -*-
"""検証⑪-A: 選別チャンピオン/チャレンジャーの遡及バックテスト(紙上専用)

    py -X utf8 test/verify_challengers.py

backtest.pyと同じウォークフォワード(2025-12-01以降を5分割、各期間はそれより前の
データだけで学習したfoldモデルで予測)。現行MODEL_PATHは使わない。
買い目は全選別器とも predictors.ken_portfolio の荒れ注意構成(検証済みV2)・
1日上限10レースに統一し、比較の変数を「選別」だけに絞る。

選別器:
- チャンピオン: 現行「1位勝率(生値)35%未満=荒れ注意」
- 挑戦者②β1(1位-2位差) / β2(エントロピー): 閾値は遡及期間全体で選別数が
  チャンピオンと±10%に揃うよう較正(challengers.calibrate_threshold参照)
- 挑戦者①市場相違型: 15分前スナップショットの人気順位とモデル順位の
  スピアマン距離。スナップショット無しは判定不能=選別しない(件数を明示)。
  閾値は暫定値(challengers.MARKET_DIVERGENCE_THRESHOLD、8月末に(d)層別で確定)
- 挑戦者③C条件型: スタブ(蓄積待ち。大穴一撃フラグ構想と同件)

検算: チャンピオンの数字は src/backtest.py の「ken現行構成[5場]荒れ注意」と
同一計算経路であり、検証⑧(5〜6月固定・単一モデル)の152.6%とはfold境界と
学習データが異なるため完全一致はしないが、同水準になることを確認する。

出力: test/backtest_report_challengers.html /
      test/verify_challengers_results.json(ガミり監視の基準値もここに保存)
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path
from statistics import pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, JST, PROJECT_DIR, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

TEST_DIR = PROJECT_DIR / "test"

SMALL_SAMPLE_NOTE = ("⚠ 少サンプル注意: fold単位・差分レースの数字は数十件規模。"
                     "傾向の参考に留め、順位の断定は9/1の判断材料が揃ってから。")


# ===== 共通データ構築(verify_compositions.pyからも流用する) =====

def load_market_orders(conn) -> dict[str, list[int]]:
    """15分前スナップショットから各レースの人気順位(1着含意確率の降順)を作る。
    final-backfill行(最終オッズ)はスナップショットではないため使わない"""
    win_imp: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for rid, comb, odds in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連単' AND fetched_at != 'final-backfill' AND odds > 0"
    ):
        win_imp[rid][int(comb.split("-")[0])] += 1.0 / odds
    return {rid: sorted(d, key=lambda l: -d[l]) for rid, d in win_imp.items()}


def load_results_and_payouts(conn) -> tuple[dict, dict]:
    actual = defaultdict(dict)
    for rid, lane, order in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,),
    ):
        actual[rid][order] = lane
    payout_map = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,),
    ):
        payout_map[rid][(bt, comb)] = amt or 0
    return actual, payout_map


def walk_forward_contexts(conn) -> tuple[list[list[dict]], dict, dict]:
    """fold別のレースコンテキスト一覧と(actual, payout_map)を返す。

    ctx = {race_id, date, fold, ranked(予測降順), probs(正規化), market_order(無ければNone)}
    backtest.pyと同一のfold分割・学習手順(train_fold)を使う。
    """
    df = build_training_set(conn)
    market_orders = load_market_orders(conn)
    actual, payout_map = load_results_and_payouts(conn)

    test_df = df[df["date"] >= TEST_START]
    dates = sorted(test_df["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

    folds: list[list[dict]] = []
    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)
                     & (df["venue_code"].isin(TARGET_VENUE_CODES))].copy()
        print(f"fold{i+1} 学習中({f_start}〜, 学習{len(train_df):,}行)...")
        booster = train_fold(train_df)
        fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])

        contexts = []
        for rid, g in fold_df.groupby("race_id"):
            if 1 not in actual[rid]:
                continue
            g_sorted = g.sort_values("pred", ascending=False)
            ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                      for _, r in g_sorted.iterrows()]
            probs = P.normalize_probs(ranked)
            if len(probs) < 4:
                continue
            contexts.append({
                "race_id": rid, "date": str(g["date"].iloc[0]), "fold": i + 1,
                "ranked": ranked, "probs": probs,
                "market_order": market_orders.get(rid),
            })
        contexts.sort(key=lambda c: (c["date"], c["race_id"]))
        folds.append(contexts)
    return folds, actual, payout_map


def plan_for(ctx: dict) -> list[tuple]:
    """全選別器共通の買い目(検証済みV2構成)。b_picksは荒れ注意構成では未使用"""
    return P.ken_portfolio("荒れ注意", ctx["ranked"], [], P.picks_katsu(ctx["probs"]))


def grade(ctx: dict, actual: dict, payout_map: dict) -> dict:
    """1レースの紙上採点: stake/ret/決着分類"""
    plan = plan_for(ctx)
    pay = payout_map[ctx["race_id"]]
    stake = sum(y for _, _, y, _ in plan)
    ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
    res = actual[ctx["race_id"]]
    santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
    return {"stake": stake, "ret": ret,
            "outcome": CH.classify_outcome(stake, ret, santan)}


def select_with_cap(contexts: list[dict], score_fn) -> tuple[set[str], int]:
    """日ごとにscore降順・上限10で選別。(選別race_id集合, 判定不能件数)を返す"""
    selected: set[str] = set()
    unjudgeable = 0
    for _d, grp in groupby(contexts, key=lambda c: c["date"]):
        cands = []
        for ctx in grp:
            score = score_fn(ctx)
            if score is None:
                if score_fn is not None and ctx.get("_unjudgeable"):
                    unjudgeable += 1
                continue
            cands.append((score, ctx["race_id"]))
        selected.update(CH.daily_cap(cands))
    return selected, unjudgeable


def agg_stats(rids: set[str], graded: dict[str, dict]) -> dict:
    rows = [graded[r] for r in rids if r in graded]
    n = len(rows)
    hits = [r for r in rows if r["ret"]]
    stake = sum(r["stake"] for r in rows)
    ret = sum(r["ret"] for r in rows)
    max_ret = max((r["ret"] for r in rows), default=0)
    dist = {k: sum(1 for r in hits if r["outcome"] == k) for k in ("順当", "中波乱", "万舟")}
    return {
        "n": n, "hits": len(hits), "hit_rate": len(hits) / n if n else 0.0,
        "stake": stake, "ret": ret, "roi": ret / stake if stake else 0.0,
        "roi_excl_max": (ret - max_ret) / stake if stake else 0.0,
        "dist": dist,
        "junto_rate": dist["順当"] / len(hits) if hits else 0.0,  # ガミり監視の基準
    }


def main():
    conn = db.connect(DB_PATH)
    folds, actual, payout_map = walk_forward_contexts(conn)
    conn.close()

    all_ctx = [c for fold in folds for c in fold]
    graded = {c["race_id"]: grade(c, actual, payout_map) for c in all_ctx}
    n_days = len({c["date"] for c in all_ctx})

    # --- 閾値の較正(チャンピオンのcap前選別数に揃える) ---
    champion_raw = [c for c in all_ctx if CH.champion_score(c["ranked"]) is not None]
    target = len(champion_raw)
    th_b1 = CH.calibrate_threshold([CH.top_gap(c["probs"]) for c in all_ctx], target, "below")
    th_b2 = CH.calibrate_threshold([CH.entropy(c["probs"]) for c in all_ctx], target, "above")
    print(f"較正: チャンピオンcap前{target}レースに合わせ β1閾値={th_b1:.4f} / β2閾値={th_b2:.4f}")

    def sc_champion(c):
        return CH.champion_score(c["ranked"])

    def sc_b1(c):
        return CH.gap_score(c["probs"], th_b1)

    def sc_b2(c):
        return CH.entropy_score(c["probs"], th_b2)

    def sc_mkt(c):
        if c["market_order"] is None:
            c["_unjudgeable"] = True
            return None
        model_order = [x["lane"] for x in c["ranked"]]
        return CH.divergence_score(model_order, c["market_order"])

    selectors = [("チャンピオン(現行)", sc_champion), ("挑戦者②β1(差)", sc_b1),
                 ("挑戦者②β2(エントロピー)", sc_b2), ("挑戦者①市場相違", sc_mkt)]

    results = {"calibration": {"target": target, "th_b1": th_b1, "th_b2": th_b2},
               "divergence_threshold": CH.MARKET_DIVERGENCE_THRESHOLD,
               "n_days": n_days, "selectors": {}, "folds": {}}

    champ_sets: dict[int, set[str]] = {}
    for name, fn in selectors:
        per_fold = []
        total_set: set[str] = set()
        unjudge_total = 0
        for i, fold in enumerate(folds, 1):
            sel, unjudge = select_with_cap(fold, fn)
            unjudge_total += unjudge
            total_set |= sel
            st = agg_stats(sel, graded)
            fold_days = len({c["date"] for c in fold})
            st["per_day"] = st["n"] / fold_days if fold_days else 0.0
            if name.startswith("チャンピオン"):
                champ_sets[i] = sel
            else:
                champ = champ_sets[i]
                inter, union = len(sel & champ), len(sel | champ)
                st["jaccard"] = inter / union if union else 0.0
                st["only_self"] = agg_stats(sel - champ, graded)
                st["only_champ"] = agg_stats(champ - sel, graded)
            per_fold.append(st)
        total = agg_stats(total_set, graded)
        total["per_day"] = total["n"] / n_days if n_days else 0.0
        total["unjudgeable"] = unjudge_total
        if not name.startswith("チャンピオン"):
            champ_all = set().union(*champ_sets.values())
            inter, union = len(total_set & champ_all), len(total_set | champ_all)
            total["jaccard"] = inter / union if union else 0.0
            total["only_self"] = agg_stats(total_set - champ_all, graded)
            total["only_champ"] = agg_stats(champ_all - total_set, graded)
        results["selectors"][name] = total
        results["folds"][name] = per_fold
        print(f"{name}: {total['n']}R({total['per_day']:.1f}/日) 的中{total['hit_rate']:.1%} "
              f"回収率{total['roi']:.1%} (最大1発除き{total['roi_excl_max']:.1%}) "
              f"順当率{total['junto_rate']:.1%} 判定不能{total['unjudgeable']}件")

    # --- ガミり監視(市場レポートD)の基準値: チャンピオンの順当決着率 ---
    champ_folds = results["folds"]["チャンピオン(現行)"]
    fold_junto = [f["junto_rate"] for f in champ_folds if f["hits"]]
    results["gami_baseline"] = {
        "junto_rate": results["selectors"]["チャンピオン(現行)"]["junto_rate"],
        "fold_junto_rates": fold_junto,
        "fold_std": pstdev(fold_junto) if len(fold_junto) > 1 else 0.0,
    }
    print(f"ガミり監視基準値(順当決着率): {results['gami_baseline']['junto_rate']:.1%} "
          f"(fold標準偏差 {results['gami_baseline']['fold_std']:.1%})")

    results["updated"] = datetime.now(JST).isoformat(timespec="seconds")
    json_path = TEST_DIR / "verify_challengers_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"集計JSON出力: {json_path}")

    (TEST_DIR / "backtest_report_challengers.html").write_text(
        render_report(results), encoding="utf-8")
    print(f"レポート出力: {TEST_DIR / 'backtest_report_challengers.html'}")


# ===== レポート =====

def _fmt_stats_row(name: str, s: dict, adopt: bool = False) -> str:
    cls = ' class="adopt"' if adopt else ""
    roi_cls = "pos" if s["roi"] >= 1 else "neg"
    ex_cls = "pos" if s["roi_excl_max"] >= 1 else "neg"
    d = s["dist"]
    jac = f"{s['jaccard']:.2f}" if "jaccard" in s else "—"
    return (f"<tr{cls}><td>{name}</td><td class='num'>{s['n']}</td>"
            f"<td class='num'>{s['per_day']:.1f}</td>"
            f"<td class='num'>{s['hit_rate']:.1%}</td>"
            f"<td class='num {roi_cls}'>{s['roi']:.1%}</td>"
            f"<td class='num {ex_cls}'>{s['roi_excl_max']:.1%}</td>"
            f"<td class='num'>{d['順当']}/{d['中波乱']}/{d['万舟']}</td>"
            f"<td class='num'>{jac}</td></tr>")


def _gami_threshold_note(g: dict) -> str:
    """発動閾値10pt(暫定)がfoldのばらつきに対して妥当かの注記を組み立てる"""
    base = ("発動条件(暫定): 実戦の直近順当決着率が「基準値+10pt超」を60日継続したら"
            "選別の再検証を発動。")
    if g["fold_std"] <= 0:
        return base + "fold間のばらつきが算出できないため、σ換算の妥当性評価は保留。"
    sigma = 0.10 / g["fold_std"]
    note = base + f"10ptはfold標準偏差({g['fold_std']:.1%})の約{sigma:.1f}σに相当。"
    if sigma < 2:
        note += (f"2σ未満は正常なゆらぎでも誤発動しやすいため、"
                 f"2σ={2 * g['fold_std']:.0%}への引き上げを提案する(コード側コメントにも記載)。")
    else:
        note += "2σ以上あり誤発動しにくい水準。"
    return note


def render_report(r: dict) -> str:
    total_rows = []
    for name, s in r["selectors"].items():
        total_rows.append(_fmt_stats_row(name, s, adopt=name.startswith("チャンピオン")))

    diff_rows = []
    for name, s in r["selectors"].items():
        if "only_self" not in s:
            continue
        a, b = s["only_self"], s["only_champ"]
        diff_rows.append(
            f"<tr><td>{name}</td>"
            f"<td class='num'>{a['n']}R / 回収率{a['roi']:.1%}</td>"
            f"<td class='num'>{b['n']}R / 回収率{b['roi']:.1%}</td></tr>")

    fold_rows = []
    for name in r["folds"]:
        for i, s in enumerate(r["folds"][name], 1):
            if s["n"] == 0:
                continue
            roi_cls = "pos" if s["roi"] >= 1 else "neg"
            fold_rows.append(
                f"<tr><td>fold{i}</td><td>{name}</td><td class='num'>{s['n']}</td>"
                f"<td class='num'>{s['hit_rate']:.1%}</td>"
                f"<td class='num {roi_cls}'>{s['roi']:.1%}</td>"
                f"<td class='num'>{s['junto_rate']:.1%}</td></tr>")

    g = r["gami_baseline"]
    cal = r["calibration"]
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>検証⑪-A 選別チャンピオン/チャレンジャー(紙上)</title>
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
<h1>検証⑪-A 選別チャンピオン/チャレンジャー(すべて紙上・本番不変)</h1>
<p class="note">生成: {r['updated']} / 枠組み: backtest.py と同一のウォークフォワード
({TEST_START}以降を{N_FOLDS}分割・foldモデルで予測、現行MODEL_PATH不使用)。
買い目は全員 検証済みV2構成(荒れ注意)・1日上限10レースに統一し、変数は選別のみ。
導入判断は行わない(2026-09-01のv2判断材料)。</p>
<div class="warn">{SMALL_SAMPLE_NOTE}</div>

<div class="card">
  <h2 style="margin-top:0">全fold合計(5場・{r['n_days']}日)</h2>
  <table>
    <tr><th>選別器</th><th class="num">選別数</th><th class="num">選別/日</th>
        <th class="num">的中率</th><th class="num">回収率</th><th class="num">最大1発除き</th>
        <th class="num">順当/中波乱/万舟</th><th class="num">Jaccard</th></tr>
    {''.join(total_rows)}
    <tr><td>挑戦者③C条件型</td><td class='num' colspan='7'>蓄積待ち
      (ledger.jsonのC的中明細+市場レポート(f)が数ヶ月分貯まってから定義。大穴一撃フラグ構想と同件)</td></tr>
  </table>
  <p class="note">挑戦者①の判定不能(スナップショット無し)レース:
  {r['selectors'].get('挑戦者①市場相違', {}).get('unjudgeable', 0):,}件
  (スナップショット収集は2026-07-08開始のため遡及期間の大半が判定不能。
  ①の数字は8月の日次紙上採点で貯めるのが本命で、ここでは器の動作確認)。
  閾値: ①=スピアマン距離{r['divergence_threshold']}以上(暫定・8月末に(d)層別で確定) /
  ②β1={cal['th_b1']:.4f}未満・β2={cal['th_b2']:.4f}超
  (チャンピオンのcap前選別数{cal['target']}件±10%に揃うよう較正。成績での最適化はしていない)。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">チャンピオンとの差分レースの成績</h2>
  <table>
    <tr><th>挑戦者</th><th class="num">挑戦者のみが選んだレース</th>
        <th class="num">チャンピオンのみが選んだレース</th></tr>
    {''.join(diff_rows)}
  </table>
  <p class="note">「挑戦者のみ」が高回収率なら、その挑戦者はチャンピオンが逃すレースを
  拾えている。逆に「チャンピオンのみ」が高いなら、挑戦者は取りこぼしている。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">fold別(選別器×期間)</h2>
  <table>
    <tr><th>fold</th><th>選別器</th><th class="num">選別数</th>
        <th class="num">的中率</th><th class="num">回収率</th><th class="num">順当決着率</th></tr>
    {''.join(fold_rows)}
  </table>
</div>

<div class="card">
  <h2 style="margin-top:0">ガミり監視の基準値(市場レポートDが参照)</h2>
  <p style="font-size:.9rem">チャンピオンの順当決着率(的中のうち払戻&lt;掛金だった割合):
  <b>{g['junto_rate']:.1%}</b> / fold間の標準偏差: {g['fold_std']:.1%}</p>
  <p class="note">{_gami_threshold_note(g)}</p>
</div>

<p class="note">検算: チャンピオンは src/backtest.py の「ken現行構成[5場]荒れ注意」と同一の計算経路。
検証⑧(5〜6月固定・train&lt;05-01の単一モデル)の152.6%とはfold分割・学習データが異なるため
完全一致はしないが、同水準であることを確認して整合とみなす。<br>
再実行: py -X utf8 test/verify_challengers.py(冪等・全期間を毎回再計算)</p>
</body>
</html>
"""


if __name__ == "__main__":
    main()
