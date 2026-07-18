# -*- coding: utf-8 -*-
"""市場分析レポート生成CLI(約束①: 15分前vs最終オッズの歪み分析+紙上対決)

    py -X utf8 test/analyze_market.py

事前に py -X utf8 test/collect_final_odds.py で最終オッズを取得しておくこと。
出力: test/market_report.html / test/paper_results.json(毎回全期間を再計算)

本番の予想・購入・採点には一切影響しない読み取り専用の分析。
- 15分前スナップショット: oddsテーブル(fetched_at≠'final-backfill'の行のみ)
- 確定最終オッズ: odds_finalテーブル
- kenの実プラン・勝負所判定: docs/data/picks_*.json(=採点対象の朝版)
- C的中明細: docs/data/ledger.json

モデル確率が必要な分析((c)(d)(f))は現行MODEL_PATHで再計算する。月次再学習が
あるため「当時の予想を出したモデルと完全一致ではない」注記をレポートに入れる。
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import challengers as CH
import db
import predictors as P
from config import DB_PATH, JST, MODEL_PATH, PROJECT_DIR, VENUE_NAMES

TEST_DIR = PROJECT_DIR / "test"
PICKS_DIR = PROJECT_DIR / "docs" / "data"

# 少サンプル注記(固定文言)。紙上対決など全セクション共通で使う
SMALL_SAMPLE_NOTE = ("⚠ 少サンプル注意: スナップショット蓄積開始直後のため、"
                     "ここの数字は傾向の参考に留め、断定しないこと。"
                     "本レポートは再実行のたびに全期間を再計算する。")


# ===== 純粋関数(unittest対象) =====

def normalize_implied(odds_table: dict[str, float]) -> dict[str, float]:
    """{買い目: オッズ} -> 合計1に正規化した含意確率(控除率の影響を除く)"""
    raw = {k: 1.0 / o for k, o in odds_table.items() if o}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def pct_change(snap: float, final: float) -> float:
    """オッズ変動率。正=最終の方が高い(15分前に見た額より実払戻が有利)"""
    return final / snap - 1.0


def calibration_rows(obs: list[tuple[float, bool]], bins: list[float]) -> list[dict]:
    """(含意確率, 的中したか)の列をビン分けし、含意確率と実際の的中率を突き合わせる"""
    rows = []
    for lo, hi in zip(bins, bins[1:]):
        seg = [(p, hit) for p, hit in obs if lo <= p < hi]
        if not seg:
            continue
        implied = mean(p for p, _ in seg)
        actual = mean(1.0 if hit else 0.0 for _, hit in seg)
        rows.append({"lo": lo, "hi": hi, "n": len(seg), "implied": implied,
                     "actual": actual,
                     "ratio": actual / implied if implied else 0.0})
    return rows


def extract_r1(plan: list[list]) -> int | None:
    """荒れ注意プランからモデル予測1位(r1)を復元する。

    検証済み構成の3連単2点は「r3-r1-r2」「r4-r1-r2」で2着・3着が共通のため、
    2点の2着番号が一致すればそれがr1。復元できない構成ならNone。
    """
    seconds = []
    for bt, comb, _yen, src in plan:
        if bt == "3連単" and src == "検証済み":
            parts = comb.split("-")
            if len(parts) == 3:
                seconds.append(parts[1])
    if len(seconds) >= 2 and len(set(seconds)) == 1:
        return int(seconds[0])
    return None


def classify_miss(r1: int, top3: set[int]) -> str:
    """外れた本命勝負所の敗因分類。軸飛び=予測1位が3着内に残らなかった"""
    return "軸飛び" if r1 not in top3 else "ヒモ抜け"


def pick_ninki(snap3t: dict[str, float]) -> tuple[str, float] | None:
    """おっちゃんA(本命党): 15分前オッズの3連単1番人気(最低オッズ)を1点"""
    valid = [(comb, o) for comb, o in snap3t.items() if o]
    if not valid:
        return None
    return min(valid, key=lambda x: x[1])


def pick_manshu(snap3t: dict[str, float], min_odds: float = 100.0) -> tuple[str, float] | None:
    """おっちゃんB(万舟党): 100倍以上の3連単で最低オッズの目を1点。該当なしは見送り"""
    cands = [(comb, o) for comb, o in snap3t.items() if o and o >= min_odds]
    if not cands:
        return None
    return min(cands, key=lambda x: x[1])


def perm_shares(tri_probs: dict[str, float], boats: tuple[int, int, int]) -> dict[str, float]:
    """同一3艇の6順列の確率を組内で正規化した配分 {順列: シェア}"""
    s = set(boats)
    perms = {k: v for k, v in tri_probs.items()
             if set(int(x) for x in k.split("-")) == s}
    total = sum(perms.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in perms.items()}


# ===== データ読み込み =====

def load_odds_pairs(conn) -> dict[str, dict[str, dict[str, float]]]:
    """スナップショットと最終の両方があるレース {race_id: {"snap": {...}, "final": {...}}}
    内側は {"3連単": {comb: odds}, "3連複": {comb: odds}}"""
    def load(table: str, cond: str) -> dict:
        out = defaultdict(lambda: {"3連単": {}, "3連複": {}})
        for rid, bt, comb, o in conn.execute(
            f"SELECT race_id, bet_type, combination, odds FROM {table} WHERE {cond}"
        ):
            out[rid][bt][comb] = o
        return out

    snap = load("odds", "fetched_at != 'final-backfill'")
    final = load("odds_final", "1=1")
    return {rid: {"snap": snap[rid], "final": final[rid]}
            for rid in snap if rid in final}


def load_picks() -> dict[str, dict]:
    """docs/data/picks_*.json を {race_id: レース辞書(dateつき)} に統合"""
    races = {}
    for path in sorted(PICKS_DIR.glob("picks_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data.get("races", []):
            r["date"] = data["date"]
            races[r["race_id"]] = r
    return races


def load_results(conn) -> tuple[dict, dict]:
    """(race_id -> {着順: 枠番}, race_id -> {(券種, 買い目): 払戻円})"""
    actual = defaultdict(dict)
    for rid, lane, order in conn.execute(
        "SELECT race_id, lane, arrival_order FROM results WHERE arrival_order IS NOT NULL"
    ):
        actual[rid][order] = lane
    payout = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT race_id, bet_type, combination, amount_yen FROM payouts"
    ):
        payout[rid][(bt, comb)] = amt or 0
    return actual, payout


def compute_model_probs(conn, race_ids: list[str]) -> dict[str, list[dict]]:
    """現行MODEL_PATHでモデル勝率を再計算 {race_id: ranked(降順)}。
    当時の予想を出したモデルとは完全一致しない(月次再学習のため)"""
    import lightgbm as lgb
    from features import FEATURE_COLUMNS, build_program_features
    df = build_program_features(conn, race_ids)
    booster = lgb.Booster(model_str=MODEL_PATH.read_text(encoding="utf-8"))
    df["prob"] = booster.predict(df[FEATURE_COLUMNS])
    out = {}
    for rid, g in df.groupby("race_id"):
        g = g.sort_values("prob", ascending=False)
        out[rid] = [{"lane": int(r["lane"]), "prob": float(r["prob"])}
                    for _, r in g.iterrows()]
    return out


def ken_return(picks_race: dict, payout: dict) -> tuple[int, int]:
    """picks JSONのkenプランの(投資円, 回収円)"""
    stake = ret = 0
    for bt, comb, yen, _src in picks_race.get("ken", []):
        stake += yen
        ret += payout.get((bt, comb), 0) * yen // 100
    return stake, ret


# ===== 各セクションの集計 =====

def sec_a_drift(pairs: dict, picks: dict) -> dict:
    """15分前vs最終オッズの変動率。全買い目とkenプラン対象目"""
    all_changes = {"3連単": [], "3連複": []}
    for pr in pairs.values():
        for bt in ("3連単", "3連複"):
            for comb, snap_o in pr["snap"][bt].items():
                final_o = pr["final"][bt].get(comb)
                if snap_o and final_o:
                    all_changes[bt].append(pct_change(snap_o, final_o))

    ken_changes, ken_snap_est, ken_final_est = [], 0.0, 0.0
    for rid, pr in pairs.items():
        r = picks.get(rid)
        if not r:
            continue
        for bt, comb, yen, _src in r.get("ken", []):
            snap_o = pr["snap"].get(bt, {}).get(comb)
            final_o = pr["final"].get(bt, {}).get(comb)
            if snap_o and final_o:
                ken_changes.append(pct_change(snap_o, final_o))
                ken_snap_est += snap_o * yen
                ken_final_est += final_o * yen

    def summary(changes):
        if not changes:
            return None
        up = sum(1 for c in changes if c > 0)
        return {"n": len(changes), "median": median(changes), "mean": mean(changes),
                "up_rate": up / len(changes)}

    return {
        "all": {bt: summary(cs) for bt, cs in all_changes.items()},
        "ken": summary(ken_changes),
        "ken_payout_ratio": (ken_final_est / ken_snap_est - 1.0) if ken_snap_est else None,
    }


def sec_b_calibration(pairs: dict, actual: dict) -> dict:
    """スナップショット含意確率の較正曲線(3連単/3連複別)"""
    bins = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.01]
    out = {}
    for bt, sep in (("3連単", "-"), ("3連複", "=")):
        obs = []
        for rid, pr in pairs.items():
            res = actual.get(rid, {})
            if 1 not in res or 2 not in res or 3 not in res:
                continue
            if bt == "3連単":
                win = f"{res[1]}-{res[2]}-{res[3]}"
            else:
                s = sorted([res[1], res[2], res[3]])
                win = f"{s[0]}={s[1]}={s[2]}"
            implied = normalize_implied(pr["snap"][bt])
            for comb, p in implied.items():
                obs.append((p, comb == win))
        out[bt] = calibration_rows(obs, bins)
    return out


def sec_c_permutation(pairs: dict, model_probs: dict) -> dict:
    """同一3艇: 3連単6順列の含意確率合計 vs 3連複含意確率、
    および順列内配分の市場vsモデル(Benter)比較"""
    from itertools import combinations
    ratios = []           # sum(6順列) / 3連複
    top_share_market = [] # 組内で最有力順列が占めるシェア(市場)
    top_share_model = []  # 同(モデル)
    top_agree = []        # 最有力順列が市場とモデルで一致したか
    mad_list = []         # 6順列シェアの平均絶対差(市場vsモデル)

    for rid, pr in pairs.items():
        imp3t = normalize_implied(pr["snap"]["3連単"])
        imp3f = normalize_implied(pr["snap"]["3連複"])
        ranked = model_probs.get(rid)
        model_tri = None
        if ranked and len(ranked) >= 4:
            probs = P.normalize_probs(ranked)
            model_tri = {f"{a}-{b}-{c}": p
                         for (a, b, c), p in P.trifecta_probs(probs).items()}
        lanes = sorted({int(x) for k in imp3f for x in k.split("=")})
        for boats in combinations(lanes, 3):
            key3f = f"{boats[0]}={boats[1]}={boats[2]}"
            trio_p = imp3f.get(key3f)
            mkt = perm_shares(imp3t, boats)
            if not mkt or not trio_p:
                continue
            sum6 = sum(imp3t.get(k, 0.0) for k in mkt)
            ratios.append(sum6 / trio_p)
            top_share_market.append(max(mkt.values()))
            if model_tri:
                mdl = perm_shares(model_tri, boats)
                if mdl:
                    top_share_model.append(max(mdl.values()))
                    top_agree.append(
                        max(mkt, key=mkt.get) == max(mdl, key=mdl.get))
                    mad_list.append(
                        mean(abs(mkt.get(k, 0) - mdl.get(k, 0)) for k in set(mkt) | set(mdl)))

    return {
        "n_sets": len(ratios),
        "ratio_median": median(ratios) if ratios else None,
        "top_share_market": mean(top_share_market) if top_share_market else None,
        "top_share_model": mean(top_share_model) if top_share_model else None,
        "top_agree_rate": mean([1.0 if a else 0.0 for a in top_agree]) if top_agree else None,
        "mad": mean(mad_list) if mad_list else None,
    }


def sec_d_rank_gap(pairs: dict, picks: dict, model_probs: dict, payout: dict) -> dict:
    """モデル1着確率順位 vs 人気順位(含意1着確率)のズレ × ken回収率"""
    rows = []
    for rid, pr in pairs.items():
        ranked = model_probs.get(rid)
        r = picks.get(rid)
        if not ranked or not r:
            continue
        # 人気側の1着確率: 3連単含意確率を1着艇ごとに合算
        imp3t = normalize_implied(pr["snap"]["3連単"])
        win_imp = defaultdict(float)
        for comb, p in imp3t.items():
            win_imp[int(comb.split("-")[0])] += p
        if not win_imp:
            continue
        market_order = sorted(win_imp, key=lambda l: -win_imp[l])
        model_order = [x["lane"] for x in ranked]
        market_rank = {l: i for i, l in enumerate(market_order)}
        gap = sum(abs(i - market_rank.get(l, len(market_order)))
                  for i, l in enumerate(model_order) if l in market_rank)
        stake, ret = ken_return(r, payout.get(rid, {}))
        rows.append({"race_id": rid, "gap": gap, "stake": stake, "ret": ret})

    groups = {"一致(0-2)": (0, 2), "小ズレ(3-6)": (3, 6), "大ズレ(7+)": (7, 999)}
    out = {}
    for label, (lo, hi) in groups.items():
        seg = [x for x in rows if lo <= x["gap"] <= hi and x["stake"]]
        stake = sum(x["stake"] for x in seg)
        ret = sum(x["ret"] for x in seg)
        out[label] = {"n": len(seg), "stake": stake, "ret": ret,
                      "roi": ret / stake if stake else 0.0}
    return out


def sec_e_miss_breakdown(picks: dict, actual: dict, payout: dict) -> dict:
    """外れた本命勝負所の敗因分解(軸飛び/ヒモ抜け)"""
    detail = []
    counts = {"的中": 0, "軸飛び": 0, "ヒモ抜け": 0, "分類不能": 0}
    for rid, r in picks.items():
        if r.get("shobusho") != "本命" or not r.get("ken"):
            continue
        res = actual.get(rid, {})
        if 1 not in res or 2 not in res or 3 not in res:
            continue
        stake, ret = ken_return(r, payout.get(rid, {}))
        if ret > 0:
            counts["的中"] += 1
            continue
        r1 = extract_r1(r["ken"])
        top3 = {res[1], res[2], res[3]}
        cls = classify_miss(r1, top3) if r1 is not None else "分類不能"
        counts[cls] += 1
        detail.append({
            "date": r["date"], "venue": VENUE_NAMES[r["venue_code"]],
            "race_no": r["race_no"], "r1": r1,
            "result": f"{res[1]}-{res[2]}-{res[3]}", "class": cls,
        })
    return {"counts": counts, "detail": detail}


def sec_f_c_hits(picks: dict, model_probs_all: dict) -> dict:
    """C勝万舟の的中レースの条件分布(ledger.jsonのhits明細から)"""
    ledger_path = PICKS_DIR / "ledger.json"
    if not ledger_path.exists():
        return {"hits": [], "note": "ledger.jsonなし"}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    name_to_code = {v: k for k, v in VENUE_NAMES.items()}

    hits = []
    for day in ledger:
        for h in day.get("hits", {}).get("c", []):
            venue_code = name_to_code.get(h["venue"])
            rid = db.make_race_id(h["date"], venue_code, h["race_no"]) if venue_code else None
            r = picks.get(rid, {})
            ranked = model_probs_all.get(rid)
            top_prob = ranked[0]["prob"] if ranked else None
            hits.append({
                "date": h["date"], "venue": h["venue"], "race_no": h["race_no"],
                "chaku": h.get("chaku"), "ret": h.get("ret"),
                "confidence": r.get("confidence"),
                "top_prob": top_prob,
                "half": "前半(1-6R)" if h["race_no"] <= 6 else "後半(7-12R)",
            })
    by_venue = defaultdict(int)
    by_half = defaultdict(int)
    for h in hits:
        by_venue[h["venue"]] += 1
        by_half[h["half"]] += 1
    return {"hits": hits, "by_venue": dict(by_venue), "by_half": dict(by_half)}


def sec_g_show_calibration(picks: dict, model_probs: dict, actual: dict) -> dict:
    """(C) ◎複勝較正: モデル予測1位の複勝(3着以内)確率 vs 実際の複勝率。

    「軸飛び6割」がモデル自身の複勝確率どおり(想定内)か、◎の複勝圏評価が
    甘い(想定超=系統的過大評価)かをここで裁く。荒れ注意限定の集計も併記。
    """
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    obs_all, obs_are = [], []
    for rid, r in picks.items():
        ranked = model_probs.get(rid)
        res = actual.get(rid, {})
        if not ranked or 1 not in res or 3 not in res:
            continue
        probs = P.normalize_probs(ranked)
        top = ranked[0]["lane"]
        pred = CH.show_probability(probs, top)
        hit = top in {res[1], res[2], res[3]}
        obs_all.append((pred, hit))
        if r.get("confidence") == "荒れ注意":
            obs_are.append((pred, hit))

    def rows(obs):
        return calibration_rows(obs, bins)

    all_rows, are_rows = rows(obs_all), rows(obs_are)

    # 判定文言(仕様どおり): n>=10のビンだけで判定。予測-実際>+5pt=過大評価
    judged = [x for x in all_rows if x["n"] >= 10]
    if not judged:
        verdict = f"蓄積待ち(全ビンn<10。現在{len(obs_all)}レース)"
    else:
        over = [x for x in judged if x["implied"] - x["actual"] > 0.05]
        within = all(abs(x["implied"] - x["actual"]) <= 0.05 for x in judged)
        if within:
            verdict = "前提健在(較正誤差が全ビンで±5pt以内。軸飛びはモデルの想定内)"
        elif over:
            over_bins = ", ".join("{:.0%}〜{:.0%}".format(x["lo"], x["hi"]) for x in over)
            verdict = ("◎複勝圏の過大評価あり(v2で選別・構成の再考材料)。"
                       f"過大ビン: {over_bins}")
        else:
            verdict = "過小評価方向のズレ(◎は想定より複勝圏に残っている。ヒモ側の問題を疑う)"
    return {"all": all_rows, "are": are_rows, "n": len(obs_all),
            "n_are": len(obs_are), "verdict": verdict}


def sec_h_gami_watch(picks: dict, payout: dict) -> dict:
    """(D) ガミり監視: 実戦の本命勝負所の順当決着率(的中のうち払戻<掛金)を
    バックテスト基準値(verify_challengers_results.jsonのチャンピオン順当率)と比較。

    発動条件(暫定): 「30日移動順当率が基準値+10pt超」の状態が60日連続したら
    『選別再検証の発動条件に到達』を赤字表示する(表示のみ・自動では何も変えない)。
    10ptは暫定値。基準側JSONにはfold標準偏差も保存されており、レポートで
    σ換算の妥当性を併記する(fold σ≈5%なら10pt≈2σで誤発動しにくい水準)。
    """
    base_path = TEST_DIR / "verify_challengers_results.json"
    baseline = None
    if base_path.exists():
        data = json.loads(base_path.read_text(encoding="utf-8"))
        baseline = data.get("gami_baseline")

    # 日別の(的中数, 順当数)系列を作る
    daily: dict[str, list[int]] = {}
    for rid, r in picks.items():
        if r.get("shobusho") != "本命" or not r.get("ken"):
            continue
        stake, ret = ken_return(r, payout.get(rid, {}))
        if not ret:
            continue
        d = daily.setdefault(r["date"], [0, 0])
        d[0] += 1
        if ret < stake:
            d[1] += 1

    dates = sorted(daily)

    def window_rate(days: int) -> tuple[float | None, int, int]:
        seg = dates[-days:]
        hits = sum(daily[d][0] for d in seg)
        junto = sum(daily[d][1] for d in seg)
        return (junto / hits if hits else None), hits, junto

    rate30, hits30, _ = window_rate(30)
    rate60, hits60, _ = window_rate(60)

    # 30日移動順当率が閾値超の「連続日数」(直近から遡る)
    streak = 0
    if baseline:
        th = baseline["junto_rate"] + 0.10
        for i in range(len(dates), 0, -1):
            seg = dates[max(0, i - 30):i]
            hits = sum(daily[d][0] for d in seg)
            junto = sum(daily[d][1] for d in seg)
            if hits and junto / hits > th:
                streak += 1
            else:
                break
    triggered = streak >= 60
    return {"baseline": baseline, "rate30": rate30, "rate60": rate60,
            "hits30": hits30, "hits60": hits60,
            "n_days": len(dates), "streak": streak, "triggered": triggered}


def extract_lanes(plan: list[list]) -> tuple[int, int, int, int] | None:
    """荒れ注意プラン(検証済みV2)からr1〜r4を復元する。

    3連単2点「r3-r1-r2」「r4-r1-r2」の共通2着=r1・共通3着=r2、頭がr3, r4。
    復元できない構成(標準・堅め等)はNone。
    """
    tfs = [comb.split("-") for bt, comb, _y, src in plan
           if bt == "3連単" and src == "検証済み"]
    if len(tfs) != 2 or {tfs[0][1], tfs[1][1]} != {tfs[0][1]}:
        return None
    if tfs[0][2] != tfs[1][2]:
        return None
    return int(tfs[0][1]), int(tfs[0][2]), int(tfs[0][0]), int(tfs[1][0])


def paper_challengers(picks: dict, model_probs: dict, payout: dict) -> dict:
    """(E) 選別・構成チャレンジャーの日次紙上採点。

    - 挑戦者②β1/β2: 閾値はverify_challengers_results.jsonの較正値(無ければ較正待ち)
    - 挑戦者①: 暫定閾値。スナップショットの無いレースは判定不能として件数を出す
    - 構成4案: 選別はチャンピオン固定=picksの本命勝負所。r1〜r4は当時の実プランから
      復元し、Cも当時のpicks_cを使う(構成行だけは当時モデルの実データで採点できる)
    - ②①の選別はモデル確率の再計算(現行MODEL_PATH)に基づくため当時版と完全一致しない
    """
    th_path = TEST_DIR / "verify_challengers_results.json"
    th = None
    if th_path.exists():
        th = json.loads(th_path.read_text(encoding="utf-8")).get("calibration")

    # 日付ごとにレースをまとめる(選別は日単位・上限10)
    by_date: dict[str, list[str]] = defaultdict(list)
    for rid, r in picks.items():
        by_date[r["date"]].append(rid)

    def plan_and_grade(rid) -> tuple[int, int] | None:
        ranked = model_probs.get(rid)
        if not ranked or len(ranked) < 4:
            return None
        probs = P.normalize_probs(ranked)
        plan = P.ken_portfolio("荒れ注意", ranked, [], P.picks_katsu(probs))
        pay = payout.get(rid, {})
        stake = sum(y for _, _, y, _ in plan)
        ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
        return stake, ret

    def run_selector(score_fn) -> tuple[dict, int]:
        stats = {"n": 0, "hits": 0, "stake": 0, "ret": 0}
        unjudge = 0
        for d in sorted(by_date):
            cands = []
            for rid in by_date[d]:
                ranked = model_probs.get(rid)
                if not ranked or len(ranked) < 4:
                    continue
                s = score_fn(rid, ranked)
                if s == "unjudgeable":
                    unjudge += 1
                    continue
                if s is not None:
                    cands.append((s, rid))
            for rid in CH.daily_cap(cands):
                g = plan_and_grade(rid)
                if not g:
                    continue
                stake, ret = g
                stats["n"] += 1
                stats["stake"] += stake
                stats["ret"] += ret
                stats["hits"] += 1 if ret else 0
        return stats, unjudge

    rows = {}
    if th:
        rows["挑戦者②β1(差)"], _ = run_selector(
            lambda rid, rk: CH.gap_score(P.normalize_probs(rk), th["th_b1"]))
        rows["挑戦者②β2(エントロピー)"], _ = run_selector(
            lambda rid, rk: CH.entropy_score(P.normalize_probs(rk), th["th_b2"]))
    else:
        rows["挑戦者②(β1/β2)"] = {"pending": "較正待ち(verify_challengers.py未実行)"}

    # 挑戦者①: スナップショットのあるレースだけ判定可能
    conn = db.connect(DB_PATH)
    market_orders: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for rid, comb, odds in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連単' AND fetched_at != 'final-backfill' AND odds > 0"
    ):
        market_orders[rid][int(comb.split("-")[0])] += 1.0 / odds
    # 回収ラインフィルタ用: 15分前スナップショットの3連複オッズ
    trio_snap: dict[str, dict[str, float]] = defaultdict(dict)
    for rid, comb, odds in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連複' AND fetched_at != 'final-backfill' AND odds > 0"
    ):
        trio_snap[rid][comb] = odds
    conn.close()
    market_order = {rid: sorted(d, key=lambda l: -d[l]) for rid, d in market_orders.items()}

    def sc_mkt(rid, ranked):
        mo = market_order.get(rid)
        if mo is None:
            return "unjudgeable"
        return CH.divergence_score([x["lane"] for x in ranked], mo)

    rows["挑戦者①市場相違(暫定閾値)"], unjudge = run_selector(sc_mkt)
    rows["挑戦者①市場相違(暫定閾値)"]["unjudgeable"] = unjudge

    # 構成4案(選別=picksの本命勝負所、券面は当時の実データから復元)
    comp_stats = {name: {"n": 0, "hits": 0, "stake": 0, "ret": 0}
                  for name in CH.COMPOSITION_NAMES}
    unrecoverable = 0
    for rid, r in picks.items():
        if r.get("shobusho") != "本命" or not r.get("ken"):
            continue
        lanes = extract_lanes(r["ken"])
        if lanes is None:
            unrecoverable += 1
            continue
        pseudo_ranked = [{"lane": l, "prob": 0.0} for l in lanes] + \
                        [{"lane": l, "prob": 0.0} for l in range(1, 7) if l not in lanes]
        c_picks = [tuple(x) for x in r.get("c", [])]
        pay = payout.get(rid, {})
        for name in CH.COMPOSITION_NAMES:
            plan = CH.build_composition(name, pseudo_ranked, c_picks)
            stake = sum(y for _, _, y, _ in plan)
            ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
            s = comp_stats[name]
            s["n"] += 1
            s["stake"] += stake
            s["ret"] += ret
            s["hits"] += 1 if ret else 0
    # 回収ラインフィルタ(2026-07-18ケンさん発案。8月宿題「ガミ条件のオッズ形状層別」の測定):
    # 本命勝負所のうち「主力3連複(1点目)の15分前オッズ×掛金 ≥ 1,000円」のレースだけ買う。
    # ガミ的中の多く(初回検証で11中8)は購入時点のオッズで回収不能と分かっていたため、
    # 「市場が予測できているレース=実は荒れない」の除外効果を毎日測る。紙上のみ
    filter_stats = {"買う(主力3連複が回収ライン以上)": {"n": 0, "hits": 0, "stake": 0, "ret": 0},
                    "見送る(オッズ不足=市場が順当視)": {"n": 0, "hits": 0, "stake": 0, "ret": 0}}
    filter_unjudgeable = 0
    for rid, r in picks.items():
        if r.get("shobusho") != "本命" or not r.get("ken"):
            continue
        trios = [(comb, yen) for bt, comb, yen, _ in r["ken"] if bt == "3連複"]
        o = trio_snap.get(rid, {}).get(trios[0][0]) if trios else None
        if o is None:
            filter_unjudgeable += 1
            continue
        key = ("買う(主力3連複が回収ライン以上)" if o * trios[0][1] >= 1000
               else "見送る(オッズ不足=市場が順当視)")
        pay = payout.get(rid, {})
        stake = sum(y for _, _, y, _ in r["ken"])
        ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in r["ken"])
        s = filter_stats[key]
        s["n"] += 1
        s["stake"] += stake
        s["ret"] += ret
        s["hits"] += 1 if ret else 0
    filter_stats["買う(主力3連複が回収ライン以上)"]["unjudgeable"] = filter_unjudgeable

    return {"selectors": rows, "compositions": comp_stats,
            "unrecoverable": unrecoverable, "odds_filter": filter_stats}


def paper_battle(pairs: dict, picks: dict, payout: dict) -> dict:
    """紙上対決(遡及・毎回再計算。本番ledgerには一切書かない)"""
    races = []
    for rid in sorted(pairs):
        r = picks.get(rid)
        snap3t = pairs[rid]["snap"]["3連単"]
        pay = payout.get(rid, {})

        row = {"race_id": rid,
               "date": r["date"] if r else rid[:4] + "-" + rid[4:6] + "-" + rid[6:8],
               "venue": VENUE_NAMES[int(rid.split("_")[1])],
               "race_no": int(rid.split("_")[2]),
               "shobusho": r.get("shobusho") if r else None}

        a = pick_ninki(snap3t)
        if a:
            comb, odds_v = a
            ret = pay.get(("3連単", comb), 0) * 10  # 1,000円=100円あたり払戻×10
            row["a"] = {"comb": comb, "odds": odds_v, "stake": 1000, "ret": ret}
        b = pick_manshu(snap3t)
        if b:
            comb, odds_v = b
            ret = pay.get(("3連単", comb), 0) * 10
            row["b"] = {"comb": comb, "odds": odds_v, "stake": 1000, "ret": ret}
        if r and r.get("ken"):
            stake, ret = ken_return(r, pay)
            row["ken"] = {"stake": stake, "ret": ret}
        races.append(row)

    def agg(rows, player):
        seg = [x[player] for x in rows if player in x]
        n = len(seg)
        stake = sum(x["stake"] for x in seg)
        ret = sum(x["ret"] for x in seg)
        hits = sum(1 for x in seg if x["ret"])
        return {"n": n, "hits": hits, "hit_rate": hits / n if n else 0.0,
                "stake": stake, "ret": ret,
                "roi": ret / stake if stake else 0.0}

    honmei = [x for x in races if x.get("shobusho") == "本命"]
    summary = {
        "全レース": {p: agg(races, p) for p in ("a", "b", "ken")},
        "本命勝負所のみ": {p: agg(honmei, p) for p in ("a", "b", "ken")},
    }
    return {"races": races, "summary": summary}


# ===== HTMLレポート =====

def _pct(x, digits=1):
    return f"{x:+.{digits}%}" if x is not None else "-"


def _render_show_calibration(g: dict) -> str:
    """(C)◎複勝較正カードのHTML"""
    def rows(items):
        out = []
        for x in items:
            diff = x["implied"] - x["actual"]
            cls = "neg" if diff > 0.05 else ("pos" if abs(diff) <= 0.05 else "")
            out.append(
                f"<tr><td>{x['lo']:.0%}〜{x['hi']:.0%}</td><td class='num'>{x['n']}</td>"
                f"<td class='num'>{x['implied']:.1%}</td><td class='num'>{x['actual']:.1%}</td>"
                f"<td class='num {cls}'>{diff:+.1%}</td></tr>")
        return "".join(out) or "<tr><td colspan='5'>データなし</td></tr>"

    return f"""
<div class="card">
  <h2>(g) ◎複勝較正: 予測1位の複勝(3着以内)確率 vs 実際の複勝率</h2>
  <p style="font-size:.9rem"><b>判定: {g['verdict']}</b></p>
  <table>
    <tr><th>予測複勝確率ビン</th><th class="num">n</th><th class="num">予測平均</th>
        <th class="num">実際の複勝率</th><th class="num">予測-実際(+は過大評価)</th></tr>
    <tr><th colspan="5">全レース({g['n']}件)</th></tr>
    {rows(g['all'])}
    <tr><th colspan="5">荒れ注意レース限定({g['n_are']}件)</th></tr>
    {rows(g['are'])}
  </table>
  <p class="note">複勝確率はBenter展開(現行モデルで再計算)の3着以内周辺化。
  「軸飛び」の多さがモデル想定内か想定超かをこの表で裁く。判定はn≥10のビンのみ。</p>
</div>"""


def _render_gami_watch(h: dict) -> str:
    """(D)ガミり監視カードのHTML(表示のみ・自動では何も変えない)"""
    b = h["baseline"]
    if not b:
        body = ("<p>基準値待ち: 先に py -X utf8 test/verify_challengers.py を実行すると"
                "バックテスト基準値(チャンピオンの順当決着率)が保存される。</p>")
    else:
        th = b["junto_rate"] + 0.10
        r30 = f"{h['rate30']:.1%}({h['hits30']}的中)" if h["rate30"] is not None else "的中なし"
        r60 = f"{h['rate60']:.1%}({h['hits60']}的中)" if h["rate60"] is not None else "的中なし"
        status = ("<span style='color:#cf222e;font-weight:bold'>『選別再検証の発動条件に到達』"
                  "(基準値+10pt超が60日継続)</span>" if h["triggered"]
                  else f"未到達(閾値{th:.1%}超の連続日数: {h['streak']}日/60日。"
                       f"実戦データ{h['n_days']}日分)")
        body = f"""
  <table>
    <tr><th>指標</th><th class="num">値</th></tr>
    <tr><td>バックテスト基準値(チャンピオン順当決着率)</td><td class="num">{b['junto_rate']:.1%}
      (foldσ {b['fold_std']:.1%})</td></tr>
    <tr><td>実戦・直近30日の順当決着率</td><td class="num">{r30}</td></tr>
    <tr><td>実戦・直近60日の順当決着率</td><td class="num">{r60}</td></tr>
    <tr><td>発動判定</td><td class="num">{status}</td></tr>
  </table>
  <p class="note">順当決着率=本命勝負所の的中のうち払戻&lt;掛金(ガミ)だった割合。
  これが上がる=「荒れ注意と選んだのに順当に決まる」=選別のエッジ鈍化のシグナル。
  10ptは暫定(foldσの約2σ)。発動しても表示のみで、再検証の実施は人間が判断する。</p>"""
    return f"""
<div class="card">
  <h2>(h) ガミり監視(選別エッジ鈍化の引き金・表示のみ)</h2>
  {body}
</div>"""


def _render_paper_challengers(pc: dict) -> str:
    """(E)紙上対決への追記: 選別・構成チャレンジャーの行"""
    def stat_row(name, s):
        if "pending" in s:
            return f"<tr><td>{name}</td><td colspan='6'>{s['pending']}</td></tr>"
        n, hits, stake, ret = s["n"], s["hits"], s["stake"], s["ret"]
        roi = ret / stake if stake else 0.0
        profit = ret - stake
        extra = f"(判定不能{s['unjudgeable']}件)" if "unjudgeable" in s else ""
        return (f"<tr><td>{name}{extra}</td><td class='num'>{n}</td>"
                f"<td class='num'>{hits / n if n else 0:.1%}</td>"
                f"<td class='num'>{stake:,}円</td><td class='num'>{ret:,}円</td>"
                f"<td class='num {'pos' if roi >= 1 else 'neg'}'>{roi:.1%}</td>"
                f"<td class='num {'pos' if profit >= 0 else 'neg'}'>{profit:+,}円</td></tr>")

    sel_rows = "".join(stat_row(k, v) for k, v in pc["selectors"].items())
    comp_rows = "".join(stat_row(f"構成:{k}", v) for k, v in pc["compositions"].items())
    filter_rows = "".join(stat_row(k, v) for k, v in pc.get("odds_filter", {}).items())
    return f"""
  <h3 style="font-size:.9rem">選別・構成チャレンジャー(検証⑪の日次紙上採点)</h3>
  <table>
    <tr><th>打ち手</th><th class="num">レース数</th><th class="num">的中率</th>
        <th class="num">投資</th><th class="num">回収</th><th class="num">回収率</th><th class="num">損益</th></tr>
    <tr><th colspan="7">選別チャレンジャー(全picksレースから日次選別・上限10)</th></tr>
    {sel_rows}
    <tr><td>挑戦者③C条件型</td><td colspan="6">蓄積待ち(大穴一撃フラグ構想と同件)</td></tr>
    <tr><th colspan="7">構成4案(選別=本番の本命勝負所に固定・券面は当時の実データから復元)</th></tr>
    {comp_rows}
    <tr><th colspan="7">回収ラインフィルタ(本命のうち主力3連複の15分前オッズで買う/見送るを分けた場合)</th></tr>
    {filter_rows}
  </table>
  <p class="note">固定注記: ①②の選別はモデル確率を現行版で再計算(当時版と完全一致しない)。
  ①の閾値は暫定(8月末に(d)層別で確定)。{SMALL_SAMPLE_NOTE}
  構成のr1〜r4復元不能レース: {pc['unrecoverable']}件。</p>"""


def render_report(d: dict) -> str:
    a, b, c, dd, e, f, paper = (d["a"], d["b"], d["c"], d["d"], d["e"], d["f"],
                                d["paper"])

    def player_rows(scope):
        names = {"a": "おっちゃんA(本命党: 3連単1番人気1点)",
                 "b": "おっちゃんB(万舟党: 100倍以上の最低オッズ1点)",
                 "ken": "予想屋ken(現行構成・実プラン)"}
        rows = []
        for p in ("a", "b", "ken"):
            s = paper["summary"][scope][p]
            roi_cls = "pos" if s["roi"] >= 1 else "neg"
            profit = s["ret"] - s["stake"]
            rows.append(
                f"<tr><td>{names[p]}</td><td class='num'>{s['n']}</td>"
                f"<td class='num'>{s['hit_rate']:.1%}</td>"
                f"<td class='num'>{s['stake']:,}円</td><td class='num'>{s['ret']:,}円</td>"
                f"<td class='num {roi_cls}'>{s['roi']:.1%}</td>"
                f"<td class='num {'pos' if profit >= 0 else 'neg'}'>{profit:+,}円</td></tr>")
        return "".join(rows)

    def drift_row(label, s):
        if not s:
            return f"<tr><td>{label}</td><td class='num' colspan='4'>データなし</td></tr>"
        return (f"<tr><td>{label}</td><td class='num'>{s['n']:,}</td>"
                f"<td class='num'>{_pct(s['median'])}</td>"
                f"<td class='num'>{_pct(s['mean'])}</td>"
                f"<td class='num'>{s['up_rate']:.1%}</td></tr>")

    def calib_rows(bt):
        rows = []
        for row in b[bt]:
            cls = "pos" if row["ratio"] >= 1 else "neg"
            rows.append(
                f"<tr><td>{row['lo']:.1%}〜{row['hi']:.1%}</td>"
                f"<td class='num'>{row['n']:,}</td>"
                f"<td class='num'>{row['implied']:.2%}</td>"
                f"<td class='num'>{row['actual']:.2%}</td>"
                f"<td class='num {cls}'>{row['ratio']:.2f}</td></tr>")
        return "".join(rows)

    rank_rows = "".join(
        f"<tr><td>{label}</td><td class='num'>{s['n']}</td>"
        f"<td class='num'>{s['stake']:,}円</td><td class='num'>{s['ret']:,}円</td>"
        f"<td class='num {'pos' if s['roi'] >= 1 else 'neg'}'>{s['roi']:.1%}</td></tr>"
        for label, s in dd.items())

    miss = e["counts"]
    miss_detail = "".join(
        f"<tr><td>{m['date']}</td><td>{m['venue']}{m['race_no']}R</td>"
        f"<td class='num'>{m['r1']}</td><td class='num'>{m['result']}</td>"
        f"<td>{m['class']}</td></tr>"
        for m in e["detail"])

    c_hit_parts = []
    for h in f["hits"]:
        top = f"{h['top_prob']:.1%}" if h["top_prob"] else "-"
        c_hit_parts.append(
            f"<tr><td>{h['date']}</td><td>{h['venue']}{h['race_no']}R</td>"
            f"<td>{h['confidence'] or '-'}</td><td class='num'>{top}</td>"
            f"<td class='num'>{h['chaku'] or '-'}</td>"
            f"<td class='num'>{h['ret']:,}円</td></tr>")
    c_hit_rows = "".join(c_hit_parts)

    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>市場分析レポート(約束①+紙上対決)</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 12px; background: #f6f8fa; max-width: 900px; margin: auto; }}
  h1 {{ font-size: 1.2rem; margin: 10px 4px; }}
  h2 {{ font-size: 1rem; margin-top: 0; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin-bottom: 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 6px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #1a7f37; font-weight: bold; }}
  .neg {{ color: #cf222e; }}
  .note {{ font-size: .78rem; color: #57606a; margin: 6px 4px; }}
  .warn {{ background: #fff8c5; border: 1px solid #d4a72c66; border-radius: 8px;
           padding: 8px 12px; font-size: .8rem; margin-bottom: 14px; }}
</style>
</head>
<body>
<h1>市場分析レポート(約束①+紙上対決)</h1>
<p class="note">生成: {updated} / 対象: 15分前スナップショットと最終オッズが揃った {d['n_races']}レース
({d['date_range']})。本番の予想・購入・採点には一切影響しない読み取り専用の分析。</p>
<div class="warn">{SMALL_SAMPLE_NOTE}</div>

<div class="card">
  <h2>(a) 15分前 → 最終オッズの変動</h2>
  <table>
    <tr><th>対象</th><th class="num">買い目数</th><th class="num">変動率中央値</th>
        <th class="num">平均</th><th class="num">上昇率(=有利化した割合)</th></tr>
    {drift_row("全買い目 3連単", a["all"]["3連単"])}
    {drift_row("全買い目 3連複", a["all"]["3連複"])}
    {drift_row("kenプラン対象目のみ", a["ken"])}
  </table>
  <p class="note">正=最終オッズの方が高い(15分前に見た想定払戻より実払戻が有利)。
  kenプランを金額加重した想定払戻の変化: <b>{_pct(a["ken_payout_ratio"])}</b>。
  ここが大きく負なら「15分前オッズでのEV判断は本番で目減りする」ことを意味する。</p>
</div>

<div class="card">
  <h2>(b) 市場の較正曲線(15分前スナップショットの含意確率 vs 実際の的中率)</h2>
  <table>
    <tr><th>含意確率ビン</th><th class="num">買い目数</th><th class="num">含意確率平均</th>
        <th class="num">実際の的中率</th><th class="num">実際/含意</th></tr>
    <tr><th colspan="5">3連単</th></tr>
    {calib_rows("3連単")}
    <tr><th colspan="5">3連複</th></tr>
    {calib_rows("3連複")}
  </table>
  <p class="note">実際/含意が1未満=市場が過大評価(買われすぎ)、1超=過小評価。
  低確率ビンで1未満なら定説どおりの本命-大穴バイアス(大穴の買われすぎ)。
  1超のビンはモデルと独立に妙味がある領域の候補。</p>
</div>

<div class="card">
  <h2>(c) 順列内の歪み: 同一3艇の3連単6順列 vs 3連複(BOX置換仮説の判定材料)</h2>
  <table>
    <tr><th>指標</th><th class="num">値</th></tr>
    <tr><td>評価した3艇組(組×レース)</td><td class="num">{c['n_sets']:,}</td></tr>
    <tr><td>Σ6順列含意確率 ÷ 3連複含意確率(中央値)</td><td class="num">{c['ratio_median']:.3f}</td></tr>
    <tr><td>組内シェア最大の順列: 市場平均</td><td class="num">{c['top_share_market']:.1%}</td></tr>
    <tr><td>同: 現行モデル(Benter)平均</td><td class="num">{c['top_share_model']:.1%}</td></tr>
    <tr><td>最有力順列が市場とモデルで一致する率</td><td class="num">{c['top_agree_rate']:.1%}</td></tr>
    <tr><td>6順列シェアの平均絶対差(市場 vs モデル)</td><td class="num">{c['mad']:.3f}</td></tr>
  </table>
  <p class="note">比が1超=同じ3艇でも3連単側の売上配分が3連複より厚い(順列指定が過熱)。
  モデルの順列配分が市場より尖っている(top_share: モデル>市場)なら順列指定に、
  市場並みかつ一致率が低いならBOX置換(v2候補)に分がある、という読み方をする。</p>
</div>

<div class="card">
  <h2>(d) モデル順位 vs 人気順位のズレ × ken回収率</h2>
  <table>
    <tr><th>順位ズレ(6艇の順位差合計)</th><th class="num">レース数</th>
        <th class="num">投資</th><th class="num">回収</th><th class="num">回収率</th></tr>
    {rank_rows}
  </table>
  <p class="note">「市場と意見が割れたレースほど儲かるか」の観察。
  ※モデル確率は現行MODEL_PATHでの再計算値であり、月次再学習のため
  当時の予想を出したモデルと完全一致ではない((c)(f)の1位勝率も同様)。</p>
</div>

<div class="card">
  <h2>(e) 敗因分解: 外れた本命勝負所(ヒモ拡張仮説の入口判定)</h2>
  <table>
    <tr><th>分類</th><th class="num">件数</th></tr>
    <tr><td>的中</td><td class="num">{miss['的中']}</td></tr>
    <tr><td>軸飛び(予測1位が3着内から消滅) → ヒモを拡張しても救えない</td><td class="num">{miss['軸飛び']}</td></tr>
    <tr><td>ヒモ抜け(軸は残ったが相手が圏外) → ヒモ拡張(v2候補)で救えた可能性</td><td class="num">{miss['ヒモ抜け']}</td></tr>
    <tr><td>分類不能(プラン構成が復元できない)</td><td class="num">{miss['分類不能']}</td></tr>
  </table>
  <table style="margin-top:8px">
    <tr><th>日付</th><th>レース</th><th class="num">予測1位</th><th class="num">結果</th><th>敗因</th></tr>
    {miss_detail}
  </table>
</div>

<div class="card">
  <h2>(f) C勝万舟の的中条件の偏り(検証⑧=大穴一撃フラグの素材)</h2>
  <p class="note">場別: {f['by_venue'] or 'なし'} / 前後半: {f['by_half'] or 'なし'}</p>
  <table>
    <tr><th>日付</th><th>レース</th><th>荒れ度</th><th class="num">1位勝率(現行モデル)</th>
        <th class="num">着順</th><th class="num">回収</th></tr>
    {c_hit_rows}
  </table>
</div>

{_render_show_calibration(d["g"])}

{_render_gami_watch(d["h"])}

<div class="card">
  <h2>紙上対決(遡及シミュレーション・本番成績とは別枠)</h2>
  <h3 style="font-size:.9rem">全レース(スナップショットのある5場レース)</h3>
  <table>
    <tr><th>打ち手</th><th class="num">レース数</th><th class="num">的中率</th>
        <th class="num">投資</th><th class="num">回収</th><th class="num">回収率</th><th class="num">損益</th></tr>
    {player_rows("全レース")}
  </table>
  <h3 style="font-size:.9rem">kenの本命勝負所のみ</h3>
  <table>
    <tr><th>打ち手</th><th class="num">レース数</th><th class="num">的中率</th>
        <th class="num">投資</th><th class="num">回収</th><th class="num">回収率</th><th class="num">損益</th></tr>
    {player_rows("本命勝負所のみ")}
  </table>
  <p class="note">{SMALL_SAMPLE_NOTE}<br>
  A/Bの購入判断は15分前スナップショットのオッズのみ・払戻は実際の確定払戻。
  kenはpicks JSON(=採点対象の朝版プラン)の実額。明細は paper_results.json。</p>
  {_render_paper_challengers(d["paper_ch"])}
</div>

<p class="note">再実行手順: py -X utf8 test/collect_final_odds.py → py -X utf8 test/analyze_market.py</p>
</body>
</html>
"""


def main():
    conn = db.connect(DB_PATH)
    pairs = load_odds_pairs(conn)
    picks = load_picks()
    actual, payout = load_results(conn)

    if not pairs:
        print("スナップショット+最終オッズの揃ったレースがありません。"
              "先に test/collect_final_odds.py を実行してください。")
        conn.close()
        return

    # モデル確率の再計算対象: スナップショットのあるレース + picksの全レース
    need_probs = sorted(set(pairs) | set(picks))
    need_probs = [rid for rid in need_probs
                  if conn.execute("SELECT 1 FROM entries WHERE race_id = ? LIMIT 1",
                                  (rid,)).fetchone()]
    print(f"モデル確率を再計算中({len(need_probs)}レース)...")
    model_probs = compute_model_probs(conn, need_probs)

    dates = sorted({rid[:8] for rid in pairs})
    data = {
        "n_races": len(pairs),
        "date_range": f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]} 〜 "
                      f"{dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]}",
        "a": sec_a_drift(pairs, picks),
        "b": sec_b_calibration(pairs, actual),
        "c": sec_c_permutation(pairs, model_probs),
        "d": sec_d_rank_gap(pairs, picks, model_probs, payout),
        "e": sec_e_miss_breakdown(picks, actual, payout),
        "f": sec_f_c_hits(picks, model_probs),
        "g": sec_g_show_calibration(picks, model_probs, actual),
        "h": sec_h_gami_watch(picks, payout),
        "paper": paper_battle(pairs, picks, payout),
        "paper_ch": paper_challengers(picks, model_probs, payout),
    }
    conn.close()

    paper_path = TEST_DIR / "paper_results.json"
    paper_path.write_text(
        json.dumps({"updated": datetime.now(JST).isoformat(timespec="seconds"),
                    **data["paper"]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    report_path = TEST_DIR / "market_report.html"
    report_path.write_text(render_report(data), encoding="utf-8")
    print(f"出力: {report_path}")
    print(f"出力: {paper_path}")


if __name__ == "__main__":
    main()
