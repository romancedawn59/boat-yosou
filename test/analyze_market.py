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
        "paper": paper_battle(pairs, picks, payout),
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
