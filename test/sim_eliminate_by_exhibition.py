# -*- coding: utf-8 -*-
"""展示データによる消去が機能するか(2026-09-21ケンさん依頼・条件は依頼文のまま固定)

    py -X utf8 test/sim_eliminate_by_exhibition.py [--stage2]

第1段階(オッズ不要): 各軸の「6艇中ボトムの艇」の実際の着順分布。
  軸1 ex_st(展示STが最も遅い艇。負値=フライングは別扱い、空=出遅れ等は除外)
  軸2 exhibition_time(周回展示タイムが最も遅い艇)
  軸3 ex_course(展示進入が最も外の艇)
  軸4 モデルの1着確率ボトム(月次walk-forward台帳 wf_ledger_pos)
  同じ値で並んだ場合は枠番の大きい方をボトムとする(レース前に決まる規則)。
第2段階(--stage2): 帯A/W/C × E0消去なし/E1モデルボトムを1着位置から/E2展示ボトムを全位置から/E3併用。
  odds_final優先・無ければodds。1点200円・払戻上限なし・完全一致のみ・返還艇を含む目は除外。
"""
import sqlite3
import sys
from collections import defaultdict

import pandas as pd

LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_pos_202605_202609.csv"
YEN = 200
BANDS = {"帯A 55〜90倍": (55.0, 90.0), "帯W 55〜150倍": (55.0, 150.0),
         "帯C 55倍以上": (55.0, float("inf"))}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
exh = defaultdict(dict)
for rid, lane, t, st, course in conn.execute(
        "SELECT race_id, lane, exhibition_time, ex_st, ex_course FROM exhibition"):
    exh[rid][lane] = (t, st, course)
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-07-01'"):
    arr[rid][lane] = ao
L = pd.read_csv(LEDGER, encoding="utf-8-sig", dtype={"refund": str})
model_bot = {rid: int(wo.split("-")[-1]) for rid, wo in zip(L["race_id"], L["win_order"])}
refund_of = {rid: {int(x) for x in rf.split("-") if x}
             for rid, rf in zip(L["race_id"], L["refund"].fillna(""))}


def bottom(vals: dict[int, float]):
    """値が最大(=最も遅い/最も外)の艇。同値なら枠番の大きい方"""
    return max(vals, key=lambda l: (vals[l], l))


axes = {"軸1 展示ST最遅": {}, "軸2 展示タイム最遅": {}, "軸3 展示進入が最も外": {},
        "軸4 モデル1着確率ボトム": {}}
n_f_excluded = 0
for rid, e in exh.items():
    if len(e) < 6 or rid not in model_bot or len(arr.get(rid, {})) < 6:
        continue
    times = {l: v[0] for l, v in e.items() if v[0] is not None}
    sts = {l: v[1] for l, v in e.items() if v[1] is not None and v[1] >= 0}   # F(負値)は別扱いで除外
    n_f_excluded += sum(1 for v in e.values() if v[1] is not None and v[1] < 0)
    courses = {l: v[2] for l, v in e.items() if v[2] is not None}
    if len(times) == 6:
        axes["軸2 展示タイム最遅"][rid] = bottom(times)
        axes["軸4 モデル1着確率ボトム"][rid] = model_bot[rid]
    if len(sts) == 6:
        axes["軸1 展示ST最遅"][rid] = bottom(sts)
    if len(courses) == 6:
        axes["軸3 展示進入が最も外"][rid] = bottom(courses)

print("===== 第1段階: ボトム艇の実際の着順(全艇平均は 1着16.7% / 3着以内50.0%) =====")
print(f"{'軸':<20}{'R数':>7}{'1着率':>8}{'2着率':>8}{'3着率':>8}{'3着以内率':>10}{'平均との差':>10}{'軸4と同じ艇':>12}")
for name, d in axes.items():
    n = len(d)
    if n == 0:
        print(f"{name:<20}{0:>7}  (データなし)")
        continue
    a = [arr[rid].get(l) for rid, l in d.items()]
    r1, r2, r3 = (sum(1 for x in a if x == k) / n for k in (1, 2, 3))
    same = sum(1 for rid, l in d.items() if model_bot[rid] == l) / n
    note = "  ※母数が少なく判断不能" if n < 300 else ""
    print(f"{name:<20}{n:>7,}{r1:>8.1%}{r2:>8.1%}{r3:>8.1%}{r1 + r2 + r3:>10.1%}{(r1 + r2 + r3 - 0.5) * 100:>+9.1f}pt"
          f"{same:>12.1%}{note}")

d2, d4 = axes["軸2 展示タイム最遅"], axes["軸4 モデル1着確率ボトム"]
print("\n― 軸2と軸4の関係(同じレース) ―")
for label, cond in (("両方が同じ艇", lambda r: d2[r] == d4[r]), ("違う艇", lambda r: d2[r] != d4[r])):
    rs = [r for r in d2 if cond(r)]
    for nm, d in (("展示タイム最遅の艇", d2), ("モデルボトムの艇", d4)):
        t3 = sum(1 for r in rs if (arr[r].get(d[r]) or 9) <= 3) / max(1, len(rs))
        w = sum(1 for r in rs if arr[r].get(d[r]) == 1) / max(1, len(rs))
        print(f"  {label}({len(rs):,}R) {nm}: 1着率{w:.1%} / 3着以内率{t3:.1%}")
print("― 軸2の月別 ―")
by_m = defaultdict(list)
for r, l in d2.items():
    by_m[r[:6]].append((arr[r].get(l) or 9, arr[r].get(d4[r]) or 9))
for m, v in sorted(by_m.items()):
    print(f"  {m}: {len(v):,}R 展示タイム最遅 1着{sum(x[0] == 1 for x in v) / len(v):.1%}・3着以内{sum(x[0] <= 3 for x in v) / len(v):.1%}"
          f" / モデルボトム 1着{sum(x[1] == 1 for x in v) / len(v):.1%}・3着以内{sum(x[1] <= 3 for x in v) / len(v):.1%}")
print(f"(展示STが負値=フライングだった艇は{n_f_excluded}艇。軸1では別扱いとして最遅の判定から外した。空=出遅れ等も除外)")

if "--stage2" not in sys.argv:
    conn.close()
    sys.exit()

# ---------------- 第2段階 ----------------
odds, src = {}, {}
for tbl, cond in (("odds", "fetched_at != 'final-backfill'"), ("odds_final", "1=1")):
    tmp = defaultdict(dict)
    for rid, comb, o in conn.execute(
            f"SELECT race_id, combination, odds FROM {tbl} WHERE bet_type='3連単' "
            f"AND odds IS NOT NULL AND odds > 0 AND {cond}"):
        tmp[rid][comb] = o
    for rid, d in tmp.items():
        if len(d) >= 110:
            odds[rid], src[rid] = d, tbl
pay3 = {rid: (c, a or 0) for rid, c, a in conn.execute(
    "SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連単'")}
conn.close()

PATS = ["E0", "E1", "E2", "E3", "E2で落ちる目"]
rows = []
for rid, ex_bot in d2.items():
    o = odds.get(rid)
    if not o or rid not in pay3:
        continue
    mb, refund = d4[rid], refund_of.get(rid, set())
    win_comb, win_amt = pay3[rid]
    rec = {"race_id": rid, "month": rid[:6], "src": src[rid], "amt": win_amt}
    for b in BANDS:
        for p in PATS:
            rec[f"{b}|{p}|n"] = rec[f"{b}|{p}|h"] = 0
    for comb, x in o.items():
        lanes = [int(v) for v in comb.split("-")]
        if x < 55.0 or set(lanes) & refund:
            continue
        e1 = lanes[0] != mb
        e2 = ex_bot not in lanes
        hit = int(comb == win_comb)
        for b, (lo, hi) in BANDS.items():
            if lo <= x < hi:
                for p, ok in (("E0", True), ("E1", e1), ("E2", e2), ("E3", e1 and e2),
                              ("E2で落ちる目", not e2)):
                    if ok:
                        rec[f"{b}|{p}|n"] += 1
                        rec[f"{b}|{p}|h"] += hit
    rows.append(rec)
D = pd.DataFrame(rows)


def stat(sub, b, p):
    n, h = sub[f"{b}|{p}|n"], sub[f"{b}|{p}|h"]
    ret = int((h * sub["amt"]).sum()) * YEN // 100
    inv = int(n.sum()) * YEN
    return int((n > 0).sum()), n.sum() / max(1, (n > 0).sum()), int(h.sum()), h.sum() / max(1, n.sum()), ret, inv, ret / max(1, inv)


def table(sub, title):
    print(f"\n==================== 第2段階 {title}: {len(sub):,}R ====================")
    for b in BANDS:
        print(f"[{b}]")
        print(f"{'':<12}{'対象R':>7}{'点数/R':>8}{'的中':>6}{'的中率':>8}{'払戻合計':>14}{'投資合計':>14}{'回収率':>8}")
        for p in PATS:
            r, ppr, h, hr, ret, inv, roi = stat(sub, b, p)
            print(f"{p:<12}{r:>7,}{ppr:>8.1f}{h:>6,}{hr:>8.2%}{ret:>13,}円{inv:>13,}円{roi:>8.1%}")


def monthly(sub, title):
    print(f"\n― 月別の回収率(的中本数): {title} ―")
    for b in BANDS:
        print(f"[{b}]  月: R数 | " + " | ".join(PATS))
        for m, g in sub.groupby("month"):
            print(f"   {m}: {len(g):>5,} | " + " | ".join(
                f"{stat(g, b, p)[6]:>6.1%}({stat(g, b, p)[2]})" for p in PATS))


F = D[D["src"] == "odds_final"]
print(f"\n第2段階の母数: {len(D):,}R(確定オッズ {len(F):,}R / 締切15分前のみ {len(D) - len(F):,}R)")
print("  確定オッズの月別: " + " / ".join(f"{m} {n:,}R" for m, n in F.groupby("month").size().items()))
table(F, "主: odds_final のみ")
monthly(F, "odds_final のみ")
table(D, "参考: odds 込み")
monthly(D, "odds 込み")
