# -*- coding: utf-8 -*-
"""消去パターンの比較: 全位置1艇 vs 2着位置のみ2艇(2026-09-21ケンさん依頼・条件は依頼文のまま固定)

    py -X utf8 test/sim_eliminate_patterns.py

順位付けは test/build_wf_ledger_positions.py の台帳(月次walk-forward・2026-05〜09)。
  m_win=1着確率 / m_place2=2着確率 / m_top3=3着以内確率(「3着ちょうど」ではない点に注意)
P0 消去なし / P1 1着位置から1着確率ボトム / P2 全位置で1艇(1着・2着・3着以内のボトム) /
P3 P2に加えて2着位置から2着確率5位も除外(3着位置には残す)
帯A 55〜90 / 帯B 90〜150 / 帯W 55〜150 / 帯C 55倍以上。odds_final優先・無ければodds。
1点200円・払戻の上限なし・3連単の完全一致のみ・返還艇を含む目は除外。
"""
import sqlite3
from collections import Counter, defaultdict

import pandas as pd

LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_pos_202605_202609.csv"
YEN = 200
BANDS = {"帯A 55〜90倍": (55.0, 90.0), "帯B 90〜150倍": (90.0, 150.0),
         "帯W 55〜150倍": (55.0, 150.0), "帯C 55倍以上": (55.0, float("inf"))}
PATS = ["P0", "P1", "P2", "P3"]
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡",
         8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門",
         15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋",
         22: "福岡", 23: "唐津", 24: "大村"}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
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
n_final_db = conn.execute("SELECT COUNT(DISTINCT race_id) FROM odds_final").fetchone()[0]
todo = conn.execute(
    "SELECT COUNT(*) FROM races r WHERE EXISTS(SELECT 1 FROM payouts p WHERE p.race_id=r.race_id) "
    "AND NOT EXISTS(SELECT 1 FROM odds_final f WHERE f.race_id=r.race_id)").fetchone()[0]
conn.close()

L = pd.read_csv(LEDGER, encoding="utf-8-sig", dtype={"refund": str})
rows, dropped_hits = [], []
second_dist = {b: {p: Counter() for p in ("P2", "P3")} for b in BANDS}
for rid, month, p1, wo, po, to, rf in zip(L["race_id"], L["month"], L["p1"], L["win_order"],
                                          L["p2_order"], L["top3_order"], L["refund"].fillna("")):
    o = odds.get(rid)
    if not o or rid not in pay3:
        continue
    w = [int(x) for x in wo.split("-")]
    p2 = [int(x) for x in po.split("-")]
    t3 = [int(x) for x in to.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    win_bot, p2_bot, p2_5th, t3_bot = w[-1], p2[-1], p2[-2], t3[-1]
    win_comb, win_amt = pay3[rid]
    rec = {"race_id": rid, "month": month, "src": src[rid], "amt": win_amt}
    for b in BANDS:
        for p in PATS + ["D"]:              # D = P2にあってP3で落ちる目
            rec[f"{b}|{p}|n"] = 0
            rec[f"{b}|{p}|h"] = 0
    for comb, x in o.items():
        a, s, t = (int(v) for v in comb.split("-"))
        if x < 55.0 or {a, s, t} & refund:
            continue
        in_p1 = a != win_bot
        in_p2 = in_p1 and s != p2_bot and t != t3_bot
        in_p3 = in_p2 and s != p2_5th
        hit = int(comb == win_comb)
        for b, (lo, hi) in BANDS.items():
            if not (lo <= x < hi):
                continue
            for p, ok in (("P0", True), ("P1", in_p1), ("P2", in_p2), ("P3", in_p3),
                          ("D", in_p2 and not in_p3)):
                if ok:
                    rec[f"{b}|{p}|n"] += 1
                    rec[f"{b}|{p}|h"] += hit
            if hit and in_p2:
                second_dist[b]["P2"][s] += 1
                if in_p3:
                    second_dist[b]["P3"][s] += 1
                else:
                    dropped_hits.append((b, win_amt, rid, comb, x, src[rid], p2.index(s) + 1,
                                         w.index(a) + 1, p1))
    rows.append(rec)
D = pd.DataFrame(rows)


def stat(sub, b, p):
    n, h = sub[f"{b}|{p}|n"], sub[f"{b}|{p}|h"]
    act = n[n > 0]
    ret = int((h * sub["amt"]).sum()) * YEN // 100
    inv = int(n.sum()) * YEN
    return {"R": int((n > 0).sum()), "avg": act.mean() if len(act) else 0,
            "min": int(act.min()) if len(act) else 0, "max": int(act.max()) if len(act) else 0,
            "hit": int(h.sum()), "rate": h.sum() / max(1, n.sum()), "ret": ret, "inv": inv,
            "roi": ret / max(1, inv), "pts": int(n.sum())}


def table(sub, title):
    print(f"\n==================== {title}: {len(sub):,}R ====================")
    for b in BANDS:
        print(f"\n[{b}]")
        print(f"{'':<5}{'対象R':>7}{'点数/R 平均':>10}{'最小':>5}{'最大':>5}{'的中':>6}{'的中率':>8}"
              f"{'払戻合計':>14}{'投資合計':>14}{'回収率':>8}")
        st = {p: stat(sub, b, p) for p in PATS + ["D"]}
        for p in PATS:
            s = st[p]
            print(f"{p:<5}{s['R']:>7,}{s['avg']:>10.1f}{s['min']:>5}{s['max']:>5}{s['hit']:>6,}{s['rate']:>8.2%}"
                  f"{s['ret']:>13,}円{s['inv']:>13,}円{s['roi']:>8.1%}")
        d = st["D"]
        verdict = ("追加除外は正しい(P3を採る)" if d["roi"] < st["P3"]["roi"]
                   else "追加除外は利益を削っている(P2で止める)")
        print(f"  P2−P3で落ちる目: {d['pts']:,}点(1Rあたり{d['pts'] / max(1, len(sub)):.1f}点) 的中{d['hit']}本 "
              f"払戻{d['ret']:,}円 投資{d['inv']:,}円 回収率{d['roi']:.1%} ⇔ P3残存{st['P3']['roi']:.1%} → {verdict}")


def monthly(sub, title):
    print(f"\n― 月別の回収率(的中本数): {title} ―")
    for b in BANDS:
        print(f"[{b}]  月: R数 | P0 | P1 | P2 | P3 | 落ちる目")
        for m, g in sub.groupby("month"):
            print(f"   {m}: {len(g):>5,} | " + " | ".join(
                f"{stat(g, b, p)['roi']:>6.1%}({stat(g, b, p)['hit']})" for p in PATS + ["D"]))


F = D[D["src"] == "odds_final"]
print(f"検証に使ったレース: {len(D):,}R(期間 {D['race_id'].min()[:8]}〜{D['race_id'].max()[:8]})")
print(f"  内訳: 確定オッズ(odds_final) {len(F):,}R / 締切15分前(odds)のみ {len(D) - len(F):,}R")
print("  確定オッズの月別: " + " / ".join(f"{m} {n:,}R" for m, n in F.groupby("month").size().items()))
print(f"  遡及取得の進捗: odds_final に {n_final_db:,}R(全期間の未取得は残り {todo:,}R)")
table(F, "主: odds_final のみ")
monthly(F, "odds_final のみ")
table(D, "参考: odds 込み(odds_final優先)")
monthly(D, "odds 込み")

print("\n===== P3で落ちた目のうち、的中して払戻が高かった上位10件(帯C=55倍以上の全体から) =====")
print(f"{'払戻(100円)':>11}  {'日付':<11}{'場':<5}{'R':>3}  {'当選目':<8}{'オッズ':>8}{'値段':>7}"
      f"{'2着艇の2着確率順位':>14}{'1着艇の1着確率順位':>14}{'1位勝率':>8}")
for b, amt, rid, comb, x, s, r2, r1, p1 in sorted(
        (t for t in dropped_hits if t[0] == "帯C 55倍以上"), key=lambda t: -t[1])[:10]:
    print(f"{amt:>10,}円  {rid[:4]}-{rid[4:6]}-{rid[6:8]} {VENUE[int(rid[9:11])]:<5}{int(rid[12:14]):>3}  "
          f"{comb:<8}{x:>8.1f}{('確定' if s == 'odds_final' else '15分前'):>7}{r2:>13}位{r1:>13}位{p1:>8.1%}")
for b in BANDS:
    t = [x for x in dropped_hits if x[0] == b]
    outer = sum(1 for x in t if int(x[3].split("-")[1]) >= 5)
    link = Counter(f"{x[3].split('-')[0]}→{x[3].split('-')[1]}" for x in t)
    print(f"  {b}: 落ちた的中{len(t)}本 / 2着が5・6号艇{outer}本 / 多い連結 "
          + " ".join(f"{k}:{v}" for k, v in link.most_common(5)))

print("\n===== 的中した目の2着艇の艇番分布(odds込み) =====")
for b in BANDS:
    for p in ("P2", "P3"):
        c = second_dist[b][p]
        tot = sum(c.values())
        print(f"  {b} {p}: 的中{tot:>5}本 | " + " ".join(f"{l}号艇{c[l]:>4}({c[l] / max(1, tot):.0%})"
                                                     for l in range(1, 7)))
