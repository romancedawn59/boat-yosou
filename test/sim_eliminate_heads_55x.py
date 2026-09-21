# -*- coding: utf-8 -*-
"""55倍以上の目に対する「1着候補の消去」効果の測定(2026-09-21ケンさん依頼・条件は依頼文のまま固定)

    py -X utf8 test/sim_eliminate_heads_55x.py

候補   = 3連単オッズ55.0倍以上の目。odds_final(確定)を優先、無ければ odds(締切15分前)。
         どちらを使ったかをレース単位で記録し、表を分ける。
消去艇 = モデルの1着確率が最下位の1艇 / 下位2艇
         (予想順位は data_raw/wf_ledger_all_202510_202609.csv = 2025-10〜2026-08は
          月次walk-forward、2026-09は本番配信。オッズがあるのは2026-05〜)
消去群 = 消去艇を1着位置に持つ目 / 残存群 = それ以外
的中   = 3連単の完全一致のみ。払戻は実際の確定払戻・上限なし。1点200円換算。
返還艇(欠場等)を含む目は買えないので候補から外す。
判定基準: 残存群の的中率3.0%以上 / 残存群の回収率165%以上 / 消去群の回収率が75%を明確に下回る
"""
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
MIN_ODDS = 55.0
YEN = 200
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡",
         8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門",
         15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋",
         22: "福岡", 23: "唐津", 24: "大村"}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
odds, src = defaultdict(dict), {}
for tbl, cond in (("odds", "fetched_at != 'final-backfill'"), ("odds_final", "1=1")):
    tmp = defaultdict(dict)
    for rid, comb, o in conn.execute(
            f"SELECT race_id, combination, odds FROM {tbl} WHERE bet_type='3連単' "
            f"AND odds IS NOT NULL AND odds > 0 AND {cond}"):
        tmp[rid][comb] = o
    for rid, d in tmp.items():
        if len(d) >= 110:                 # 後から読む odds_final が上書き=確定を優先
            odds[rid], src[rid] = d, tbl
pay3 = {rid: (c, a or 0) for rid, c, a in conn.execute(
    "SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連単'")}
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
rows, top = [], []
for rid, month, venue, p1, pred, rf in zip(L["race_id"], L["month"], L["venue"], L["p1"],
                                           L["pred"], L["refund"].fillna("")):
    o = odds.get(rid)
    if not o or rid not in pay3:
        continue
    order = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    win_comb, win_amt = pay3[rid]
    n = {"all": 0, "r6": 0, "r56": 0}
    hit = {"all": 0, "r6": 0, "r56": 0}
    for comb, x in o.items():
        lanes = [int(t) for t in comb.split("-")]
        if x < MIN_ODDS or set(lanes) & refund:
            continue
        head_rank = order.index(lanes[0]) + 1
        is_hit = comb == win_comb
        n["all"] += 1
        hit["all"] += is_hit
        if head_rank == 6:
            n["r6"] += 1
            hit["r6"] += is_hit
        if head_rank >= 5:
            n["r56"] += 1
            hit["r56"] += is_hit
            if is_hit:
                top.append((win_amt, rid, comb, head_rank, x, src[rid], p1, order))
    rows.append((rid, month, src[rid], n["all"], n["r6"], n["r56"],
                 hit["all"], hit["r6"], hit["r56"], win_amt))
D = pd.DataFrame(rows, columns=["race_id", "month", "src", "n_all", "n_r6", "n_r56",
                                "h_all", "h_r6", "h_r56", "amt"])


def stat(sub, n_col, h_col, neg=None):
    """(対象R, 平均点数, 的中, 的中率, 払戻, 投資, 回収率)。negがあれば全体から引いた残存群"""
    if neg:
        pts = sub["n_all"] - sub[neg[0]]
        hits = sub["h_all"] - sub[neg[1]]
    else:
        pts, hits = sub[n_col], sub[h_col]
    races = int((pts > 0).sum())
    tot = int(pts.sum())
    h = int(hits.sum())
    ret = int((hits * sub["amt"]).sum() * YEN // 100)
    inv = tot * YEN
    return races, tot / max(1, races), h, h / max(1, tot), ret, inv, ret / max(1, inv)


def table(sub, title):
    print(f"\n===== {title}: {len(sub):,}R =====")
    print(f"{'条件':<10}{'群':<6}{'対象R':>7}{'点数/R':>8}{'的中':>6}{'的中率':>8}{'払戻合計':>14}{'投資合計':>14}{'回収率':>8}")
    for name, n_col, h_col in (("0艇除外", None, None), ("1艇除外", "n_r6", "h_r6"),
                               ("2艇除外", "n_r56", "h_r56")):
        groups = [("残存群", stat(sub, "n_all", "h_all") if n_col is None
                   else stat(sub, None, None, neg=(n_col, h_col)))]
        if n_col:
            groups.append(("消去群", stat(sub, n_col, h_col)))
        for g, (r, ppr, h, hr, ret, inv, roi) in groups:
            print(f"{name:<10}{g:<6}{r:>7,}{ppr:>8.1f}{h:>6,}{hr:>8.2%}{ret:>13,}円{inv:>13,}円{roi:>8.1%}")


def monthly(sub, title):
    print(f"\n― 月別: {title} ―")
    print(f"{'月':<9}{'R数':>6} | {'0艇 的中率':>9}{'回収率':>8} | {'1艇 残存 的中率':>13}{'回収率':>8}{'消去 回収率':>10}"
          f" | {'2艇 残存 的中率':>13}{'回収率':>8}{'消去 回収率':>10}")
    for m, g in sub.groupby("month"):
        b = stat(g, "n_all", "h_all")
        r1, e1 = stat(g, None, None, neg=("n_r6", "h_r6")), stat(g, "n_r6", "h_r6")
        r2, e2 = stat(g, None, None, neg=("n_r56", "h_r56")), stat(g, "n_r56", "h_r56")
        print(f"{m:<9}{len(g):>6,} | {b[3]:>9.2%}{b[6]:>8.1%} | {r1[3]:>13.2%}{r1[6]:>8.1%}{e1[6]:>10.1%}"
              f" | {r2[3]:>13.2%}{r2[6]:>8.1%}{e2[6]:>10.1%}")


print(f"候補の定義: 3連単{MIN_ODDS}倍以上 / 1点{YEN}円換算 / 払戻の上限なし")
print(f"使えるレース: 確定オッズ(odds_final) {int((D['src'] == 'odds_final').sum()):,}R / "
      f"締切15分前(odds)のみ {int((D['src'] == 'odds').sum()):,}R")
F = D[D["src"] == "odds_final"]
table(F, "主: odds_final のみ(実際の確定の値段)")
monthly(F, "odds_final のみ")
table(D, "参考: odds 込み(odds_final優先・無ければ締切15分前)")
monthly(D, "odds 込み")
table(D[D["src"] == "odds"], "参考: 締切15分前(odds)だけ")

print("\n===== 消去群で的中した目のうち払戻が高かった上位10件(2艇除外の消去群=頭が予想5位・6位) =====")
print(f"{'払戻(100円)':>11}  {'日付':<10}{'場':<5}{'R':>3}  {'当選目':<8}{'頭の予想順位':>10}{'オッズ':>9}{'値段の種類':>10}{'1位勝率':>8}  予想(1位→6位)")
for amt, rid, comb, hr, x, s, p1, order in sorted(top, reverse=True)[:10]:
    print(f"{amt:>10,}円  {rid[:4]}-{rid[4:6]}-{rid[6:8]} {VENUE[int(rid[9:11])]:<5}{int(rid[12:14]):>3}  "
          f"{comb:<8}{hr:>9}位{x:>9.1f}{('確定' if s == 'odds_final' else '15分前'):>10}{p1:>8.1%}  "
          + "-".join(map(str, order)))
for label, k in (("予想6位が頭", 6), ("予想5位が頭", 5)):
    t = [a for a, *_r, in top if _r[2] == k]
    print(f"  {label}の的中: {len(t)}本 / 払戻合計(100円) {sum(t):,}円 / うち万舟 {sum(1 for a in t if a >= 10000)}本 "
          f"/ 5万円以上 {sum(1 for a in t if a >= 50000)}本")
