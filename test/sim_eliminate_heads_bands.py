# -*- coding: utf-8 -*-
"""オッズ帯を絞ったうえでの「1着候補の消去」効果(2026-09-21ケンさん再依頼・条件は依頼文のまま固定)

    py -X utf8 test/sim_eliminate_heads_bands.py

帯A 55〜90倍未満 / 帯B 90〜150倍未満 / 帯W 55〜150倍未満 / 帯C 55倍以上(前回条件)
odds_final(確定)を優先、無ければ odds(締切15分前)。レース単位で記録して表を分ける。
消去艇 = ウォークフォワード予測(2026-09は本番配信)の1着確率が最下位の1艇 / 下位2艇。
消去群 = 消去艇を1着位置に持つ目。1点200円・払戻の上限なし・3連単の完全一致のみ。
返還艇を含む目は候補から外す。
判定基準: 残存群の回収率100%超 / 消去群の回収率が同じ帯の0艇除外を明確に下回るか /
          帯ごとの消去の効き幅(残存群−0艇除外)
"""
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
YEN = 200
BANDS = {"帯A 55〜90倍": (55.0, 90.0), "帯B 90〜150倍": (90.0, 150.0),
         "帯W 55〜150倍": (55.0, 150.0), "帯C 55倍以上": (55.0, float("inf"))}
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
        if len(d) >= 110:                 # 後から読む odds_final が上書き=確定を優先
            odds[rid], src[rid] = d, tbl
pay3 = {rid: (c, a or 0) for rid, c, a in conn.execute(
    "SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連単'")}
n_final_db = conn.execute("SELECT COUNT(DISTINCT race_id) FROM odds_final").fetchone()[0]
todo = conn.execute(
    "SELECT COUNT(*) FROM races r WHERE date>='2026-07-21' AND date<='2026-09-20' "
    "AND EXISTS(SELECT 1 FROM payouts p WHERE p.race_id=r.race_id) "
    "AND NOT EXISTS(SELECT 1 FROM odds_final f WHERE f.race_id=r.race_id)").fetchone()[0]
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
rows = []
top = {b: [] for b in BANDS}
for rid, month, p1, pred, rf in zip(L["race_id"], L["month"], L["p1"], L["pred"],
                                    L["refund"].fillna("")):
    o = odds.get(rid)
    if not o or rid not in pay3:
        continue
    order = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    win_comb, win_amt = pay3[rid]
    rec = {"race_id": rid, "month": month, "src": src[rid], "amt": win_amt}
    for b in BANDS:
        for k in ("n", "n6", "n5", "h", "h6", "h5"):
            rec[f"{b}|{k}"] = 0
    for comb, x in o.items():
        lanes = [int(t) for t in comb.split("-")]
        if x < 55.0 or set(lanes) & refund:
            continue
        hr = order.index(lanes[0]) + 1
        is_hit = int(comb == win_comb)
        for b, (lo, hi) in BANDS.items():
            if lo <= x < hi:
                rec[f"{b}|n"] += 1
                rec[f"{b}|h"] += is_hit
                if hr == 6:
                    rec[f"{b}|n6"] += 1
                    rec[f"{b}|h6"] += is_hit
                elif hr == 5:
                    rec[f"{b}|n5"] += 1
                    rec[f"{b}|h5"] += is_hit
                if is_hit and hr >= 5:
                    top[b].append((win_amt, rid, comb, hr, x, src[rid], p1, order))
    rows.append(rec)
D = pd.DataFrame(rows)


def stat(sub, b, part):
    """part: all / keep1 / drop1 / keep2 / drop2 / only5"""
    n, n6, n5 = sub[f"{b}|n"], sub[f"{b}|n6"], sub[f"{b}|n5"]
    h, h6, h5 = sub[f"{b}|h"], sub[f"{b}|h6"], sub[f"{b}|h5"]
    pts, hits = {"all": (n, h), "keep1": (n - n6, h - h6), "drop1": (n6, h6),
                 "keep2": (n - n6 - n5, h - h6 - h5), "drop2": (n6 + n5, h6 + h5),
                 "only5": (n5, h5)}[part]
    races = int((pts > 0).sum())
    tot, hh = int(pts.sum()), int(hits.sum())
    ret = int((hits * sub["amt"]).sum()) * YEN // 100
    inv = tot * YEN
    return races, tot / max(1, races), hh, hh / max(1, tot), ret, inv, ret / max(1, inv)


ROWS = (("0艇除外", "残存群", "all"), ("1艇除外", "残存群", "keep1"), ("1艇除外", "消去群", "drop1"),
        ("2艇除外", "残存群", "keep2"), ("2艇除外", "消去群", "drop2"))


def table(sub, title):
    print(f"\n==================== {title}: {len(sub):,}R ====================")
    for b in BANDS:
        print(f"\n[{b}]")
        print(f"{'条件':<9}{'群':<6}{'対象R':>7}{'点数/R':>8}{'的中':>6}{'的中率':>8}{'払戻合計':>14}{'投資合計':>14}{'回収率':>8}")
        for cname, g, part in ROWS:
            r, ppr, h, hr, ret, inv, roi = stat(sub, b, part)
            print(f"{cname:<9}{g:<6}{r:>7,}{ppr:>8.1f}{h:>6,}{hr:>8.2%}{ret:>13,}円{inv:>13,}円{roi:>8.1%}")
        base, k1, k2 = stat(sub, b, "all")[6], stat(sub, b, "keep1")[6], stat(sub, b, "keep2")[6]
        o5 = stat(sub, b, "only5")
        print(f"  効き幅: 1艇除外 {(k1 - base) * 100:+.1f}pt / 2艇除外 {(k2 - base) * 100:+.1f}pt "
              f"/ 2艇目だけ(予想5位が頭)の回収率 {o5[6]:.1%}({o5[2]}本)")


def monthly(sub, title):
    print(f"\n― 月別の回収率: {title} ―")
    for b in BANDS:
        print(f"[{b}]  月: R数 | 0艇(的中率) | 1艇 残存/消去 | 2艇 残存/消去")
        for m, g in sub.groupby("month"):
            a, k1, d1, k2, d2 = (stat(g, b, p) for p in ("all", "keep1", "drop1", "keep2", "drop2"))
            print(f"   {m}: {len(g):>5,} | {a[6]:>6.1%}({a[3]:.2%}) | {k1[6]:>6.1%} / {d1[6]:>6.1%}"
                  f" | {k2[6]:>6.1%} / {d2[6]:>6.1%}")


F = D[D["src"] == "odds_final"]
print(f"検証に使ったレース: {len(D):,}R(期間 {D['race_id'].min()[:8]}〜{D['race_id'].max()[:8]})")
print(f"  内訳: 確定オッズ(odds_final) {len(F):,}R / 締切15分前(odds)のみ {len(D) - len(F):,}R")
print("  確定オッズの月別: " + " / ".join(f"{m} {n:,}R" for m, n in F.groupby("month").size().items()))
print(f"  遡及取得の進捗: odds_final に {n_final_db:,}R(7/21〜9/20の未取得は残り {todo:,}R)")
table(F, "主: odds_final のみ")
monthly(F, "odds_final のみ")
table(D, "参考: odds 込み(odds_final優先)")
monthly(D, "odds 込み")

for b in ("帯A 55〜90倍", "帯W 55〜150倍"):
    print(f"\n===== {b}: 消去群(頭が予想5位・6位)で的中した目の払戻上位10件 =====")
    print(f"{'払戻(100円)':>11}  {'日付':<11}{'場':<5}{'R':>3}  {'当選目':<8}{'頭の予想順位':>8}{'オッズ':>8}{'値段':>7}{'1位勝率':>8}  予想(1位→6位)")
    for amt, rid, comb, hr, x, s, p1, order in sorted(top[b], reverse=True)[:10]:
        print(f"{amt:>10,}円  {rid[:4]}-{rid[4:6]}-{rid[6:8]} {VENUE[int(rid[9:11])]:<5}{int(rid[12:14]):>3}  "
              f"{comb:<8}{hr:>7}位{x:>8.1f}{('確定' if s == 'odds_final' else '15分前'):>7}{p1:>8.1%}  "
              + "-".join(map(str, order)))
    t = top[b]
    print(f"  的中{len(t)}本(予想6位頭{sum(1 for a in t if a[3] == 6)}本・予想5位頭{sum(1 for a in t if a[3] == 5)}本) / "
          f"払戻合計(100円){sum(a[0] for a in t):,}円 / 最高{max((a[0] for a in t), default=0):,}円")
