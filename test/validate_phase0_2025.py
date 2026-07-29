# -*- coding: utf-8 -*-
"""Phase 0: 2025年8-10月データでのアウトオブサンプル検証(ケン理論 最終ルール)

    py -X utf8 test/validate_phase0_2025.py

【重要な制約】
odds_backfill.dbは「過去日付ページは最終(確定後)オッズを表示し続ける」裏技で
取得したものなので、ここに入っているのは締切15分前スナップショットではなく
決着後の確定オッズ。2026年5-7月の検証で使った`odds`テーブル(真の15分前
snapshot・931R)とは取得タイミングが異なる。今日の実戦で確認した通り
締切前後でオッズは数倍単位で動くため、この検証は「15分前に見えていた窓」の
完全な再現ではなく近似(prox)である。ただし判定ロジック自体は不変で、
N数が931→万単位に増える利点は大きいので、傾向の再現性チェックとして扱う。

対象: odds_backfill.db(3並列取得を統合済み)の全レース × 本体DBのentries/results/payouts
検証: ①複4点(本線) ②T3差され単8点 ③複オッズ水準で車両切替(A組/B組)
"""
import sys
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
import db
from config import DB_PATH, PROJECT_DIR

CLASS_PT = {"A1": 4.0, "A2": 3.0, "B1": 2.0, "B2": 1.0}
ODDS_DB = PROJECT_DIR / "odds_backfill.db"


def z(vals):
    xs = [v for v in vals.values() if v is not None]
    if len(xs) < 2:
        return {k: 0.0 for k in vals}
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
    return {k: ((v or 0) - m) / sd for k, v in vals.items()}


def main():
    import sqlite3
    # 書き込み中のバッチと衝突しないよう読み取り専用で開く
    odb = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
    rids = [r[0] for r in odb.execute(
        "SELECT DISTINCT race_id FROM odds_final")]
    print(f"odds_backfill対象レース: {len(rids):,}R")
    if not rids:
        print("データなし。バッチ未完了の可能性。")
        return
    ph = ",".join("?" for _ in rids)

    tan_by, fuku_by = defaultdict(dict), defaultdict(dict)
    for rid, bt, comb, o in odb.execute(
        "SELECT race_id, bet_type, combination, odds FROM odds_final"):
        if o:
            (tan_by if bt == "3連単" else fuku_by)[rid][comb] = o
    odb.close()

    conn = db.connect(DB_PATH)
    races, dates = defaultdict(list), {}
    for rid, d, lane, cls, nwr, lwr, n3, m2, m3 in conn.execute(f"""
        SELECT e.race_id, r.date, e.lane, e.racer_class, e.national_win_rate,
               e.local_win_rate, e.national_3rate, e.motor_2rate, e.motor_3rate
        FROM entries e JOIN races r ON r.race_id = e.race_id
        WHERE e.race_id IN ({ph})""", rids):
        races[rid].append(dict(lane=lane, cls=cls, nwr=nwr, lwr=lwr,
                               n3=n3, m2=m2, m3=m3))
        dates[rid] = d
    arrivals = defaultdict(dict)
    for rid, lane, o in conn.execute(f"""
        SELECT race_id, lane, arrival_order FROM results
        WHERE arrival_order IS NOT NULL AND race_id IN ({ph})""", rids):
        arrivals[rid][o] = lane
    payout = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(f"""
        SELECT race_id, bet_type, combination, amount_yen FROM payouts
        WHERE race_id IN ({ph})""", rids):
        payout[rid][(bt, comb)] = amt or 0
    conn.close()

    signals = []
    for rid, rows in races.items():
        if len(rows) != 6:
            continue
        a = arrivals.get(rid, {})
        if not all(a.get(i) for i in (1, 2, 3)):
            continue
        tan = tan_by.get(rid)
        if not tan or len(tan) < 100:
            continue
        fav = min(tan.values())
        lanes = [r["lane"] for r in rows]
        g = lambda k: {r["lane"]: r[k] for r in rows}
        zm, zl, zn = z(g("m2")), z(g("lwr")), z(g("nwr"))
        zc = z({r["lane"]: CLASS_PT.get(r["cls"], 2.0) for r in rows})
        lead = {ln: zm[ln] + zl[ln] + zn[ln] + zc[ln] for ln in lanes}
        rk = sorted(lanes, key=lambda l: -lead[l])
        head = defaultdict(float)
        for comb, v in tan.items():
            try:
                head[int(comb.split("-")[0])] += 1 / v
            except (ValueError, IndexError):
                continue
        if len(head) < 6:
            continue
        mrank = {ln: i + 1 for i, ln in
                 enumerate(sorted(head, key=lambda k: -head[k]))}
        div = max(mrank[l] - (rk.index(l) + 1) for l in rk[:2])
        fo = fuku_by.get(rid, {})
        ticket = ["=".join(map(str, sorted(rk[:2] + [t]))) for t in rk[2:]]
        known = [fo[c] for c in ticket if c in fo]
        avg_fo = sum(known) / len(known) if len(known) >= 3 else None
        signals.append((rid, dates[rid], rk, fav, div, avg_fo))

    print(f"6艇・結果確定・オッズ完備: {len(signals):,}R\n")

    def run(sel, kind):
        stake = ret = hits = 0
        for rid, d, rk, fav, div, avg_fo in sel:
            a1, a2 = rk[0], rk[1]
            rest = rk[2:]
            bets = []
            if kind == "複4点":
                bets = [("3連複", "=".join(map(str, sorted([a1, a2, t]))), 200)
                        for t in rest]
            elif kind == "T3差され8点":
                bets = [("3連単", f"{t}-{x}-{y}", 100)
                        for t in rest for x, y in ((a1, a2), (a2, a1))]
            got_any = 0
            for bt, c, yen in bets:
                stake += yen
                got = payout[rid].get((bt, c), 0) * yen // 100
                ret += got
                got_any += got
            if got_any:
                hits += 1
        return stake, ret, hits

    for div_th, label in ((0, "窓のみ(乖離なし)"), (3, "★最終ルール(窓+乖離3)")):
        sel = [s for s in signals if 15 < s[3] <= 18 and s[4] >= div_th]
        print(f"=== {label}: {len(sel):,}R ===")
        for kind in ("複4点", "T3差され8点"):
            st, rt, h = run(sel, kind)
            if st:
                print(f"  {kind:<14} 的中{h:>4}R  投資{st:>10,}円  回収{rt:>10,}円  "
                      f"回収率{rt/st:>7.1%}")
        # 条件付き車両切替(A組/B組)
        selA = [s for s in sel if s[5] is not None and s[5] < 15]
        selB = [s for s in sel if s[5] is not None and s[5] >= 15]
        stA, rtA, hA = run(selA, "複4点")
        stB, rtB, hB = run(selB, "T3差され8点")
        if stA + stB:
            print(f"  {'条件付き切替':<14} 的中{hA+hB:>4}R  投資{stA+stB:>10,}円  "
                  f"回収{rtA+rtB:>10,}円  回収率{(rtA+rtB)/(stA+stB):>7.1%}  "
                  f"(A組{len(selA)}R複4点+B組{len(selB)}R差され単)")

        # 月別(最終ルールのみ)
        if div_th == 3:
            print("  月別内訳(複4点):")
            monthly = defaultdict(lambda: [0, 0, 0])
            for rid, d, rk, fav, div, avg_fo in sel:
                m = d[:7]
                st, rt, h = run([(rid, d, rk, fav, div, avg_fo)], "複4点")
                monthly[m][0] += st
                monthly[m][1] += rt
                monthly[m][2] += 1
            for m in sorted(monthly):
                st, rt, n = monthly[m]
                if st:
                    print(f"    {m}: {n:>3}R 投資{st:>7,}円 回収{rt:>7,}円 "
                          f"回収率{rt/st:>7.1%}")
        print()

    print("【この検証の制約】オッズは確定後(=最終)値。締切15分前スナップショット\n"
          "ではないため、当日の実戦で確認した通りの終盤の値動きは反映できていない。\n"
          "傾向(方向・概ねの水準)の確認として扱うこと。")


if __name__ == "__main__":
    main()
