# -*- coding: utf-8 -*-
"""対象場の選び方の検証(2026-09-21ケンさん「5場のセレクト方法を考えた方が良さそう」)

    py -X utf8 test/sim_venue_select_0921.py

帯 = 1位勝率22.5〜30%(全24場)。2026-01〜08は月次walk-forward台帳、09月は本番配信picks。
採点月 = 2026-05〜09。各採点月の場選びは、その月より前の帯レースだけで行う(後出し禁止)。
選び方(いずれも過去20R以上の場から上位5場):
  ROI順    過去のドンピシャ回収率
  的中率順  過去のドンピシャ的中率
  1位1着順  予想1位が1着になった率
  上位3艇順 予想上位3艇で決着した率
  配当順    3連単の平均配当
  期待値順  ドンピシャの(発生確率×的中時の平均配当)。場ごとの標本が小さいので
            確率は全場平均へ100R分、配当は全場平均へ5本分だけ寄せて推定(縮小推定)
比較: 現行5場固定 / 全24場。買い方: ドンピシャ1点500円 / 中間案1,000円。
■ 合格基準: 現行5場固定を両方の買い方で上回り、かつ最大1本除外後も100%超。
"""
import csv
import glob
import json
import sqlite3
from collections import Counter, defaultdict

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}
NAME = {v: k for k, v in VEN.items()}
FIVE = {3, 4, 8, 13, 20}
EVAL = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
MIN_R = 20
P_PRIOR = 100     # 発生確率を全場平均へ寄せる強さ(R数換算)
PAY_PRIOR = 5     # 平均配当を全場平均へ寄せる強さ(的中本数換算)

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type IN ('3連単','3連複')"):
    pay[rid][(bt, comb)] = amt or 0
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-09-01' "
        "AND res.arrival_order IS NOT NULL"):
    arr[rid][lane] = ao
conn.close()

races = []   # (month, venue, pred, rid, 結果の並び, 3連単配当)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    p1 = float(r["モデル1位勝率"].rstrip("%"))
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if 22.5 <= p1 < 30 and len(pred) >= 6 and len(res) >= 3 and rid in pay:
        races.append((r["日付"][:7], VEN[r["場"]], pred, rid,
                      tuple(pred.index(l) + 1 for l in res),
                      pay[rid].get(("3連単", r["結果(3連単)"]), 0)))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        rid = r["race_id"]
        a = arr.get(rid)
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in pay \
                or not a or len(a) < 3 or not (0.225 <= r["ranked"][0][1] < 0.30):
            continue
        pred = [x[0] for x in r["ranked"]]
        top3 = sorted(a, key=a.get)[:3]
        races.append((pk["date"][:7], r["venue_code"], pred, rid,
                      tuple(pred.index(l) + 1 for l in top3),
                      pay[rid].get(("3連単", "-".join(map(str, top3))), 0)))


def trio(*x):
    s = sorted(x)
    return ("3連複", f"{s[0]}={s[1]}={s[2]}")


def don(l):
    return [("3連単", f"{l[0]}-{l[1]}-{l[2]}", 500)]


def chukan(l):
    r1, r2, r3, r4 = l[:4]
    return [(*trio(r1, r2, r3), 100), (*trio(r2, r3, r4), 100),
            ("3連単", f"{r1}-{r2}-{r3}", 300), ("3連単", f"{r3}-{r1}-{r2}", 200),
            ("3連単", f"{r4}-{r1}-{r2}", 100), ("3連単", f"{r4}-{r2}-{r1}", 200)]


def venue_stats(hist):
    st = defaultdict(lambda: [0, 0, 0, 0, 0])   # R, ドンピシャ数, ドンピシャ払戻, 1位1着, 上位3艇
    paysum = Counter()
    for _m, v, _pred, _rid, pt, amt in hist:
        s = st[v]
        s[0] += 1
        s[1] += pt == (1, 2, 3)
        s[2] += amt if pt == (1, 2, 3) else 0
        s[3] += pt[0] == 1
        s[4] += max(pt) <= 3
        paysum[v] += amt
    n_all = sum(s[0] for s in st.values())
    h_all = sum(s[1] for s in st.values())
    p0 = h_all / n_all
    pay0 = sum(s[2] for s in st.values()) / max(1, h_all)
    return {v: {"ROI順": s[2] / (s[0] * 100), "的中率順": s[1] / s[0],
                "1位1着順": s[3] / s[0], "上位3艇順": s[4] / s[0],
                "配当順": paysum[v] / s[0],
                "期待値順": ((s[1] + P_PRIOR * p0) / (s[0] + P_PRIOR))
                * ((s[2] + PAY_PRIOR * pay0) / (s[1] + PAY_PRIOR)) / 100,
                "_raw": (s[0], s[1], s[2], p0, pay0)}
            for v, s in st.items() if s[0] >= MIN_R}


METHODS = ["ROI順", "的中率順", "1位1着順", "上位3艇順", "配当順", "期待値順"]
chosen = {m: {} for m in METHODS}
for em in EVAL:
    vs = venue_stats([x for x in races if x[0] < em])
    for m in METHODS:
        chosen[m][em] = set(sorted(vs, key=lambda v: -vs[v][m])[:5])

print("===== 各採点月に選ばれた場(その月より前のデータだけで選定) =====")
for m in METHODS:
    print(f"― {m} ―")
    for em in EVAL:
        print(f"  {em[5:]}月: " + "・".join(NAME[v] for v in sorted(chosen[m][em])))

print("\n===== 場ごとのドンピシャ期待値(全期間1〜9月・100円あたり) =====")
print(f"{'場':<8}{'R数':>5}{'的中':>4}{'発生確率':>8}{'平均配当':>9}{'確率×配当':>9}"
      f"{'寄せた確率':>9}{'寄せた配当':>10}{'期待値(寄せ後)':>12}")
vs_all = venue_stats(races)
for v in sorted(vs_all, key=lambda v: -vs_all[v]["期待値順"]):
    n, h, psum, p0, pay0 = vs_all[v]["_raw"]
    ps = (h + P_PRIOR * p0) / (n + P_PRIOR)
    pays = (psum + PAY_PRIOR * pay0) / (h + PAY_PRIOR)
    print(f"{NAME[v] + ('★' if v in FIVE else ''):<8}{n:>5}{h:>4}{h / n:>8.1%}"
          f"{(psum // h if h else 0):>8,}円{psum / (n * 100):>9.1%}{ps:>9.1%}"
          f"{pays:>9,.0f}円{ps * pays / 100:>12.1%}")
print(f"(全場平均: 発生確率{p0:.1%} × 平均配当{pay0:,.0f}円 = {p0 * pay0 / 100:.1%})")

arms = [("現行5場固定", lambda em: FIVE), ("全24場", lambda em: set(NAME))] \
    + [(m, (lambda mm: lambda em: chosen[mm][em])(m)) for m in METHODS]
for pname, fn in (("ドンピシャ1点500円", don), ("中間案1,000円", chukan)):
    print(f"\n===== {pname}(採点5〜9月) =====")
    print(f"{'選び方':<12}{'R数':>5}{'的中':>5}{'的中率':>7}{'回収率':>8}{'損益':>11}{'最大1本除く':>10}"
          f"  月別回収率")
    for aname, pick in arms:
        n = hit = st = rt = best = 0
        mon = defaultdict(lambda: [0, 0])
        for em, v, pred, rid, _pt, _amt in races:
            if em not in EVAL or v not in pick(em):
                continue
            bets = fn(pred)
            s = sum(y for *_x, y in bets)
            g = sum(pay[rid].get((bt, cb), 0) * y // 100 for bt, cb, y in bets)
            n += 1
            hit += g > 0
            st += s
            rt += g
            best = max(best, g)
            mon[em][0] += s
            mon[em][1] += g
        ms = " ".join(f"{k[5:]}月{v[1] / v[0]:.0%}" for k, v in sorted(mon.items()))
        print(f"{aname:<12}{n:>5}{hit:>5}{hit / max(1, n):>7.1%}{rt / max(1, st):>8.1%}"
              f"{rt - st:>+10,}円{(rt - best) / max(1, st):>10.1%}  {ms}")
