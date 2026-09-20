# -*- coding: utf-8 -*-
"""「来ない所を除外し、残りの確率の高い所を取る」は成り立つか
(2026-09-21ケンさん発案)

    py -X utf8 test/verify_exclude_then_pick.py

■ 方法(事前登録)
本命帯の台帳(全24場×1位勝率20〜35%・2026-01〜08)。予想順位の並び120通りの
発生率を1〜4月だけで測り、5〜8月で採点する。
 ① 除外の確かさ: 1〜4月の発生率が低い順に除外した並びが、5〜8月に何%来たか。
    来たときの平均配当(=スルーする損)も出す。
 ② 残りから取る: 1〜4月の発生率上位K通りを5〜8月に各100円で買った回収率。
■ 合格基準: ②のどれかのKで5〜8月の回収率100%超、かつ最大1本除外後も95%超。
"""
import csv
from collections import Counter

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")

rows = []
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    if len(pred) < 6 or len(res) < 3 or not r["払戻(100円)"]:
        continue
    rows.append((r["日付"], tuple(pred.index(l) + 1 for l in res),
                 int(r["払戻(100円)"]), bool(r["対象5場"])))
tr = [x for x in rows if x[0] < "2026-05-01"]
te = [x for x in rows if x[0] >= "2026-05-01"]
cnt = Counter(x[1] for x in tr)
from itertools import permutations
order = sorted(permutations(range(1, 7), 3), key=lambda p: -cnt[p])
print(f"設計 1〜4月 {len(tr):,}R / 採点 5〜8月 {len(te):,}R")

print("\n===== ① 除外の確かさ(1〜4月で来なかった順に除外 → 5〜8月) =====")
print(f"{'除外する並び数':<12}{'1〜4月の発生率':>12}{'5〜8月に来た率':>13}{'来たときの平均配当':>14}")
for n_ex in (20, 40, 60, 80, 100):
    ex = set(order[-n_ex:])
    tr_rate = sum(cnt[p] for p in ex) / len(tr)
    came = [x for x in te if x[1] in ex]
    avg = sum(x[2] for x in came) // max(1, len(came))
    print(f"{n_ex:>6}通り{'':<4}{tr_rate:>12.1%}{len(came) / len(te):>13.1%}{avg:>13,}円")

print("\n===== ② 残りの発生率上位K通りを各100円(5〜8月) =====")
print(f"{'K':<6}{'的中率':>7}{'回収率(全24場)':>13}{'最大1本除く':>10}{'回収率(5場)':>11}  上位の並び")
for k in (1, 3, 6, 10, 20, 40):
    buy = set(order[:k])
    hits = [x for x in te if x[1] in buy]
    st = len(te) * k * 100
    rt = sum(x[2] for x in hits)
    best = max((x[2] for x in hits), default=0)
    te5 = [x for x in te if x[3]]
    rt5 = sum(x[2] for x in te5 if x[1] in buy)
    lab = " ".join("r" + "-r".join(map(str, p)) for p in order[:min(k, 6)])
    print(f"{k:<6}{len(hits) / len(te):>7.1%}{rt / st:>13.1%}{(rt - best) / st:>10.1%}"
          f"{rt5 / (len(te5) * k * 100):>11.1%}  {lab}")
allrt = sum(x[2] for x in te) / (len(te) * 12000)
print(f"(参考) 120通り全部買い: {allrt:.1%}")
