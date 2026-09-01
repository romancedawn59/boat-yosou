# -*- coding: utf-8 -*-
"""月次反省レポート: 本番の予想(picks JSON)と結果から「予想の仕方」を点検する
(2026-09-01ケンさん承認「①進めて」)

    py -X utf8 test/reflect_monthly.py 2026-08   # 月指定(省略時は当月)

材料は配信済みpicks JSON(本番モデルの予想そのもの)と結果だけ。学習不要・
リークは原理的に混入しない。月末の判定会の定型入力として使う。

出力するもの:
1. 自信の答え合わせ(較正): 帯別・確率十分位・枠別・場別に「予想確率 vs 実際」
   → ズレたマスが次の特徴量候補(KR注入はこの型の反省から生まれた)
2. 超混戦の軸生存率: 現行順位 vs 🧪専用順位(top3_order・2026-09-01から記録)
   と、それぞれで⑬を組んだ場合の紙上回収率
3. 外れの分類: モデル1位が4着以下に沈んだレースの決まり手内訳
   (まくられた=展開事故 / その他)
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
from config import DB_PATH, PROJECT_DIR, jst_today

KM = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}
month = sys.argv[1] if len(sys.argv) > 1 else jst_today().isoformat()[:7]

files = sorted(glob.glob(str(PROJECT_DIR / "docs" / "data"
                             / f"picks_{month}-*.json")))
if not files:
    print(f"{month}: picks JSONが見つかりません")
    sys.exit(1)

races = []
for path in files:
    p = json.load(open(path, encoding="utf-8"))
    races.extend(p["races"])
rids = [r["race_id"] for r in races]

conn = db.connect(DB_PATH)
arr_map = defaultdict(dict)
for rid, lane, ao in conn.execute(
        f"SELECT race_id, lane, arrival_order FROM results "
        f"WHERE race_id IN ({','.join('?' * len(rids))}) "
        f"AND arrival_order IS NOT NULL", rids):
    arr_map[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        f"SELECT race_id, bet_type, combination, amount_yen FROM payouts "
        f"WHERE race_id IN ({','.join('?' * len(rids))})", rids):
    payout_map[rid][(bt, comb)] = amt or 0
tech_map = dict(conn.execute(
    f"SELECT race_id, winning_technique_number FROM races "
    f"WHERE race_id IN ({','.join('?' * len(rids))})", rids))
conn.close()

scored = [r for r in races if len(arr_map[r["race_id"]]) >= 3]
print(f"===== 月次反省レポート {month} =====")
print(f"配信{len(races)}R / 結果あり{len(scored)}R\n")

# ---- 1. 自信の答え合わせ(較正) -------------------------------------------------
print("― 1. 自信の答え合わせ(予想確率 vs 実際の1着率) ―")
bands = [("超混戦帯(<20%)", 0, 0.20), ("本命帯(20-35%)", 0.20, 0.35),
         ("標準帯(35-50%)", 0.35, 0.50), ("堅め帯(50%+)", 0.50, 1.01)]
for name, lo, hi in bands:
    sub = [r for r in scored if lo <= r["ranked"][0][1] < hi]
    if not sub:
        continue
    pred = sum(r["ranked"][0][1] for r in sub) / len(sub)
    act = sum(arr_map[r["race_id"]].get(r["ranked"][0][0]) == 1
              for r in sub) / len(sub)
    flag = " ←ズレ大" if abs(pred - act) >= 0.05 else ""
    print(f"  {name:<14} {len(sub):>4}R 予想{pred:>6.1%} 実際{act:>6.1%}{flag}")

decile = defaultdict(lambda: [0, 0.0, 0])
lane_cal = defaultdict(lambda: [0, 0.0, 0])
venue_cal = defaultdict(lambda: [0, 0.0, 0])
for r in scored:
    arr = arr_map[r["race_id"]]
    for lane, prob in r["ranked"]:
        won = arr.get(lane) == 1
        d = min(int(prob * 10), 9)
        for agg, key in ((decile, d), (lane_cal, lane),
                         (venue_cal, r["venue_code"])):
            a = agg[key]
            a[0] += 1
            a[1] += prob
            a[2] += won
print("  確率十分位(全艇):")
for d in sorted(decile):
    n, ps, w = decile[d]
    flag = " ←ズレ大" if abs(ps / n - w / n) >= 0.05 and n >= 30 else ""
    print(f"    {d*10:>2}-{d*10+10}%帯 {n:>5}艇 予想{ps/n:>6.1%} 実際{w/n:>6.1%}{flag}")
print("  枠別(全艇):")
for lane in sorted(lane_cal):
    n, ps, w = lane_cal[lane]
    flag = " ←ズレ大" if abs(ps / n - w / n) >= 0.03 and n >= 30 else ""
    print(f"    枠{lane}: {n:>4}艇 予想{ps/n:>6.1%} 実際{w/n:>6.1%}{flag}")
worst = sorted(venue_cal.items(),
               key=lambda kv: -abs(kv[1][1] / kv[1][0] - kv[1][2] / kv[1][0]))
print("  場別のズレ上位3:")
for vc, (n, ps, w) in worst[:3]:
    print(f"    場{vc}: {n:>4}艇 予想{ps/n:>6.1%} 実際{w/n:>6.1%}")


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan13(lanes):
    r1, r2, r3, r4, r5 = lanes[:5]
    bets = [("3連単", f"{a}-{b}-{c}", 100)
            for a, b, c in permutations((r1, r2, r3))]
    bets += [("3連単", f"{a}-{b}-{c}", 100)
             for a, b, c in permutations((r1, r2, r4))]
    bets += [("3連単", f"{r3}-{r1}-{r2}", 300),
             ("3連単", f"{r4}-{r1}-{r2}", 300),
             ("3連複", trio(r3, r4, r5), 200)]
    return bets


def score13(lanes, pay):
    merged = defaultdict(int)
    for bt, comb, y in plan13(lanes):
        merged[(bt, comb)] += y
    return (sum(merged.values()),
            sum(pay.get(k, 0) * y // 100 for k, y in merged.items()))


# ---- 2. 超混戦: 現行順位 vs 専用順位 -------------------------------------------
print("\n― 2. 超混戦の軸生存率(現行 vs 🧪専用順位) ―")
kon = [r for r in scored if r["ranked"][0][1] < 0.20 and len(r["ranked"]) >= 5]
arms = {"現行順位": lambda r: [l for l, _p in r["ranked"]],
        "専用順位": lambda r: r.get("top3_order")}
for name, get in arms.items():
    n = ax = st = rt = 0
    for r in kon:
        lanes = get(r)
        if not lanes:
            continue
        arr = arr_map[r["race_id"]]
        top3 = sorted(arr, key=arr.get)[:3]
        n += 1
        ax += lanes[0] in top3 and lanes[1] in top3
        s, t = score13(lanes, payout_map[r["race_id"]])
        st += s
        rt += t
    if n:
        print(f"  {name}: {n:>3}R 軸生存{ax / n:>6.1%} "
              f"⑬紙上回収率{rt / st:>7.1%}({rt - st:+,}円)")
    else:
        print(f"  {name}: 記録なし(top3_orderは2026-09-01配信分から)")

# ---- 3. 外れの分類(モデル1位の沈没) --------------------------------------------
print("\n― 3. 外れの分類(本命帯+超混戦帯でモデル1位が4着以下) ―")
target = [r for r in scored if r["ranked"][0][1] < 0.35]
sinks = []
for r in target:
    arr = arr_map[r["race_id"]]
    ao = arr.get(r["ranked"][0][0])
    if ao is None or ao >= 4:
        sinks.append(r)
tc = Counter(KM.get(tech_map.get(r["race_id"]), "不明") for r in sinks)
print(f"  対象{len(target)}R中、沈没{len(sinks)}R({len(sinks) / max(1, len(target)):.1%})")
for k, v in tc.most_common():
    note = " ←展開事故(Z1-2の守備範囲)" if k in ("まくり", "まくり差し") else ""
    print(f"    {k}: {v}R ({v / len(sinks):.0%}){note}")
print("\n(使い方: ズレ大のマスが次の特徴量候補。月末判定会でこのレポートを見ながら"
      "専用順位の昇格・特徴量の追加を決める)")
