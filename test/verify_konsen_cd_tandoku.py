# -*- coding: utf-8 -*-
"""超混戦: 廃止予定のC複・D複を「3連単2点」で置き換えて的中を捉え直せるか
(2026-07-29判断会中のケンさん発案)

    py -X utf8 test/verify_konsen_cd_tandoku.py

■ 問い
案1はC複{r1,r3,r4}とD複{r2,r3,r4}(いずれも単体赤字)を廃止して複を厚くする。
その200円で3連単2点を買えば、C/D帯の的中を「複の薄利」ではなく「単の厚利」で
拾い直せるのではないか——E/F単(差され単248%/301%)と同じ発見の形。

■ 事前登録(実行前に固定)
単の並びは理論(差され構造)から選ぶ。マスを見てから選ばない:
  C帯{r1,r3,r4} = 2位が飛ぶ日。E/Fと同じ「頭=下位・2着=r1残存」形
    → C'単 = r3-r1-r4 / r4-r1-r3
  D帯{r2,r3,r4} = 1位が飛ぶ日。繰り上がり差され「頭=下位・2着=r2残存」形
    → D'単 = r3-r2-r4 / r4-r2-r3
変種(全て1,000円・この2つだけ):
  変形イ: A200 B100 E200 F200 G100 + C'(r3-r1-r4)100 + D'(r3-r2-r4)100
  変形ロ: A200 B100 E200 F200 G100 + C'(r4-r1-r3)100 + D'(r4-r2-r3)100
判定基準: 変形が案1を回収率で上回り、かつガミ率が案1+5pt以内、かつ
月次で1発依存でない場合のみ「案1より優先」として判断会に提出。
それ以外は案1のまま。
参考表示(採否に使わない): C/D各トリオの6並び別成績と、
トリオ的中時にどの並びで決まったかの分布(=単で捉えられる理論上の上限)。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
races_month = {}
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
for rid, d in conn.execute(
        "SELECT race_id, date FROM races WHERE date >= ?", (TEST_START,)):
    races_month[rid] = d[:7]
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

ctxs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if 1 not in actual[rid] or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        top3 = {actual[rid][o] for o in (1, 2, 3) if o in actual[rid]}
        if len(top3) != 3:
            continue
        ctxs.append({"rid": rid, "lanes": [r["lane"] for r in ranked],
                     "month": races_month.get(rid, "?")})

n = len(ctxs)
print(f"\n超混戦帯(1位20%未満): {n:,}レース")


def build_bets(c, variant):
    r1, r2, r3, r4, r5 = c["lanes"][:5]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"
    base = {
        "A": ("3連複", trio(r1, r2, r3)), "B": ("3連複", trio(r1, r2, r4)),
        "C": ("3連複", trio(r1, r3, r4)), "D": ("3連複", trio(r2, r3, r4)),
        "E": ("3連単", f"{r3}-{r1}-{r2}"), "F": ("3連単", f"{r4}-{r1}-{r2}"),
        "G": ("3連複", trio(r3, r4, r5)),
        "C'イ": ("3連単", f"{r3}-{r1}-{r4}"), "D'イ": ("3連単", f"{r3}-{r2}-{r4}"),
        "C'ロ": ("3連単", f"{r4}-{r1}-{r3}"), "D'ロ": ("3連単", f"{r4}-{r2}-{r3}"),
    }
    return [(base[k], yen) for k, yen in variant.items() if yen]


VARIANTS = {
    "現行Q案": {"A": 200, "B": 100, "C": 100, "D": 100, "E": 200, "F": 200, "G": 100},
    "案1拾える複厚": {"A": 300, "B": 200, "E": 200, "F": 200, "G": 100},
    "変形イ(r3頭で置換)": {"A": 200, "B": 100, "C'イ": 100, "D'イ": 100,
                          "E": 200, "F": 200, "G": 100},
    "変形ロ(r4頭で置換)": {"A": 200, "B": 100, "C'ロ": 100, "D'ロ": 100,
                          "E": 200, "F": 200, "G": 100},
}

print(f"\n--- 変種比較(レース単位・1,000円) ---")
print(f"{'変種':<16}{'回収率':>8}{'完全外れ':>9}{'ガミ':>7}{'プラス':>7}{'最大1発除き':>11}")
summary = {}
for vname, w in VARIANTS.items():
    st = rt = fm = gm = pl = 0
    monthly = defaultdict(lambda: [0, 0])
    best_hit = 0
    for c in ctxs:
        pay = payout_map[c["rid"]]
        rs = rr = 0
        for (bt, comb), yen in build_bets(c, w):
            rs += yen
            rr += pay.get((bt, comb), 0) * yen // 100
        st += rs
        rt += rr
        m = monthly[c["month"]]
        m[0] += rs
        m[1] += rr
        best_hit = max(best_hit, rr)
        if rr == 0:
            fm += 1
        elif rr < rs:
            gm += 1
        else:
            pl += 1
    ex = (rt - best_hit) / st if st else 0
    summary[vname] = {"roi": rt / st, "gami": gm / n, "ex": ex,
                      "monthly": dict(monthly)}
    print(f"{vname:<16}{rt/st:>8.1%}{fm/n:>9.1%}{gm/n:>7.1%}{pl/n:>7.1%}{ex:>11.1%}")

print(f"\n--- 月次(回収率) ---")
months = sorted({m for v in summary.values() for m in v["monthly"]})
header = "月       " + "".join(f"{v:<14}" for v in VARIANTS)
print(header)
for m in months:
    row = f"{m}  "
    for vname in VARIANTS:
        s, r = summary[vname]["monthly"].get(m, [0, 0])
        row += f"{(r/s if s else 0):>7.1%}       "
    print(row)

# --- 参考: C/Dトリオの並び別成績(100円買い・全レース) ---
print(f"\n--- 参考(採否に使わない): トリオ内の並び別成績(各100円・{n:,}R) ---")
for name, key_head in (("C帯{r1,r3,r4}", "C"), ("D帯{r2,r3,r4}", "D")):
    print(f"  ◆{name}")
    trio_hits = 0
    order_stat = {}
    for c in ctxs:
        r1, r2, r3, r4, r5 = c["lanes"][:5]
        members = (r1, r3, r4) if key_head == "C" else (r2, r3, r4)
        label = {"C": {r1: "r1", r3: "r3", r4: "r4"},
                 "D": {r2: "r2", r3: "r3", r4: "r4"}}[key_head]
        pay = payout_map[c["rid"]]
        from itertools import permutations
        for perm in permutations(members):
            comb = f"{perm[0]}-{perm[1]}-{perm[2]}"
            got = pay.get(("3連単", comb), 0)
            k = "-".join(label[x] for x in perm)
            s = order_stat.setdefault(k, [0, 0, 0])
            s[0] += 100
            s[1] += got
            if got:
                s[2] += 1
    for k, (s0, s1, hits) in sorted(order_stat.items(), key=lambda x: -x[1][1] / x[1][0]):
        avg = s1 / hits if hits else 0
        print(f"    {k:<10} 的中{hits:>3}本 回収率{s1/s0:>7.1%} 平均払戻{avg:>8,.0f}円")

ok = []
for vname in ("変形イ(r3頭で置換)", "変形ロ(r4頭で置換)"):
    v, a1 = summary[vname], summary["案1拾える複厚"]
    passed = (v["roi"] > a1["roi"] and v["gami"] <= a1["gami"] + 0.05
              and v["ex"] > a1["ex"])
    ok.append((vname, passed))
print(f"\n===== 事前登録基準の判定 =====")
for vname, passed in ok:
    print(f"  {vname}: {'案1超え・提出可' if passed else '基準未達 → 案1のまま'}")
