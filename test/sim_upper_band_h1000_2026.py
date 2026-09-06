# -*- coding: utf-8 -*-
"""上の帯(1位勝率31〜45%)にH1000を当てたら(2026-09-06ケンさん質問)

    py -X utf8 test/sim_upper_band_h1000_2026.py

月次walk-forward・全艇ランク(クリーン)・返還処理あり。2026-01〜08。
帯: 0.31 <= p1 < 0.45。構成: H静的スリム1000円(複r1r2r3 200/複r2r3r4 100/
単r1-r2-r3 100/単r3-r1-r2 200/単r4-r1-r2 200/単r4-r2-r1 200)。
集計: 全24場 / 5場 / 帯内の細分(31-35, 35-40, 40-45) /
      5場×低い順×1日4Rまで(現行の選び方を上の帯に適用)。参考に現行9行も併記。
"""
import sys
from collections import defaultdict
import pandas as pd
sys.path.insert(0, r"Y:\マイドライブ\boat\src")
import db
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import (FEATURE_COLUMNS, _ENTRY_COLS, _attach_extra_features, _encode,
                      build_training_set, compute_form_features)
EVAL_MONTHS = ["2026-01","2026-02","2026-03","2026-04","2026-05","2026-06","2026-07","2026-08"]
LO, HI = 0.31, 0.45
print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-01-01'""", conn)
eval_df = _encode(eval_df).merge(compute_form_features(conn), on=["race_id","lane"], how="left")
eval_df = _attach_extra_features(eval_df, conn)
paymap = defaultdict(dict)
for rid, bt, comb, amt in conn.execute("SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01'"):
    paymap[rid][(bt, comb)] = amt or 0
conn.close()
def trio(a,b,c):
    s = sorted([a,b,c]); return f"{s[0]}={s[1]}={s[2]}"
def h1000(l):
    r1,r2,r3,r4 = l[:4]
    return [("3連複",trio(r1,r2,r3),200),("3連複",trio(r2,r3,r4),100),("3連単",f"{r1}-{r2}-{r3}",100),
            ("3連単",f"{r3}-{r1}-{r2}",200),("3連単",f"{r4}-{r1}-{r2}",200),("3連単",f"{r4}-{r2}-{r1}",200)]
def nine(l):
    r1,r2,r3,r4 = l[:4]
    return [("3連複",trio(r1,r2,r3),200),("3連複",trio(r1,r2,r4),200),("3連複",trio(r1,r3,r4),100),
            ("3連単",f"{r3}-{r1}-{r2}",200),("3連単",f"{r4}-{r1}-{r2}",200),("3連複",trio(r2,r3,r4),100),
            ("3連単",f"{r3}-{r2}-{r1}",100),("3連単",f"{r4}-{r2}-{r1}",300)]
def score(bets, pay, refund):
    m = defaultdict(int)
    for bt,comb,y in bets: m[(bt,comb)] += y
    st = rt = 0
    for (bt,comb),y in m.items():
        sep = "-" if bt=="3連単" else "="
        mem = {int(x) for x in comb.split(sep)}
        st += y; rt += y if mem & refund else pay.get((bt,comb),0)*y//100
    return st, rt
agg = defaultdict(lambda: [0,0,0,0])   # {(セル,月): [st,rt,n,hit1]}
def add(cell, m, st, rt, w):
    a = agg[(cell,m)]; a[0]+=st; a[1]+=rt; a[2]+=1; a[3]+=w
for m in EVAL_MONTHS:
    tr = train_all[train_all["date"] < f"{m}-01"]; ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty: continue
    print(f"{m}: 学習{len(tr):,}行", flush=True)
    b = train_fold(tr); md = ev.copy(); md["pred"] = b.predict(md[FEATURE_COLUMNS])
    daily = defaultdict(list)
    for rid, grp in md.groupby("race_id"):
        pay = paymap[rid]
        if not pay: continue
        gs = grp.sort_values("pred", ascending=False); lanes = [int(x) for x in gs["lane"]]; p1 = float(gs["pred"].iloc[0])
        if not (LO <= p1 < HI) or len(lanes) < 4: continue
        arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows() if pd.notna(r["arrival_order"])}
        if len(arr) < 3: continue
        nonfin = {int(r["lane"]) for _, r in grp.iterrows() if pd.isna(r["arrival_order"])}
        refund = {l for l in nonfin if pd.isna(grp[grp["lane"]==l]["st_time"].iloc[0])}
        w = arr.get(lanes[0]) == 1
        st, rt = score(h1000(lanes), pay, refund); st9, rt9 = score(nine(lanes), pay, refund)
        add("全24場・H1000", m, st, rt, w); add("全24場・9行", m, st9, rt9, w)
        sub = "31-35%" if p1 < 0.35 else "35-40%" if p1 < 0.40 else "40-45%"
        add(f"全24場・H1000・{sub}", m, st, rt, w)
        if int(gs["venue_code"].iloc[0]) in TARGET_VENUE_CODES:
            add("5場・H1000", m, st, rt, w)
            daily[gs["date"].iloc[0]].append((p1, lanes, pay, refund, w))
    for d, rs in daily.items():
        for p1, lanes, pay, refund, w in sorted(rs, key=lambda x: x[0])[:4]:
            st, rt = score(h1000(lanes), pay, refund); add("5場×低い順×cap4・H1000", m, st, rt, w)
print("\n===== 1位勝率31〜45%帯にH1000(2026-01〜08) =====")
cells = ["全24場・H1000","全24場・9行","全24場・H1000・31-35%","全24場・H1000・35-40%","全24場・H1000・40-45%","5場・H1000","5場×低い順×cap4・H1000"]
for c in cells:
    tot = [0,0,0,0]; line = []; ok = 0
    for m in EVAL_MONTHS:
        a = agg[(c,m)]
        for i in range(4): tot[i] += a[i]
        if a[0]: line.append(f"{m[5:]}月{a[1]/a[0]:>4.0%}"); ok += a[1] > a[0]
    st, rt, n, h = tot
    if not st: continue
    print(f"\n{c}: {n}R 1位的中{h/n:.1%} 回収率{rt/st:.1%} 損益{rt-st:+,}円 100%超の月{ok}/8")
    print("  " + " ".join(line))
