# -*- coding: utf-8 -*-
"""場ごとの相性(2026-10-25ケンさん指示「持っているデータをすべて用いて」)

    py -X utf8 test/venue_affinity_1025.py

材料: 1年分のwalk-forward台帳(2025-10〜2026-09・52,693R)+本番配信(2026-10-01〜10-25)、全帯。
相性の物差し(場ごと):
  A 予想精度   : 予想1位の1着率 − モデルが出した1位勝率(較正のずれ)、ドンピシャ率
  B 市場との差 : 予想1位のlogloss(モデル − 市場の暗黙確率)。負なら市場より正確。
                 市場=3連単オッズ(確定を優先・無ければ締切15分前)の1/オッズをレース内正規化して艇ごとに合算
  C 回収率     : ドンピシャ1点500円(全帯 / 22.5〜30%帯)、3期間の黒字数
  D 偶然との区別: 場ラベルをシャッフルして「最良の場の回収率」「場間のばらつき」が偶然で出る確率
"""
import glob, json, math, os, random, sqlite3
from collections import defaultdict
import pandas as pd

V={1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",9:"津",10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}
YEN=500
c=sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro",uri=True,timeout=120)
pay={r:(cb,a or 0) for r,cb,a in c.execute("select race_id, combination, amount_yen from payouts where bet_type='3連単'")}
win={r:l for r,l in c.execute("select race_id, lane from results where arrival_order=1")}
odds={}
for tbl,cond in (("odds","fetched_at!='final-backfill'"),("odds_final","1=1")):
    tmp=defaultdict(dict)
    for r,cb,o in c.execute(f"select race_id, combination, odds from {tbl} where bet_type='3連単' and odds>0 and {cond}"): tmp[r][cb]=o
    for r,d in tmp.items():
        if len(d)>=110: odds[r]=d
c.close()
rows=[]  # (venue, period, p1, pred list, rid)
L=pd.read_csv(r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv",encoding="utf-8-sig",dtype={"refund":str})
for rid,m,v,p1,pred in zip(L["race_id"],L["month"],L["venue"],L["p1"],L["pred"]):
    per=0 if m<="2026-01" else 1 if m<="2026-05" else 2
    rows.append((int(v),per,float(p1),[int(x) for x in pred.split("-")],rid))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-10-*.json")):
    for r in json.load(open(path,encoding="utf-8"))["races"]:
        if r.get("ranked") and len(r["ranked"])>=6: rows.append((r["venue_code"],2,r["ranked"][0][1],[x[0] for x in r["ranked"]],r["race_id"]))
rows=[x for x in rows if x[4] in pay and x[4] in win]
print(f"対象 {len(rows):,}R(全24場・全帯・2025-10〜2026-10-25) / 市場比較に使えるレース {sum(1 for x in rows if x[4] in odds):,}R")

S=defaultdict(lambda: defaultdict(float))
for v,per,p1,pred,rid in rows:
    s=S[v]; cb,amt=pay[rid]; hit=cb=="-".join(map(str,pred[:3])); w=win[rid]==pred[0]
    s["n"]+=1; s["p1"]+=p1; s["w"]+=w; s["don"]+=hit; s["ret"]+=amt*YEN//100 if hit else 0; s[f"ret{per}"]+=amt*YEN//100 if hit else 0; s[f"n{per}"]+=1
    if .225<=p1<.30: s["bn"]+=1; s["bret"]+=amt*YEN//100 if hit else 0; s["bdon"]+=hit
    if rid in odds:
        o=odds[rid]; tot=sum(1/x for x in o.values()); q=sum(1/x for cb2,x in o.items() if cb2.startswith(f"{pred[0]}-"))/tot
        pm=min(max(p1,1e-4),1-1e-4); q=min(max(q,1e-4),1-1e-4)
        s["mn"]+=1; s["ll_m"]+=-(math.log(pm) if w else math.log(1-pm)); s["ll_q"]+=-(math.log(q) if w else math.log(1-q)); s["q"]+=q
print(f"\n{'場':<5}{'R数':>6}{'1位勝率(出した)':>12}{'実際の1着率':>9}{'ずれ':>7}{'ドンピシャ率':>8}{'回収率(全帯)':>9}{'3期間黒字':>7} | {'帯R':>5}{'帯回収率':>8} | {'市場比較R':>7}{'市場の1位確率':>10}{'logloss差(モデル−市場)':>16}")
out=[]
for v,s in S.items():
    n=s["n"]; d=(s["ll_m"]-s["ll_q"])/s["mn"] if s["mn"] else float("nan")
    out.append((d,v,s))
for d,v,s in sorted(out,key=lambda x:(math.isnan(x[0]),x[0])):
    n=s["n"]; ok=sum(1 for k in range(3) if s[f"n{k}"] and s[f"ret{k}"]>s[f"n{k}"]*YEN)
    print(f"{V[v]:<5}{int(n):>6,}{s['p1']/n:>12.1%}{s['w']/n:>9.1%}{(s['w']-s['p1'])/n*100:>+6.1f}pt{s['don']/n:>8.1%}{s['ret']/(n*YEN):>9.1%}{ok:>6}/3 | {int(s['bn']):>5}{(s['bret']/(s['bn']*YEN) if s['bn'] else 0):>8.0%} | {int(s['mn']):>7,}{(s['q']/s['mn'] if s['mn'] else 0):>10.1%}{d:>+16.4f}")
tot=defaultdict(float)
for s in S.values():
    for k,x in s.items(): tot[k]+=x
n=tot["n"]; print(f"{'全場':<5}{int(n):>6,}{tot['p1']/n:>12.1%}{tot['w']/n:>9.1%}{(tot['w']-tot['p1'])/n*100:>+6.1f}pt{tot['don']/n:>8.1%}{tot['ret']/(n*YEN):>9.1%}{'':>9} | {int(tot['bn']):>5}{tot['bret']/(tot['bn']*YEN):>8.0%} | {int(tot['mn']):>7,}{tot['q']/tot['mn']:>10.1%}{(tot['ll_m']-tot['ll_q'])/tot['mn']:>+16.4f}")

# D シャッフル検定
def venue_stats(lbls):
    ret=defaultdict(float); cnt=defaultdict(int)
    for lb,(v,per,p1,pred,rid) in zip(lbls,rows):
        cb,amt=pay[rid]; ret[lb]+=amt*YEN//100 if cb=="-".join(map(str,pred[:3])) else 0; cnt[lb]+=1
    r=[ret[k]/(cnt[k]*YEN) for k in cnt]; return max(r), (max(r)-min(r))
labels=[x[0] for x in rows]; real_best,real_spread=venue_stats(labels)
rng=random.Random(7); ge_b=ge_s=0; N=300
for _ in range(N):
    rng.shuffle(labels); b,sp=venue_stats(labels); ge_b+=b>=real_best; ge_s+=sp>=real_spread
print(f"\nD 偶然との区別(場ラベルを{N}回シャッフル): 最良の場の全帯回収率{real_best:.1%}以上が出る確率 {ge_b/N:.1%} / 場間のばらつき(最良−最悪 {real_spread*100:.0f}pt)以上が出る確率 {ge_s/N:.1%}")
