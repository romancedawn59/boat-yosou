# -*- coding: utf-8 -*-
"""月次の再学習で予想順位がどれだけ動くかを測る(2026-09-21ケンさん承認・土台づくり)

    py -X utf8 test/measure_rank_stability.py

同じレース(2026-09-01〜)を、本番で実際に使った2つのモデルで予測して比べる。
  旧 = 2026-08-01再学習のモデル(git 6a4f507・v2.1の特徴量)
  新 = 2026-09-01再学習のモデル(git ced222a・v2.2の特徴量)
さらに特徴量を揃えた比較として、現行特徴量で学習期間だけ変えた2モデル
(〜2026-07-31 / 〜2026-08-31)も比べる。
見る数字: 予想1位の一致、予想1-2-3位の並びの一致、1位勝率22.5〜30%帯の出入り。
"""
import subprocess
import sys
import tempfile

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

REPO = r"Y:\マイドライブ\boat"
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}


def model_from_git(commit):
    txt = subprocess.run(["git", "-C", REPO, "show", f"{commit}:models/lgbm_win.txt"],
                         capture_output=True, check=True).stdout
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    f.write(txt)
    f.close()
    return lgb.Booster(model_file=f.name)


def train_until(train_all, end):
    tr = train_all[train_all["date"] < end].sort_values("date")
    cutoff = tr["date"].iloc[int(len(tr) * 0.9)]
    a, b = tr[tr["date"] < cutoff], tr[tr["date"] >= cutoff]
    ds = lgb.Dataset(a[FEATURE_COLUMNS], label=a["is_winner"],
                     categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(PARAMS, ds, num_boost_round=500,
                     valid_sets=[lgb.Dataset(b[FEATURE_COLUMNS], label=b["is_winner"],
                                             reference=ds)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
ev = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}
    FROM entries e JOIN races r ON r.race_id = e.race_id
    WHERE r.date >= '2026-09-01'
""", conn)
ev = _encode(ev)
ev = ev.merge(compute_form_features(conn), on=["race_id", "lane"], how="left")
ev = _attach_extra_features(ev, conn)
conn.close()
full = ev.groupby("race_id")["lane"].transform("count") == 6
ev = ev[full].copy()


def order_of(model, cols):
    e = ev.copy()
    e["p"] = model.predict(e[cols])
    out = {}
    for rid, g in e.groupby("race_id"):
        gs = g.sort_values("p", ascending=False)
        out[rid] = ([int(x) for x in gs["lane"]], float(gs["p"].iloc[0]))
    return out


def compare(title, A, B):
    rids = sorted(set(A) & set(B))
    n = len(rids)
    same1 = sum(A[r][0][0] == B[r][0][0] for r in rids)
    same3 = sum(A[r][0][:3] == B[r][0][:3] for r in rids)
    set3 = sum(set(A[r][0][:3]) == set(B[r][0][:3]) for r in rids)
    inA = {r for r in rids if 0.225 <= A[r][1] < 0.30}
    inB = {r for r in rids if 0.225 <= B[r][1] < 0.30}
    both = inA & inB
    dp = sum(abs(A[r][1] - B[r][1]) for r in rids) / n
    print(f"\n===== {title}: {n:,}R =====")
    print(f"  予想1位が同じ {same1 / n:.1%} / 予想1-2-3位の並びが同じ {same3 / n:.1%} / "
          f"上位3艇の顔ぶれが同じ {set3 / n:.1%} / 1位勝率の差(平均) {dp * 100:.1f}pt")
    print(f"  22.5〜30%帯: 片方{len(inA)}R・もう片方{len(inB)}R・両方{len(both)}R "
          f"(重なり {len(both) / max(1, len(inA | inB)):.1%})")
    if both:
        print(f"  両方が帯に入れたレースで 予想1位が同じ {sum(A[r][0][0] == B[r][0][0] for r in both) / len(both):.1%}"
              f" / 並びが同じ {sum(A[r][0][:3] == B[r][0][:3] for r in both) / len(both):.1%}")


old, new = model_from_git("6a4f507"), model_from_git("ced222a")
compare("本番モデル 8/1版 vs 9/1版(特徴量も v2.1→v2.2 に変更)",
        order_of(old, old.feature_name()), order_of(new, new.feature_name()))
print("\n学習期間だけ変えた2モデルを学習中...", flush=True)
m_a, m_b = train_until(train_all, "2026-08-01"), train_until(train_all, "2026-09-01")
compare("現行特徴量で学習期間だけ1か月違い(〜7/31 vs 〜8/31)",
        order_of(m_a, FEATURE_COLUMNS), order_of(m_b, FEATURE_COLUMNS))
