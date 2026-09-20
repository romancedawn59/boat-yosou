# -*- coding: utf-8 -*-
"""検証の作法を道具にしたもの(2026-09-21ケンさん承認)

買い方・場・並びを試すスクリプトは、ここの関数で次の3点を必ず確認する。
  1. 標本の下限      enough_hits(): 的中10本未満の数字では判断しない
  2. 偶然との区別    label_shuffle_pvalue(): 場や並びのように選択肢が多いものは、
                     ラベルをシャッフルしても同じ見た目になる確率を出す
  3. 事前宣言        紙上で追うものは logs/preregistered.log に宣言してから数える
                     (採点は test/track_preregistered.py が宣言日より後のレースだけで行う)
採点に使う予想は本番配信のpicks JSONだけ(test/score_production_picks.py と同じ考え方)。
"""
import math
import random

MIN_HITS = 10


def enough_hits(hits: int, min_hits: int = MIN_HITS) -> bool:
    """的中本数が判断に足りるか"""
    return hits >= min_hits


def wilson(hits: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """発生確率の幅(既定は90%区間)"""
    if n <= 0:
        return 0.0, 0.0
    p = hits / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return max(0.0, (c - w) / d), min(1.0, (c + w) / d)


def label_shuffle_pvalue(labels: list, payoffs: list[float], stake: float,
                         stat=None, n_iter: int = 2000, seed: int = 7) -> float:
    """ラベル(場など)に実力差が無いと仮定したとき、実データ以上の見た目が出る確率。

    labels[i] = i番目のレースのラベル、payoffs[i] = そのレースの払戻、stake = 1レースの投資。
    stat = {ラベル: 回収率} を受け取って1つの数字を返す関数(既定は最良ラベルの回収率)。
    """
    stat = stat or (lambda roi: max(roi.values()))

    def roi_by_label(lbls):
        tot, cnt = {}, {}
        for l, g in zip(lbls, payoffs):
            tot[l] = tot.get(l, 0.0) + g
            cnt[l] = cnt.get(l, 0) + 1
        return {l: tot[l] / (cnt[l] * stake) for l in tot}

    real = stat(roi_by_label(labels))
    rng = random.Random(seed)
    shuffled = list(labels)
    ge = 0
    for _ in range(n_iter):
        rng.shuffle(shuffled)
        ge += stat(roi_by_label(shuffled)) >= real
    return ge / n_iter
