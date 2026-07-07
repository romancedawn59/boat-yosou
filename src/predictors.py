"""3人の予想者(A石橋渡/B山田三連単/C勝万舟)と予想屋kenのポートフォリオ

すべてモデルの各艇勝率から派生計算する(追加の学習・データ取得は不要)。
決着確率はHarville法: P(a-b-c) = pa * pb/(1-pa) * pc/(1-pa-pb)

- A 石橋渡: 硬い予想。2連複・3連複の全組み合わせから発生確率上位5点
- B 山田三連単: 3連単の全120通りから発生確率上位5点
- C 勝万舟: 万舟圏(発生確率0.5%以下)の3連単から確率上位5点
- 予想屋ken: 3人の案を基に1レース1,000円のポートフォリオを構成。
  C案を必ず1点以上100円で購入、他は100〜400円単位。
"""
from itertools import combinations, permutations

# 万舟圏の判定ライン(較正済み: この確率以下の決着は実績平均払戻22,442円・万舟率59%)
MANSHU_PROB_MAX = 0.005


def normalize_probs(ranked: list[dict]) -> dict[int, float]:
    """予測勝率を合計1に正規化して {枠番: 勝率} を返す"""
    total = sum(r["prob"] for r in ranked)
    if total <= 0:
        return {}
    return {r["lane"]: r["prob"] / total for r in ranked}


def trifecta_probs(probs: dict[int, float]) -> dict[tuple, float]:
    """3連単全順列の発生確率"""
    out = {}
    for a, b, c in permutations(probs, 3):
        d1 = 1 - probs[a]
        d2 = 1 - probs[a] - probs[b]
        if d1 <= 0 or d2 <= 0:
            continue
        out[(a, b, c)] = probs[a] * (probs[b] / d1) * (probs[c] / d2)
    return out


def quinella_prob(probs: dict[int, float], a: int, b: int) -> float:
    """2連複{a,b}の発生確率(a-b着順とb-a着順の和)"""
    pa, pb = probs[a], probs[b]
    p = 0.0
    if 1 - pa > 0:
        p += pa * pb / (1 - pa)
    if 1 - pb > 0:
        p += pb * pa / (1 - pb)
    return p


def picks_ishibashi(probs: dict[int, float]) -> list[tuple[str, str, float]]:
    """A 石橋渡: 2連複15通り+3連複20通りから発生確率上位5点"""
    tri = trifecta_probs(probs)
    cands = []
    for a, b in combinations(sorted(probs), 2):
        cands.append(("2連複", f"{a}={b}", quinella_prob(probs, a, b)))
    for trio_set in combinations(sorted(probs), 3):
        p = sum(v for k, v in tri.items() if set(k) == set(trio_set))
        cands.append(("3連複", "=".join(map(str, trio_set)), p))
    return sorted(cands, key=lambda x: -x[2])[:5]


def picks_yamada(probs: dict[int, float]) -> list[tuple[str, str, float]]:
    """B 山田三連単: 3連単の発生確率上位5点"""
    tri = trifecta_probs(probs)
    top = sorted(tri.items(), key=lambda x: -x[1])[:5]
    return [("3連単", f"{a}-{b}-{c}", p) for (a, b, c), p in top]


def picks_katsu(probs: dict[int, float]) -> list[tuple[str, str, float]]:
    """C 勝万舟: 万舟圏(発生確率0.5%以下)の3連単から確率上位5点"""
    tri = trifecta_probs(probs)
    cands = sorted(
        ((k, p) for k, p in tri.items() if p <= MANSHU_PROB_MAX),
        key=lambda x: -x[1],
    )[:5]
    return [("3連単", f"{a}-{b}-{c}", p) for (a, b, c), p in cands]


def bucket_of(top_prob: float) -> str:
    if top_prob >= 0.50:
        return "堅め"
    if top_prob >= 0.35:
        return "標準"
    return "荒れ注意"


def ken_portfolio(
    confidence: str,
    ranked: list[dict],
    a_picks: list[tuple[str, str, float]],
    b_picks: list[tuple[str, str, float]],
    c_picks: list[tuple[str, str, float]],
) -> list[tuple[str, str, int, str]]:
    """予想屋ken: 1レース1,000円のポートフォリオ。(券種, 組み合わせ, 金額, 出典)のリスト。

    - C勝万舟の案を必ず1点以上・100円で購入
    - 荒れ注意はウォークフォワード検証済みの構成(3連複軸1流し+3連単穴)を核に維持する
    - 堅め・標準はA(堅い)を厚く、B(3連単)を添える
    """
    lanes = [r["lane"] for r in ranked]
    if len(lanes) < 4 or not c_picks:
        return []
    r1, r2, r3, r4 = lanes[:4]

    def trio(a, b, c):
        s = sorted([a, b, c])
        return f"{s[0]}={s[1]}={s[2]}"

    if confidence == "荒れ注意":
        # 検証済み構成を核に、3点目の3連複を100円に減らしてC枠を捻出(V2案)。
        # 5-6月の同一条件比較: 現行106.1% → V2 113.5%(エッジは削れない)
        plan = [
            ("3連複", trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", trio(r1, r2, r4), 200, "検証済み"),
            ("3連複", trio(r1, r3, r4), 100, "検証済み"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
        ]
    elif confidence == "堅め":
        plan = [
            (a_picks[0][0], a_picks[0][1], 400, "石橋"),
            (a_picks[1][0], a_picks[1][1], 300, "石橋"),
            (b_picks[0][0], b_picks[0][1], 200, "山田"),
        ]
    else:  # 標準
        plan = [
            (a_picks[0][0], a_picks[0][1], 300, "石橋"),
            (a_picks[1][0], a_picks[1][1], 200, "石橋"),
            (b_picks[0][0], b_picks[0][1], 200, "山田"),
            (b_picks[1][0], b_picks[1][1], 200, "山田"),
        ]

    # C勝万舟から、既にプランにある組み合わせと重複しない最初の1点を100円で追加
    existing = {(bt, comb) for bt, comb, _, _ in plan}
    for bt, comb, _p in c_picks:
        if (bt, comb) not in existing:
            plan.append((bt, comb, 100, "勝万舟"))
            break
    else:
        return plan  # C全点が重複(理論上ほぼ起きない)

    return plan


def select_shobusho(races: list[dict], max_races: int = 10) -> None:
    """勝負所を選定し、各レースに shobusho キー(None/'本命'/'準')を設定する。

    - 荒れ注意レースは全て「本命」(検証済みエッジ)。10を超える場合は1位勝率が低い順に10まで
    - 枠が余れば、標準レースから1位勝率が低い順(波乱含みの順)に「準」で補充
    """
    for r in races:
        r["shobusho"] = None

    are = sorted(
        (r for r in races if r["bets"]["confidence"] == "荒れ注意" and r["bets"]["plan"]),
        key=lambda r: r["ranked"][0]["prob"],
    )
    for r in are[:max_races]:
        r["shobusho"] = "本命"

    remaining = max_races - min(len(are), max_races)
    if remaining > 0:
        standards = sorted(
            (r for r in races if r["bets"]["confidence"] == "標準" and r["bets"]["plan"]),
            key=lambda r: r["ranked"][0]["prob"],
        )
        for r in standards[:remaining]:
            r["shobusho"] = "準"
