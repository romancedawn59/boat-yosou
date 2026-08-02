"""3人の予想者(A石橋渡/B山田三連単/C勝万舟)と予想屋kenのポートフォリオ

すべてモデルの各艇勝率から派生計算する(追加の学習・データ取得は不要)。
決着確率はHarville法: P(a-b-c) = pa * pb/(1-pa) * pc/(1-pa-pb)

- A 石橋渡: 硬い予想。2連複・3連複の全組み合わせから発生確率上位5点
- B 山田三連単: 3連単の全120通りから発生確率上位10点
- C 勝万舟: 万舟圏(発生確率0.5%以下)の3連単から確率上位5点
- 予想屋ken: 3人の案を基に1レース1,000円のポートフォリオを構成。
  荒れ注意(本命帯)の6点目は保険複r2r3r4(2026-07-29判断会でC枠から置換)。
  堅め・標準はC案を1点100円で購入(C候補0点のレースはC無し・計900円)。
  2連複は購入しない(Aの2連複は判断材料。検証⑦: 3連複への置換で
  標準83.3%→85.7%・ken全体は92.5%→92.4%と同水準のため採用)。
"""
from itertools import combinations, permutations

# Benter割引: 2着・3着の条件付き確率を勝率のべき乗で減衰させる。
# P(2着=b|1着=a) = pb^λ / Σ(残りp^λ)。λ=μ=1が素のHarville法。
# 2025-12〜2026-04のウォークフォワード予測21,398レースで対数尤度を最大化して推定
# (Harville比 +2,119。「2着以降は勝率ほど順当に決まらない」を反映)。
LAMBDA2 = 0.70  # 2着の減衰
LAMBDA3 = 0.50  # 3着の減衰

# 万舟圏の判定ライン。Benter割引後の確率で再較正済み(2025-12〜2026-04、21,398レース):
# 発生確率0.005以下で決まったレースの実払戻は平均約27,000円・万舟率約70%
MANSHU_PROB_MAX = 0.005


def normalize_probs(ranked: list[dict]) -> dict[int, float]:
    """予測勝率を合計1に正規化して {枠番: 勝率} を返す"""
    total = sum(r["prob"] for r in ranked)
    if total <= 0:
        return {}
    return {r["lane"]: r["prob"] / total for r in ranked}


def trifecta_probs(probs: dict[int, float], lam: float = LAMBDA2, mu: float = LAMBDA3) -> dict[tuple, float]:
    """3連単全順列の発生確率(Benter割引つきHarville法)"""
    pow2 = {k: v ** lam for k, v in probs.items()}
    pow3 = {k: v ** mu for k, v in probs.items()}
    sum2 = sum(pow2.values())
    sum3 = sum(pow3.values())
    out = {}
    for a, b, c in permutations(probs, 3):
        d2 = sum2 - pow2[a]
        d3 = sum3 - pow3[a] - pow3[b]
        if d2 <= 0 or d3 <= 0:
            continue
        out[(a, b, c)] = probs[a] * (pow2[b] / d2) * (pow3[c] / d3)
    return out


def quinella_prob(probs: dict[int, float], a: int, b: int, lam: float = LAMBDA2) -> float:
    """2連複{a,b}の発生確率(割引つき。a-b着順とb-a着順の和)"""
    pow2 = {k: v ** lam for k, v in probs.items()}
    sum2 = sum(pow2.values())
    p = 0.0
    if sum2 - pow2[a] > 0:
        p += probs[a] * pow2[b] / (sum2 - pow2[a])
    if sum2 - pow2[b] > 0:
        p += probs[b] * pow2[a] / (sum2 - pow2[b])
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
    """B 山田三連単: 3連単の発生確率上位10点(フォーメーション買いに近い形になる)"""
    tri = trifecta_probs(probs)
    top = sorted(tri.items(), key=lambda x: -x[1])[:10]
    return [("3連単", f"{a}-{b}-{c}", p) for (a, b, c), p in top]


def trio_top(probs: dict[int, float], n: int = 2) -> list[tuple[str, float]]:
    """3連複20通りの発生確率上位n点 [(組み合わせ, 確率)]"""
    tri = trifecta_probs(probs)
    agg: dict[str, float] = {}
    for (a, b, c), p in tri.items():
        s = sorted([a, b, c])
        key = f"{s[0]}={s[1]}={s[2]}"
        agg[key] = agg.get(key, 0.0) + p
    return sorted(agg.items(), key=lambda x: -x[1])[:n]


def combo_prob(bet_type: str, combination: str, probs: dict[int, float]) -> float:
    """買い目1点の発生確率(=自信ポイント)。3連複は同じ組の順列を合算する。

    このシステムは朝買いで締切前のオッズを見ない(見ると市場に引きずられ、
    検証⑥でEVフィルタ・市場ブレンドとも素通しより悪化した)。そのため
    「この目はいくらつくか」はオッズではなくこの確率から逆算する。
    較正はtest/verify_slot_performance.pyで確認済み(自信0.48%→実際0.49%、
    1.54%→1.43%、14.20%→14.05%)なので、確率はそのまま信頼してよい。
    """
    tri = trifecta_probs(probs)
    try:
        if bet_type == "3連単":
            a, b, c = (int(x) for x in combination.split("-"))
            return tri.get((a, b, c), 0.0)
        if bet_type == "3連複":
            s = {int(x) for x in combination.split("=")}
            return sum(p for k, p in tri.items() if set(k) == s)
        if bet_type == "2連複":
            a, b = (int(x) for x in combination.split("="))
            return quinella_prob(probs, a, b)
    except (ValueError, KeyError):
        return 0.0
    return 0.0


def implied_odds(prob: float, takeout: float = 0.75) -> float:
    """自信ポイントから想定配当(倍)を逆算する。オッズを見ない設計の代替指標。
    takeout=0.75は3連単・3連複の払戻率(控除率25%)。確率0なら0を返す"""
    return takeout / prob if prob > 0 else 0.0


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
    b_picks: list[tuple[str, str, float]],
    c_picks: list[tuple[str, str, float]],
    konsen: bool = False,
) -> list[tuple[str, str, int, str]]:
    """予想屋ken: 1レース1,000円のポートフォリオ。(券種, 組み合わせ, 金額, 出典)のリスト。

    - 荒れ注意(本命帯)の6点目は保険複r2r3r4(2026-07-29判断会・議題B採用)。
      旧C枠(勝万舟100円)は唯一の赤字スロット(74.8%)で、置換で90.5%→95.6%
      (test/verify_honmei_c_orthodox_off.py)。意味は「オーソドックス1着(モデル1位)が
      消えた世界」の順不同保険——ケンさんの展開理論の実装。単指定(r2-r3-r4)は
      沈没日の並びが混沌のため複に劣る(92.5%)。Z1-2(沈没予測)完成後は
      P(沈没)ゲート付き発動に進化させる構想(2026-09)
    - 堅め・標準はC勝万舟の案を1点100円で購入(C候補が0点のレースはC無し・計900円)
    - 堅め・標準は3連複上位を厚く、B(3連単)を添える。2連複は購入しない
      (Aの2連複は判断材料。検証⑦: 3連複置換で標準83.3%→85.7%)
    - konsen=True(超混戦帯)は⑬「BOX+差され傾斜」(2026-08-01ケンさん決定・2,000円)。
      A/Bトリオ(1-2-3位・1-2-4位)の3連単BOX各100円で面を張り、実測最高値マスの
      E/F差され単(3位-1位-2位・4位-1位-2位)だけ+300円の傾斜、G複(3=4=5位)200円の
      全滅保険。月次8か月190.1%・+899,560円・最低月122.4%(test/verify_konsen_box_plus600.py)。
      経緯: Q案(7/21)→案1「拾える複厚」(7/29判断会)→2,000円増資(7/31)→⑬(8/1)。
      旧案・対抗案(案1×2/家族拡張/BOX素)は紙上で継続計測し9/1に再判定。
      棄却済み: C/D複・軸外し・D家族(複/単/BOXの三形態全て)・A-BOX倍厚・
      5位頭差され単・少額広範囲(5点外の海は63.7%)
    """
    lanes = [r["lane"] for r in ranked]
    # c_picksが空でも検証済みプランは返す。確率が平坦なレース(例: 1位20%)では
    # 3連単全120通りが万舟圏の閾値MANSHU_PROB_MAXを超えてC候補が0点になりうる。
    # 以前はここでc_picksも必須にしていたため、最も荒れたレースほどプランごと
    # 空になり勝負所から静かに脱落するバグがあった
    if len(lanes) < 4:
        return []
    r1, r2, r3, r4 = lanes[:4]
    r5 = lanes[4] if len(lanes) >= 5 else None

    def trio(a, b, c):
        s = sorted([a, b, c])
        return f"{s[0]}={s[1]}={s[2]}"

    if confidence == "荒れ注意" and konsen and r5 is not None:
        # ⑬「BOX+差され傾斜」(2026-08-01ケンさん決定・計2,000円)。
        # A/Bトリオを単BOXで面張り(各100円)+最高値マスのE/F差され単に+300円の傾斜
        # +G複(全滅保険)。月次8か月190.1%・最低月122.4%で案1×2を全指標で上回った
        # (test/verify_konsen_box_plus600.py)。旧4案は紙上で継続計測し9/1に再判定。
        # 購入は5操作: BOX2回+E/F追加2点+G複(金額編集なし)。
        # 表記も購入操作と同じ形にする(2026-08-02ケンさん要望):
        # BOXは全並び各100円で並べ、E/Fの差され傾斜は「+300円」の追加行で表す
        # (同一買い目が2行になるが採点は行ごとに合算されるため等価)
        plan = []
        for members in ((r1, r2, r3), (r1, r2, r4)):
            for a, b, c in permutations(members):
                plan.append(("3連単", f"{a}-{b}-{c}", 100, "BOX"))
        plan.append(("3連単", f"{r3}-{r1}-{r2}", 300, "差され追加"))
        plan.append(("3連単", f"{r4}-{r1}-{r2}", 300, "差され追加"))
        plan.append(("3連複", trio(r3, r4, r5), 200, "深い波乱"))
        return plan

    if confidence == "荒れ注意":
        # 検証済み構成を核に、6点目は保険複(2026-07-29判断会・議題B)。
        # 旧: C勝万舟100円 → 新: r2r3r4複100円(モデル1位が飛ぶ日の順不同保険)
        return [
            ("3連複", trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", trio(r1, r2, r4), 200, "検証済み"),
            ("3連複", trio(r1, r3, r4), 100, "検証済み"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
            ("3連複", trio(r2, r3, r4), 100, "保険複"),
        ]
    else:
        probs = normalize_probs(ranked)
        trios = trio_top(probs, 2)
        if confidence == "堅め":
            plan = [
                ("3連複", trios[0][0], 400, "本線"),
                ("3連複", trios[1][0], 300, "本線"),
                (b_picks[0][0], b_picks[0][1], 200, "山田"),
            ]
        else:  # 標準
            plan = [
                ("3連複", trios[0][0], 300, "本線"),
                ("3連複", trios[1][0], 200, "本線"),
                (b_picks[0][0], b_picks[0][1], 200, "山田"),
                (b_picks[1][0], b_picks[1][1], 200, "山田"),
            ]

    # (堅め・標準のみ)C勝万舟から、既にプランにある組み合わせと重複しない
    # 最初の1点を100円で追加。C候補が0点ならこのループはスキップされ、
    # 本線構成のみ(計900円)を返す
    existing = {(bt, comb) for bt, comb, _, _ in plan}
    for bt, comb, _p in c_picks:
        if (bt, comb) not in existing:
            plan.append((bt, comb, 100, "勝万舟"))
            break
    else:
        return plan  # C全点が重複(理論上ほぼ起きない)

    return plan


def select_shobusho(races: list[dict], honmei_venues: list[int],
                    honmei_cap: int = 6, konsen_max: float = 0.20,
                    attention_cap: int = 4, honmei_prob_max: float = 0.30) -> None:
    """v2選別(ケンさん案・2026-07-18): 各レースに shobusho キーを設定する。

    - 超混戦: 全場で1位勝率(モデル生値)がkonsen_max未満。市場も予測できない本物の
      混戦=エッジの本体(walk-forward 387%/最大1発除き312%。検証はtest/verify_ken_v2*.py)
    - 本命: honmei_venues(検証済み5場)の荒れ注意のうち1位勝率がhonmei_prob_max未満から
      低い順にhonmei_cap件。閾値は2026-07-20に35%→30%(28〜35%帯は利益貢献ゼロの詰め物。
      利益維持のまま投資-25%・DD-30%。検証はtest/verify_honmei_threshold.py)。
      cap6はcap10より回収率・ドローダウンとも良い(薄い帯の尻尾が削れるため)
    - 要注目: 観測専用・購入なし。閾値で外れた30〜35%帯・本命から溢れた対象場の荒れ注意
      +対象場の標準(1位勝率が低い順)で計attention_cap件。「注目に値したか(中波乱・万舟で
      決着)/標準だったか」を採点で記録し、荒れ判定境界の教師データにする。30〜35%帯の
      観測は閾値30%採用日を起点にした前向き検証を兼ねる(8月末に答え合わせ)
    購入対象は「本命+超混戦」のみ。対象場のレースが両条件を満たす場合は本命と表示する
    (購入は1回。和集合の意味は変わらない)。
    """
    for r in races:
        r["shobusho"] = None

    # 超混戦(全場)。プランが組めるレースのみ
    for r in races:
        if r["ranked"][0]["prob"] < konsen_max and r["bets"]["plan"]:
            r["shobusho"] = "超混戦"

    # 本命(対象場の荒れ注意×閾値未満・1位勝率が低い順にcap件)。
    # 超混戦と重複したら本命表示を優先
    are = sorted(
        (r for r in races
         if r["venue_code"] in honmei_venues
         and r["bets"]["confidence"] == "荒れ注意" and r["bets"]["plan"]),
        key=lambda r: r["ranked"][0]["prob"],
    )
    honmei_pool = [r for r in are if r["ranked"][0]["prob"] < honmei_prob_max]
    for r in honmei_pool[:honmei_cap]:
        r["shobusho"] = "本命"

    # 要注目(観測専用): 買わない超混戦(プラン不成立等)は購入0点として必ず載せ、
    # 続いて本命に入らなかった対象場の荒れ注意(閾値超の30〜35%帯・capからの溢れ)
    # → 足りなければ標準から補充
    konsen_unbought = [r for r in races
                       if r["ranked"][0]["prob"] < konsen_max
                       and r["shobusho"] is None]
    for r in konsen_unbought:
        r["shobusho"] = "要注目"
    attention = [r for r in are if r["shobusho"] is None]
    if len(attention) < attention_cap:
        standards = sorted(
            (r for r in races
             if r["venue_code"] in honmei_venues
             and r["bets"]["confidence"] == "標準" and r["bets"]["plan"]),
            key=lambda r: r["ranked"][0]["prob"],
        )
        attention += standards[:attention_cap - len(attention)]
    for r in attention[:attention_cap]:
        r["shobusho"] = "要注目"
