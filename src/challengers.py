"""検証⑪: 選別チャンピオン/チャレンジャーと構成チャレンジャーの共通ロジック(紙上専用)

本番の予想・購入・勝負所判定・採点・表示には一切使わない。
test/verify_challengers.py(選別の比較) / test/verify_compositions.py(構成の比較) /
test/analyze_market.py(日次の紙上採点・監視) から参照される測定用モジュール。

用語:
- チャンピオン: 現行の選別「1位勝率(モデル生値)35%未満=荒れ注意」
- 挑戦者①市場相違型 / ②混戦度型(β1差・β2エントロピー) / ③C条件型(スタブ)
- 構成4案: 現行 / 攻撃型 / 守備型 / 軸分散型(選別はチャンピオン固定で比較する)

方針: 買い目構成は全選別器で predictors.ken_portfolio の荒れ注意構成(検証済みV2)に
統一し、1日上限10レースも共通にする。比較の変数を「選別」だけに絞るため、
挑戦者が標準/堅めバケットのレースを選んでも構成は変えない。
"""
from math import log

import predictors as P

# 挑戦者①(市場相違型)の暫定閾値。スピアマン順位距離がこの値以上なら選別。
# analyze_market.py(d)の「大ズレ(7+)」区分に合わせた仮置き。
# TODO(2026-08末): (d)の層別成績が貯まったら、その結果でこの値を確定させる
MARKET_DIVERGENCE_THRESHOLD = 7

# 万舟の定義: 決着した3連単の払戻(100円あたり)がこの額以上
MANSHU_PAYOUT_MIN = 10_000

# 1日に選別してよい上限レース数(現行select_shobushoのmax_racesと同じ)
DAILY_CAP = 10


# ===== 選別指標 =====

def top_gap(probs: dict[int, float]) -> float:
    """正規化確率の1位-2位差。小さいほど混戦(挑戦者②β1)"""
    ps = sorted(probs.values(), reverse=True)
    return ps[0] - ps[1] if len(ps) >= 2 else 1.0


def entropy(probs: dict[int, float]) -> float:
    """6艇正規化確率のシャノンエントロピー(自然対数)。大きいほど混戦(挑戦者②β2)。
    6艇均等で最大 ln(6)≈1.792"""
    return -sum(p * log(p) for p in probs.values() if p > 0)


def market_divergence(model_order: list[int], market_order: list[int]) -> int:
    """モデル1着確率順位と人気順位の相違度(挑戦者①)。

    スピアマンのフットルール距離(順位差の絶対値の合計、L1)を採用する。
    L2(順位差の二乗和)より外れ1艇の影響が穏やかで、整数のため閾値が読みやすい。
    完全一致=0、6艇完全逆順=18。
    """
    market_rank = {lane: i for i, lane in enumerate(market_order)}
    return sum(abs(i - market_rank[lane])
               for i, lane in enumerate(model_order) if lane in market_rank)


def calibrate_threshold(values: list[float], target: int, mode: str) -> float:
    """挑戦者②の閾値較正: 選別数がチャンピオンの選別数(target)に揃う分位点を返す。

    手順(レポート・コードの両方に記録):
    1. 遡及期間の全対象レースについて指標値を計算し昇順に並べる
    2. mode="below"(値<閾値で選別)なら小さい方からtarget番目と次の中点を閾値に、
       mode="above"(値>閾値で選別)なら大きい方からtarget番目と次の中点を閾値にする
    3. これで選別数がほぼtargetに一致する(同値の重なりで±数件ずれることはある)
    ※較正は「選別数を揃える正規化」であり、成績を見て閾値を選ぶ操作ではない
      (成績由来の最適化をすると多重比較になるため行わない)
    """
    # 較正データが無い場合は「何も選別しない」安全側の閾値を返す
    if not values or target <= 0:
        return float("inf") if mode == "above" else float("-inf")
    s = sorted(values)
    target = min(target, len(s))
    if mode == "below":
        hi = s[target - 1]
        nxt = s[target] if target < len(s) else hi
        return (hi + nxt) / 2
    else:  # above
        lo = s[-target]
        prv = s[-target - 1] if target < len(s) else lo
        return (lo + prv) / 2


# ===== 選別器(scoreが大きいほど優先。Noneは非選別/判定不能) =====

def champion_score(ranked: list[dict]) -> float | None:
    """現行選別: 1位勝率(生値)35%未満=荒れ注意。優先度は1位勝率が低い順"""
    top = ranked[0]["prob"]
    return -top if P.bucket_of(top) == "荒れ注意" else None


def gap_score(probs: dict[int, float], threshold: float) -> float | None:
    """挑戦者②β1: 1位-2位差が閾値未満なら選別。差が小さいほど優先"""
    g = top_gap(probs)
    return -g if g < threshold else None


def entropy_score(probs: dict[int, float], threshold: float) -> float | None:
    """挑戦者②β2: エントロピーが閾値超なら選別。大きいほど優先"""
    e = entropy(probs)
    return e if e > threshold else None


def divergence_score(model_order: list[int], market_order: list[int] | None,
                     threshold: int = MARKET_DIVERGENCE_THRESHOLD) -> float | None:
    """挑戦者①: 市場相違度が閾値以上なら選別。market_order=None(スナップショット無し)は
    判定不能としてNone(選別しない)。呼び出し側で判定不能件数を数えて明示すること"""
    if market_order is None:
        return None
    d = market_divergence(model_order, market_order)
    return float(d) if d >= threshold else None


def challenger_c_condition(ranked: list[dict], c_picks: list, context: dict) -> float | None:
    """挑戦者③ C条件型(スタブ)。

    「C勝万舟が特に強い条件のレースだけ選ぶ」大穴一撃フラグ構想
    (notes/HANDOVER.md §7の検証候補=検証⑧構想と同件)。
    ledger.jsonのC的中明細と市場レポート(f)の条件分布が数ヶ月分貯まってから
    条件を定義する。それまでは未実装(レポートには「蓄積待ち」と表示する)。
    """
    raise NotImplementedError("挑戦者③は蓄積待ち(大穴一撃フラグ構想と同件。notes/HANDOVER.md参照)")


def daily_cap(candidates: list[tuple[float, str]], limit: int = DAILY_CAP) -> list[str]:
    """(score, race_id)の候補から score降順で上限limit件のrace_idを返す"""
    return [rid for _score, rid in sorted(candidates, reverse=True)[:limit]]


# ===== 決着分類・集計ヘルパー =====

def classify_outcome(stake: int, ret: int, santan_payout: int) -> str | None:
    """的中レースの決着3分類。不的中はNone。

    - 順当: 的中したが払戻<掛金(=ガミ)。「本命どおり決まって儲からない」の行動的定義
    - 万舟: 3連単払戻が10,000円以上のレース(順当でない場合)
    - 中波乱: それ以外の的中
    優先順位は順当を先に判定する: ガミり監視(市場レポートD)の目的が
    「荒れ注意と選んだのに順当に決まる比率の変化」の検出であり、
    万舟レースでガミる(C不発で3連複だけ薄く当たる)のも「取り损ね」として
    順当側に数える方が監視の意図に合うため。
    """
    if not ret:
        return None
    if ret < stake:
        return "順当"
    if santan_payout >= MANSHU_PAYOUT_MIN:
        return "万舟"
    return "中波乱"


def show_probability(probs: dict[int, float], lane: int) -> float:
    """指定艇の3着以内(複勝圏)確率。Benter展開(trifecta_probs)の周辺化"""
    tri = P.trifecta_probs(probs)
    return sum(p for (a, b, c), p in tri.items() if lane in (a, b, c))


def max_drawdown(daily_pnl: list[float]) -> float:
    """日次損益系列の最大ドローダウン(累積損益の最高値からの最大下落幅、負値で返す)"""
    peak = cum = 0.0
    dd = 0.0
    for x in daily_pnl:
        cum += x
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return dd


def longest_losing_streak(daily_pnl: list[float]) -> int:
    """日次損益系列の最長連敗(マイナス日が連続した最大日数)"""
    best = cur = 0
    for x in daily_pnl:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


# ===== 構成4案(検証⑪-B。選別はチャンピオン固定で比較する) =====

COMPOSITION_NAMES = ("現行", "攻撃型", "守備型", "軸分散型")


def _trio(a: int, b: int, c: int) -> str:
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def build_composition(name: str, ranked: list[dict],
                      c_picks: list[tuple[str, str, float]]) -> list[tuple[str, str, int, str]]:
    """構成4案の買い目リスト(券種, 買い目, 金額, 出典)。

    - 現行: predictors.ken_portfolio の荒れ注意構成をそのまま流用(検証済みV2)
    - 攻撃型: 3連複を2点に絞り、3連単の穴頭を4点に広げる(各100円)。
      原案の「3連複250円×2」は舟券が100円単位のため購入不可であり、
      結果を見る前(実装時点)に300/200へ事前調整した(発生確率の高い1=2=3を厚く。
      現行が3連複を200/200/100と前重みにしているのと同じ考え方)
    - 守備型: 3連単を買わず3連複4点に厚く。配分は仮置きの300/300/200/100が
      そのまま「合計900円+C100円=1,000円・100円単位」の制約を満たすため無調整で採用
      (事前登録の精神: 結果を見てから配分は変えない)
    - 軸分散型: 現行の3連複3点目(1=3=4の100円)を予測1位不在の2=3=4に置換。他は現行と同一
    いずれもC勝万舟1点100円を現行と同じ規則で追加する(C候補0点なら900円構成)。
    """
    if name == "現行":
        return P.ken_portfolio("荒れ注意", ranked, [], c_picks)

    lanes = [r["lane"] for r in ranked]
    if len(lanes) < 4:
        return []
    r1, r2, r3, r4 = lanes[:4]

    if name == "攻撃型":
        plan = [
            ("3連複", _trio(r1, r2, r3), 300, "攻撃"),
            ("3連複", _trio(r1, r2, r4), 200, "攻撃"),
            ("3連単", f"{r3}-{r1}-{r2}", 100, "攻撃"),
            ("3連単", f"{r4}-{r1}-{r2}", 100, "攻撃"),
            ("3連単", f"{r3}-{r1}-{r4}", 100, "攻撃"),
            ("3連単", f"{r4}-{r1}-{r3}", 100, "攻撃"),
        ]
    elif name == "守備型":
        plan = [
            ("3連複", _trio(r1, r2, r3), 300, "守備"),
            ("3連複", _trio(r1, r2, r4), 300, "守備"),
            ("3連複", _trio(r1, r3, r4), 200, "守備"),
            ("3連複", _trio(r2, r3, r4), 100, "守備"),
        ]
    elif name == "軸分散型":
        plan = [
            ("3連複", _trio(r1, r2, r3), 200, "軸分散"),
            ("3連複", _trio(r1, r2, r4), 200, "軸分散"),
            ("3連複", _trio(r2, r3, r4), 100, "軸分散"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "軸分散"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "軸分散"),
        ]
    else:
        raise ValueError(f"未知の構成名: {name}")

    # C勝万舟の追加は現行(ken_portfolio)と同じ規則: 重複しない最初の1点を100円
    existing = {(bt, comb) for bt, comb, _, _ in plan}
    for bt, comb, _p in c_picks:
        if (bt, comb) not in existing:
            plan.append((bt, comb, 100, "勝万舟"))
            break
    return plan
