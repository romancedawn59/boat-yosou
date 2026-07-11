"""10:00に締切前レースの最新オッズを取得し、「オッズ反映版」タブ付きでサイトを再生成するCLI

    python noon_update.py             # 今日(JST)。締切前のレースのみオッズ取得
    python noon_update.py 2026-07-07  # 日付指定(検証用。--allで全レース対象)

方針:
- オッズ反映版は成績対象外の参考情報。朝の予想(タブ1)・勝負所判定・picks JSON・
  採点・アーカイブには一切影響しない(HTMLページだけを再生成する)
- 取得したオッズはDBに保存しない(15分前スナップショットの蓄積データと分離するため)
"""
import json
import sys
from datetime import date, datetime

import odds as odds_mod
import predict
import predictors as P
from config import JST, PAGES_URL, PROJECT_DIR, VENUE_NAMES, jst_today


def _apply_morning_picks(races: list[dict], d: date) -> bool:
    """朝のワークフローが出力した picks JSON を正として予測表示を上書きする。

    noonは predict_day() で予測を再計算するため、scheduleの遅延で朝の予想より
    先に走ると、古いDB由来の予測で勝負所判定が朝版(=採点対象)とズレたページを
    公開しうる。picks JSONが見つかればそちらの判定・買い目で固定する。
    見つからなければFalseを返し、呼び出し側は再計算表示にフォールバックする。
    """
    candidates = [
        predict.SITE_DIR / "data" / f"picks_{d.isoformat()}.json",
        PROJECT_DIR / "docs" / "data" / f"picks_{d.isoformat()}.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return False

    picks = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["race_id"]: r for r in picks.get("races", [])}
    for race in races:
        m = by_id.get(race["race_id"])
        if m is None:
            continue  # 朝版にないレースは再計算値のまま
        race["bets"]["confidence"] = m["confidence"]
        race["shobusho"] = m["shobusho"]
        race["bets"]["plan"] = [tuple(x) for x in m["ken"]]
        race["picks_a"] = [tuple(x) for x in m["a"]]
        race["picks_b"] = [tuple(x) for x in m["b"]]
        race["picks_c"] = [tuple(x) for x in m["c"]]
    print(f"朝のpicks JSONを反映: {path}")
    return True


def build_odds_view(race: dict, odds_data: dict, fetched_label: str) -> dict:
    """レースのオッズ反映ビュー(kenプラン各点のオッズ/想定払戻/EV + 妙味候補)を作る"""
    probs = P.normalize_probs(race["ranked"])
    tri = P.trifecta_probs(probs)
    trio_p: dict[tuple, float] = {}
    for k, v in tri.items():
        key = tuple(sorted(k))
        trio_p[key] = trio_p.get(key, 0.0) + v

    def model_prob(bt: str, key: tuple) -> float:
        return tri.get(key, 0.0) if bt == "3連単" else trio_p.get(tuple(sorted(key)), 0.0)

    ken_rows = []
    for bt, comb, yen, _src in race["bets"]["plan"]:
        sep = "-" if bt == "3連単" else "="
        key = tuple(int(x) for x in comb.split(sep))
        if bt == "3連複":
            key = tuple(sorted(key))
        o = odds_data.get(bt, {}).get(key)
        est = int(o * yen) if o else 0
        ev = model_prob(bt, key) * o if o else 0.0
        ken_rows.append((bt, comb, o, est, ev))

    # オッズ妙味: モデル×市場ブレンド確率のEV上位3点(実験枠・未検証)
    cands = []
    for bt, table in odds_data.items():
        raw = {k: 1.0 / o for k, o in table.items() if o}
        tot = sum(raw.values())
        if not tot:
            continue
        for key, inv in raw.items():
            market_p = inv / tot
            blend = 0.5 * model_prob(bt, key) + 0.5 * market_p
            o = table[key]
            sep = "-" if bt == "3連単" else "="
            cands.append((blend * o, bt, sep.join(map(str, key)), o))
    cands.sort(reverse=True)
    value = [(bt, comb, o) for _ev, bt, comb, o in cands[:3]]

    return {"fetched": fetched_label, "ken_rows": ken_rows, "value": value}


def build_notify_text(fetched_label: str, races: list, odds_panes: dict) -> str:
    """オッズ反映版のLINE通知文(ワークフロー側でその日の初回のみ送信される)"""
    venues = sorted({r["venue_code"] for r in races if r["race_id"] in odds_panes})
    names = "・".join(VENUE_NAMES[v] for v in venues)
    return (f"⏱オッズ反映版を公開しました（{fetched_label}時点）\n"
            f"{names}の締切前{len(odds_panes)}レースにオッズ・想定払戻つきの予想を掲載\n"
            f"{PAGES_URL}/")


def run(d: date, include_all: bool = False) -> bool:
    races = predict.predict_day(d)
    if races is None:
        print(f"{d}: 対象5場はすべて非開催。")
        return False

    if not _apply_morning_picks(races, d):
        print(f"警告: 朝のpicks JSON(picks_{d.isoformat()}.json)が見つからないため、"
              "再計算した予測で表示します(勝負所判定が朝版とズレる可能性あり)")

    now = datetime.now(JST)
    fetched_label = now.strftime("%H:%M")
    odds_panes: dict[str, str] = {}
    for race in races:
        if not race["bets"]["plan"]:
            continue
        if not include_all:
            deadline = race["deadline"]
            if not deadline:
                continue
            deadline_dt = datetime.strptime(deadline, "%Y-%m-%d %H:%M:%S").replace(tzinfo=JST)
            if deadline_dt <= now:
                continue  # 締切済みレースはオッズ版なし(朝版のみ表示)
        try:
            odds_data = odds_mod.fetch_odds(race["venue_code"], race["race_no"], d)
        except Exception as e:
            print(f"{race['race_id']}: オッズ取得失敗 ({e})")
            continue
        if not odds_data["3連単"]:
            continue
        view = build_odds_view(race, odds_data, fetched_label)
        odds_panes[race["race_id"]] = predict._render_odds_pane(view)

    predict.SITE_DIR.mkdir(parents=True, exist_ok=True)
    for venue, slug in predict.VENUE_SLUGS.items():
        html = predict.render_venue_page(d, venue, races, odds_panes)
        (predict.SITE_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    (predict.SITE_DIR / "index.html").write_text(
        predict.render_venue_page(d, predict.TOP_VENUE, races, odds_panes), encoding="utf-8")

    # オッズを1レースでも反映できたら通知文を書き出す(送信判断はワークフロー側。
    # *.htmlしかdocsへコピーされないため、このファイルがサイトに載ることはない)
    if odds_panes:
        data_dir = predict.SITE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "noon_notify.txt").write_text(
            build_notify_text(fetched_label, races, odds_panes), encoding="utf-8")

    print(f"{d}: {len(odds_panes)}レースにオッズ反映タブを追加してサイトを再生成 -> {predict.SITE_DIR}")
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all"]
    include_all = "--all" in sys.argv
    target = date.fromisoformat(args[0]) if args else jst_today()
    run(target, include_all=include_all)
