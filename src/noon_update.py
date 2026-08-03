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
import time
from datetime import date, datetime

import odds as odds_mod
import predict
import predictors as P
from config import JST, PAGES_URL, PROJECT_DIR, TARGET_VENUE_CODES, VENUE_NAMES, jst_today

# 要オッズ確認の全レース紙上記録(2026-08-04ケンさん指示)。
# 表示・購入ルールは従来どおり5場×30〜35帯のみ。記録だけ全レースに広げ、
# 帯別・場別の追検証データを自動で貯める(docs/data/odds_check_日付.json)。
# 負荷=1レース2リクエスト(3連単/3連複ページ)×開催全レース×3回/日。
LOG_ALL_RACES = True
FETCH_INTERVAL_SEC = 0.3  # 公式サイトへの連続アクセス間隔(負荷配慮)


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

    # 要オッズ確認(2026-08-02ケンさん発案・表示のみ): 5場×1位生値30〜35%帯で、
    # プランの3連複のうち一桁オッズ(10倍未満)が2点以下なら「購入チャンス(裁量)」。
    # 検証: test/verify_odds_single_digit_rule.py(一桁≤2=100.8%/ガミ18.2% vs
    # 一桁≥3=82.5%/ガミ43.7%、一桁ちょうど2点は129.4%。逆に20-30本命帯では
    # 符号が逆転するため本命帯には適用しない=オッズで本命を見送らない原則は不変)
    # 判定は全レースで計算し、検証済みスコープ(5場×30〜35帯)かをフラグで区別する。
    # 表示はvalidated=検証値つきの色付き、それ以外=記録用の控えめ表示(predict側)
    odds_check = build_odds_check(ken_rows)
    if odds_check is not None:
        top_raw = race["ranked"][0]["prob"] if race.get("ranked") else None
        odds_check["validated"] = bool(
            top_raw is not None and 0.30 <= top_raw < 0.35
            and race.get("venue_code") in TARGET_VENUE_CODES)

    return {"fetched": fetched_label, "ken_rows": ken_rows, "value": value,
            "odds_check": odds_check}


def build_odds_check(ken_rows: list) -> dict | None:
    """プラン3連複の一桁オッズ数から要オッズ確認の判定を作る(帯・場の制限なし)。

    判定は「ちょうど2点」のみチャンス(2026-08-02ケンさん指摘で厳格化。
    ≤2で括ると0-1点[53.4%]が混ざり129.4%が100.8%に薄まる)
    """
    fuku = [o for bt, _c, o, _e, _v in ken_rows if bt == "3連複" and o]
    if not fuku:
        return None
    singles = sum(1 for o in fuku if o < 10.0)
    verdict = "chance" if singles == 2 else "chaos" if singles <= 1 else "cheap"
    return {"singles": singles, "verdict": verdict, "n_fuku": len(fuku)}


def _append_odds_check_log(d: date, fetched_label: str, records: list) -> None:
    """要オッズ確認の全レース紙上記録をスナップショット単位で追記保存する。

    Actionsは毎回まっさらなチェックアウトで走るため、既存分はdocs/data側から
    読み継ぐ(_apply_morning_picksと同じ2段フォールバック)。同時刻ラベルの
    再実行は上書き。出力先はreports/site/data(公開はワークフローがコピー)。
    更新後のログ全体を返す(サイト表示用)。
    """
    fname = f"odds_check_{d.isoformat()}.json"
    data_dir = predict.SITE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log = None
    for p in (data_dir / fname, PROJECT_DIR / "docs" / "data" / fname):
        if p.exists():
            log = json.loads(p.read_text(encoding="utf-8"))
            break
    if log is None:
        log = {"date": d.isoformat(), "snapshots": []}
    log["snapshots"] = [s for s in log["snapshots"]
                        if s["fetched"] != fetched_label]
    log["snapshots"].append({"fetched": fetched_label, "races": records})
    (data_dir / fname).write_text(
        json.dumps(log, ensure_ascii=False), encoding="utf-8")
    return log


def merged_odds_check(log: dict) -> list[dict]:
    """日次ログの全スナップショットをレース単位で最新値にマージし締切順で返す。

    サイトの全レース判定テーブル用(締切済みレースも最後の判定を残して表示する)。
    """
    latest: dict[str, dict] = {}
    for snap in log.get("snapshots", []):
        for rec in snap["races"]:
            latest[rec["race_id"]] = {**rec, "fetched": snap["fetched"]}
    return sorted(latest.values(), key=lambda r: r.get("deadline") or "9999")


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
    log_records: list[dict] = []
    for race in races:
        # オッズタブの表示は従来どおり勝負所(本命/超混戦/要注目)のみ。
        # LOG_ALL_RACES時は記録用に締切前の全レースからも取得する
        is_shobusho = bool(race.get("shobusho"))
        if not race["bets"]["plan"] or not (is_shobusho or LOG_ALL_RACES):
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
        finally:
            time.sleep(FETCH_INTERVAL_SEC)
        if not odds_data["3連単"]:
            continue
        view = build_odds_view(race, odds_data, fetched_label)
        # オッズタブ=勝負所(従来)+ページのある5場の全レース(2026-08-04〜)
        if is_shobusho or race["venue_code"] in predict.VENUE_SLUGS:
            odds_panes[race["race_id"]] = predict._render_odds_pane(view)
        # 標準4組(モデル上位4艇の 1=2=3/1=2=4/1=3=4/2=3=4)のオッズも全レースで
        # 記録する。プランの3連複は帯で構成が違い(⑰=4行/堅め標準=trio_top2行)、
        # 帯横断の追検証には共通基準の生オッズが要るため(0.0=投票未形成)
        std_trio = []
        lanes = [b["lane"] for b in race.get("ranked") or []][:4]
        if len(lanes) == 4:
            r1, r2, r3, r4 = lanes
            for combo in ((r1, r2, r3), (r1, r2, r4), (r1, r3, r4), (r2, r3, r4)):
                key = tuple(sorted(combo))
                std_trio.append(["=".join(map(str, key)),
                                 odds_data["3連複"].get(key)])
        log_records.append({
            "race_id": race["race_id"],
            "venue_code": race["venue_code"],
            "race_no": race["race_no"],
            "shobusho": race.get("shobusho"),
            "p1": race["ranked"][0]["prob"] if race.get("ranked") else None,
            "deadline": race["deadline"],
            "check": view["odds_check"],
            "plan_odds": [[bt, comb, o]
                          for bt, comb, o, _e, _v in view["ken_rows"]],
            "std_trio_odds": std_trio,
        })

    odds_check_records = None
    if log_records:
        log = _append_odds_check_log(d, fetched_label, log_records)
        odds_check_records = merged_odds_check(log)

    predict.SITE_DIR.mkdir(parents=True, exist_ok=True)
    for venue, slug in predict.VENUE_SLUGS.items():
        html = predict.render_venue_page(d, venue, races, odds_panes)
        (predict.SITE_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    # トップ=買い目一覧(v2)+全レース一桁オッズ判定テーブル
    (predict.SITE_DIR / "index.html").write_text(
        predict.render_shopping_page(d, races, odds_panes, odds_check_records),
        encoding="utf-8")

    # オッズを1レースでも反映できたら通知文を書き出す(送信判断はワークフロー側。
    # *.htmlしかdocsへコピーされないため、このファイルがサイトに載ることはない)
    if odds_panes:
        data_dir = predict.SITE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "noon_notify.txt").write_text(
            build_notify_text(fetched_label, races, odds_panes), encoding="utf-8")

    print(f"{d}: {len(odds_panes)}レースにオッズ反映タブを追加してサイトを再生成 -> {predict.SITE_DIR}")
    print(f"{d}: 要オッズ確認の紙上記録={len(log_records)}レース({fetched_label}時点)")
    return True


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all"]
    include_all = "--all" in sys.argv
    target = date.fromisoformat(args[0]) if args else jst_today()
    run(target, include_all=include_all)
