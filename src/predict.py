"""予測対象5場(江戸川・平和島・常滑・尼崎・若松)の指定日レースを予測し、
買い目を1枚のHTMLレポートに出力するCLI

    python predict.py             # 明日分
    python predict.py today       # 今日分
    python predict.py 2026-07-11  # 日付指定

HTMLはreports/に保存される。プロジェクトがGoogle Drive配下にあるため、
スマホのDriveアプリからそのまま閲覧できる。

買い目はウォークフォワード検証(backtest.py)の結果に基づく:
- 荒れ注意(予測1位の勝率35%未満)だけが期待値プラス
  → このレースのみ「勝負」として予算1000円を配分
- 堅め・標準はどの買い方も一貫してマイナス → 見送り推奨(参考買い目のみ)
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb

import db
import weather
from config import DB_PATH, MODEL_PATH, PROJECT_DIR, TARGET_VENUE_CODES, VENUE_NAMES
from downloader import download_day
from features import FEATURE_COLUMNS, build_program_features
from parser_b import parse_program

REPORTS_DIR = PROJECT_DIR / "reports"


def recommend_bets(ranked: list[dict]) -> dict:
    """モデルの勝率降順の艇リストから買い目プランを組む。

    ウォークフォワード検証で期待値プラスが確認できた「荒れ注意」レースのみ
    予算1000円の「勝負」プラン(3連複軸1流し600円+3連単穴400円)。
    堅め・標準は見送り推奨とし、少額の参考買い目だけ付ける。
    planの要素は (券種, 組み合わせ, 金額円, 区分)。
    """
    lanes = [r["lane"] for r in ranked]
    top_prob = ranked[0]["prob"]

    if top_prob >= 0.50:
        confidence = "堅め"
    elif top_prob >= 0.35:
        confidence = "標準"
    else:
        confidence = "荒れ注意"

    def trio(a, b, c):
        s = sorted([a, b, c])
        return f"{s[0]}={s[1]}={s[2]}"

    r1, r2, r3, r4 = (lanes + [None] * 4)[:4]

    if confidence == "荒れ注意" and r4 is not None:
        stance = "勝負"
        plan = [
            ("3連複", trio(r1, r2, r3), 200, "負けにくい"),
            ("3連複", trio(r1, r2, r4), 200, "負けにくい"),
            ("3連複", trio(r1, r3, r4), 200, "負けにくい"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "大穴"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "大穴"),
        ]
    elif confidence == "標準" and r4 is not None:
        stance = "見送り推奨"
        plan = [
            ("3連複", trio(r1, r2, r3), 100, "参考"),
            ("3連複", trio(r1, r2, r4), 100, "参考"),
            ("3連複", trio(r1, r3, r4), 100, "参考"),
        ]
    else:
        stance = "見送り推奨"
        plan = []
        if r2 is not None:
            plan.append(("2連複", f"{min(r1, r2)}={max(r1, r2)}", 100, "参考"))
        if r3 is not None:
            plan.append(("3連単", f"{r1}-{r2}-{r3}", 100, "参考"))

    return {"confidence": confidence, "stance": stance, "plan": plan}


def _ensure_program(conn, d: date) -> bool:
    """指定日の対象場の番組表がDBになければダウンロードして格納する"""
    def target_count():
        ph = ",".join("?" * len(TARGET_VENUE_CODES))
        return conn.execute(
            f"SELECT COUNT(*) FROM races WHERE date = ? AND venue_code IN ({ph})",
            (d.isoformat(), *TARGET_VENUE_CODES),
        ).fetchone()[0]

    if target_count():
        return True

    paths = download_day(d)
    if paths["program"] is None:
        return False

    program_data = parse_program(json.loads(paths["program"].read_text(encoding="utf-8")))
    for race in program_data["races"]:
        db.upsert_race(conn, race)
    for entry in program_data["entries"]:
        db.upsert_entry(conn, entry)
    conn.commit()
    return target_count() > 0


def _fetch_weather_by_race(conn, race_meta: dict) -> dict[str, dict]:
    """レースIDごとのレース前予報(Open-Meteo)を返す。表示専用(モデルには使わない)。

    バックテストでモデルの特徴量に加えると回収率が悪化したため(features.py参照)、
    予測には使わずレポート上で人間が判断材料にするための参考情報として提供する。
    取得失敗した場はスキップする。
    """
    hourly_by_venue = {}
    for venue in {meta["venue_code"] for meta in race_meta.values()}:
        try:
            hourly_by_venue[venue] = weather.fetch_hourly(venue)
        except Exception as e:
            print(f"警告: {VENUE_NAMES[venue]}の気象予報取得に失敗({e})。表示なしで続行します。")

    result = {}
    for race_id, meta in race_meta.items():
        hourly = hourly_by_venue.get(meta["venue_code"])
        if not hourly or not meta["deadline"]:
            continue
        wx = weather.lookup(hourly, meta["deadline"])
        if wx is None:
            continue
        wind_speed, wind_deg, temperature = wx
        result[race_id] = {
            "wind_speed_m": wind_speed,
            "wind_dir": weather.compass_name(wind_deg),
            "temperature": temperature,
            "wave_height_cm": weather.estimate_wave_height_cm(conn, meta["venue_code"], wind_speed),
        }
    return result


def predict_day(d: date) -> list[dict] | None:
    """1日分・対象5場の予測。どの場も開催がなければNone"""
    conn = db.connect(DB_PATH)
    if not _ensure_program(conn, d):
        conn.close()
        return None

    ph = ",".join("?" * len(TARGET_VENUE_CODES))
    rows = conn.execute(
        f"SELECT race_id, venue_code, race_no, deadline_time FROM races "
        f"WHERE date = ? AND venue_code IN ({ph}) ORDER BY venue_code, race_no",
        (d.isoformat(), *TARGET_VENUE_CODES),
    ).fetchall()
    race_meta = {
        r[0]: {"venue_code": r[1], "race_no": r[2], "deadline": r[3]} for r in rows
    }

    df = build_program_features(conn, list(race_meta.keys()))
    race_weather = _fetch_weather_by_race(conn, race_meta)
    conn.close()

    # 日本語を含むパスをLightGBMネイティブに渡せないため、Python側で読み込む
    booster = lgb.Booster(model_str=MODEL_PATH.read_text(encoding="utf-8"))
    df["prob"] = booster.predict(df[FEATURE_COLUMNS])

    races = []
    for race_id, meta in race_meta.items():
        race_df = df[df["race_id"] == race_id].sort_values("prob", ascending=False)
        ranked = [
            {
                "lane": int(row["lane"]),
                "name": row["racer_name"],
                "racer_class": row["racer_class"],
                "prob": float(row["prob"]),
            }
            for _, row in race_df.iterrows()
        ]
        races.append({
            "venue_code": meta["venue_code"],
            "venue_name": VENUE_NAMES[meta["venue_code"]],
            "race_no": meta["race_no"],
            "deadline": meta["deadline"],
            "weather": race_weather.get(race_id),
            "ranked": ranked,
            "bets": recommend_bets(ranked),
        })
    return races


_CONFIDENCE_COLORS = {"堅め": "#1a7f37", "標準": "#9a6700", "荒れ注意": "#cf222e"}


def render_html(d: date, races: list[dict]) -> str:
    shobu = [r for r in races if r["bets"]["stance"] == "勝負"]
    if shobu:
        labels = "、".join(f"{r['venue_name']}{r['race_no']}R" for r in shobu)
        summary = (f"本日の勝負レース: <b>{labels}</b>"
                   f"(予算 {len(shobu) * 1000:,}円)。それ以外は見送り推奨。")
    else:
        summary = "本日は勝負レースなし(全レース見送り推奨)。"

    venues_today = sorted({r["venue_code"] for r in races})
    sections = []
    for venue in venues_today:
        venue_races = [r for r in races if r["venue_code"] == venue]
        cards = [_render_race_card(r) for r in venue_races]
        sections.append(
            f"<h2>{VENUE_NAMES[venue]}</h2>\n" + "".join(cards)
        )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{d} 買い目予想(5場)</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 8px; background: #f6f8fa; }}
  h1 {{ font-size: 1.2rem; margin: 8px 4px; }}
  h2 {{ font-size: 1.05rem; margin: 18px 4px 8px; border-left: 4px solid #0969da;
       padding-left: 8px; }}
  .note {{ font-size: .75rem; color: #57606a; margin: 0 4px 12px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 12px; margin-bottom: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  .head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }}
  .rno {{ font-size: 1.3rem; font-weight: bold; }}
  .deadline {{ color: #57606a; font-size: .85rem; }}
  .conf {{ margin-left: auto; color: #fff; font-size: .75rem; padding: 3px 10px;
          border-radius: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  td {{ padding: 4px 6px; border-bottom: 1px solid #eee; }}
  .lane {{ width: 2em; text-align: center; font-weight: bold; border-radius: 4px; }}
  .l1 {{ background: #fff; border: 1px solid #ccc; }} .l2 {{ background: #222; color: #fff; }}
  .l3 {{ background: #d32f2f; color: #fff; }} .l4 {{ background: #1565c0; color: #fff; }}
  .l5 {{ background: #fbc02d; }} .l6 {{ background: #2e7d32; color: #fff; }}
  .prob {{ text-align: right; font-weight: bold; }}
  .weather {{ font-size: .78rem; color: #57606a; background: #f0f6ff; border-radius: 6px;
             padding: 5px 8px; margin-bottom: 8px; }}
  .wx-note {{ display: block; font-size: .68rem; color: #8c959f; }}
  .card.dim {{ opacity: .62; }}
  .stance {{ font-size: .75rem; padding: 3px 10px; border-radius: 12px; color: #fff;
            font-weight: bold; }}
  .stance.go {{ background: #cf222e; }}
  .stance.skip {{ background: #8c959f; }}
  .summary {{ background: #fff8c5; border: 1px solid #d4a72c66; border-radius: 8px;
             padding: 10px 12px; margin: 0 0 12px; font-size: .9rem; }}
  .plan {{ margin-top: 10px; background: #f6f8fa; border-radius: 8px; padding: 8px; }}
  .plan h3 {{ margin: 0 0 6px; font-size: .8rem; }}
  .plan-table td {{ border: none; padding: 2px 6px; font-size: .95rem; }}
  .plan-table .tag {{ font-size: .7rem; color: #57606a; width: 5em; }}
  .plan-table .bt {{ font-size: .8rem; color: #57606a; width: 4em; }}
  .plan-table .yen {{ text-align: right; }}
</style>
</head>
<body>
<h1>{d} 買い目予想(対象5場)</h1>
<div class="summary">{summary}</div>
<p class="note">勝率はモデル予測値。「勝負」=検証で回収率100%超の荒れ注意レースのみ。
堅め・標準レースはどの買い方も期待値マイナスのため見送り推奨。購入は自己責任で。</p>
{''.join(sections)}
</body>
</html>
"""


def _render_race_card(race: dict) -> str:
    deadline = (race["deadline"] or "")[-8:-3]  # HH:MM
    bets = race["bets"]
    conf = bets["confidence"]
    color = _CONFIDENCE_COLORS[conf]
    is_shobu = bets["stance"] == "勝負"

    boat_rows = "".join(
        f"<tr><td class='lane l{b['lane']}'>{b['lane']}</td>"
        f"<td>{b['name']}</td><td>{b['racer_class']}</td>"
        f"<td class='prob'>{b['prob']:.0%}</td></tr>"
        for b in race["ranked"]
    )
    plan_rows = "".join(
        f"<tr><td class='tag'>{tag}</td><td class='bt'>{bt}</td>"
        f"<td>{comb}</td><td class='yen'>{yen}円</td></tr>"
        for bt, comb, yen, tag in bets["plan"]
    )
    total = sum(yen for _, _, yen, _ in bets["plan"])
    stance_html = (
        "<span class='stance go'>勝負</span>" if is_shobu
        else "<span class='stance skip'>見送り推奨</span>"
    )
    wx = race.get("weather")
    weather_html = (
        f"<div class='weather'>予報: 風速{wx['wind_speed_m']:.1f}m/s({wx['wind_dir']}の風) "
        f"波高目安{wx['wave_height_cm']:.1f}cm 気温{wx['temperature']:.0f}℃"
        f"<span class='wx-note'>※参考情報・予測には未使用</span></div>"
        if wx else ""
    )
    plan_title = (
        f"買い目プラン(計{total:,}円)" if is_shobu
        else f"参考買い目(期待値マイナス・計{total:,}円)"
    )

    return f"""
  <div class="card{' dim' if not is_shobu else ''}">
    <div class="head">
      <span class="rno">{race['race_no']}R</span>
      <span class="deadline">締切 {deadline}</span>
      {stance_html}
      <span class="conf" style="background:{color}">{conf}</span>
    </div>
    {weather_html}
    <table>{boat_rows}</table>
    <div class="plan">
      <h3>{plan_title}</h3>
      <table class="plan-table">{plan_rows}</table>
    </div>
  </div>"""


def run(d: date) -> Path | None:
    races = predict_day(d)
    if races is None:
        print(f"{d}: 対象5場はすべて非開催(または番組表未公開)")
        return None

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{d.isoformat()}_picks.html"
    out.write_text(render_html(d, races), encoding="utf-8")
    venues = "、".join(sorted({r["venue_name"] for r in races}))
    print(f"{d}: {len(races)}レース({venues})を出力 -> {out}")
    return out


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"モデルが見つかりません: {MODEL_PATH}\n先に train_model.py を実行してください。")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "today":
        targets = [date.today()]
    elif len(sys.argv) > 1:
        targets = [date.fromisoformat(sys.argv[1])]
    else:
        targets = [date.today() + timedelta(days=1)]

    for target in targets:
        run(target)
