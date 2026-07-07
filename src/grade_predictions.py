"""当日の予想(picks JSON)を確定結果で採点し、通算成績ページを更新するCLI

    python grade_predictions.py             # 今日分を採点
    python grade_predictions.py 2026-07-08  # 日付指定

- A/B/Cは各点100円の均等買いとして採点(通算成績の見える化用)
- 予想屋kenはポートフォリオの実額で採点(全レース、勝負所[本命/準]の内訳つき)
- 結果は docs用の ledger.json に日次追記(再実行時は同日を上書き=冪等)
- stats.html(通算成績ページ)を再生成する
"""
import json
import sys
from datetime import date
from pathlib import Path

import db
from config import DB_PATH, PROJECT_DIR, jst_today

SITE_DIR = PROJECT_DIR / "reports" / "site"
DATA_DIR = SITE_DIR / "data"

PREDICTOR_LABELS = {
    "a": "A 石橋渡",
    "b": "B 山田三連単",
    "c": "C 勝万舟",
    "ken": "予想屋ken(全レース)",
    "ken_hon": "ken 本命勝負所",
    "ken_jun": "ken 準勝負所",
}


def _zero():
    return {"stake": 0, "ret": 0, "races": 0, "hits": 0}


def grade_day(picks: dict, conn) -> dict | None:
    """1日分のpicksを採点。結果未確定(払戻ゼロ件)ならNone"""
    day = {k: _zero() for k in PREDICTOR_LABELS}
    graded_races = 0

    for race in picks["races"]:
        rid = race["race_id"]
        payout = {(bt, comb): amt for bt, comb, amt in conn.execute(
            "SELECT bet_type, combination, amount_yen FROM payouts WHERE race_id = ?", (rid,))}
        if not payout:
            continue  # 未確定 or 中止
        graded_races += 1

        for key in ("a", "b", "c"):
            s = day[key]
            s["races"] += 1
            ret = sum(payout.get((bt, comb), 0) for bt, comb, _p in race[key])
            s["stake"] += 100 * len(race[key])
            s["ret"] += ret
            s["hits"] += 1 if ret else 0

        ken = race["ken"]
        if ken:
            stake = sum(y for _, _, y, _ in ken)
            ret = sum(payout.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in ken)
            for key in ["ken"] + (
                ["ken_hon"] if race["shobusho"] == "本命"
                else ["ken_jun"] if race["shobusho"] == "準" else []
            ):
                s = day[key]
                s["races"] += 1
                s["stake"] += stake
                s["ret"] += ret
                s["hits"] += 1 if ret else 0

    if graded_races == 0:
        return None
    return day


def load_ledger() -> list:
    path = DATA_DIR / "ledger.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_ledger(ledger: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")


def render_stats(ledger: list) -> str:
    totals = {k: _zero() for k in PREDICTOR_LABELS}
    for entry in ledger:
        for k, s in entry["stats"].items():
            if k not in totals:
                continue
            for f in ("stake", "ret", "races", "hits"):
                totals[k][f] += s[f]

    def row(label, s, highlight=False):
        if s["races"] == 0:
            return ""
        roi = s["ret"] / s["stake"] if s["stake"] else 0
        cls = " class='hl'" if highlight else ""
        color = "pos" if s["ret"] >= s["stake"] else "neg"
        return (f"<tr{cls}><td>{label}</td><td class='num'>{s['races']:,}</td>"
                f"<td class='num'>{s['hits'] / s['races']:.1%}</td>"
                f"<td class='num {color}'>{roi:.1%}</td>"
                f"<td class='num {color}'>{s['ret'] - s['stake']:+,}円</td></tr>")

    total_rows = "".join(
        row(PREDICTOR_LABELS[k], totals[k], highlight=k.startswith("ken"))
        for k in PREDICTOR_LABELS
    )

    daily_rows = []
    for entry in sorted(ledger, key=lambda e: e["date"], reverse=True)[:30]:
        ken = entry["stats"].get("ken_hon", _zero())
        all_ken = entry["stats"].get("ken", _zero())
        pnl = ken["ret"] - ken["stake"]
        color = "pos" if pnl >= 0 else "neg"
        daily_rows.append(
            f"<tr><td>{entry['date']}</td>"
            f"<td class='num'>{ken['races']}</td>"
            f"<td class='num {color}'>{pnl:+,}円</td>"
            f"<td class='num'>{all_ken['races']}</td>"
            f"<td class='num'>{all_ken['ret'] - all_ken['stake']:+,}円</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>通算成績</title>
<style>
  body {{ font-family: sans-serif; margin: 0; padding: 8px; background: #f6f8fa; }}
  h1 {{ font-size: 1.15rem; margin: 8px 4px; }}
  h2 {{ font-size: 1.0rem; margin: 18px 4px 8px; }}
  .nav {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 10px; }}
  .nav a {{ text-decoration: none; font-size: .82rem; padding: 5px 10px; border-radius: 14px;
           background: #fff; color: #0969da; border: 1px solid #d0d7de; }}
  .card {{ background: #fff; border-radius: 10px; padding: 12px; margin-bottom: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 6px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #1a7f37; font-weight: bold; }}
  .neg {{ color: #cf222e; }}
  tr.hl td {{ background: #d6efff55; }}
  .note {{ font-size: .75rem; color: #57606a; margin: 6px 4px; }}
</style>
</head>
<body>
<h1>通算成績</h1>
<div class="nav">
  <a href="index.html">平和島</a><a href="edogawa.html">江戸川</a>
  <a href="tokoname.html">常滑</a><a href="amagasaki.html">尼崎</a>
  <a href="wakamatsu.html">若松</a>
</div>
<div class="card">
  <h2 style="margin-top:0">予想者別 通算成績(採点開始 {ledger[0]['date'] if ledger else '-'} 〜)</h2>
  <table>
    <tr><th>予想者</th><th class="num">レース数</th><th class="num">的中率</th>
        <th class="num">回収率</th><th class="num">損益</th></tr>
    {total_rows}
  </table>
  <p class="note">A/B/Cは1点100円の均等買い換算。予想屋kenはポートフォリオ実額(1レース1,000円)。
  水色行がken。推奨運用は「ken 本命勝負所」のみ購入。</p>
</div>
<div class="card">
  <h2 style="margin-top:0">日別(直近30日)</h2>
  <table>
    <tr><th>日付</th><th class="num">本命勝負所</th><th class="num">本命損益</th>
        <th class="num">全レース</th><th class="num">全レース損益</th></tr>
    {''.join(daily_rows)}
  </table>
</div>
<p class="note">毎晩23時台に自動更新。舟券の購入は自己責任で。</p>
</body>
</html>
"""


def main(d: date) -> None:
    picks_path = DATA_DIR / f"picks_{d.isoformat()}.json"
    if not picks_path.exists():
        print(f"{d}: picksファイルがありません({picks_path})。予想未実行の日はスキップ。")
        return

    picks = json.loads(picks_path.read_text(encoding="utf-8"))
    conn = db.connect(DB_PATH)
    day = grade_day(picks, conn)
    conn.close()

    if day is None:
        print(f"{d}: 結果がまだ確定していないため採点をスキップ。")
        return

    ledger = [e for e in load_ledger() if e["date"] != d.isoformat()]
    ledger.append({"date": d.isoformat(), "stats": day})
    ledger.sort(key=lambda e: e["date"])
    save_ledger(ledger)

    (SITE_DIR / "stats.html").write_text(render_stats(ledger), encoding="utf-8")

    ken = day["ken_hon"]
    print(f"{d}: 採点完了。本命勝負所 {ken['races']}レース 損益{ken['ret'] - ken['stake']:+,}円。"
          f"通算 {len(ledger)}日分をstats.htmlへ出力。")


if __name__ == "__main__":
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else jst_today()
    main(target)
