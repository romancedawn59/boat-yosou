# -*- coding: utf-8 -*-
"""評価点システム: 本命勝負所の的中レースを払戻率で採点し「なぜ100点未満か」を記録するCLI

    py -X utf8 test/study_log.py

2026-07-18ケンさん発案:
- 払戻率をそのまま評価点にする(100%以上=100点が合格。ガミ=100点未満)
- ガミ幅は70%まで許容する方針だが、100点未満の1件1件について
  「なぜその点数だったのか」を記録し、少しずつ勉強・向上の糧にする
- 特に「的中が3連単のみ」(現行構成では実質C勝万舟の単独的中)は、
  万舟圏想定の目がいくらで決まったかを重点的に見る

出力: docs/study.html(スマホ閲覧用の独立ページ。予想・採点・通算成績には影響しない)
データ源: docs/data/ledger.json(読み取り専用)+15分前スナップショット(参考注記)
更新は手動(このコマンド)。v2でワークフロー組込みを検討。
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db as db_mod
from config import DB_PATH, JST, PROJECT_DIR, VENUE_NAMES
from tax_summary import PURCHASE_START, parse_line

DATA_DIR = PROJECT_DIR / "docs" / "data"
OUT_PATH = PROJECT_DIR / "docs" / "study.html"

NAME_TO_CODE = {v: k for k, v in VENUE_NAMES.items()}


def score_of(stake: int, ret: int) -> int:
    """評価点: 払戻率そのまま(100%以上は100点で頭打ち)"""
    if stake <= 0:
        return 0
    return min(100, round(ret * 100 / stake))


def classify(lines: list[tuple[str, str, int, int]], stake: int, ret: int,
             snap_note: str = "") -> tuple[str, str]:
    """的中レースの(タグ, 学びメモ)。linesは[(券種, 買い目, 券購入額, 払戻額)]。

    現行構成では穴頭3連単(3-1-2等)の的中は3連複と同時的中になるため、
    「3連単のみ」は実質C勝万舟の単独的中を意味する。
    """
    kinds = {bt for bt, _, _, _ in lines}
    only_santan = kinds == {"3連単"}
    has_santan = "3連単" in kinds
    s = score_of(stake, ret)

    if only_santan:
        tag = "3連単のみ(C単独)"
        if s >= 100:
            memo = "合格点。万舟圏の一撃が仕事をした型"
        else:
            memo = ("万舟圏想定の目が低配当で決着。モデルの確率と市場の評価が乖離"
                    "(オッズ変動か確率の過小見積もりを要研究)")
    elif has_santan:
        tag = "3連単+3連複"
        if s >= 100:
            memo = "合格点。穴頭決着を厚く取れた型"
        else:
            memo = ("穴頭で決まったのに配当が薄い=市場は穴頭まで織り込み済み。"
                    "「荒れの質」(誰が来て荒れるか)の見極めが課題")
    else:
        tag = "3連複のみ"
        if s >= 100:
            memo = "3連複だけで掛金超え(中波乱を面で取れた型)"
        else:
            memo = "順当決着で本命サイドの3連複のみ薄く的中。荒れ判定が外れた=選別の課題"
    if snap_note and s < 100:
        memo += f"。{snap_note}"
    return tag, memo


def load_snapshot_trio(conn) -> dict[str, dict[str, float]]:
    snap: dict[str, dict[str, float]] = {}
    for rid, comb, o in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連複' AND fetched_at != 'final-backfill' AND odds > 0"):
        snap.setdefault(rid, {})[comb] = o
    return snap


def build_rows() -> tuple[list[dict], dict]:
    ledger = json.loads((DATA_DIR / "ledger.json").read_text(encoding="utf-8"))
    ledger = [d for d in ledger if d["date"] >= PURCHASE_START]  # 実購入分のみ
    conn = sqlite3.connect(DB_PATH)
    snap = load_snapshot_trio(conn)
    conn.close()

    rows = []
    total = {"races": 0, "hits": 0, "score_sum": 0, "perfect": 0}
    for day in ledger:
        total["races"] += day["stats"]["ken_hon"]["races"]
        total["races"] += day["stats"].get("ken_konsen", {}).get("races", 0)
        for h in (day.get("hits", {}).get("ken_hon", [])
                  + day.get("hits", {}).get("ken_konsen", [])):
            lines = []
            for ln in h.get("lines", []):
                parsed = parse_line(ln["label"])
                if parsed:
                    bt, comb, lstake = parsed
                    lines.append((bt, comb, lstake, ln["payout"]))
            rid = db_mod.make_race_id(h["date"], NAME_TO_CODE[h["venue"]], h["race_no"])
            # 参考: 購入時点(15分前)に主力3連複が回収ライン未満だったか
            snap_note = ""
            trio_lines = [(c, st) for bt, c, st, _p in lines if bt == "3連複"]
            if rid in snap and trio_lines:
                comb, lstake = trio_lines[0]
                o = snap[rid].get(comb)
                if o and o * lstake < 1000:
                    snap_note = f"購入時点で市場は順当視(的中3連複は{o:.1f}倍)"
            s = score_of(h["stake"], h["ret"])
            tag, memo = classify(lines, h["stake"], h["ret"], snap_note)
            rows.append({"date": h["date"], "venue": h["venue"], "race_no": h["race_no"],
                         "chaku": h.get("chaku"), "lines": lines,
                         "stake": h["stake"], "ret": h["ret"],
                         "score": s, "tag": tag, "memo": memo})
            total["hits"] += 1
            total["score_sum"] += s
            total["perfect"] += 1 if s >= 100 else 0
    rows.sort(key=lambda r: (r["date"], r["venue"], r["race_no"]), reverse=True)
    return rows, total


def render(rows: list[dict], total: dict) -> str:
    avg = total["score_sum"] / total["hits"] if total["hits"] else 0
    body_rows = []
    for r in rows:
        tickets = " / ".join(f"{bt}{comb}({st}円→{pay:,}円)" for bt, comb, st, pay in r["lines"])
        cls = "perfect" if r["score"] >= 100 else ("solo" if "C単独" in r["tag"] else "")
        body_rows.append(
            f"<tr class='{cls}'><td class='num'>{r['score']}</td>"
            f"<td>{r['date'][5:]}<br>{r['venue']}{r['race_no']}R</td>"
            f"<td class='num'>{r['chaku']}</td>"
            f"<td>{tickets}<br><span class='memo'>{r['memo']}</span></td></tr>")

    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>評価点と学びのログ</title>
<style>
  body {{ font-family: sans-serif; margin: 0 auto; padding: 10px; background: #f6f8fa; max-width: 720px; }}
  h1 {{ font-size: 1.15rem; margin: 10px 4px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 12px; margin-bottom: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 5px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px 5px; border-bottom: 1px solid #eee; vertical-align: top; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  tr.perfect td:first-child {{ color: #1a7f37; font-weight: bold; }}
  tr:not(.perfect) td:first-child {{ color: #cf222e; font-weight: bold; }}
  tr.solo {{ background: #f3ecff66; }}
  .memo {{ font-size: .72rem; color: #57606a; }}
  .note {{ font-size: .75rem; color: #57606a; margin: 6px 2px; }}
  .nav a {{ text-decoration: none; font-size: .8rem; color: #0969da; }}
</style>
</head>
<body>
<h1>評価点と学びのログ(実購入=本命+超混戦の的中)</h1>
<p class="nav"><a href="index.html">← 今日の予想</a> / <a href="stats.html">通算成績</a> / <a href="review.html">実戦レビュー</a></p>
<p class="note">評価点=払戻率(100%以上は100点)。100点未満の1件ずつに「なぜか」を記録し、
選別・モデル改善の糧にする(2026-07-18ケンさん発案)。ガミ許容ラインは70%(市場レポートで監視)。
紫の行=3連単のみの的中(C勝万舟単独)。</p>

<div class="card">
  <h2 style="margin-top:0;font-size:.95rem">サマリー(実購入{PURCHASE_START}〜)</h2>
  <table>
    <tr><th>本命レース</th><th class="num">的中</th><th class="num">100点</th>
        <th class="num">的中の平均点</th></tr>
    <tr><td class="num">{total['races']}</td><td class="num">{total['hits']}</td>
        <td class="num">{total['perfect']}</td><td class="num">{avg:.0f}点</td></tr>
  </table>
</div>

<div class="card">
  <h2 style="margin-top:0;font-size:.95rem">的中レースの採点(新しい順)</h2>
  <table>
    <tr><th class="num">点</th><th>レース</th><th class="num">決着</th><th>的中券と学び</th></tr>
    {''.join(body_rows)}
  </table>
</div>

<p class="note">生成: {updated} / 更新は手動(py -X utf8 test/study_log.py)。
v2でワークフローへの組込みを検討。外れレースの敗因(軸飛び/ヒモ抜け)は市場レポート(e)参照。</p>
</body>
</html>
"""


if __name__ == "__main__":
    rows, total = build_rows()
    OUT_PATH.write_text(render(rows, total), encoding="utf-8")
    avg = total["score_sum"] / total["hits"] if total["hits"] else 0
    print(f"的中{total['hits']}R(100点{total['perfect']}) 平均{avg:.0f}点 -> {OUT_PATH}")
