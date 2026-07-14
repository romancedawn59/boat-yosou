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
from datetime import date, datetime
from pathlib import Path

import db
from config import DB_PATH, JST, PROJECT_DIR, VENUE_NAMES, jst_today

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


def _chaku(payout: dict) -> str:
    """払戻の3連単キーからそのレースの決着(1着-2着-3着)を拾う"""
    for bt, comb in payout:
        if bt == "3連単":
            return comb
    return "?"


def collect_hits(picks: dict, conn, day_iso: str) -> dict:
    """1日分のpicksから、各予想者が「的中した(払戻があった)」レースの明細を集める。

    返り値: {predictor_key: [ {date,venue,race_no,chaku,stake,ret,lines:[{label,payout}]} ]}
    A/B/Cは各点100円換算、kenはポートフォリオ実額で払戻を計算する。
    """
    hits = {k: [] for k in PREDICTOR_LABELS}

    for race in picks["races"]:
        rid = race["race_id"]
        payout = {(bt, comb): amt for bt, comb, amt in conn.execute(
            "SELECT bet_type, combination, amount_yen FROM payouts WHERE race_id = ?", (rid,))}
        if not payout:
            continue
        base = {
            "date": day_iso,
            "venue": VENUE_NAMES.get(race["venue_code"], str(race["venue_code"])),
            "race_no": race["race_no"],
            "chaku": _chaku(payout),
        }

        for key in ("a", "b", "c"):
            lines, ret = [], 0
            for bt, comb, _p in race[key]:
                amt = payout.get((bt, comb), 0)
                if amt:
                    lines.append({"label": f"{bt} {comb}", "payout": amt})
                    ret += amt
            if ret:
                hits[key].append({**base, "stake": 100 * len(race[key]), "ret": ret, "lines": lines})

        ken = race["ken"]
        if ken:
            stake = sum(y for _, _, y, _ in ken)
            lines, ret = [], 0
            for bt, comb, y, _ in ken:
                amt = payout.get((bt, comb), 0)
                if amt:
                    r = amt * y // 100
                    lines.append({"label": f"{bt} {comb}（{y}円）", "payout": r})
                    ret += r
            if ret:
                detail = {**base, "stake": stake, "ret": ret, "lines": lines}
                hits["ken"].append(detail)
                sub = ("ken_hon" if race["shobusho"] == "本命"
                       else "ken_jun" if race["shobusho"] == "準" else None)
                if sub:
                    hits[sub].append(detail)

    return hits


def load_ledger() -> list:
    path = DATA_DIR / "ledger.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_ledger(ledger: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")


# ビューワーに表示する日数(3日+今日=4日)。ledger.jsonのログ自体は全期間保存する
SHOW_DAYS = 4


def render_stats(ledger: list) -> str:
    # 表示専用の生成時刻(JST)。採点結果そのものには影響しない
    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    totals = {k: _zero() for k in PREDICTOR_LABELS}
    for entry in ledger:
        for k, s in entry["stats"].items():
            if k not in totals:
                continue
            for f in ("stake", "ret", "races", "hits"):
                totals[k][f] += s[f]

    # 表示対象の日付(新しい順に最大SHOW_DAYS日)。通算成績の合計は全期間のまま
    recent = set(sorted({e["date"] for e in ledger}, reverse=True)[:SHOW_DAYS])

    # 各予想者の「的中(払戻あり)レース」明細を表示対象日ぶん集約(新しい順)
    all_hits = {k: [] for k in PREDICTOR_LABELS}
    for entry in ledger:
        if entry["date"] not in recent:
            continue
        for k, lst in entry.get("hits", {}).items():
            if k in all_hits:
                all_hits[k].extend(lst)
    for k in all_hits:
        all_hits[k].sort(key=lambda h: (h["date"], h["venue"], h["race_no"]), reverse=True)

    # 予想者ごとの「的中があった日」一覧(その日の損益・的中レース数つき、新しい順)。
    # 損益はその日そのレースだけでなく、その予想者のその日全体(ret-stake)。
    day_pnl = {k: {} for k in PREDICTOR_LABELS}
    for entry in ledger:
        for k, s in entry.get("stats", {}).items():
            if k in day_pnl:
                day_pnl[k][entry["date"]] = s["ret"] - s["stake"]
    days = {k: [] for k in PREDICTOR_LABELS}
    for k in PREDICTOR_LABELS:
        counts = {}
        for h in all_hits[k]:
            counts[h["date"]] = counts.get(h["date"], 0) + 1
        days[k] = sorted(
            ({"date": d, "n": n, "pnl": day_pnl[k].get(d, 0)} for d, n in counts.items()),
            key=lambda x: x["date"], reverse=True)

    def row(key, label, s, highlight=False):
        if s["races"] == 0:
            return ""
        roi = s["ret"] / s["stake"] if s["stake"] else 0
        n_hit = len(days.get(key, []))
        cls = "row hl" if highlight else "row"
        color = "pos" if s["ret"] >= s["stake"] else "neg"
        arrow = f"<span class='arrow'>›</span>" if n_hit else ""
        return (f"<tr class='{cls}' data-key='{key}'><td>{label}{arrow}</td>"
                f"<td class='num'>{s['races']:,}</td>"
                f"<td class='num'>{s['hits'] / s['races']:.1%}</td>"
                f"<td class='num {color}'>{roi:.1%}</td>"
                f"<td class='num {color}'>{s['ret'] - s['stake']:+,}円</td></tr>")

    total_rows = "".join(
        row(k, PREDICTOR_LABELS[k], totals[k], highlight=k.startswith("ken"))
        for k in PREDICTOR_LABELS
    )

    hits_json = json.dumps(all_hits, ensure_ascii=False)
    days_json = json.dumps(days, ensure_ascii=False)
    labels_json = json.dumps(PREDICTOR_LABELS, ensure_ascii=False)

    daily_rows = []
    for entry in sorted(ledger, key=lambda e: e["date"], reverse=True)[:SHOW_DAYS]:
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
  tr.row {{ cursor: pointer; transition: background .12s; }}
  tr.row:hover td {{ background: #eef4ff; }}
  tr.row.active td {{ background: #cfe6ff; }}
  .arrow {{ color: #0969da; font-weight: bold; margin-left: 5px; }}
  .note {{ font-size: .75rem; color: #57606a; margin: 6px 4px; }}
  #hits-card {{ display: none; }}
  #hits-card h2 {{ display: flex; justify-content: space-between; align-items: center; }}
  #hits-close {{ font-size: .8rem; color: #0969da; cursor: pointer; padding: 2px 8px;
                border: 1px solid #d0d7de; border-radius: 12px; background: #fff; }}
  .hit-lines {{ font-size: .8rem; line-height: 1.45; }}
  .hit-lines b {{ color: #1a7f37; }}
  tr.day-row {{ cursor: pointer; }}
  tr.day-row:hover td {{ background: #eef4ff; }}
  .back {{ display: inline-block; color: #0969da; cursor: pointer; font-size: .85rem;
          margin-bottom: 10px; }}
  .back:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>通算成績</h1>
<p class="note">最終更新: {updated}(自動採点は毎晩23時台)</p>
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
  水色行がken。推奨運用は「ken 本命勝負所」のみ購入。
  <b>予想者の行をタップ→日付→その日の当たったレースの順で、的中履歴が見られます</b>
  (表示は直近{SHOW_DAYS}日分。記録自体は全期間保存)。</p>
</div>
<div class="card" id="hits-card">
  <h2 style="margin-top:0"><span id="hits-title"></span><span id="hits-close">閉じる</span></h2>
  <div id="hits-body"></div>
</div>
<div class="card">
  <h2 style="margin-top:0">日別(直近{SHOW_DAYS}日)</h2>
  <table>
    <tr><th>日付</th><th class="num">本命勝負所</th><th class="num">本命損益</th>
        <th class="num">全レース</th><th class="num">全レース損益</th></tr>
    {''.join(daily_rows)}
  </table>
</div>
<p style="text-align:center;margin:16px 0">
  <a href="https://github.com/romancedawn59/boat-yosou/actions/workflows/grade.yml"
     style="display:inline-block;background:#1a7f37;color:#fff;font-weight:bold;
            padding:10px 22px;border-radius:20px;text-decoration:none;font-size:.9rem">
    ▶ 手動で採点を更新(GitHubが開きます)</a>
</p>
<p class="note">上のボタン→「Run workflow」→日付欄に採点したい日(YYYY-MM-DD、空なら今日)→緑のボタン。
1〜2分でこのページに反映されます。夜間の自動採点が日付をまたいで飛んだ日の復旧にも同じ手順を使えます。</p>
<p class="note">最終更新: {updated} / 毎晩23時台に自動更新。舟券の購入は自己責任で。</p>
<script>
const HITS = {hits_json};
const DAYS = {days_json};
const LABELS = {labels_json};
const yen = n => n.toLocaleString('ja-JP');
const signed = n => (n >= 0 ? '+' : '') + yen(n) + '円';
const card = document.getElementById('hits-card');
const titleEl = document.getElementById('hits-title');
const bodyEl = document.getElementById('hits-body');

// 第1階層: 予想者をタップ → 的中があった日付の一覧(その日の損益つき)
function showDays(key, scroll = true) {{
  document.querySelectorAll('tr.row').forEach(tr =>
    tr.classList.toggle('active', tr.dataset.key === key));
  const list = DAYS[key] || [];
  titleEl.textContent = LABELS[key] + ' の的中履歴';
  if (!list.length) {{
    bodyEl.innerHTML = '<p class="note">まだ当たったレースがありません。</p>';
  }} else {{
    const rows = list.map(d => {{
      const cls = d.pnl >= 0 ? 'pos' : 'neg';
      return '<tr class="day-row" data-key="' + key + '" data-date="' + d.date + '">'
        + '<td>' + d.date + '<span class="arrow">›</span></td>'
        + '<td class="num">' + d.n + 'レース的中</td>'
        + '<td class="num ' + cls + '">' + signed(d.pnl) + '</td></tr>';
    }}).join('');
    bodyEl.innerHTML = '<p class="note">日付をタップすると、その日の当たったレースが見られます。</p>'
      + '<table><tr><th>日付</th><th class="num">的中</th><th class="num">その日の損益</th></tr>'
      + rows + '</table>';
    bodyEl.querySelectorAll('tr.day-row').forEach(tr =>
      tr.addEventListener('click', () => showDayHits(tr.dataset.key, tr.dataset.date)));
  }}
  card.style.display = 'block';
  if (scroll) card.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
}}

// 第2階層: 日付をタップ → その日その予想者の当たったレース明細
function showDayHits(key, date) {{
  const list = (HITS[key] || []).filter(h => h.date === date);
  titleEl.textContent = LABELS[key] + '　' + date;
  const rows = list.map(h => {{
    const pnl = h.ret - h.stake;
    const cls = pnl >= 0 ? 'pos' : 'neg';
    const lines = h.lines.map(l => l.label + ' <b>' + yen(l.payout) + '円</b>').join('<br>');
    return '<tr><td>' + h.venue + h.race_no + 'R</td><td>' + h.chaku + '</td>'
      + '<td class="hit-lines">' + lines + '</td>'
      + '<td class="num ' + cls + '">' + signed(pnl) + '</td></tr>';
  }}).join('');
  bodyEl.innerHTML = '<div class="back" data-key="' + key + '">‹ 日付一覧へ戻る</div>'
    + '<table><tr><th>レース</th><th>決着</th>'
    + '<th>的中した買い目（払戻）</th><th class="num">損益</th></tr>' + rows + '</table>';
  bodyEl.querySelector('.back').addEventListener('click', () => showDays(key));
  card.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
}}

document.querySelectorAll('tr.row').forEach(tr =>
  tr.addEventListener('click', () => showDays(tr.dataset.key)));
document.getElementById('hits-close').addEventListener('click', () => {{
  card.style.display = 'none';
  document.querySelectorAll('tr.row').forEach(tr => tr.classList.remove('active'));
}});

// 初期表示: 推奨運用の「ken 本命勝負所」を開いておく(スクロールはしない)
if ((DAYS['ken_hon'] || []).length) showDays('ken_hon', false);
</script>
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
    hits = collect_hits(picks, conn, d.isoformat()) if day is not None else None
    conn.close()

    if day is None:
        print(f"{d}: 結果がまだ確定していないため採点をスキップ。")
        return

    ledger = [e for e in load_ledger() if e["date"] != d.isoformat()]
    ledger.append({"date": d.isoformat(), "stats": day, "hits": hits})
    ledger.sort(key=lambda e: e["date"])
    save_ledger(ledger)

    (SITE_DIR / "stats.html").write_text(render_stats(ledger), encoding="utf-8")

    ken = day["ken_hon"]
    print(f"{d}: 採点完了。本命勝負所 {ken['races']}レース 損益{ken['ret'] - ken['stake']:+,}円。"
          f"通算 {len(ledger)}日分をstats.htmlへ出力。")


if __name__ == "__main__":
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else jst_today()
    main(target)
