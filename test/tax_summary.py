# -*- coding: utf-8 -*-
"""確定申告用の払戻・当たり舟券購入費の集計CLI(読み取り専用・本番不干渉)

    py -X utf8 test/tax_summary.py

docs/data/ledger.json(夜間採点の確定明細)から、暦年ごとに
- 払戻金合計
- 当たり舟券の購入費合計(一時所得で経費にできるのはここだけ。外れ舟券は不可)
- 参考: 総投資額・実収支
- 一時所得の試算((払戻 − 当たり券費 − 特別控除50万円) × 1/2)
を「実購入(本命+超混戦) / 要注目(観測) / 全レース(紙上参考)」の3スコープで集計する。

重要な注意(レポートにも明記):
- 申告の一次資料はテレボートの投票履歴(公式サイトからCSVで取得可能)。
  本レポートはシステム側の記録による補助計算であり、実際の購入と必ず突き合わせること
- 特別控除50万円と1/2は「その年の一時所得全体」に対して適用されるため、
  他に一時所得(懸賞・保険一時金等)がある場合は通算が必要
- 税額は給与等と合算した総合課税で決まる。最終判断は税理士に確認すること
- ledger.jsonへの書き込みは一切しない(読み取り専用)
"""
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import JST, PROJECT_DIR

TEST_DIR = PROJECT_DIR / "test"
DATA_DIR = PROJECT_DIR / "docs" / "data"

DEDUCTION = 500_000  # 一時所得の特別控除

# 実購入の開始日(2026-07-12ケンさん申告)。2026-07-07はシステム試運転日で
# 実際には購入していないため、申告集計から除外する。ledger(システム成績)は
# 7/7を含むが、税金は「実際に買った券」だけが対象。
PURCHASE_START = "2026-07-08"

# 的中行ラベル「3連複 1=2=5（200円）」から(券種, 買い目, 購入額)を取り出す
_LINE_RE = re.compile(r"^(\S+)\s+(\S+)（(\d+)円）$")

# v2(2026-07-18〜): 実購入=本命+超混戦。要注目(旧・準)は観測専用で購入なし
SCOPES = {
    "実購入(本命+超混戦)": ["ken_hon", "ken_konsen"],
    "要注目(観測・参考)": ["ken_jun"],
    "全レース(紙上・参考)": ["ken"],
}


def parse_line(label: str) -> tuple[str, str, int] | None:
    """的中行ラベル -> (券種, 買い目, 購入額円)。形式が違えばNone"""
    m = _LINE_RE.match(label)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def collect_hits(ledger: list[dict], keys: list[str]) -> list[dict]:
    """指定スコープの的中明細を行単位に展開する"""
    rows = []
    for day in ledger:
        for key in keys:
            for h in day.get("hits", {}).get(key, []):
                for line in h.get("lines", []):
                    parsed = parse_line(line["label"])
                    if parsed is None:
                        # 想定外形式は購入額0で載せて人間の確認対象にする(黙って落とさない)
                        rows.append({"date": h["date"], "venue": h["venue"],
                                     "race_no": h["race_no"], "chaku": h.get("chaku"),
                                     "bt": "?", "comb": line["label"], "line_stake": 0,
                                     "line_payout": line["payout"], "scope_key": key})
                        continue
                    bt, comb, stake = parsed
                    rows.append({"date": h["date"], "venue": h["venue"],
                                 "race_no": h["race_no"], "chaku": h.get("chaku"),
                                 "bt": bt, "comb": comb, "line_stake": stake,
                                 "line_payout": line["payout"], "scope_key": key})
    return rows


def scope_stats(ledger: list[dict], keys: list[str]) -> dict[str, dict]:
    """日別statsから総投資・総回収を年・月に積み上げる {期間キー: {stake, ret}}"""
    agg: dict[str, dict] = defaultdict(lambda: {"stake": 0, "ret": 0, "races": 0, "hits": 0})
    for day in ledger:
        for key in keys:
            s = day.get("stats", {}).get(key)
            if not s:
                continue
            for period in (day["date"][:4], day["date"][:7]):
                a = agg[period]
                a["stake"] += s["stake"]
                a["ret"] += s["ret"]
                a["races"] += s["races"]
                a["hits"] += s["hits"]
    return agg


def ichiji_shotoku(payout: int, winning_cost: int) -> tuple[int, int]:
    """(一時所得, 課税対象額=1/2後)。マイナスは0(他の所得と損益通算不可)"""
    base = max(0, payout - winning_cost - DEDUCTION)
    return base, base // 2


def missing_grade_dates(ledger: list[dict]) -> list[str]:
    """picksが存在するのにledgerに採点が無い日(=集計漏れの恐れ)を検出する"""
    graded = {d["date"] for d in ledger}
    pick_dates = {p.stem.replace("picks_", "") for p in DATA_DIR.glob("picks_*.json")}
    today = datetime.now(JST).strftime("%Y-%m-%d")
    return sorted(d for d in pick_dates - graded if d < today)  # 当日はまだ採点前なので除く


def main():
    ledger_path = DATA_DIR / "ledger.json"
    if not ledger_path.exists():
        print("ledger.jsonが見つかりません(採点がまだ動いていない環境)")
        return
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    # 実購入開始日より前(試運転期間)は申告対象外
    excluded = [d["date"] for d in ledger if d["date"] < PURCHASE_START]
    ledger = [d for d in ledger if d["date"] >= PURCHASE_START]
    if not ledger:
        print("実購入開始日以降の採点記録がまだありません")
        return
    years = sorted({d["date"][:4] for d in ledger})
    missing = [d for d in missing_grade_dates(ledger) if d >= PURCHASE_START]

    for year in years:
        year_ledger = [d for d in ledger if d["date"].startswith(year)]
        report = {"year": year, "scopes": {}, "missing": missing,
                  "excluded": [d for d in excluded if d.startswith(year)],
                  "period": f"{year_ledger[0]['date']} 〜 {year_ledger[-1]['date']}",
                  "updated": datetime.now(JST).strftime("%Y-%m-%d %H:%M")}

        for scope_name, keys in SCOPES.items():
            hits = [h for h in collect_hits(year_ledger, keys)]
            stats = scope_stats(year_ledger, keys)
            payout = sum(h["line_payout"] for h in hits)
            wcost = sum(h["line_stake"] for h in hits)
            base, taxable = ichiji_shotoku(payout, wcost)
            months = sorted(k for k in stats if len(k) == 7)
            report["scopes"][scope_name] = {
                "payout": payout, "winning_cost": wcost,
                "stake": stats.get(year, {}).get("stake", 0),
                "ret": stats.get(year, {}).get("ret", 0),
                "ichiji": base, "taxable": taxable,
                "monthly": {m: stats[m] for m in months},
                "hits": hits,
            }

        # 明細CSV(税理士・申告書作成の根拠資料)。
        # 実購入スコープの的中明細(v2: 本命+超混戦。旧期間はken_konsenが無いため自然に本命のみ)
        csv_path = TEST_DIR / f"tax_details_{year}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["日付", "場", "レース", "区分", "決着", "券種", "買い目",
                        "当たり券購入額(経費側)", "払戻額"])
            label = {"ken_hon": "本命", "ken_konsen": "超混戦"}
            for h in collect_hits(year_ledger, ["ken_hon", "ken_konsen"]):
                w.writerow([h["date"], h["venue"], f"{h['race_no']}R",
                            label.get(h["scope_key"], h["scope_key"]), h["chaku"],
                            h["bt"], h["comb"], h["line_stake"], h["line_payout"]])

        html_path = TEST_DIR / f"tax_report_{year}.html"
        html_path.write_text(render_report(report), encoding="utf-8")
        s = report["scopes"]["実購入(本命+超混戦)"]
        print(f"{year}年(実購入): 払戻{s['payout']:,}円 / 当たり券費{s['winning_cost']:,}円 / "
              f"一時所得{s['ichiji']:,}円 / 課税対象{s['taxable']:,}円")
        print(f"出力: {html_path}")
        print(f"出力: {csv_path}")
    if missing:
        print(f"⚠ 採点漏れの疑いがある日: {', '.join(missing)}(レポートにも警告表示)")


def render_report(r: dict) -> str:
    scope_rows = []
    for name, s in r["scopes"].items():
        profit = s["ret"] - s["stake"]
        scope_rows.append(
            f"<tr{' class=adopt' if '実購入' in name else ''}>"
            f"<td>{name}</td>"
            f"<td class='num'>{s['payout']:,}円</td>"
            f"<td class='num'>{s['winning_cost']:,}円</td>"
            f"<td class='num'><b>{s['ichiji']:,}円</b></td>"
            f"<td class='num'><b>{s['taxable']:,}円</b></td>"
            f"<td class='num'>{s['stake']:,}円</td>"
            f"<td class='num {'pos' if profit >= 0 else 'neg'}'>{profit:+,}円</td></tr>")

    hon = r["scopes"]["実購入(本命+超混戦)"]
    month_rows = []
    for m, s in hon["monthly"].items():
        profit = s["ret"] - s["stake"]
        month_rows.append(
            f"<tr><td>{m}</td><td class='num'>{s['races']}</td>"
            f"<td class='num'>{s['hits']}</td>"
            f"<td class='num'>{s['stake']:,}円</td><td class='num'>{s['ret']:,}円</td>"
            f"<td class='num {'pos' if profit >= 0 else 'neg'}'>{profit:+,}円</td></tr>")

    missing_html = ""
    if r["missing"]:
        missing_html = (f'<div class="warn">⚠ <b>採点漏れの疑い:</b> {", ".join(r["missing"])} は'
                        f'予想(picks)があるのにledgerに採点記録がない。この日に購入していた場合、'
                        f'本集計から漏れている。夜間採点の手動実行で補完してから再集計すること。</div>')

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{r['year']}年 確定申告用集計(舟券払戻)</title>
<style>
  body {{ font-family: sans-serif; margin: 0 auto; padding: 12px; background: #f6f8fa; max-width: 860px; }}
  h1 {{ font-size: 1.2rem; margin: 10px 4px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin-bottom: 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 6px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #1a7f37; font-weight: bold; }}
  .neg {{ color: #cf222e; }}
  .adopt {{ background: #d6efff55; }}
  .note {{ font-size: .78rem; color: #57606a; margin: 6px 4px; }}
  .warn {{ background: #fff8c5; border: 1px solid #d4a72c66; border-radius: 8px;
           padding: 8px 12px; font-size: .85rem; margin-bottom: 14px; }}
</style>
</head>
<body>
<h1>{r['year']}年 確定申告用集計(舟券払戻・一時所得)</h1>
<p class="note">対象期間: {r['period']}(採点済みの実戦記録・実購入開始日{PURCHASE_START}以降のみ) /
生成: {r['updated']} / 再実行: py -X utf8 test/tax_summary.py(冪等・毎回全期間を再集計)</p>
{f'<p class="note">除外した試運転日(未購入): {", ".join(r["excluded"])}</p>' if r["excluded"] else ''}
{missing_html}

<div class="card">
  <h2 style="margin-top:0">年間サマリー(スコープ別)</h2>
  <table>
    <tr><th>スコープ</th><th class="num">①払戻合計</th><th class="num">②当たり券購入費</th>
        <th class="num">一時所得<br>(①-②-50万)</th><th class="num">課税対象<br>(×1/2)</th>
        <th class="num">参考:総投資</th><th class="num">参考:実収支</th></tr>
    {''.join(scope_rows)}
  </table>
  <p class="note"><b>実購入は「本命+超混戦」(v2・2026-07-18〜。それ以前は本命のみで数字は連続)</b>。
  申告にはこの行と明細CSVを使う。「要注目」「全レース」はシステムの紙上採点の参考値で
  申告対象ではない。一時所得がマイナスの場合は0(他の所得との損益通算は不可)。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">月別(実購入=本命+超混戦)</h2>
  <table>
    <tr><th>月</th><th class="num">レース数</th><th class="num">的中数</th>
        <th class="num">投資</th><th class="num">払戻</th><th class="num">実収支</th></tr>
    {''.join(month_rows)}
  </table>
</div>

<div class="card">
  <h2 style="margin-top:0">申告時の注意(必読)</h2>
  <ul style="font-size:.85rem">
    <li><b>一次資料はテレボートの投票履歴</b>(公式サイト→投票履歴照会でCSV取得可)。
        本レポートはシステム記録による補助計算のため、必ず突き合わせること</li>
    <li><b>外れ舟券は経費にならない</b>(一時所得)。経費にできるのは②当たり券の購入費のみ</li>
    <li>特別控除50万円と1/2は「その年の一時所得全体」への適用。懸賞・保険一時金など
        他の一時所得があれば通算して計算する</li>
    <li>給与所得者は「給与以外の所得20万円以下なら所得税の確定申告不要」の特例があるが、
        <b>住民税の申告は別途必要</b>。課税対象欄が20万円以下でも市区町村への申告を忘れない</li>
    <li>的中1行ごとの明細は tax_details_{r['year']}.csv(このフォルダ)。税理士にはこれを渡す</li>
    <li>本レポートは税務助言ではない。最終判断は税理士・税務署に確認すること</li>
  </ul>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
