"""予測対象5場のレースを予測し、場別ページ(A/B/C予想+予想屋kenのポートフォリオ)を出力するCLI

    python predict.py             # 明日分
    python predict.py today       # 今日分
    python predict.py 2026-07-11  # 日付指定

出力(reports/site/): index.html(=平和島) / 各場ページ / data/picks_日付.json(採点用)
ワークフローがdocs/へコピーしてGitHub Pagesで公開する。

予想の構成(predictors.py):
- A 石橋渡: 堅い2連複・3連複を5点
- B 山田三連単: 発生確率上位の3連単を10点
- C 勝万舟: 万舟圏(発生確率0.5%以下)から確率上位5点
- 予想屋ken: 3人の案から1レース1,000円のポートフォリオ(C案を必ず100円以上含む)
- 勝負所: 荒れ注意=本命(検証済みエッジ)+標準から補充の準、最大10レース/日
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb

import db
import predictors as P
import weather
from config import (
    ATTENTION_CAP, DB_PATH, HONMEI_CAP, HONMEI_PROB_MAX, KONSEN_PROB_MAX, MODEL_PATH,
    PAGES_URL, PROJECT_DIR, TARGET_VENUE_CODES, VENUE_COORDS, VENUE_NAMES, is_buyable,
    jst_today,
)
from downloader import download_day
from features import FEATURE_COLUMNS, build_program_features
from parser_b import parse_program

SITE_DIR = PROJECT_DIR / "reports" / "site"

# 場コード -> ページファイル名(検証済み5場のみ個別ページを持つ)。
# v2(2026-07)からトップページ(index.html)は「本日の買い目一覧」。
# 他19場のレース(超混戦)は一覧ページにだけ載せ、場別ページは作らない
VENUE_SLUGS = {4: "heiwajima", 3: "edogawa", 8: "tokoname", 13: "amagasaki", 20: "wakamatsu"}


def _ensure_program(conn, d: date) -> bool:
    """指定日の番組表がDBになければダウンロードして格納する(v2: 全場が予測対象)"""
    def target_count():
        return conn.execute(
            "SELECT COUNT(*) FROM races WHERE date = ?", (d.isoformat(),),
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
    """レースIDごとのレース前予報(Open-Meteo)。表示専用(モデルには使わない)。
    座標登録のある場(検証済み5場)のみ取得し、他場は表示なし"""
    hourly_by_venue = {}
    for venue in {meta["venue_code"] for meta in race_meta.values()}:
        if venue not in VENUE_COORDS:
            continue
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
    """1日分・全24場の予測(v2)。開催がなければNone"""
    conn = db.connect(DB_PATH)
    if not _ensure_program(conn, d):
        conn.close()
        return None

    rows = conn.execute(
        "SELECT race_id, venue_code, race_no, deadline_time FROM races "
        "WHERE date = ? ORDER BY venue_code, race_no",
        (d.isoformat(),),
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
        probs = P.normalize_probs(ranked)
        confidence = P.bucket_of(ranked[0]["prob"])
        a = P.picks_ishibashi(probs) if len(probs) >= 4 else []
        b = P.picks_yamada(probs) if len(probs) >= 4 else []
        c = P.picks_katsu(probs) if len(probs) >= 4 else []
        ken = P.ken_portfolio(confidence, ranked, b, c)
        # 各点の自信ポイント(発生確率)。オッズを見ない設計のため、これが
        # 「この目はいくらつくか」の代替指標になる(較正確認済み)
        ken_conf = [P.combo_prob(bt, comb, probs) for bt, comb, _y, _s in ken]

        races.append({
            "race_id": race_id,
            "venue_code": meta["venue_code"],
            "venue_name": VENUE_NAMES[meta["venue_code"]],
            "race_no": meta["race_no"],
            "deadline": meta["deadline"],
            "buyable": is_buyable(meta["deadline"]),  # メンテ等の購入不可窓に締切があればFalse
            "weather": race_weather.get(race_id),
            "ranked": ranked,
            "picks_a": a,
            "picks_b": b,
            "picks_c": c,
            "bets": {"confidence": confidence, "plan": ken, "conf": ken_conf},
        })

    P.select_shobusho(races, honmei_venues=TARGET_VENUE_CODES,
                      honmei_cap=HONMEI_CAP, konsen_max=KONSEN_PROB_MAX,
                      attention_cap=ATTENTION_CAP, honmei_prob_max=HONMEI_PROB_MAX)
    return races


def shobu_summary(races: list[dict]) -> tuple[list[str], list[str], list[str], int, list[str]]:
    """(買える本命, 買える超混戦, 要注目, 購入予算円, 購入不可の本命・超混戦)。

    理想(システム推奨=本命/超混戦のラベル)は不変のまま、メンテ等で買えないレース
    (buyable=False)は予算と①②のリストから外し、別枠(blocked)で返す。
    「理想と実際の分離」: 採点側は理想全体と実際(買えた分)を両方記録する。
    """
    def label(r):
        return f"{r['venue_name']}{r['race_no']}R"

    def names(mark, buyable_only=True):
        return [label(r) for r in races
                if r.get("shobusho") == mark and (not buyable_only or r.get("buyable", True))]

    blocked = [label(r) for r in races
               if r.get("shobusho") in ("本命", "超混戦") and not r.get("buyable", True)]
    budget = sum(
        sum(y for _, _, y, _ in r["bets"]["plan"])
        for r in races
        if r.get("shobusho") in ("本命", "超混戦") and r.get("buyable", True)
    )
    return names("本命"), names("超混戦"), names("要注目", False), budget, blocked


def build_notify_text(d: date, races: list[dict]) -> str:
    """LINE通知(v2): ①本命 ②超混戦 ③購入予算。メンテ等で買えないレースがある日だけ
    「購入不可」行を追加して知らせる(買い間違い防止)"""
    honmei, konsen, attention, budget, blocked = shobu_summary(races)
    lines = [f"【競艇予想】{d}"]
    if honmei:
        lines.append(f"本命: {'、'.join(honmei)}")
    if konsen:
        lines.append(f"超混戦: {'、'.join(konsen)}")
    if honmei or konsen:
        lines.append(f"購入予算: {budget:,}円(1レース1,000円)")
    else:
        lines.append("本日は購入対象なし(全レース見送り推奨)")
    if blocked:
        lines.append(f"⚠メンテ等で購入不可: {'、'.join(blocked)}(買わないこと)")
    # 要注目は通知しない(2026-07-18ユーザー指示。観測枠はサイト下部のみ)
    lines.append("")
    lines.append(PAGES_URL)
    return "\n".join(lines)


_CONFIDENCE_COLORS = {"堅め": "#1a7f37", "標準": "#9a6700", "荒れ注意": "#cf222e"}

_CSS = """
  body { font-family: sans-serif; margin: 0; padding: 8px; background: #f6f8fa; }
  h1 { font-size: 1.15rem; margin: 8px 4px; }
  .nav { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 10px; }
  .nav a { text-decoration: none; font-size: .82rem; padding: 5px 10px; border-radius: 14px;
           background: #fff; color: #0969da; border: 1px solid #d0d7de; }
  .nav a.active { background: #0969da; color: #fff; border-color: #0969da; }
  .note { font-size: .75rem; color: #57606a; margin: 0 4px 12px; }
  .card { background: #fff; border-radius: 10px; padding: 12px; margin-bottom: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }
  .head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .rno { font-size: 1.3rem; font-weight: bold; }
  .deadline { color: #57606a; font-size: .85rem; }
  .conf { margin-left: auto; color: #fff; font-size: .75rem; padding: 3px 10px;
          border-radius: 12px; }
  .sho { font-size: .75rem; padding: 3px 10px; border-radius: 12px; color: #fff; font-weight: bold; }
  .sho.hon { background: #cf222e; }
  .sho.kon { background: #6f42c1; }
  .sho.att { background: #6e7781; }
  .sho.blk { background: #8250df; border: 2px solid #fff; }
  .sec-h { font-size: .95rem; margin: 14px 4px 8px; }
  .venue-tag { font-size: .8rem; color: #57606a; font-weight: bold; }
  table { width: 100%; border-collapse: collapse; font-size: .9rem; }
  td { padding: 4px 6px; border-bottom: 1px solid #eee; }
  .lane { width: 2em; text-align: center; font-weight: bold; border-radius: 4px; }
  .l1 { background: #fff; border: 1px solid #ccc; } .l2 { background: #222; color: #fff; }
  .l3 { background: #d32f2f; color: #fff; } .l4 { background: #1565c0; color: #fff; }
  .l5 { background: #fbc02d; } .l6 { background: #2e7d32; color: #fff; }
  .prob { text-align: right; font-weight: bold; }
  .weather { font-size: .78rem; color: #57606a; background: #f0f6ff; border-radius: 6px;
             padding: 5px 8px; margin-bottom: 8px; }
  .wx-note { display: block; font-size: .68rem; color: #8c959f; }
  .summary { background: #fff8c5; border: 1px solid #d4a72c66; border-radius: 8px;
             padding: 10px 12px; margin: 0 0 12px; font-size: .9rem; }
  .picks { margin-top: 8px; border-radius: 8px; padding: 8px 10px; background: #f6f8fa; }
  .picks h3 { margin: 0 0 4px; font-size: .8rem; }
  .picks .items { font-size: .88rem; line-height: 1.8; }
  .picks .p { color: #57606a; font-size: .75rem; }
  .ken { margin-top: 8px; background: #d6efff; border: 1px solid #54aeff88;
         border-radius: 8px; padding: 8px 10px; }
  .ken h3 { margin: 0 0 6px; font-size: .85rem; }
  .ken-table td { border: none; padding: 2px 6px; font-size: .95rem; }
  .ken-table .src { font-size: .7rem; color: #57606a; width: 5em; }
  .ken-table .bt { font-size: .8rem; color: #57606a; width: 4em; }
  .ken-table .yen { text-align: right; font-weight: bold; }
  .ken-table .cf { text-align: right; font-size: .8rem; color: #0969da; width: 3.5em; }
  .ken-table .io { text-align: right; font-size: .8rem; color: #57606a; width: 5em; }
  .ken-table th { font-size: .68rem; color: #8c959f; font-weight: normal; padding: 0 6px; }
  .ken-note { font-size: .68rem; color: #57606a; margin: 6px 0 0; }
  .tabs { display: flex; gap: 6px; margin-top: 10px; }
  .tabbtn { font-size: .8rem; padding: 5px 12px; border-radius: 14px 14px 0 0;
            border: 1px solid #d0d7de; border-bottom: none; background: #eef1f4;
            color: #57606a; cursor: pointer; }
  .tabbtn.active { background: #fff; color: #0969da; font-weight: bold;
                   border-color: #0969da; }
  .pane { display: none; }
  .pane.active { display: block; }
  .odds-view { margin-top: 6px; background: #fff8f0; border: 1px solid #bc4c0044;
               border-radius: 8px; padding: 8px 10px; }
  .odds-meta { font-size: .72rem; color: #bc4c00; margin: 0 0 6px; }
  .odds-table th { font-size: .72rem; background: #fff3e8; padding: 3px 6px; }
  .odds-table td { font-size: .88rem; padding: 3px 6px; border-bottom: 1px solid #f0e0d0; }
  .odds-table .num { text-align: right; font-variant-numeric: tabular-nums; }
  .odds-note { font-size: .68rem; color: #8c959f; margin: 6px 0 0; }
"""

_TAB_JS = """
<script>
function swTab(btn, paneId) {
  const card = btn.closest('.card');
  card.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
  card.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}
</script>
"""


def _nav_html(active_venue: int | None, venues_today: set[int]) -> str:
    """ナビ。v2からトップ=買い目一覧、5場は各自のページを持つ"""
    cls_top = ' class="active"' if active_venue is None else ""
    links = [f'<a href="index.html"{cls_top}>本日の買い目</a>']
    for venue, slug in VENUE_SLUGS.items():
        cls = " class=\"active\"" if venue == active_venue else ""
        mark = "" if venue in venues_today else "・休"
        links.append(f'<a href="{slug}.html"{cls}>{VENUE_NAMES[venue]}{mark}</a>')
    links.append('<a href="stats.html">通算成績</a>')
    return '<div class="nav">' + "".join(links) + "</div>"


def _summary_html(races: list[dict]) -> str:
    honmei, konsen, attention, budget, blocked = shobu_summary(races)
    parts = []
    if honmei:
        parts.append(f"🔴本命(5場・上位{HONMEI_CAP}): <b>{'、'.join(honmei)}</b>")
    if konsen:
        parts.append(f"🟣超混戦(全場・1位勝率{KONSEN_PROB_MAX:.0%}未満): <b>{'、'.join(konsen)}</b>")
    if honmei or konsen:
        parts.append(f"購入予算 {budget:,}円(1レース1,000円)")
    else:
        parts.append("本日は購入対象なし(全レース見送り推奨)。")
    if blocked:
        parts.append(f"🚫メンテ等で購入不可(買わないこと): {'、'.join(blocked)}")
    # 要注目はサマリーに載せない(ユーザー指示。ページ下部の観測セクションのみ)
    return '<div class="summary">' + "<br>".join(parts) + "</div>"


def _picks_html(title: str, picks: list[tuple[str, str, float]]) -> str:
    if not picks:
        return ""
    # 小数点以下3桁表示(2026-07-18ユーザー指示・表示のみの凍結例外)。
    # C勝万舟の閾値0.5%際で「0.50%に見えて実は0.495%」の情報が潰れるのを防ぐ
    items = " / ".join(
        f"{bt}{comb}<span class='p'>({p:.3%})</span>"
        for bt, comb, p in picks
    )
    return f"<div class='picks'><h3>{title}</h3><div class='items'>{items}</div></div>"


def _render_odds_pane(view: dict) -> str:
    """オッズ反映ペイン(12:00参考版)のHTML。viewはnoon_update.build_odds_viewの出力"""
    ken_rows = "".join(
        f"<tr><td class='bt'>{bt}</td><td>{comb}</td>"
        f"<td class='num'>{('%.1f' % o) + '倍' if o else '-'}</td>"
        f"<td class='num'>{est:,}円</td>"
        f"<td class='num'>{ev:.2f}</td></tr>"
        for bt, comb, o, est, ev in view["ken_rows"]
    )
    value_items = " / ".join(
        f"{bt}{comb}<span class='p'>({o:.1f}倍)</span>" for bt, comb, o in view["value"]
    ) or "なし"
    return f"""
      <div class='odds-view'>
        <p class='odds-meta'>オッズ取得: {view['fetched']} 時点(参考・成績対象外。朝の勝負所判定は変わりません)</p>
        <table class='odds-table'>
          <tr><th>券種</th><th>買い目</th><th>オッズ</th><th>想定払戻</th><th>EV※</th></tr>
          {ken_rows}
        </table>
        <p class='odds-note'>※EV=モデル確率×オッズ。1.00超はモデルが市場より強気の目。
        検証ではEVによる目の選別は逆効果だったため、判断材料の提示にとどめる。</p>
        <div class='picks'><h3>オッズ妙味(実験枠・未検証)</h3><div class='items'>{value_items}</div></div>
      </div>"""


def _render_race_card(race: dict, odds_pane: str | None = None,
                      show_venue: bool = False) -> str:
    deadline = (race["deadline"] or "")[-8:-3]
    conf = race["bets"]["confidence"]
    color = _CONFIDENCE_COLORS[conf]
    shobusho = race.get("shobusho")

    sho_html = ""
    if shobusho == "本命":
        sho_html = "<span class='sho hon'>本命</span>"
    elif shobusho == "超混戦":
        sho_html = "<span class='sho kon'>超混戦</span>"
    elif shobusho == "要注目":
        sho_html = "<span class='sho att'>要注目(観測)</span>"
    if shobusho in ("本命", "超混戦") and not race.get("buyable", True):
        # 理想(推奨)ラベルは残し、買えないことだけ明示(理想と実際の分離)
        sho_html += "<span class='sho blk'>🚫購入不可</span>"

    boat_rows = "".join(
        f"<tr><td class='lane l{b['lane']}'>{b['lane']}</td>"
        f"<td>{b['name']}</td><td>{b['racer_class']}</td>"
        f"<td class='prob'>{b['prob']:.0%}</td></tr>"
        for b in race["ranked"]
    )
    wx = race.get("weather")
    weather_html = (
        f"<div class='weather'>予報: 風速{wx['wind_speed_m']:.1f}m/s({wx['wind_dir']}の風) "
        f"波高目安{wx['wave_height_cm']:.1f}cm 気温{wx['temperature']:.0f}℃"
        f"<span class='wx-note'>※参考情報・予測には未使用</span></div>"
        if wx else ""
    )

    picks_html = (
        _picks_html("A 石橋渡(堅実・2連複/3連複)", race["picks_a"])
        + _picks_html("B 山田三連単(のびのび3連単)", race["picks_b"])
        + _picks_html("C 勝万舟(万舟圏・発生率順)", race["picks_c"])
    )

    ken_plan = race["bets"]["plan"]
    if ken_plan:
        total = sum(y for _, _, y, _ in ken_plan)
        # 自信ポイントと、そこから逆算した想定配当(オッズを見ない設計の代替指標)
        confs = race["bets"].get("conf") or [0.0] * len(ken_plan)
        ken_rows = "".join(
            f"<tr><td class='src'>{src}</td><td class='bt'>{bt}</td>"
            f"<td>{comb}</td><td class='yen'>{yen}円</td>"
            f"<td class='cf'>{p:.1%}</td>"
            f"<td class='io'>{('約' + format(P.implied_odds(p), ',.0f') + '倍') if p > 0 else '—'}</td></tr>"
            for (bt, comb, yen, src), p in zip(ken_plan, confs)
        )
        ken_html = (
            f"<div class='ken'><h3>予想屋ken のポートフォリオ(計{total:,}円)</h3>"
            f"<table class='ken-table'>"
            f"<tr><th></th><th></th><th></th><th class='yen'>金額</th>"
            f"<th class='cf'>自信</th><th class='io'>想定配当</th></tr>"
            f"{ken_rows}</table>"
            f"<p class='ken-note'>自信=モデルが見た的中確率。想定配当=自信から逆算"
            f"(オッズは見ない設計)。実際の配当は市場しだいで前後します</p></div>"
        )
    else:
        ken_html = ""

    morning_pane = picks_html + ken_html
    if odds_pane is None:
        body = morning_pane
    else:
        rid = race["race_id"]
        body = f"""
    <div class="tabs">
      <button class="tabbtn active" onclick="swTab(this,'m-{rid}')">朝の予想</button>
      <button class="tabbtn" onclick="swTab(this,'o-{rid}')">オッズ反映⏱</button>
    </div>
    <div id="m-{rid}" class="pane active">{morning_pane}</div>
    <div id="o-{rid}" class="pane">{odds_pane}</div>"""

    venue_html = f"<span class='venue-tag'>{race['venue_name']}</span>" if show_venue else ""
    return f"""
  <div class="card">
    <div class="head">
      {venue_html}<span class="rno">{race['race_no']}R</span>
      <span class="deadline">締切 {deadline}</span>
      {sho_html}
      <span class="conf" style="background:{color}">{conf}</span>
    </div>
    {weather_html}
    <table>{boat_rows}</table>
    {body}
  </div>"""


def render_venue_page(d: date, venue: int, races: list[dict],
                      odds_panes: dict[str, str] | None = None) -> str:
    venues_today = {r["venue_code"] for r in races}
    venue_races = [r for r in races if r["venue_code"] == venue]
    odds_panes = odds_panes or {}

    if venue_races:
        body = "".join(
            _render_race_card(r, odds_panes.get(r["race_id"])) for r in venue_races
        )
    else:
        body = '<div class="card">本日この場は非開催です。上のメニューから開催場をご覧ください。</div>'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{d} {VENUE_NAMES[venue]} 買い目予想</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{d} {VENUE_NAMES[venue]} 買い目予想</h1>
{_nav_html(venue, venues_today)}
{_summary_html(races)}
<p class="note">A/B/Cは3人の予想者の視点(購入額なし・通算成績は「通算成績」ページ)。
水色枠の予想屋kenが実際の購入プラン(1レース1,000円)。「本命勝負所」だけ買うのが検証済みの推奨運用。
確率はモデル予測値。購入は自己責任で。</p>
{body}
{_TAB_JS}
</body>
</html>
"""


def render_shopping_page(d: date, races: list[dict],
                         odds_panes: dict[str, str] | None = None) -> str:
    """トップページ「本日の買い目一覧」(v2)。区分ごとに締切時刻順で並べた買い物リスト"""
    odds_panes = odds_panes or {}
    venues_today = {r["venue_code"] for r in races}

    def section(title, mark):
        rs = sorted((r for r in races if r.get("shobusho") == mark),
                    key=lambda r: r["deadline"] or "9999")
        if not rs:
            return ""
        cards = "".join(
            _render_race_card(r, odds_panes.get(r["race_id"]), show_venue=True)
            for r in rs)
        return f"<h2 class='sec-h'>{title}</h2>{cards}"

    body = (section(f"🔴 本命(検証済み5場・上位{HONMEI_CAP})", "本命")
            + section(f"🟣 超混戦(全場・1位勝率{KONSEN_PROB_MAX:.0%}未満)", "超混戦")
            + section("👀 要注目(観測のみ・購入0点)", "要注目"))
    if not body:
        body = '<div class="card">本日は購入対象なし(全レース見送り推奨)。</div>'

    # 購入不可窓(メンテ等)で買い目から外れたレースがあれば注記
    maint = ""
    if any(r.get("buyable") is False for r in races):
        maint = ('<div class="summary" style="background:#ffe9e0;border-color:#cf222e55">'
                 '⚠ システムメンテナンス等で購入できない時間帯のレースは買い目から外し、'
                 '要注目(観測・購入0点)に回しています。</div>')

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{d} 本日の買い目</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{d} 本日の買い目</h1>
{_nav_html(None, venues_today)}
{_summary_html(races)}
{maint}
{body}
{_TAB_JS}
</body>
</html>
"""


def _picks_json(d: date, races: list[dict]) -> dict:
    return {
        "date": d.isoformat(),
        "races": [
            {
                "race_id": r["race_id"],
                "venue_code": r["venue_code"],
                "race_no": r["race_no"],
                "confidence": r["bets"]["confidence"],
                "shobusho": r.get("shobusho"),
                "buyable": r.get("buyable", True),
                # 予測順位と1位勝率(生値)。事後分析でモデルの見立てを復元するために残す
                # (2026-07-21まで未保存で、過去日の分析はwalk-forward再実行が必要だった)
                "ranked": [[r2["lane"], round(r2["prob"], 6)] for r2 in r["ranked"]],
                "a": [[bt, comb, p] for bt, comb, p in r["picks_a"]],
                "b": [[bt, comb, p] for bt, comb, p in r["picks_b"]],
                "c": [[bt, comb, p] for bt, comb, p in r["picks_c"]],
                "ken": [[bt, comb, yen, src] for bt, comb, yen, src in r["bets"]["plan"]],
                "ken_conf": [round(p, 6) for p in r["bets"].get("conf") or []],
            }
            for r in races
        ],
    }


def run(d: date) -> Path | None:
    races = predict_day(d)
    if races is None:
        print(f"{d}: 対象5場はすべて非開催(または番組表未公開)")
        return None

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "data").mkdir(exist_ok=True)

    for venue, slug in VENUE_SLUGS.items():
        html = render_venue_page(d, venue, races)
        (SITE_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    # トップページ=本日の買い目一覧(v2)
    (SITE_DIR / "index.html").write_text(
        render_shopping_page(d, races), encoding="utf-8")

    picks_path = SITE_DIR / "data" / f"picks_{d.isoformat()}.json"
    picks_path.write_text(
        json.dumps(_picks_json(d, races), ensure_ascii=False, indent=1), encoding="utf-8")

    notify_path = SITE_DIR / "data" / f"notify_{d.isoformat()}.txt"
    notify_path.write_text(build_notify_text(d, races), encoding="utf-8")

    venues = "、".join(sorted({r["venue_name"] for r in races}))
    print(f"{d}: {len(races)}レース({venues})のサイトを出力 -> {SITE_DIR}")
    return SITE_DIR


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"モデルが見つかりません: {MODEL_PATH}\n先に train_model.py を実行してください。")
        sys.exit(1)

    # クラウドランナーはUTCのためJSTで「今日」を判定する(date.today()はUTC日付になり1日ずれる)
    if len(sys.argv) > 1 and sys.argv[1] == "today":
        targets = [jst_today()]
    elif len(sys.argv) > 1:
        targets = [date.fromisoformat(sys.argv[1])]
    else:
        targets = [jst_today() + timedelta(days=1)]

    for target in targets:
        run(target)
