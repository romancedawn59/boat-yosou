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
- 予想屋ken: 本命1,400円/超混戦2,000円のポートフォリオ(2026-08-04予算制)
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
    ATTENTION_CAP, DAILY_BUDGET, DB_PATH, HONMEI_CAP, HONMEI_PROB_MAX, HONMEI_UNIT,
    KONSEN_PROB_MAX, KONSEN_UNIT, MODEL_PATH, PAGES_URL, PROJECT_DIR,
    TARGET_VENUE_CODES, VENUE_COORDS, VENUE_NAMES, is_buyable, jst_today,
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
    """指定日の番組表がDBになければダウンロードして格納する(v2: 全場が予測対象)

    レース枠(races)は結果JSON側からも作られるため、出走表の実体である
    entriesの件数で判定する。races件数で判定すると、番組表が未公開のまま
    結果JSONだけ取れた日に「番組表あり」と誤判定して素通りし、
    特徴量が空のまま予測に進んでしまう。
    """
    def entry_count():
        return conn.execute(
            "SELECT COUNT(*) FROM entries e JOIN races r ON e.race_id = r.race_id "
            "WHERE r.date = ?", (d.isoformat(),),
        ).fetchone()[0]

    if entry_count():
        return True

    # 朝の収集時点で番組表が未公開(404)でも、予測時には公開済みのことがある
    paths = download_day(d)
    if paths["program"] is not None:
        raw = json.loads(paths["program"].read_text(encoding="utf-8"))
    else:
        # 上流の当日公開遅延(2026-07-19/20/30・08-02で4回)への恒久フォールバック:
        # 公式サイトの出走表を直接パースする(約150リクエスト・5〜8分)。
        # PC不要=クラウド(Actions)でもこのまま発動する(2026-08-02ケンさん指示)
        try:
            from official_programs import fetch_official_programs
            raw = fetch_official_programs(d)
        except Exception as e:
            print(f"公式サイトフォールバック失敗: {e}")
            return False
        if not raw.get("programs"):
            return False

    program_data = parse_program(raw)
    for race in program_data["races"]:
        db.upsert_race(conn, race)
    for entry in program_data["entries"]:
        db.upsert_entry(conn, entry)
    conn.commit()
    return entry_count() > 0


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


def _rising_lanes(conn, d: date, race_ids: list[str]) -> dict[str, list[int]]:
    """★伸び盛り: 直近90日の実測2連対率が番組表の全国2連率を+10pt超上回る艇。

    表示専用(2026-07-29判断会・議題C採用)。購入ロジックには使わない。
    根拠: 単勝回収77-78%(横ばい59-65%)・3連対57%実測(test/verify_rising_racer.py)、
    超混戦帯では伸び盛り艇ありのレースが回収率226.9%vs189.9%
    (test/verify_rising_race_selection.py。48Rの小標本のため8月紙上追跡中)。
    リークなし: 当日より前の結果のみ参照。12走未満は判定しない。
    """
    rows = conn.execute(
        f"SELECT e.race_id, e.lane, e.reg_no, e.national_2rate FROM entries e "
        f"WHERE e.race_id IN ({','.join('?' * len(race_ids))})", race_ids,
    ).fetchall()
    regs = sorted({r[2] for r in rows if r[2] is not None})
    if not regs:
        return {}
    d0 = (d - timedelta(days=90)).isoformat()
    hist: dict[int, list[int]] = {}
    for reg, top2 in conn.execute(
        f"SELECT e.reg_no, (res.arrival_order <= 2) FROM entries e "
        f"JOIN races r ON r.race_id = e.race_id "
        f"JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane "
        f"WHERE e.reg_no IN ({','.join('?' * len(regs))}) "
        f"AND r.date >= ? AND r.date < ? AND res.arrival_order IS NOT NULL",
        [*regs, d0, d.isoformat()],
    ):
        hist.setdefault(reg, []).append(top2)
    out: dict[str, list[int]] = {}
    for rid, lane, reg, n2 in rows:
        h = hist.get(reg)
        if not h or len(h) < 12 or n2 is None:
            continue
        if sum(h) / len(h) - n2 / 100.0 > 0.10:
            out.setdefault(rid, []).append(int(lane))
    return out


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
    rising = _rising_lanes(conn, d, list(race_meta.keys()))
    conn.close()

    if df.empty:
        raise RuntimeError(
            f"{d}: 出走表の特徴量が0件です(races={len(race_meta)}件)。"
            "番組表が未取得のままレース枠だけが結果JSONから作られている可能性があります。"
        )

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
        if not ranked:
            # 上流の遅延公開時など、レース枠だけあって出走表特徴量が空のことがある
            continue
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
            "rising": rising.get(race_id, []),   # ★伸び盛り艇の枠番(表示専用)
            "picks_a": a,
            "picks_b": b,
            "picks_c": c,
            "bets": {"confidence": confidence, "plan": ken, "conf": ken_conf},
        })

    P.select_shobusho(races, honmei_venues=TARGET_VENUE_CODES,
                      honmei_cap=HONMEI_CAP, konsen_max=KONSEN_PROB_MAX,
                      attention_cap=ATTENTION_CAP, honmei_prob_max=HONMEI_PROB_MAX,
                      daily_budget=DAILY_BUDGET, konsen_unit=KONSEN_UNIT,
                      honmei_unit=HONMEI_UNIT)

    # 超混戦帯(1位生値20%未満)のレースを⑬構成へ差し替える。
    # 2026-08-04(検証⑮・GO): 5場で「本命」表示に吸われた20%未満のレースも
    # 帯が基準なので⑬を適用する(従来は本命構成1,000円のまま=適用漏れ。
    # 104RでROI・除き・ガミ率とも⑬優位: test/verify_konsen_absorbed_honmei.py)。
    # 表示ラベル(本命/超混戦)と勝負所カウントは従来どおり変えない
    for r in races:
        if not r["bets"]["plan"] or not r.get("ranked"):
            continue
        if r.get("shobusho") not in ("超混戦", "本命"):
            continue
        if r["ranked"][0]["prob"] >= KONSEN_PROB_MAX:
            continue
        probs = P.normalize_probs(r["ranked"])
        plan = P.ken_portfolio(r["bets"]["confidence"], r["ranked"], [],
                               P.picks_katsu(probs), konsen=True)
        if plan:
            r["bets"]["plan"] = plan
            r["bets"]["conf"] = [P.combo_prob(bt, comb, probs)
                                 for bt, comb, _y, _s in plan]
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
        lines.append(f"購入予算: {budget:,}円(本命1,400円/超混戦2,000円)")
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
  .rising { color: #bf5b04; font-size: .72rem; font-weight: bold; }
  .rising-note { font-size: .68rem; color: #8c959f; margin: 4px 0 0; }
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

# 古いページの検知と自動更新(2026-08-02ケンさん要望「買い目ページで日付更新」)。
# ページ生成日(PAGE_DATE)とJSTの今日を比べ、古ければバナー表示。
# さらに今日のpicks JSONが既に公開されていれば一度だけ自動リロードで最新に切り替える
# (sessionStorageでリロードループを防止)。すべて表示のみ・購入ロジック不変
_STALE_JS_TMPL = """
<script>
(function() {{
  const pageDate = "{page_date}";
  const now = new Date(Date.now() + 9 * 3600 * 1000);
  const today = now.getUTCFullYear() + "-" +
    String(now.getUTCMonth() + 1).padStart(2, "0") + "-" +
    String(now.getUTCDate()).padStart(2, "0");
  if (pageDate === today) return;
  fetch("data/picks_" + today + ".json?_=" + Date.now(), {{cache: "no-store"}})
    .then(r => {{
      if (r.ok && sessionStorage.getItem("reloaded-" + today) !== "1") {{
        sessionStorage.setItem("reloaded-" + today, "1");
        location.reload();
        return;
      }}
      const div = document.createElement("div");
      div.style.cssText = "position:sticky;top:0;z-index:99;background:#fff8c5;" +
        "border-bottom:2px solid #d4a72c;padding:10px 14px;font-size:.9rem;";
      div.innerHTML = r.ok
        ? "⚠️ この予想は" + pageDate + "のものです。<a href=''>タップで最新(" + today + ")に更新</a>"
        : "⚠️ この予想は" + pageDate + "のものです。本日分は未配信 " +
          "<a href='status.html'>🔄配信状況を確認</a>";
      document.body.prepend(div);
    }}).catch(() => {{}});
}})();
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
    # 更新チェック(docs/status.html は手書きの静的ページ。配信状況と上流の番組表
    # 公開状況をその場で確認できる。2026-08-02ケンさん要望・上流遅延5回目を受けて)
    links.append('<a href="status.html">🔄更新チェック</a>')
    return '<div class="nav">' + "".join(links) + "</div>"


def _summary_html(races: list[dict]) -> str:
    honmei, konsen, attention, budget, blocked = shobu_summary(races)
    parts = []
    if honmei:
        parts.append(f"🔴本命(5場・上位{HONMEI_CAP}): <b>{'、'.join(honmei)}</b>")
    if konsen:
        parts.append(f"🟣超混戦(全場・1位勝率{KONSEN_PROB_MAX:.0%}未満): <b>{'、'.join(konsen)}</b>")
    if honmei or konsen:
        parts.append(f"購入予算 {budget:,}円(本命1,400円/超混戦2,000円)")
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
    oc = view.get("odds_check")
    oc_html = ""
    if oc and not oc.get("validated", True):
        # 検証外スコープ(5場×30〜35帯以外)は記録用の控えめ表示のみ(2026-08-04〜)
        marks = {"chance": "○相当(ちょうど2点)", "chaos": "△相当(0〜1点)",
                 "cheap": "×相当(3点以上)"}
        oc_html = (f"<p style='background:#f6f8fa;border:1px solid #d0d7de;"
                   f"border-radius:8px;padding:8px 12px;font-size:.9rem;color:#57606a'>"
                   f"🔍一桁オッズ判定(記録用・検証外の帯): 3連複{oc['n_fuku']}点中、"
                   f"一桁オッズが{oc['singles']}点={marks.get(oc.get('verdict'), '-')}。"
                   f"検証済みの帯(5場×30〜35%)ではないため購入判断には使わない</p>")
    elif oc:
        v = oc.get("verdict")
        if v == "chance":
            oc_html = (f"<p style='background:#ddf4e4;border:1px solid #1a7f37;"
                       f"border-radius:8px;padding:8px 12px;font-size:.9rem'>"
                       f"🔍<b>要オッズ確認: ○購入チャンス(裁量)</b> — "
                       f"3連複{oc['n_fuku']}点中、一桁オッズがちょうど2点(本線堅く・ヒモに配当が乗る形)。"
                       f"検証値: 回収率129.4%・ガミ率17.1%(買う場合は裁量枠として報告を)</p>")
        elif v == "cheap":
            oc_html = (f"<p style='background:#ffebe9;border:1px solid #cf222e66;"
                       f"border-radius:8px;padding:8px 12px;font-size:.9rem'>"
                       f"🔍要オッズ確認: ×見送り — 一桁オッズが{oc['singles']}点"
                       f"(3点以上=全体が安いガミ地獄の形。検証値: 回収率82.5%・ガミ率43.7%)</p>")
        else:
            oc_html = (f"<p style='background:#fff8c5;border:1px solid #d4a72c66;"
                       f"border-radius:8px;padding:8px 12px;font-size:.9rem'>"
                       f"🔍要オッズ確認: △見送り — 一桁オッズが{oc['singles']}点"
                       f"(0〜1点=市場は総混沌と見ておりプランと形が不一致。検証値: 回収率53.4%)</p>")
    return f"""
      <div class='odds-view'>
        <p class='odds-meta'>オッズ取得: {view['fetched']} 時点(参考・成績対象外。朝の勝負所判定は変わりません)</p>
        {oc_html}
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
        # 20%未満の帯は本命表示でも⑬構成2,000円(2026-08-04・検証⑮)
        if any(src == "深い波乱" for _b, _c, _y, src in race["bets"]["plan"] or []):
            sho_html += "<span class='sho kon'>⑬適用(超混戦帯)</span>"
    elif shobusho == "超混戦":
        sho_html = "<span class='sho kon'>超混戦</span>"
    elif shobusho == "要注目":
        sho_html = "<span class='sho att'>要注目(観測)</span>"
        # 要オッズ確認(2026-08-02): 5場×30〜35%帯は昼のオッズタブで
        # 「3連複の一桁≤2点」判定が出る(購入は裁量・記録は裁量枠)
        if (race.get("venue_code") in VENUE_SLUGS and race.get("ranked")
                and 0.30 <= race["ranked"][0]["prob"] < 0.35):
            sho_html += ("<span class='sho' style='background:#1a7f37'>"
                         "🔍要オッズ確認</span>")
    if shobusho in ("本命", "超混戦") and not race.get("buyable", True):
        # 理想(推奨)ラベルは残し、買えないことだけ明示(理想と実際の分離)
        sho_html += "<span class='sho blk'>🚫購入不可</span>"

    rising = set(race.get("rising") or [])
    boat_rows = "".join(
        f"<tr><td class='lane l{b['lane']}'>{b['lane']}</td>"
        f"<td>{b['name']}{'<span class=rising> ★伸び盛り</span>' if b['lane'] in rising else ''}</td>"
        f"<td>{b['racer_class']}</td>"
        f"<td class='prob'>{b['prob']:.0%}</td></tr>"
        for b in race["ranked"]
    )
    rising_note = (
        "<p class='rising-note'>★伸び盛り=直近90日の実測2連対率が番組表を10pt超上回る選手"
        "(市場の値付けが古い可能性。表示のみ・買い目には未反映)</p>"
        if rising else ""
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
        # フォーメーション入力ガイド(2026-07-31ケンさん要望「購入を楽に」)。
        # 金額編集なしでプランを組める操作手順を表示する(表示のみ・購入ロジック不変)
        srcs = {src for _, _, _, src in ken_plan}
        lanes6 = [b["lane"] for b in race["ranked"]]
        guide = ""
        if "深い波乱" in srcs and len(lanes6) >= 5:      # 超混戦(⑬BOX+差され傾斜・2,000円)
            g1, g2, g3, g4, g5 = lanes6[:5]

            def tri(a, b, c):
                s = sorted([a, b, c])
                return f"{s[0]}={s[1]}={s[2]}"
            guide = (f"①3連単BOX [{g1},{g2},{g3}] 各100円 "
                     f"②3連単BOX [{g1},{g2},{g4}] 各100円 "
                     f"③3連単 {g3}-{g1}-{g2} に300円追加 "
                     f"④3連単 {g4}-{g1}-{g2} に300円追加 "
                     f"⑤3連複 {tri(g3, g4, g5)} 200円(金額編集なしの5操作)")
        elif "保険複" in srcs and len(lanes6) >= 4:      # 本命(⑰③案・1,400円)
            g1, g2, g3, g4 = lanes6[:4]
            guide = (f"①3連複F {g1}={g2}−[{g3},{g4}] 各200円 "
                     f"②3連複F {g3}={g4}−[{g1},{g2}] 各100円 "
                     f"③3連単F [{g3},{g4}]−{g1}−{g2} 各200円 "
                     f"④3連単F [{g3},{g4}]−{g2}−{g1} 各100円 "
                     f"⑤3連単 {g4}-{g2}-{g1} に200円追加(5操作)")
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
            + (f"<p class='ken-note'>📱入力ガイド: {guide}</p>" if guide else "")
            + f"<p class='ken-note'>自信=モデルが見た的中確率。想定配当=自信から逆算"
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
    {rising_note}
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
水色枠の予想屋kenが実際の購入プラン(本命1,400円/超混戦2,000円)。「本命勝負所」だけ買うのが検証済みの推奨運用。
確率はモデル予測値。購入は自己責任で。</p>
{body}
{_TAB_JS}
{_STALE_JS_TMPL.format(page_date=d.isoformat())}
</body>
</html>
"""


# 選択レースパネル(2026-08-09ケンさん要望「オッズ追っかけ用に場とレースを入れたら
# 買い目が出るシステム」)。静的サイトのままJSで実現: 公開済みのpicks JSONと
# odds_check JSON(あれば)を読んで任意レースのプランを表示する。
# 一桁オッズ判定テーブルの行クリックでも呼び出される。表示のみ・購入ロジック不変
_SELRACE_TMPL = """
<div class='card' id='selrace' style='border:2px solid #0969da'>
  <h2 style='margin:0 0 8px'>🎯 選択レース(オッズ追っかけ用)</h2>
  <p class='note'>場とレースを選ぶと買い目候補を表示します。下の一桁オッズ判定表の行クリックでも呼び出せます。
  勝負所以外は<b>購入対象外の参考プラン</b>です。</p>
  <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap'>
    <select id='selrace-v'></select>
    <select id='selrace-r'></select>
    <button onclick='selShow()' style='padding:6px 16px;cursor:pointer'>表示</button>
  </div>
  <div id='selrace-out' style='margin-top:10px'></div>
</div>
<script>
const SEL_DATE = "__DATE__";
const SEL_VENUES = __VENUES__;
const SEL_T5 = __T5__;
let _selPicks = null, _selOdds = null;
(function () {
  const v = document.getElementById("selrace-v");
  for (const [code, name] of Object.entries(SEL_VENUES))
    v.insertAdjacentHTML("beforeend", `<option value='${code}'>${name}</option>`);
  const r = document.getElementById("selrace-r");
  for (let i = 1; i <= 12; i++)
    r.insertAdjacentHTML("beforeend", `<option value='${i}'>${i}R</option>`);
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("button.kentobtn");
    if (btn) {                       // 買い目予想取得 → 購入検討タブへ倉庫入れ
      kentoAdd(+btn.dataset.v, +btn.dataset.r, btn);
      e.stopPropagation();
      return;
    }
    const row = e.target.closest("tr.ocrow");
    if (!row) return;
    v.value = row.dataset.v; r.value = row.dataset.r;
    selShow();
  });
  kentoRender();                     // 倉庫はlocalStorageに残る(リロード復元)
})();
async function _selLoad() {
  if (!_selPicks) {
    _selPicks = await (await fetch(`data/picks_${SEL_DATE}.json`)).json();
    try { _selOdds = await (await fetch(`data/odds_check_${SEL_DATE}.json`)).json(); }
    catch (e) { _selOdds = null; }
  }
}
function _selOddsFor(rid) {
  if (!_selOdds) return null;
  for (let i = _selOdds.snapshots.length - 1; i >= 0; i--) {
    const rec = _selOdds.snapshots[i].races.find((x) => x.race_id === rid);
    if (rec) return { fetched: _selOdds.snapshots[i].fetched, rec };
  }
  return null;
}
function _raceCard(v, rno, race) {
  const lanes = race.ranked.map((x) => x[0]);
  const ranked = race.ranked.map(([l, p]) => `${l}号艇 ${(p * 100).toFixed(0)}%`).join(" → ");
  const sho = race.shobusho
    ? `<span style='background:${race.shobusho === "本命" ? "#cf222e" : race.shobusho === "超混戦" ? "#8250df" : "#9a6700"};color:#fff;border-radius:6px;padding:2px 8px'>${race.shobusho}</span>`
    : "<span style='background:#57606a;color:#fff;border-radius:6px;padding:2px 8px'>勝負所外・購入対象外(参考)</span>";
  const oc = _selOddsFor(race.race_id);
  const oddsMap = [];
  if (oc && oc.rec.plan_odds) for (const [bt, comb, o] of oc.rec.plan_odds) oddsMap.push([bt, comb, o]);
  function popOdds(bt, comb) {
    const i = oddsMap.findIndex((x) => x[0] === bt && x[1] === comb);
    if (i < 0) return null;
    return oddsMap.splice(i, 1)[0][2];
  }
  let total = 0;
  const rows = race.ken.map(([bt, comb, yen, src]) => {
    total += yen;
    const o = popOdds(bt, comb);
    return `<tr><td>${src}</td><td>${bt}</td><td><b>${comb}</b></td><td style='text-align:right'>${yen}円</td><td style='text-align:right'>${o ? o.toFixed(1) + "倍" : "-"}</td></tr>`;
  }).join("");
  let guide = "";
  const srcs = new Set(race.ken.map((x) => x[3]));
  const tri = (a, b, c) => [a, b, c].sort().join("=");
  if (srcs.has("深い波乱") && lanes.length >= 5) {
    const [g1, g2, g3, g4, g5] = lanes;
    guide = `①3連単BOX [${g1},${g2},${g3}] 各100円 ②3連単BOX [${g1},${g2},${g4}] 各100円 ③3連単 ${g3}-${g1}-${g2} に300円追加 ④3連単 ${g4}-${g1}-${g2} に300円追加 ⑤3連複 ${tri(g3, g4, g5)} 200円`;
  } else if (srcs.has("保険複") && lanes.length >= 4) {
    const [g1, g2, g3, g4] = lanes;
    guide = `①3連複F ${g1}=${g2}−[${g3},${g4}] 各200円 ②3連複F ${g3}=${g4}−[${g1},${g2}] 各100円 ③3連単F [${g3},${g4}]−${g1}−${g2} 各200円 ④3連単F [${g3},${g4}]−${g2}−${g1} 各100円 ⑤3連単 ${g4}-${g2}-${g1} に200円追加`;
  }
  const valid5 = SEL_T5.includes(v) && race.ranked[0][1] >= 0.30 && race.ranked[0][1] < 0.35;
  const ocLine = oc && oc.rec.check
    ? `一桁オッズ判定(${oc.fetched}時点): 一桁${oc.rec.check.singles}点/${oc.rec.check.n_fuku}点 → ${oc.rec.check.verdict === "chance" ? "○ちょうど2点" : oc.rec.check.verdict === "chaos" ? "△0-1点(混沌)" : "×3点以上(安い)"}${valid5 ? " 🟢検証済み帯" : "(検証外帯・参考)"}`
    : "オッズ未取得(9:00/10:30/12:00の反映後に表示)";
  const jcd = String(v).padStart(2, "0");
  const hd = SEL_DATE.replaceAll("-", "");
  return `
    <div style='margin-bottom:6px'><b>${SEL_VENUES[v]} ${rno}R</b> ${sho}
      <button class='kentobtn' data-v='${v}' data-r='${rno}' style='margin-left:8px;cursor:pointer'>🛒検討に入れる</button></div>
    <div class='note'>モデル予測順位: ${ranked}</div>
    <div class='note'>${ocLine}</div>
    <div style='overflow-x:auto'><table class='odds-table'>
      <tr><th></th><th>券種</th><th>買い目</th><th>金額</th><th>最終取得オッズ</th></tr>${rows}
    </table></div>
    <div class='note'>計${total.toLocaleString()}円${guide ? " / 📱入力ガイド: " + guide : ""}</div>
    <div class='note'><a href='https://www.boatrace.jp/owpc/pc/race/odds3t?rno=${rno}&jcd=${jcd}&hd=${hd}' target='_blank'>公式サイトで今のオッズを見る→</a></div>`;
}
async function selShow() {
  const v = +document.getElementById("selrace-v").value;
  const rno = +document.getElementById("selrace-r").value;
  const out = document.getElementById("selrace-out");
  out.innerHTML = "読み込み中…";
  try { await _selLoad(); } catch (e) { out.innerHTML = "picksの読み込みに失敗しました"; return; }
  const race = _selPicks.races.find((x) => +x.venue_code === v && +x.race_no === rno);
  if (!race) { out.innerHTML = `${SEL_VENUES[v]}${rno}Rは本日の対象データにありません(非開催など)`; return; }
  out.innerHTML = _raceCard(v, rno, race);
  document.getElementById("selrace").scrollIntoView({ behavior: "smooth" });
}
// ---- 購入検討(倉庫)。localStorageに日付キーで保存しリロード後も残る ----
function _kentoKey() { return `kento_${SEL_DATE}`; }
function _kentoList() {
  try { return JSON.parse(localStorage.getItem(_kentoKey())) || []; }
  catch (e) { return []; }
}
function _kentoSave(list) {
  try { localStorage.setItem(_kentoKey(), JSON.stringify(list)); } catch (e) {}
}
async function kentoAdd(v, rno, btn) {
  const list = _kentoList();
  if (!list.some((x) => x[0] === v && x[1] === rno)) {
    list.push([v, rno]);
    _kentoSave(list);
  }
  if (btn) { btn.textContent = "✓検討中"; btn.disabled = true; }
  await kentoRender();
  document.getElementById("kento").scrollIntoView({ behavior: "smooth" });
}
async function kentoRemove(v, rno) {
  _kentoSave(_kentoList().filter((x) => !(x[0] === v && x[1] === rno)));
  await kentoRender();
}
async function kentoClear() { _kentoSave([]); await kentoRender(); }
async function kentoRender() {
  const box = document.getElementById("kento-list");
  const count = document.getElementById("kento-count");
  if (!box) return;
  const list = _kentoList();
  count.textContent = list.length;
  if (!list.length) {
    box.innerHTML = "<p class='note'>倉庫は空です。各表の「買い目取得」ボタンでここに追加されます。</p>";
    return;
  }
  try { await _selLoad(); } catch (e) { box.innerHTML = "picksの読み込みに失敗しました"; return; }
  box.innerHTML = list.map(([v, rno]) => {
    const race = _selPicks.races.find((x) => +x.venue_code === v && +x.race_no === rno);
    const inner = race ? _raceCard(v, rno, race)
      : `<b>${SEL_VENUES[v] || v} ${rno}R</b> データなし`;
    return `<div style='border:1px solid #d0d7de;border-radius:8px;padding:10px;margin-top:10px'>
      ${inner}
      <div style='margin-top:6px'><button onclick='kentoRemove(${v},${rno})' style='cursor:pointer'>倉庫から外す</button></div>
    </div>`;
  }).join("");
  document.querySelectorAll("#kento-list button.kentobtn").forEach((b) => b.remove());
}
</script>"""


_KENTO_SHELL = """
<div class='card' id='kento' style='border:2px solid #9a6700'>
  <h2 style='margin:0 0 8px'>🛒 購入検討(倉庫: <span id='kento-count'>0</span>件)</h2>
  <p class='note'>各表の「買い目取得」ボタンで気になるレースをここに貯められます(リロードしても残ります)。
  購入対象外レースの購入は裁量枠なので、買ったら報告を。</p>
  <div style='margin-bottom:4px'><button onclick='kentoClear()' style='cursor:pointer'>全部外す</button></div>
  <div id='kento-list'></div>
</div>"""


def _render_selrace_panel(d: date, races: list[dict]) -> str:
    venues = {int(r["venue_code"]): VENUE_NAMES[r["venue_code"]] for r in races}
    return (_SELRACE_TMPL
            .replace("__DATE__", d.isoformat())
            .replace("__VENUES__", json.dumps(venues, ensure_ascii=False))
            .replace("__T5__", json.dumps(sorted(TARGET_VENUE_CODES))))


def _render_oc_target_section(races: list[dict],
                              records: list[dict] | None) -> str:
    """要オッズ確認(5場×1位30〜35%)の追っかけ対象一覧(2026-08-09ケンさん要望)。

    朝から常設し「今日どのレースのオッズを追えばよいか」を一目で示す。
    昼のオッズ反映後は最新の一桁判定を併記。行クリックで選択レースパネルへ。
    """
    targets = [r for r in races
               if r.get("venue_code") in TARGET_VENUE_CODES and r.get("ranked")
               and 0.30 <= r["ranked"][0]["prob"] < 0.35]
    head = "🔍 要オッズ確認・オッズ追っかけ対象(5場×1位30〜35%)"
    if not targets:
        return (f"<div class='card'><b>{head}</b>"
                "<p class='note'>本日の対象レースはありません。</p></div>")
    by_id = {rec["race_id"]: rec for rec in (records or [])}
    rows = []
    for r in sorted(targets, key=lambda x: x["deadline"] or "9999"):
        rec = by_id.get(r["race_id"])
        oc = rec.get("check") if rec else None
        if oc:
            v = oc["verdict"]
            label = ("🟢 ○購入チャンス(一桁ちょうど2点)" if v == "chance"
                     else f"△見送り(一桁{oc['singles']}点=混沌)" if v == "chaos"
                     else f"×見送り(一桁{oc['singles']}点=安い)")
            label += f" [{rec.get('fetched', '')}時点]"
            style = ("background:#ddf4e4;font-weight:bold"
                     if v == "chance" else "")
        else:
            label = "判定待ち(オッズ反映は9:00/10:30/12:00ごろ)"
            style = "color:#57606a"
        deadline = (r["deadline"] or "")[-8:-3]
        rows.append(
            f"<tr class='ocrow' data-v='{int(r['venue_code'])}' "
            f"data-r='{int(r['race_no'])}' style='cursor:pointer;{style}'>"
            f"<td>{deadline}</td>"
            f"<td>{VENUE_NAMES[r['venue_code']]}{r['race_no']}R</td>"
            f"<td class='num'>{r['ranked'][0]['prob']:.1%}</td>"
            f"<td>{label}</td>"
            f"<td><button class='kentobtn' data-v='{int(r['venue_code'])}' "
            f"data-r='{int(r['race_no'])}' style='cursor:pointer'>"
            f"買い目取得</button></td></tr>")
    return f"""
<div class='card' style='border:2px solid #1a7f37'>
  <b>{head}</b>
  <p class='note'>🟢○(3連複の一桁オッズがちょうど2点)が出たら裁量チャンス(検証値129.4%・買ったら報告を)。
  △/×は見送り。行をクリックすると下の選択レースに買い目候補が出ます。</p>
  <div style='overflow-x:auto'><table class='odds-table'>
  <tr><th>締切</th><th>レース</th><th>1位確率</th><th>一桁オッズ判定</th><th>買い目予想</th></tr>
  {''.join(rows)}
  </table></div>
</div>"""


def _render_odds_check_section(records: list[dict]) -> str:
    """全レース一桁オッズ判定テーブル(2026-08-04〜テスト運用・noon実行時のみ)。

    判定=プラン3連複のうち一桁オッズ(10倍未満)の点数。検証済みの帯
    (5場×30〜35%)の行だけ🟢で強調し、それ以外は記録・観測用。
    """
    if not records:
        return ""
    labels = {"chance": "○ ちょうど2点", "chaos": "△ 0〜1点(混沌)",
              "cheap": "× 3点以上(安い)"}
    counts = {"chance": 0, "chaos": 0, "cheap": 0}
    rows = []
    for rec in records:
        oc = rec.get("check")
        deadline = (rec.get("deadline") or "")[-8:-3]
        p1 = rec.get("p1")
        p1_txt = f"{p1:.0%}" if p1 is not None else "-"
        venue = VENUE_NAMES.get(rec["venue_code"], str(rec["venue_code"]))
        if oc is None:
            label, style = "判定不能(3連複オッズ未形成)", "color:#57606a"
        else:
            counts[oc["verdict"]] = counts.get(oc["verdict"], 0) + 1
            label = f"{labels.get(oc['verdict'], '-')} [{oc['singles']}/{oc['n_fuku']}]"
            style = ("background:#ddf4e4;font-weight:bold" if oc.get("validated")
                     else "")
        badge = "🟢検証済み帯 " if oc and oc.get("validated") else ""
        sho = rec.get("shobusho") or ""
        rows.append(
            f"<tr class='ocrow' data-v='{int(rec['venue_code'])}' "
            f"data-r='{int(rec['race_no'])}' style='cursor:pointer;{style}' "
            f"title='クリックで選択レースに買い目を表示'>"
            f"<td>{deadline}</td><td>{venue}</td>"
            f"<td>{rec['race_no']}R</td><td class='num'>{p1_txt}</td>"
            f"<td>{badge}{label}</td><td>{sho}</td>"
            f"<td><button class='kentobtn' data-v='{int(rec['venue_code'])}' "
            f"data-r='{int(rec['race_no'])}' style='cursor:pointer'>"
            f"買い目取得</button></td></tr>")
    summary = (f"○{counts['chance']} / △{counts['chaos']} / ×{counts['cheap']}"
               f"(全{len(records)}レース)")
    return f"""
<details class='card' style='margin-top:24px'>
<summary style='cursor:pointer'><b>🔍 一桁オッズ判定・全レース記録(テスト運用)</b> {summary}</summary>
<p class='note'>プラン3連複のうち一桁オッズ(10倍未満)の点数による判定。
検証済みは<b>5場×30〜35%帯の「○ちょうど2点」(回収率129.4%)</b>のみ=🟢行。
それ以外の行は帯別の追検証用の記録で、購入判断には使わない。
オッズは各レース最後に取得した時点の値。</p>
<div style='overflow-x:auto'><table class='odds-table'>
<tr><th>締切</th><th>場</th><th>R</th><th>1位%</th><th>判定</th><th>区分</th><th>買い目予想</th></tr>
{''.join(rows)}
</table></div>
</details>"""


def render_shopping_page(d: date, races: list[dict],
                         odds_panes: dict[str, str] | None = None,
                         odds_check_records: list[dict] | None = None) -> str:
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
{_render_oc_target_section(races, odds_check_records)}
{_KENTO_SHELL}
{body}
{_render_selrace_panel(d, races)}
{_render_odds_check_section(odds_check_records or [])}
{_TAB_JS}
{_STALE_JS_TMPL.format(page_date=d.isoformat())}
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
                "rising": r.get("rising") or [],
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
