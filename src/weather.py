"""予測対象場ピンポイントの気象予報(Open-Meteo, 無料・APIキー不要・登録不要)

バックテストの結果、モデルの特徴量に加えても回収率が悪化した(features.py参照)ため
学習には使わない。レース前の最新の風・気温をレポートに表示する参考情報として使う。

風速・気温は数値でそのまま使えるが、波高はOpen-Meteoに項目がないため、
競走水面の波はほぼ風速だけで決まる(戸田で実測相関0.95)ことを利用し、
自前DBの当該場の実測データから風速→波高の回帰式をその都度算出して推定する。
"""
import json
import urllib.request
from datetime import datetime

from config import VENUE_COORDS

_URL_TMPL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=wind_speed_10m,wind_direction_10m,temperature_2m"
    "&wind_speed_unit=ms&timezone=Asia%2FTokyo&forecast_days=3"
)

_COMPASS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
            "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]


def compass_name(degrees: float) -> str:
    """風向(0-360度)を16方位の日本語名に変換する(表示用)"""
    idx = round(degrees / 22.5) % 16
    return _COMPASS[idx]


def fetch_hourly(venue_code: int) -> dict[str, tuple[float, float, float]]:
    """指定場の {'YYYY-MM-DDTHH:00': (風速m/s, 風向度, 気温℃)} を返す。3日先まで。"""
    lat, lon = VENUE_COORDS[venue_code]
    url = _URL_TMPL.format(lat=lat, lon=lon)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    h = data["hourly"]
    return {
        t: (ws, wd, temp)
        for t, ws, wd, temp in zip(
            h["time"], h["wind_speed_10m"], h["wind_direction_10m"], h["temperature_2m"]
        )
    }


def lookup(hourly: dict[str, tuple[float, float, float]], deadline_time: str) -> tuple[float, float, float] | None:
    """締切時刻(例 '2026-07-05 10:47:00')に最も近い時間帯の(風速, 風向度, 気温)を返す"""
    dt = datetime.strptime(deadline_time, "%Y-%m-%d %H:%M:%S")
    key = dt.replace(minute=0, second=0).strftime("%Y-%m-%dT%H:00")
    return hourly.get(key)


def estimate_wave_height_cm(conn, venue_code: int, wind_speed_m: float) -> float:
    """指定venueの実測データから風速→波高の回帰係数をその都度算出して推定する"""
    rows = conn.execute(
        "SELECT wind_speed_m, wave_height_cm FROM races "
        "WHERE venue_code = ? AND wind_speed_m IS NOT NULL AND wave_height_cm IS NOT NULL",
        (venue_code,),
    ).fetchall()
    n = len(rows)
    if n < 30:
        return 0.0

    mx = sum(r[0] for r in rows) / n
    my = sum(r[1] for r in rows) / n
    varx = sum((r[0] - mx) ** 2 for r in rows) / n
    if varx == 0:
        return my
    b = sum((r[0] - mx) * (r[1] - my) for r in rows) / n / varx
    a = my - b * mx
    return max(0.0, a + b * wind_speed_m)
