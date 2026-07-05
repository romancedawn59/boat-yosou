"""競走成績JSON(BoatraceOpenAPI results v2)をDB行の辞書へ変換する"""
from db import make_race_id

BET_TYPE_NAMES = {
    "win": "単勝",
    "place": "複勝",
    "exacta": "2連単",
    "quinella": "2連複",
    "quinella_place": "拡連複",
    "trifecta": "3連単",
    "trio": "3連複",
}


def parse_result(data: dict) -> dict:
    """戻り値: {"races": [...], "results": [...], "payouts": [...]}

    racesには気象・決まり手などresults側にしかない列を入れる。
    db._upsertが渡した列だけ更新するので、番組表由来の列は上書きされない。
    """
    races: list[dict] = []
    results: list[dict] = []
    payouts: list[dict] = []

    for r in data.get("results", []):
        race_id = make_race_id(r["race_date"], r["race_stadium_number"], r["race_number"])
        races.append({
            "race_id": race_id,
            "date": r["race_date"],
            "venue_code": r["race_stadium_number"],
            "race_no": r["race_number"],
            "weather_number": r.get("race_weather_number"),
            "wind_speed_m": r.get("race_wind"),
            "wind_direction_number": r.get("race_wind_direction_number"),
            "wave_height_cm": r.get("race_wave"),
            "temperature": r.get("race_temperature"),
            "water_temperature": r.get("race_water_temperature"),
            "winning_technique_number": r.get("race_technique_number"),
        })

        for b in r.get("boats", []):
            order = b.get("racer_place_number")
            if not isinstance(order, int) or not 1 <= order <= 6:
                order = None  # 失格・欠場・転覆等
            results.append({
                "race_id": race_id,
                "lane": b["racer_boat_number"],
                "course": b.get("racer_course_number"),
                "arrival_order": order,
                "st_time": b.get("racer_start_timing"),
            })

        for key, bet_type in BET_TYPE_NAMES.items():
            for pay in (r.get("payouts") or {}).get(key) or []:
                if pay.get("combination") is None:
                    continue  # 不成立(返還等)
                payouts.append({
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "combination": pay["combination"],
                    "amount_yen": pay.get("payout"),
                })

    return {"races": races, "results": results, "payouts": payouts}
