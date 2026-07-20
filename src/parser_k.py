"""競走成績JSON(BoatraceOpenAPI results v3)をDB行の辞書へ変換する"""
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
        race_id = make_race_id(r["date"], r["stadium_number"], r["number"])
        races.append({
            "race_id": race_id,
            "date": r["date"],
            "venue_code": r["stadium_number"],
            "race_no": r["number"],
            "weather_number": r.get("weather_number"),
            "wind_speed_m": r.get("wind_speed"),
            "wind_direction_number": r.get("wind_direction_number"),
            "wave_height_cm": r.get("wave_height"),
            "temperature": r.get("air_temperature"),
            "water_temperature": r.get("water_temperature"),
            "winning_technique_number": r.get("technique_number"),
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
                    "amount_yen": pay.get("amount"),
                })

    return {"races": races, "results": results, "payouts": payouts}
