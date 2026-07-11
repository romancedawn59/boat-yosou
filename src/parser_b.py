"""番組表JSON(BoatraceOpenAPI programs v3)をDB行の辞書へ変換する"""
from db import make_race_id

CLASS_NAMES = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}


def _extract_programs(data: dict) -> list[dict]:
    """新旧両形式のJSONからprograms配列を取り出す。

    上流BoatraceOpenAPIの2026-07-11変更(コミット「fix: include yesterday's data
    in update process」)により、直近日付のJSONは
    {"today": {"programs": [...]}, "yesterday": {"programs": [...]}} の
    envelope形式で配信される(2026-07-09以前のアーカイブは旧形式のまま)。
    各program行は自身のdateを持ちrace_idはそこから決まるため、
    today/yesterday両ブランチを連結して返しても別日が混ざる問題はない。
    """
    if "programs" in data:
        return data["programs"]  # 旧形式(確定アーカイブ)
    programs: list[dict] = []
    for branch in ("today", "yesterday"):
        programs.extend((data.get(branch) or {}).get("programs", []))
    return programs


def parse_program(data: dict) -> dict:
    """戻り値: {"races": [race_dict, ...], "entries": [entry_dict, ...]}

    race_dict / entry_dict のキーは db.py の races / entries テーブル列に対応する。
    """
    races: list[dict] = []
    entries: list[dict] = []

    for p in _extract_programs(data):
        race_id = make_race_id(p["date"], p["stadium_number"], p["number"])
        races.append({
            "race_id": race_id,
            "date": p["date"],
            "venue_code": p["stadium_number"],
            "race_no": p["number"],
            "title": p.get("title"),
            "subtitle": p.get("subtitle"),
            "grade": p.get("grade_label"),
            "day_label": p.get("day_label"),
            "distance_m": p.get("distance"),
            "deadline_time": p.get("closed_at"),
        })

        for b in p.get("boats", []):
            # 選手未確定(欠場・中止等)の枠は全項目nullで配信されることがある
            if b.get("racer_boat_number") is None or b.get("racer_number") is None:
                continue
            entries.append({
                "race_id": race_id,
                "lane": b["racer_boat_number"],
                "reg_no": b["racer_number"],
                "racer_name": b.get("racer_name"),
                "racer_class": CLASS_NAMES.get(b.get("racer_class_number")),
                "branch_number": b.get("racer_branch_number"),
                "birthplace_number": b.get("racer_birthplace_number"),
                "age": b.get("racer_age"),
                "weight_kg": b.get("racer_weight"),
                "flying_count": b.get("racer_flying_count"),
                "late_count": b.get("racer_late_count"),
                "avg_st": b.get("racer_average_start_timing"),
                "national_win_rate": b.get("racer_national_top_1_percent"),
                "national_2rate": b.get("racer_national_top_2_percent"),
                "national_3rate": b.get("racer_national_top_3_percent"),
                "local_win_rate": b.get("racer_local_top_1_percent"),
                "local_2rate": b.get("racer_local_top_2_percent"),
                "local_3rate": b.get("racer_local_top_3_percent"),
                "motor_no": b.get("racer_assigned_motor_number"),
                "motor_2rate": b.get("racer_assigned_motor_top_2_percent"),
                "motor_3rate": b.get("racer_assigned_motor_top_3_percent"),
                "boat_no": b.get("racer_assigned_boat_number"),
                "boat_2rate": b.get("racer_assigned_boat_top_2_percent"),
                "boat_3rate": b.get("racer_assigned_boat_top_3_percent"),
            })

    return {"races": races, "entries": entries}
