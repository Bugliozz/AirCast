"""Quick test of the NRT (Near Real Time) Socrata API.

Manual exploration script — run it directly (`python scripts/test_nrt_api.py`).
Everything lives inside main() so that pytest collection (the filename matches
test_*.py) does not fire live API calls or rewrite the output file on import.
"""
import requests


def main():
    out = open("scripts/nrt_test_output.txt", "w", encoding="utf-8")

    def p(msg):
        print(msg, file=out)
        print(msg)

    # 1. Find PM10 sensors in the NRT station registry
    p("=== NRT Station Registry: PM10 Sensors ===")
    r = requests.get(
        "https://www.dati.lombardia.it/resource/9xaz-9vbz.json",
        params={"$where": "nometiposensore like '%PM10%'", "$limit": "200"},
        timeout=15,
    )
    r.raise_for_status()
    nrt_pm10_sensors = r.json()
    p(f"Found {len(nrt_pm10_sensors)} PM10 sensors in NRT registry")
    for s in nrt_pm10_sensors[:8]:
        p(f"  idsensore={s['idsensore']}, idstazione={s['idstazione']}, "
          f"name={s['nomestazione']}, type={s['nometiposensore']}")

    # 2. Cross-reference with main registry
    p("\n=== Main Station Registry: PM10 Sensors ===")
    r2 = requests.get(
        "https://www.dati.lombardia.it/resource/ib47-atvt.json",
        params={"$where": "nometiposensore like '%PM10%'", "$limit": "200"},
        timeout=15,
    )
    r2.raise_for_status()
    main_pm10_sensors = r2.json()
    p(f"Found {len(main_pm10_sensors)} PM10 sensors in main registry")
    for s in main_pm10_sensors[:5]:
        p(f"  idsensore={s['idsensore']}, idstazione={s['idstazione']}, "
          f"name={s['nomestazione']}, type={s['nometiposensore']}")

    # 3. Check overlap
    nrt_ids = {s["idsensore"] for s in nrt_pm10_sensors}
    main_ids = {s["idsensore"] for s in main_pm10_sensors}
    p(f"\nNRT PM10 sensors: {len(nrt_ids)}")
    p(f"Main PM10 sensors: {len(main_ids)}")
    p(f"Overlap: {len(nrt_ids & main_ids)}")
    p(f"Only NRT: {len(nrt_ids - main_ids)}, Only Main: {len(main_ids - nrt_ids)}")

    # 4. NRT idstazione mapping
    nrt_station_map = {s["idsensore"]: s["idstazione"] for s in nrt_pm10_sensors}
    main_station_map = {s["idsensore"]: s["idstazione"] for s in main_pm10_sensors}
    p(f"\nNRT idstazione values same as main? Checking overlapping sensors...")
    for sid in sorted(nrt_ids & main_ids)[:5]:
        p(f"  sensor {sid}: NRT station={nrt_station_map[sid]}, Main station={main_station_map[sid]}, match={nrt_station_map[sid]==main_station_map[sid]}")

    # 5. Get today's hourly PM10 for one sensor
    if nrt_pm10_sensors:
        test = nrt_pm10_sensors[0]
        p(f"\n=== Today's NRT for sensor {test['idsensore']} ({test['nomestazione']}) ===")
        r3 = requests.get(
            "https://www.dati.lombardia.it/resource/ykhg-b8rs.json",
            params={
                "$where": f"idsensore = '{test['idsensore']}'",
                "$order": "data ASC",
                "$limit": "50",
            },
            timeout=15,
        )
        r3.raise_for_status()
        hourly = r3.json()
        p(f"Hourly records: {len(hourly)}")
        for h in hourly:
            p(f"  {h['data']}  val={h['valore']}")
        vals = [float(h["valore"]) for h in hourly if float(h.get("valore", -1)) > 0]
        if vals:
            p(f"  -> Daily mean so far: {sum(vals)/len(vals):.1f} ug/m3 ({len(vals)} hours)")

    out.close()


if __name__ == "__main__":
    main()
