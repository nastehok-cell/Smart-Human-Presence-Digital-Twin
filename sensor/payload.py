"""sensor:update -viestin rakentaminen (SensorUpdate, frontendin backend-contracts.ts)."""
from datetime import datetime, timezone

import numpy as np
import cv2


def iso_timestamp():
    # Sama muoto kuin JS:n toISOString(): 2026-01-01T12:00:00.000Z
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def find_zone(x, y, zones):
    """Palauttaa sen vyöhykkeen id:n jonka sisällä piste on, tai None."""
    for zone in zones:
        contour = np.array(zone["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
            return zone["id"]
    return None


def build_updates(people, zones):
    """
    people: lista (x, y, confidence) karttakoordinaateissa.
    Palauttaa yhden SensorUpdate-viestin per vyöhyke, myös tyhjille vyöhykkeille.
    Vyöhykkeiden ulkopuoliset pisteet pudotetaan.
    """
    by_zone = {zone["id"]: [] for zone in zones}
    for x, y, conf in people:
        zone_id = find_zone(x, y, zones)
        if zone_id is not None:
            by_zone[zone_id].append((x, y, conf))

    timestamp = iso_timestamp()
    updates = []
    for zone_id, pts in by_zone.items():
        confs = [c for _, _, c in pts]
        updates.append({
            "timestamp": timestamp,
            "zone_id": zone_id,
            "occupancy_count": len(pts),
            "movement_events": [
                {"x": round(float(x), 2), "y": round(float(y), 2), "confidence": round(float(c), 3)}
                for x, y, c in pts
            ],
            "confidence": round(sum(confs) / len(confs), 3) if confs else 1.0,
        })
    return updates


def emit_updates(sio, updates):
    if not sio.connected:
        return
    for update in updates:
        sio.emit("sensor:update", update)
