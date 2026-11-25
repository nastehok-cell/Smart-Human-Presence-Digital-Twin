import threading
import time

import cv2
import numpy as np

from payload import iso_timestamp

# Tarkistaa, onko piste alueen sisällä
def in_polygon(x, y, polygon):
    if not polygon:
        return False
    contour = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


class TrackManager:
    def __init__(self, checkin_polygon, lost_after, merge_distance=35):
        self.checkin_polygon = checkin_polygon
        self.lost_after = lost_after
        self.merge_distance = merge_distance
        self._tracks = {}
        self._person_of = {} 
        self._next_person = 1
        self._lock = threading.Lock()
    # Päivittää sijainnit ja liittää jokaisen trackin henkilöön.
    def update(self, camera_id, people):
        now = time.monotonic()
        with self._lock:
            for track_id, x, y in people:
                self._tracks[(camera_id, track_id)] = {
                    "x": x,
                    "y": y,
                    "in_checkin_zone": in_polygon(x, y, self.checkin_polygon),
                    "last_seen": now,
                }
            for track_id, x, y in people:
                key = (camera_id, track_id)
                if key not in self._person_of:
                    self._person_of[key] = self._find_or_new_person(camera_id, x, y)

    # Etsii toisen kameran trackin läheltä, muuten luo uuden henkilön.
    def _find_or_new_person(self, camera_id, x, y):
        taken = {pid for (cam, _), pid in self._person_of.items() if cam == camera_id}
        best, best_d = None, self.merge_distance
        for key, pid in self._person_of.items():
            if key[0] == camera_id or pid in taken or key not in self._tracks:
                continue
            t = self._tracks[key]
            d = np.hypot(x - t["x"], y - t["y"])
            if d < best_d:
                best, best_d = pid, d
        if best is not None:
            return best
        pid = self._next_person
        self._next_person += 1
        return pid

    # Poistaa trackit, joita ei ole nähty riittävän pitkään.
    def pop_lost(self):
        now = time.monotonic()
        with self._lock:
            lost = [k for k, t in self._tracks.items() if now - t["last_seen"] > self.lost_after]
            lost_persons = set()
            for key in lost:
                del self._tracks[key]
                pid = self._person_of.pop(key, None)
                if pid is not None:
                    lost_persons.add(pid)
            gone = [pid for pid in lost_persons if pid not in self._person_of.values()]
        return lost, gone

    # Lista kaikista trackeista
    def snapshot(self):
        with self._lock:
            return [
                {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "person_id": self._person_of.get((camera_id, track_id)),
                    "x": round(float(t["x"]), 2),
                    "y": round(float(t["y"]), 2),
                    "in_checkin_zone": t["in_checkin_zone"],
                }
                for (camera_id, track_id), t in self._tracks.items()
            ]

    # Yksi rivi per henkilö (molempien kameroiden trackit yhdistetty).
    def people(self):
        with self._lock:
            groups = {}
            for key, t in self._tracks.items():
                pid = self._person_of.get(key)
                if pid is not None:
                    groups.setdefault(pid, []).append(t)
            return [
                {
                    "person_id": pid,
                    "x": round(float(np.mean([t["x"] for t in ts])), 2),
                    "y": round(float(np.mean([t["y"] for t in ts])), 2),
                    "in_checkin_zone": any(t["in_checkin_zone"] for t in ts),
                }
                for pid, ts in groups.items()
            ]

    # Poistaa kaikki seurannassa olevat henkilöt.
    def clear(self):
        with self._lock:
            self._tracks.clear()
            self._person_of.clear()

    # Hakee Check-In -alueella olevat henkilöt.
    def register(self, sio):
        @sio.on("checkin:query")
        def on_checkin_query(*args):
            return {
                "people": [p for p in self.people() if p["in_checkin_zone"]],
                "tracks": [t for t in self.snapshot() if t["in_checkin_zone"]],
            }

def emit_tracks(sio, tracks, people):
    if not getattr(sio, "connected", False):
        return
    sio.emit("tracks:update", {"timestamp": iso_timestamp(), "tracks": tracks, "people": people})

def emit_lost(sio, key):
    if not getattr(sio, "connected", False):
        return
    camera_id, track_id = key
    sio.emit("track:lost", {"camera_id": camera_id, "track_id": track_id})

def emit_person_lost(sio, person_id):
    if not getattr(sio, "connected", False):
        return
    sio.emit("person:lost", {"person_id": person_id})