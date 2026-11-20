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
    def __init__(self, checkin_polygon, lost_after):
        self.checkin_polygon = checkin_polygon
        self.lost_after = lost_after
        self._tracks = {} 
        self._lock = threading.Lock()  
    # Päivittää henkilöiden sijainnit ja tunnisteet.
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
    # Poistaa henkilöt, joita kamera ei ole nähnyt riittävän pitkään.
    def pop_lost(self):
        now = time.monotonic()
        with self._lock:
            lost = [k for k, t in self._tracks.items() if now - t["last_seen"] > self.lost_after]
            for key in lost:
                del self._tracks[key]
        return lost
    
    # Lista tällä hetkellä seuratuista henkilöistä.
    def snapshot(self):
        with self._lock:
            return [
                {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "x": round(float(t["x"]), 2),
                    "y": round(float(t["y"]), 2),
                    "in_checkin_zone": t["in_checkin_zone"],
                }
                for (camera_id, track_id), t in self._tracks.items()
            ]
    # Poistaa kaikki seurannassa olevat henkilöt.
    def clear(self):
        with self._lock:
            self._tracks.clear()

    # Hakee sisäänkirjautumisalueella olevat henkilöt.
    def register(self, sio):
        @sio.on("checkin:query")
        def on_checkin_query(*args):
            return {
                "tracks": [
                    {"camera_id": t["camera_id"], "track_id": t["track_id"]}
                    for t in self.snapshot()
                    if t["in_checkin_zone"]
                ]
            }


def emit_tracks(sio, tracks):
    if not getattr(sio, "connected", False):
        return
    sio.emit("tracks:update", {"timestamp": iso_timestamp(), "tracks": tracks})


def emit_lost(sio, key):
    if not getattr(sio, "connected", False):
        return
    camera_id, track_id = key
    sio.emit("track:lost", {"camera_id": camera_id, "track_id": track_id})