"""Yhteiset apufunktiot: asetukset, /api/map ja Socket.IO-yhteys."""
import os
import sys
from pathlib import Path

import requests
import socketio
from dotenv import load_dotenv

SENSOR_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SENSOR_DIR / "calibration.json"


def load_config():
    # .env sensor-kansiosta, ajohakemistosta riippumatta
    load_dotenv(SENSOR_DIR / ".env")
    return {
        "backend_url": os.getenv("BACKEND_URL", "http://localhost:3001").rstrip("/"),
        "camera_index": int(os.getenv("CAMERA_INDEX", "0")),
        "send_interval": float(os.getenv("SEND_INTERVAL_SEC", "1.0")),
        "model_path": os.getenv("MODEL_PATH", "yolov8n.pt"),
    }


def _point(p):
    # Piste voi olla [x, y] tai {"x": .., "y": ..}
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _zone_polygon(zone):
    for key in ("polygon", "points", "vertices", "coordinates"):
        if key in zone and zone[key]:
            return [_point(p) for p in zone[key]]
    # Suorakulmio {x, y, width, height}
    if all(k in zone for k in ("x", "y", "width", "height")):
        x, y, w, h = (float(zone[k]) for k in ("x", "y", "width", "height"))
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    raise ValueError(f"Zone {zone.get('id')!r} has no polygon")


def fetch_map(backend_url):
    """Hakee kartan ja palauttaa {"width", "height", "zones": [{"id", "polygon"}]}."""
    url = f"{backend_url}/api/map"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"Could not fetch {url}: {e}")

    data = res.json()
    # Kartta voi olla kääritty {"map": ...} tai {"data": ...} sisään
    for key in ("map", "data"):
        if isinstance(data.get(key), dict):
            data = data[key]

    zones = [{"id": z["id"], "polygon": _zone_polygon(z)} for z in data.get("zones", [])]
    if not zones:
        sys.exit(f"{url} returned no zones")

    return {"width": data.get("width"), "height": data.get("height"), "zones": zones}


def connect_socket(backend_url):
    sio = socketio.Client(reconnection=True)

    @sio.event
    def connect():
        print(f"Connected to {backend_url}")

    @sio.event
    def disconnect(*args):
        print("Disconnected from backend")

    @sio.event
    def connect_error(data):
        print(f"Connection error: {data}")

    @sio.on("sensor:error")
    def on_sensor_error(data):
        data = data or {}
        print(f"sensor:error: {data.get('message')}")
        for err in data.get("errors") or []:
            print(f"  {err.get('instancePath') or '/'}: {err.get('message')}")

    try:
        sio.connect(backend_url, wait_timeout=5)
    except socketio.exceptions.ConnectionError as e:
        sys.exit(f"Could not connect to {backend_url}: {e}")
    return sio
