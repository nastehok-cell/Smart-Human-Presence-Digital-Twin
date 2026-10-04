"""
Läsnäolosensori: YOLOv8 tunnistaa ihmiset, jalkapisteet muunnetaan kartalle
ja jokaisen vyöhykkeen tila lähetetään backendille "sensor:update" -viestinä.

  python presence_sensor.py [--preview]
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common import CALIBRATION_PATH, SENSOR_DIR, connect_socket, fetch_map, load_config
from payload import build_updates, emit_updates

FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080


def load_calibration():
    if not CALIBRATION_PATH.exists():
        sys.exit(f"{CALIBRATION_PATH} not found - run calibrate.py first")
    data = json.loads(CALIBRATION_PATH.read_text())
    H = np.array(data["homography"], dtype=np.float64)
    return H, data.get("image_size")


def image_to_map(points, H):
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).tolist()


def draw_zones(frame, zones, H_inv):
    # Projisoidaan vyöhykkeet kartalta takaisin kameran kuvaan
    for zone in zones:
        pts = np.array(zone["polygon"], dtype=np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pts, H_inv).astype(np.int32)
        cv2.polylines(frame, [img_pts], True, (255, 200, 0), 2)
        x, y = img_pts[0][0]
        cv2.putText(frame, str(zone["id"]), (int(x), int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="show annotated camera frame")
    args = parser.parse_args()

    config = load_config()
    H, calib_size = load_calibration()
    H_inv = np.linalg.inv(H)
    floor_map = fetch_map(config["backend_url"])
    zones = floor_map["zones"]
    print(f"Loaded {len(zones)} zones")

    model_path = config["model_path"]
    if not (SENSOR_DIR / model_path).exists() and (SENSOR_DIR.parent / model_path).exists():
        model_path = str(SENSOR_DIR.parent / model_path)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(config["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        sys.exit(f"Camera {config['camera_index']} could not be opened")

    sio = connect_socket(config["backend_url"])
    window_name = "Presence sensor"
    if args.preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_sent = 0.0
    size_checked = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame")
                break

            if not size_checked:
                h, w = frame.shape[:2]
                if calib_size and [w, h] != list(calib_size):
                    print(f"Warning: camera gives {w}x{h} but calibration was made at "
                          f"{calib_size[0]}x{calib_size[1]} - re-run calibrate.py")
                size_checked = True

            # Vain ihmiset (COCO luokka 0)
            results = model.track(frame, persist=True, classes=[0], verbose=False)
            boxes = results[0].boxes

            feet, confs = [], []
            for (x1, y1, x2, y2), conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                feet.append(((x1 + x2) / 2, y2))  # laatikon alareunan keskikohta
                confs.append(conf)

            map_points = image_to_map(feet, H)
            people = [(x, y, c) for (x, y), c in zip(map_points, confs)]

            now = time.monotonic()
            if now - last_sent >= config["send_interval"]:
                updates = build_updates(people, zones)
                emit_updates(sio, updates)
                last_sent = now

            if args.preview:
                annotated = results[0].plot()
                draw_zones(annotated, zones, H_inv)
                for (fx, fy), (mx, my) in zip(feet, map_points):
                    cv2.circle(annotated, (int(fx), int(fy)), 6, (0, 0, 255), -1)
                    cv2.putText(annotated, f"({mx:.0f},{my:.0f})", (int(fx) + 8, int(fy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sio.disconnect()
        print("Stopped")


if __name__ == "__main__":
    main()
