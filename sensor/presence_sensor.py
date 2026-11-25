"""
Läsnäolosensori: YOLOv8 tunnistaa ihmiset, jalkapisteet muunnetaan kartalle
ja jokaisen vyöhykkeen tila lähetetään backendille "sensor:update" -viestinä.

  python presence_sensor.py [--preview]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from camera_control import CameraController
from common import get_calibration_path, SENSOR_DIR, connect_socket, fetch_map, load_config
from payload import build_updates, emit_updates
from tracking import TrackManager, emit_tracks, emit_lost, emit_person_lost

FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080


def load_calibration(camera_index):
    path = get_calibration_path(camera_index)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run calibrate.py --camera {camera_index}")
    data = json.loads(path.read_text())
    H = np.array(data["homography"], dtype=np.float64)

    return H, data.get("image_size")


def image_to_map(points, H):
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).tolist()


def merge_cameras(detections, max_distance):
    clusters = []
    for cam, people in detections.items():
        if not people:
            continue
        open_idx = [i for i, cl in enumerate(clusters) if cam not in cl["cams"]]
        matched = set()
        if open_idx:
            cents = np.array([np.mean([p[:2] for p in clusters[i]["pts"]], axis=0)
                              for i in open_idx])
            pts = np.array([p[:2] for p in people])
            cost = np.linalg.norm(pts[:, None, :] - cents[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= max_distance:
                    cl = clusters[open_idx[c]]
                    cl["pts"].append(people[r])
                    cl["cams"].add(cam)
                    matched.add(r)
        for r, p in enumerate(people):
            if r not in matched:
                clusters.append({"pts": [p], "cams": {cam}})

    out = []
    for cl in clusters:
        w = np.array([p[2] for p in cl["pts"]])
        xy = np.array([p[:2] for p in cl["pts"]])
        x, y = (xy * w[:, None]).sum(axis=0) / w.sum()  # confidence-weighted average
        out.append((float(x), float(y), float(w.max())))
    return out


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
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--zones", default=None)
    parser.add_argument(
        "--start-allowed",
        action="store_true",
        help="Paikalliseen testaukseen: aloittaa luvalla ilman backend-viestiä",
    )
    args = parser.parse_args()

    config = load_config()

    camera_indices = (
        [args.camera_index]
        if args.camera_index is not None
        else list(dict.fromkeys(config["camera_indices"]))
    )
    if not camera_indices:
        sys.exit("No cameras configured")

    floor_map = fetch_map(config["backend_url"])
    zones = floor_map["zones"]
    if args.zones:
        wanted = {z.strip() for z in args.zones.split(",")}
        zones = [z for z in zones if str(z["id"]) in wanted]
        if not zones:
            sys.exit(f"None of the zones {sorted(wanted)} exist on the map")
    print(f"Loaded {len(zones)} zones")

    model_path = config["model_path"]
    if not (SENSOR_DIR / model_path).exists() and (SENSOR_DIR.parent / model_path).exists():
        model_path = str(SENSOR_DIR.parent / model_path)

    # Valmistellaan kalibroinnit ja mallit, mutta EI avata kameroita ennen lupaa
    camera_configs = {}
    for camera_index in camera_indices:
        try:
            H, calib_size = load_calibration(camera_index)
            H_inv = np.linalg.inv(H)
        except (OSError, json.JSONDecodeError, KeyError, ValueError, np.linalg.LinAlgError) as exc:
            print(f"Skipping camera {camera_index}: calibration error: {exc}")
            continue

        camera_configs[camera_index] = {
            "model": YOLO(model_path),
            "H": H,
            "H_inv": H_inv,
            "calib_size": calib_size,
            "window_name": f"Presence sensor {camera_index}",
        }

    if not camera_configs:
        sys.exit("No usable cameras (check indices and calibration files)")

    sio = connect_socket(config["backend_url"])
    tracker = TrackManager(config["checkin_zone"], config["track_lost_sec"], config["merge_distance"])
    tracker.register(sio)

    permission_event = os.getenv("CAMERA_PERMISSION_EVENT", "camera:permission")
    controller = CameraController(
        camera_configs,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        on_close=tracker.clear,
        preview=args.preview,
    )
    controller.register(sio, event_name=permission_event)

    if args.start_allowed:
        controller.set_permission(True)
    else:
        print(f"Waiting for camera permission on '{permission_event}'...")

    last_tracks_sent = 0.0
    last_sent = 0.0
    try:
        while True:
            # Kamera avataan vasta luvalla; kun lupa loppuu, se suljetaan ja lähetys pysähtyy
            if not controller.sync():
                time.sleep(0.1)
                continue

            detections = {}
            preview_frames = {}
            failed = set()

            for camera_index, camera in list(controller.active_cameras.items()):
                cap = camera["cap"]
                model = camera["model"]
                H = camera["H"]
                H_inv = camera["H_inv"]
                calib_size = camera["calib_size"]

                ret, frame = cap.read()
                if not ret:
                    print(f"Could not read frame from camera {camera_index}")
                    failed.add(camera_index)
                    continue

                if not camera["size_checked"]:
                    h, w = frame.shape[:2]
                    if calib_size and [w, h] != list(calib_size):
                        print(f"Warning: camera {camera_index} gives {w}x{h} but calibration was made at "
                              f"{calib_size[0]}x{calib_size[1]} - re-run calibrate.py")
                    camera["size_checked"] = True

                # Vain ihmiset (COCO luokka 0)
                results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
                boxes = results[0].boxes

                feet, confs = [], []
                for (x1, y1, x2, y2), conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                    feet.append(((x1 + x2) / 2, y2))  # laatikon alareunan keskikohta
                    confs.append(conf)

                ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(feet)

                map_points = image_to_map(feet, H)
                people = [(x, y, c) for (x, y), c in zip(map_points, confs)]
                detections[camera_index] = people
                tracker.update(
                    camera_index,
                    [(tid, x, y) for tid, (x, y) in zip(ids, map_points) if tid is not None],
                )

                if args.preview:
                    annotated = results[0].plot()
                    draw_zones(annotated, zones, H_inv)
                    for (fx, fy), (mx, my) in zip(feet, map_points):
                        cv2.circle(annotated, (int(fx), int(fy)), 6, (0, 0, 255), -1)
                        cv2.putText(annotated, f"({mx:.0f},{my:.0f})", (int(fx) + 8, int(fy)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    preview_frames[camera_index] = annotated

            # Tarkistetaan vielä, ettei lupa loppunut kesken kierroksen ennen lähetystä
            if not controller.is_permitted:
                controller.sync()
                continue

            people = merge_cameras(detections, config["merge_distance"])

            now = time.monotonic()
            lost, gone = tracker.pop_lost()
            for key in lost:
                emit_lost(sio, key)
            for person_id in gone:
                emit_person_lost(sio, person_id)
            if not failed and now - last_tracks_sent >= config["send_interval"]:
                emit_tracks(sio, tracker.snapshot(), tracker.people())
                last_tracks_sent = now

            if not failed and now - last_sent >= config["send_interval"]:
                updates = build_updates(people, zones)
                emit_updates(sio, updates)
                last_sent = now

            if args.preview:
                for camera_index, annotated in preview_frames.items():
                    if camera_index in controller.active_cameras:
                        cv2.imshow(controller.active_cameras[camera_index]["window_name"], annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        controller.close_cameras()
        cv2.destroyAllWindows()
        sio.disconnect()
        print("Stopped")


if __name__ == "__main__":
    main()