"""
Kameran kalibrointi: kuvan lattiapisteet -> kartan koordinaatit (homografia).

Klikkaa kuvasta vähintään 4 lattiapistettä ja syötä jokaiselle kartan x,y terminaalissa.
  r = nollaa pisteet, s = tallenna calibration.json, q = lopeta
"""
import json
import sys

import cv2
import numpy as np

from common import CALIBRATION_PATH, fetch_map, load_config

FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080

image_pts = []
map_pts = []
pending_click = None


def click_event(event, x, y, flags, param):
    global pending_click
    if event == cv2.EVENT_LBUTTONDOWN and pending_click is None:
        pending_click = (x, y)


def ask_map_point(click):
    # Kysytään klikattua pistettä vastaava kartan koordinaatti
    while True:
        raw = input(f"Image point {click} -> map x,y (empty = skip): ").strip()
        if not raw:
            return None
        try:
            x, y = (float(v) for v in raw.replace(" ", "").split(","))
            return x, y
        except ValueError:
            print("  Give two numbers, e.g. 120,340")


def draw_points(frame):
    for i, ((ix, iy), (mx, my)) in enumerate(zip(image_pts, map_pts), start=1):
        cv2.circle(frame, (ix, iy), 6, (0, 255, 255), -1)
        cv2.putText(frame, f"{i}: ({mx:g},{my:g})", (ix + 8, iy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if len(image_pts) >= 2:
        cv2.polylines(frame, [np.array(image_pts, dtype=np.int32)], True, (0, 255, 255), 1)
    cv2.putText(frame, f"points: {len(image_pts)}  (r=reset s=save q=quit)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def save():
    if len(image_pts) < 4:
        print(f"Need at least 4 points, have {len(image_pts)}")
        return
    H, _ = cv2.findHomography(np.array(image_pts, dtype=np.float32),
                              np.array(map_pts, dtype=np.float32))
    if H is None:
        print("Homography failed - points may be collinear, try spreading them out")
        return
    data = {
        "image_size": [FRAME_WIDTH, FRAME_HEIGHT],
        "image_pts": image_pts,
        "map_pts": map_pts,
        "homography": H.tolist(),
    }
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))
    print(f"Saved {CALIBRATION_PATH}")


def main():
    global pending_click
    config = load_config()

    floor_map = fetch_map(config["backend_url"])
    print(f"Map size: {floor_map['width']} x {floor_map['height']}")
    print("Zones:", ", ".join(str(z["id"]) for z in floor_map["zones"]))

    cap = cv2.VideoCapture(config["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        sys.exit(f"Camera {config['camera_index']} could not be opened")

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame")
            break

        draw_points(frame)
        if pending_click is not None:
            cv2.circle(frame, pending_click, 6, (255, 0, 255), -1)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if pending_click is not None:
            cv2.waitKey(1)  # piirretään klikattu piste ennen kuin input() blokkaa
            map_point = ask_map_point(pending_click)
            if map_point is not None:
                image_pts.append(list(pending_click))
                map_pts.append(list(map_point))
            pending_click = None

        if key == ord("r"):
            image_pts.clear()
            map_pts.clear()
            print("Points reset")
        elif key == ord("s"):
            save()
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
