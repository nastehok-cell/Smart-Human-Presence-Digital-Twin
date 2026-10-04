"""
Testisensori ilman kameraa: 1-4 simuloitua ihmistä kävelee vyöhykkeiden sisällä
ja tila lähetetään samalla payloadilla kuin presence_sensor.py.

  python fake_sensor.py
"""
import random
import time

import cv2
import numpy as np

from common import connect_socket, fetch_map, load_config
from payload import build_updates, emit_updates

STEP = 0.05  # askeleen pituus suhteessa vyöhykkeen kokoon


def random_point_in(zone):
    # Arvotaan piste vyöhykkeen rajauslaatikosta kunnes se osuu polygonin sisään
    poly = np.array(zone["polygon"], dtype=np.float32)
    (min_x, min_y), (max_x, max_y) = poly.min(axis=0), poly.max(axis=0)
    contour = poly.reshape(-1, 1, 2)
    for _ in range(100):
        x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            return x, y
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


class Walker:
    def __init__(self, zones):
        self.zones = zones
        self.zone = random.choice(zones)
        self.x, self.y = random_point_in(self.zone)
        self.target = random_point_in(self.zone)

    def step(self):
        # Siirrytään joskus toiseen vyöhykkeeseen
        if random.random() < 0.05:
            self.zone = random.choice(self.zones)
            self.target = random_point_in(self.zone)

        poly = np.array(self.zone["polygon"], dtype=np.float32)
        speed = STEP * float(np.ptp(poly, axis=0).max())
        dx, dy = self.target[0] - self.x, self.target[1] - self.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= speed:
            self.x, self.y = self.target
            self.target = random_point_in(self.zone)
        else:
            self.x += dx / dist * speed
            self.y += dy / dist * speed

        return self.x, self.y, round(random.uniform(0.6, 0.95), 3)


def main():
    config = load_config()
    floor_map = fetch_map(config["backend_url"])
    zones = floor_map["zones"]
    print(f"Loaded {len(zones)} zones")

    sio = connect_socket(config["backend_url"])
    walkers = [Walker(zones) for _ in range(random.randint(1, 4))]
    print(f"Simulating {len(walkers)} people (Ctrl+C to stop)")

    try:
        while True:
            people = [w.step() for w in walkers]
            emit_updates(sio, build_updates(people, zones))
            time.sleep(config["send_interval"])
    except KeyboardInterrupt:
        pass
    finally:
        sio.disconnect()
        print("Stopped")


if __name__ == "__main__":
    main()
