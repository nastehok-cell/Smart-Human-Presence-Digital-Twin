# Smart-Human-Presence-Digital-Twin

## Running the sensor

The sensor detects people with YOLOv8, maps their floor position onto the backend's map and sends a `sensor:update` Socket.IO event per zone.

```bash
# 1. Virtual environment + dependencies
python3 -m venv yolov8-env
source yolov8-env/bin/activate
pip install -r sensor/requirements.txt

# 2. Settings
cp sensor/.env.example sensor/.env   # edit BACKEND_URL / CAMERA_INDEX if needed

# 3. Calibrate once per camera position (backend must be running)
python sensor/calibrate.py
#   click >= 4 floor points, type each point's map x,y in the terminal
#   r = reset, s = save sensor/calibration.json, q = quit

# 4. Run the sensor
python sensor/presence_sensor.py            # add --preview to see the annotated camera view

# ...or test without a camera
python sensor/fake_sensor.py
```

Re-run `calibrate.py` whenever the camera is moved or the map changes.
