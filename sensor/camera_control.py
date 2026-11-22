"""Kameroiden hallinta: avaus vain luvalla, sulkeminen ja tietojen tyhjennys."""
import threading
import cv2


def parse_permission(payload):
    """Tulkitsee backendin lähettämän lupaviestin (bool tai dict)."""
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, (int, float)):
        return bool(payload)
    if isinstance(payload, str):
        return payload.strip().lower() in {
            "1", "true", "on", "yes", "open", "start", "allowed", "enabled"
        }
    if isinstance(payload, dict):
        for key in (
            "enabled",
            "allowed",
            "permission",
            "active",
            "camera_enabled",
            "cameras_enabled",
            "on",
        ):
            if key in payload:
                return parse_permission(payload[key])
    return False


def default_capture_factory(camera_index, width, height):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class CameraController:
    def __init__(
        self,
        camera_configs,
        frame_width=1920,
        frame_height=1080,
        capture_factory=None,
        on_close=None,
        preview=False,
    ):
        self.camera_configs = camera_configs
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._capture_factory = capture_factory or default_capture_factory
        self._on_close = on_close
        self.preview = preview

        self._lock = threading.Lock()
        self._permitted = False
        self._session_active = False
        self.active_cameras = {}

    @property
    def is_permitted(self):
        with self._lock:
            return self._permitted

    @property
    def is_open(self):
        return bool(self.active_cameras)

    def set_permission(self, allowed):
        with self._lock:
            self._permitted = bool(allowed)

    def register(self, sio, event_name="camera:permission"):
        """Kuuntelee lupasignaalia backendilta Socket.IO:n kautta."""
        @sio.on(event_name)
        def on_camera_permission(data=None):
            allowed = parse_permission(data)
            print(f"Camera permission ({event_name}): {allowed}")
            self.set_permission(allowed)

        @sio.event
        def disconnect(*args):
            print("Disconnected from backend - closing cameras")
            self.set_permission(False)

    def open_cameras(self):
        """Avaa kamerat vasta kun lupa on voimassa."""
        if self.active_cameras:
            return self.active_cameras

        opened = {}
        for camera_index, cfg in self.camera_configs.items():
            cap = self._capture_factory(camera_index, self.frame_width, self.frame_height)
            if not cap.isOpened():
                cap.release()
                print(
                    f"Camera index {camera_index} could not be opened; "
                    "check the index and ensure the camera is connected and not in use."
                )
                continue

            if self.preview:
                cv2.namedWindow(cfg["window_name"], cv2.WINDOW_NORMAL)

            opened[camera_index] = {
                **cfg,
                "cap": cap,
                "size_checked": False,
            }

        self.active_cameras = opened
        if opened:
            self._session_active = True
            print(f"Opened cameras: {list(opened.keys())}")
        return self.active_cameras

    def close_cameras(self):
        """Sulkee kamerat, nollaa mallin trackerin ja tyhjentää seurantatiedot."""
        for camera in self.active_cameras.values():
            camera["cap"].release()
            model = camera.get("model")
            if model is not None and getattr(model, "predictor", None) is not None:
                # Nollataan ByteTrack-tila, jotta uudelleenavaus alkaa puhtaalta pöydältä
                model.predictor = None

        self.active_cameras.clear()

        if self.preview:
            cv2.destroyAllWindows()

        if self._on_close is not None:
            self._on_close()

        self._session_active = False
        print("Cameras closed and tracking data cleared")

    def sync(self):
        """
        Synkronoi kameroiden tilan pääsilmukassa (turvallista OpenCV:lle).
        Palauttaa True, jos kamerat ovat auki ja valmiina lukemaan kuvia.
        """
        permitted = self.is_permitted
        if permitted and not self.is_open:
            self.open_cameras()
        elif not permitted and (self.is_open or self._session_active):
            self.close_cameras()
        return self.is_open