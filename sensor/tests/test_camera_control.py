from camera_control import CameraController, parse_permission
from tracking import TrackManager


class FakeCapture:
    def __init__(self, camera_index, opened=True):
        self.camera_index = camera_index
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def release(self):
        self.released = True
        self._opened = False


class FakeModel:
    def __init__(self):
        self.predictor = object()


class FakeSocket:
    def __init__(self):
        self.handlers = {}

    def on(self, event_name):
        def decorator(fn):
            self.handlers[event_name] = fn
            return fn
        return decorator

    def event(self, fn):
        self.handlers[fn.__name__] = fn
        return fn

    def trigger(self, event_name, *args):
        self.handlers[event_name](*args)


def _make_controller(tracker, created_caps):
    def factory(camera_index, width, height):
        cap = FakeCapture(camera_index)
        created_caps.append(cap)
        return cap

    configs = {
        0: {
            "model": FakeModel(),
            "window_name": "Presence sensor 0",
        }
    }
    return CameraController(
        configs,
        capture_factory=factory,
        on_close=tracker.clear,
        preview=False,
    )


def test_camera_not_opened_before_permission():
    created_caps = []
    tracker = TrackManager([], lost_after=1.5)
    controller = _make_controller(tracker, created_caps)

    assert controller.is_permitted is False
    assert controller.sync() is False
    assert controller.is_open is False
    assert len(created_caps) == 0


def test_camera_opens_only_when_permission_arrives():
    created_caps = []
    tracker = TrackManager([], lost_after=1.5)
    controller = _make_controller(tracker, created_caps)

    controller.set_permission(True)
    assert controller.sync() is True
    assert controller.is_open is True
    assert len(created_caps) == 1
    assert created_caps[0].released is False


def test_camera_closes_and_clears_tracking_data_when_permission_ends():
    created_caps = []
    tracker = TrackManager([[0, 0], [100, 0], [100, 100], [0, 100]], lost_after=1.5)
    controller = _make_controller(tracker, created_caps)

    # Avataan kamera ja lisätään seurantatietoja
    controller.set_permission(True)
    assert controller.sync() is True
    tracker.update(0, [(1, 50.0, 50.0), (2, 10.0, 20.0)])
    assert len(tracker.snapshot()) == 2

    # Lupa loppuu -> kamera suljetaan, malli nollataan ja trackit tyhjennetään
    controller.set_permission(False)
    assert controller.sync() is False
    assert controller.is_open is False
    assert created_caps[0].released is True
    assert controller.camera_configs[0]["model"].predictor is None
    assert tracker.snapshot() == []


def test_camera_reopens_cleanly_after_closing():
    created_caps = []
    tracker = TrackManager([], lost_after=1.5)
    controller = _make_controller(tracker, created_caps)

    # 1. Avaus
    controller.set_permission(True)
    assert controller.sync() is True
    assert len(created_caps) == 1

    # 2. Sulkeminen
    controller.set_permission(False)
    assert controller.sync() is False
    assert created_caps[0].released is True

    # 3. Uudelleenavaus
    controller.set_permission(True)
    assert controller.sync() is True
    assert controller.is_open is True
    assert len(created_caps) == 2
    assert created_caps[1].released is False


def test_socket_events_toggle_camera_and_disconnect_closes():
    created_caps = []
    tracker = TrackManager([], lost_after=1.5)
    controller = _make_controller(tracker, created_caps)
    sio = FakeSocket()
    controller.register(sio, event_name="camera:permission")

    # Lupa tulee dict-muodossa
    sio.trigger("camera:permission", {"enabled": True})
    assert controller.sync() is True
    assert controller.is_open is True

    # Yhteys katkeaa -> kamera suljetaan
    sio.trigger("disconnect")
    assert controller.sync() is False
    assert controller.is_open is False
    assert created_caps[0].released is True

    # Lupa tulee uudelleen bool-muodossa -> kamera aukeaa uudelleen
    sio.trigger("camera:permission", True)
    assert controller.sync() is True
    assert len(created_caps) == 2


def test_parse_permission_formats():
    assert parse_permission(True) is True
    assert parse_permission(False) is False
    assert parse_permission({"enabled": True}) is True
    assert parse_permission({"allowed": False}) is False
    assert parse_permission({"camera_enabled": True}) is True
    assert parse_permission("on") is True
    assert parse_permission("off") is False