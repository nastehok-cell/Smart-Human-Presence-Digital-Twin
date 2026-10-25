import re

from payload import build_updates, emit_updates, find_zone, iso_timestamp

# Kaksi erillistä neliötä (A vasemmalla, B oikealla)
ZONES = [
    {"id": "A", "polygon": [(0, 0), (10, 0), (10, 10), (0, 10)]},
    {"id": "B", "polygon": [(20, 0), (30, 0), (30, 10), (20, 10)]},
]


def test_find_zone_inside():
    assert find_zone(5, 5, ZONES) == "A"
    assert find_zone(25, 5, ZONES) == "B"


def test_find_zone_outside():
    assert find_zone(15, 5, ZONES) is None


def test_build_updates_one_message_per_zone():
    # Kaksi ihmistä vyöhykkeessä A, ei ketään vyöhykkeessä B
    updates = build_updates([(5, 5, 0.9), (6, 6, 0.7)], ZONES)
    by_id = {u["zone_id"]: u for u in updates}

    # Jokaisesta vyöhykkeestä tulee yksi viesti, myös tyhjästä
    assert len(updates) == 2
    assert by_id["A"]["occupancy_count"] == 2
    assert by_id["A"]["confidence"] == 0.8
    assert by_id["B"]["occupancy_count"] == 0
    assert by_id["B"]["confidence"] == 1.0


def test_build_updates_drops_points_outside_zones():
    # Vyöhykkeiden ulkopuolella oleva ihminen ei näy missään viestissä
    updates = build_updates([(15, 5, 0.9)], ZONES)
    assert all(u["occupancy_count"] == 0 for u in updates)


def test_timestamp_format():
    # Timestamp muoto: 2026-01-01T12:00:00.000Z (ISO 8601, UTC)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", iso_timestamp())


class FakeSio:
    # Tallentaa lähetetyt viestit listaan, jotta niitä voi tarkistaa
    def __init__(self, connected):
        self.connected = connected
        self.sent = []

    def emit(self, event, data):
        self.sent.append((event, data))


def test_emit_updates_sends_when_connected():
    # Kun yhteys on päällä, jokainen päivitys lähetetään omana viestinään
    sio = FakeSio(True)
    emit_updates(sio, [{"zone_id": "A"}, {"zone_id": "B"}])
    assert [e for e, _ in sio.sent] == ["sensor:update"] * 2


def test_emit_updates_skips_when_disconnected():
    # Kun yhteys ei ole päällä, mitään ei lähetetä
    sio = FakeSio(False)
    emit_updates(sio, [{"zone_id": "A"}])
    assert sio.sent == []