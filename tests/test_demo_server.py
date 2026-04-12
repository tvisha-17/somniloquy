"""Tests for src/realtime/demo_server.py."""

import asyncio

import numpy as np
from fastapi.testclient import TestClient


class DummyDetector:
    def __init__(self, states):
        self.states = list(states)

    def process_window(self, window, stage_hint=None):
        return self.states.pop(0)


class DummyDecoder:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def process_window(self, window, *, timestamp, rem_probability):
        payload = dict(self.payloads.pop(0))
        payload["timestamp"] = timestamp
        payload["rem_probability"] = rem_probability
        return payload


def test_load_replay_source_round_trip(tmp_path):
    from src.realtime.demo_server import load_replay_source

    np.savez(
        tmp_path / "sub-demo_epochs.npz",
        data=np.zeros((3, 4, 8), dtype=np.float32),
        sleep_stages=np.array([1, 4, 4], dtype=np.int64),
        subject_id="demo",
        sfreq=256.0,
        ch_names=np.array(["C1", "C2", "C3", "C4"]),
        epoch_times_s=np.array([0.0, 2.0, 4.0], dtype=np.float64),
    )

    source = load_replay_source(tmp_path / "sub-demo_epochs.npz")
    assert tuple(source.data.shape) == (3, 4, 8)
    assert source.subject_id == "demo"


def test_demo_controller_playback_broadcasts_prediction_and_complete():
    from src.realtime.demo_server import ReplayEpochSource, RealtimeDemoController

    source = ReplayEpochSource(
        data=np.zeros((2, 4, 8), dtype=np.float32),
        sleep_stages=np.array([4, 4], dtype=np.int64),
        epoch_times_s=np.array([0.0, 2.0], dtype=np.float64),
        sfreq=256.0,
        subject_id="demo",
    )
    detector = DummyDetector(
        [
            {"rem_probability": 0.9, "triggered": True, "consecutive_count": 3},
            {"rem_probability": 0.9, "triggered": False, "consecutive_count": 0},
        ]
    )
    decoder = DummyDecoder(
        [
            {
                "predicted_text": "flying",
                "confidence": 0.9,
                "alternatives": ["running"],
                "abstained": False,
                "raw_top_phrase": "flying",
                "raw_scores": [0.9, 0.1],
            }
        ]
    )
    controller = RealtimeDemoController(source=source, detector=detector, decoder=decoder, playback_interval_s=0.0)
    events = []

    async def _collect(payload):
        events.append(payload)

    controller.broadcast = _collect  # type: ignore[method-assign]
    asyncio.run(controller.playback())

    assert events[0]["event"] == "prediction"
    assert events[0]["predicted_text"] == "flying"
    assert events[1]["predicted_text"] == "low confidence"
    assert events[-1]["event"] == "complete"


def test_create_app_serves_dashboard_and_health(tmp_path):
    from src.realtime.demo_server import ReplayEpochSource, RealtimeDemoController, create_app

    source = ReplayEpochSource(
        data=np.zeros((1, 4, 8), dtype=np.float32),
        sleep_stages=np.array([4], dtype=np.int64),
        epoch_times_s=np.array([0.0], dtype=np.float64),
        sfreq=256.0,
        subject_id="demo",
    )
    controller = RealtimeDemoController(
        source=source,
        detector=DummyDetector([{"rem_probability": 0.9, "triggered": False, "consecutive_count": 0}]),
        decoder=DummyDecoder([]),
        playback_interval_s=0.0,
    )
    app = create_app({"autostart": False}, controller=controller)
    client = TestClient(app)

    index_response = client.get("/")
    health_response = client.get("/health")

    assert index_response.status_code == 200
    assert "Somniloquy Realtime Demo" in index_response.text
    assert health_response.json()["status"] == "ok"
