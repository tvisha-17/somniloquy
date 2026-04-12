"""FastAPI demo server for simulated realtime playback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from src.models.zuna_decoder import ZUNAForSpeechDecoding
from src.realtime.rem_detector import REMDetector
from src.realtime.speech_decoder_realtime import PhraseBank, RealTimeSpeechDecoder, WindowStatisticsEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Somniloquy Realtime Demo</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 0; background: #f4efe6; color: #1c1b19; }
    main { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; padding: 20px; }
    section { background: rgba(255,255,255,0.82); border-radius: 18px; padding: 18px; box-shadow: 0 12px 30px rgba(0,0,0,0.08); }
    h1 { margin: 0 0 14px; font-size: 28px; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    .metric { font-size: 36px; margin: 6px 0 14px; }
    .mono { font-family: Menlo, monospace; font-size: 13px; }
    #hypnogram { display: flex; flex-wrap: wrap; gap: 6px; }
    .stage { padding: 6px 8px; border-radius: 999px; font-size: 12px; }
    .stage.rem { background: #1d7a72; color: white; }
    .stage.other { background: #ded7cb; color: #40352a; }
    #cloud { display: flex; flex-wrap: wrap; gap: 8px; }
    #cloud span { background: #efe6d8; padding: 6px 10px; border-radius: 999px; }
    ul { padding-left: 18px; }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Somniloquy Realtime Demo</h1>
      <div>Current prediction</div>
      <div class="metric" id="prediction">Waiting for playback</div>
      <div>Confidence</div>
      <div class="metric" id="confidence">0.00</div>
      <div>REM probability</div>
      <div class="metric" id="remprob">0.00</div>
      <h2>Top alternatives</h2>
      <ul id="alternatives"></ul>
    </section>
    <section>
      <h2>Hypnogram</h2>
      <div id="hypnogram"></div>
      <h2>Dream cloud</h2>
      <div id="cloud"></div>
      <h2>Latest event</h2>
      <pre class="mono" id="event">{}</pre>
    </section>
  </main>
  <script>
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');
    const remprob = document.getElementById('remprob');
    const alternatives = document.getElementById('alternatives');
    const hypnogram = document.getElementById('hypnogram');
    const cloud = document.getElementById('cloud');
    const eventBox = document.getElementById('event');
    const counts = {};
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onmessage = (message) => {
      const data = JSON.parse(message.data);
      eventBox.textContent = JSON.stringify(data, null, 2);
      if (data.event === 'complete') {
        prediction.textContent = 'Playback complete';
        return;
      }
      prediction.textContent = data.predicted_text;
      confidence.textContent = Number(data.confidence).toFixed(2);
      remprob.textContent = Number(data.rem_probability).toFixed(2);

      alternatives.innerHTML = '';
      (data.alternatives || []).forEach((item) => {
        const li = document.createElement('li');
        li.textContent = item;
        alternatives.appendChild(li);
      });

      const stage = document.createElement('span');
      stage.className = `stage ${data.sleep_stage === 4 ? 'rem' : 'other'}`;
      stage.textContent = `${data.timestamp.toFixed(1)}s · stage ${data.sleep_stage}`;
      hypnogram.appendChild(stage);
      if (hypnogram.children.length > 32) {
        hypnogram.removeChild(hypnogram.firstChild);
      }

      if (!data.abstained && data.predicted_text && data.predicted_text !== 'low confidence') {
        counts[data.predicted_text] = (counts[data.predicted_text] || 0) + 1;
      }
      cloud.innerHTML = '';
      Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12)
        .forEach(([phrase, count]) => {
          const chip = document.createElement('span');
          chip.textContent = `${phrase} ×${count}`;
          cloud.appendChild(chip);
        });
    };
  </script>
</body>
</html>"""


@dataclass
class ReplayEpochSource:
    """Replay source built from a preprocessed epoch file."""

    data: np.ndarray
    sleep_stages: np.ndarray
    epoch_times_s: np.ndarray
    ch_names: list[str]
    sfreq: float
    subject_id: str


def load_replay_source(path: Path) -> ReplayEpochSource:
    """Load a replay source from a preprocessed `.npz` file."""
    payload = np.load(str(path), allow_pickle=True)
    data = payload["data"].astype(np.float32)
    sleep_stages = payload["sleep_stages"].astype(np.int64)
    epoch_times_s = payload["epoch_times_s"].astype(np.float64)
    ch_names = [str(name) for name in payload["ch_names"].tolist()]
    sfreq = float(payload["sfreq"])
    subject_id = str(payload["subject_id"])
    logger.info(
        "load_replay_source data_shape=%s sleep_stage_shape=%s epoch_times_shape=%s",
        tuple(data.shape),
        tuple(sleep_stages.shape),
        tuple(epoch_times_s.shape),
    )
    return ReplayEpochSource(
        data=data,
        sleep_stages=sleep_stages,
        epoch_times_s=epoch_times_s,
        ch_names=ch_names,
        sfreq=sfreq,
        subject_id=subject_id,
    )


def load_realtime_model(config: dict, source: Optional[ReplayEpochSource] = None) -> torch.nn.Module:
    """Load the runtime model from checkpoint or create a baseline fallback."""
    model_mode = str(config.get("model_mode", "checkpoint"))
    if model_mode == "heuristic_baseline":
        logger.warning("Using heuristic baseline model for realtime demo.")
        return WindowStatisticsEncoder(target_embed_dim=int(config.get("target_embed_dim", 384)))

    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = dict(checkpoint.get("config", {}))
    ch_names = [] if source is None else list(source.ch_names)

    model = ZUNAForSpeechDecoding(
        ch_names=ch_names,
        zuna_model_name=str(checkpoint_config.get("zuna_model_name", config["zuna_model_name"])),
        target_embed_dim=int(checkpoint_config.get("target_embed_dim", config["target_embed_dim"])),
        dropout=float(checkpoint_config.get("dropout", config["dropout"])),
        latent_dim=int(checkpoint_config.get("latent_dim", config.get("latent_dim", 1024))),
        backbone_mode=str(checkpoint_config.get("backbone_mode", config.get("backbone_mode", "auto"))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class RealtimeDemoController:
    """Coordinates playback, detection, decoding, and WebSocket broadcast."""

    def __init__(
        self,
        source: ReplayEpochSource,
        detector: REMDetector,
        decoder: RealTimeSpeechDecoder,
        playback_interval_s: float = 0.2,
    ) -> None:
        self.source = source
        self.detector = detector
        self.decoder = decoder
        self.playback_interval_s = playback_interval_s
        self.connections: set[WebSocket] = set()
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, payload: dict) -> None:
        stale = []
        for websocket in self.connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.connections.discard(websocket)

    async def playback(self) -> None:
        """Replay epochs in timestamp order and broadcast events."""
        logger.info("Starting playback for subject=%s n_epochs=%d", self.source.subject_id, self.source.data.shape[0])
        for index in range(self.source.data.shape[0]):
            window = self.source.data[index]
            stage = int(self.source.sleep_stages[index])
            timestamp = float(self.source.epoch_times_s[index])
            rem_state = self.detector.process_window(window, stage_hint=stage)

            if rem_state["triggered"]:
                payload = self.decoder.process_window(
                    window,
                    timestamp=timestamp,
                    rem_probability=rem_state["rem_probability"],
                )
            else:
                payload = {
                    "timestamp": timestamp,
                    "predicted_text": "low confidence",
                    "confidence": 0.0,
                    "alternatives": [],
                    "abstained": True,
                    "raw_top_phrase": None,
                    "raw_scores": [],
                    "rem_probability": rem_state["rem_probability"],
                }

            payload.update(
                {
                    "sleep_stage": stage,
                    "rem_triggered": rem_state["triggered"],
                    "event": "prediction",
                    "subject_id": self.source.subject_id,
                }
            )
            await self.broadcast(payload)
            await asyncio.sleep(self.playback_interval_s)

        await self.broadcast(
            {
                "event": "complete",
                "subject_id": self.source.subject_id,
                "timestamp": float(self.source.epoch_times_s[-1]) if len(self.source.epoch_times_s) else 0.0,
            }
        )

    async def start(self) -> None:
        async with self._lock:
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self.playback())


def build_demo_controller(config: dict, model: Optional[torch.nn.Module] = None) -> RealtimeDemoController:
    """Assemble source, detector, decoder, and controller from config."""
    source = load_replay_source(Path(config["replay_epoch_file"]))
    detector = REMDetector(
        threshold=float(config.get("rem_threshold", 0.7)),
        required_consecutive=int(config.get("required_consecutive", 3)),
    )
    phrase_bank = PhraseBank.from_target_embedding_dir(Path(config["target_embeddings_dir"]))
    runtime_model = model or load_realtime_model(config, source=source)
    decoder = RealTimeSpeechDecoder(
        model=runtime_model,
        phrase_bank=phrase_bank,
        confidence_threshold=float(config.get("confidence_threshold", 0.3)),
        smoothing_window=int(config.get("smoothing_window", 5)),
        required_majority=int(config.get("required_majority", 3)),
        top_k=int(config.get("top_k", 3)),
        device=str(config.get("device", "cpu")),
    )
    return RealtimeDemoController(
        source=source,
        detector=detector,
        decoder=decoder,
        playback_interval_s=float(config.get("playback_interval_s", 0.2)),
    )


def create_app(config: dict, controller: Optional[RealtimeDemoController] = None) -> FastAPI:
    """Create the FastAPI app for the realtime demo."""
    controller = controller or build_demo_controller(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.controller = controller
        if bool(config.get("autostart", True)):
            await controller.start()
        yield

    app = FastAPI(title="Somniloquy Realtime Demo", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "subject_id": controller.source.subject_id})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await controller.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await controller.disconnect(websocket)
        except Exception:
            await controller.disconnect(websocket)

    return app
