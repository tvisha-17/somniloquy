#!/usr/bin/env python
"""Launch the Somniloquy realtime demo server."""

from __future__ import annotations

import argparse
import pathlib
import sys

import uvicorn
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.realtime.demo_server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Somniloquy realtime demo server.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "demo_realtime.yaml",
        help="Path to YAML config for the realtime demo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    app = create_app(config)
    uvicorn.run(
        app,
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 8000)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
