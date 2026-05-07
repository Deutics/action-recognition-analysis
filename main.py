"""
Main Entry Point for Posture Detection Service
Single-process architecture:
- Loads one shared inference backend
- Processes all stream frames sequentially (round-robin)
- Keeps fall tracking + notifications independent per stream
"""

import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path
import ast

from config.posture_config import PostureConfig
from config.constants import (
    STREAM_SOURCES_FILE_PATH,
    POSE_MODEL_PATH,
    HUMAN_DETECTOR_MODEL_PATH,
)
from inference.runtime_factory import InferenceRuntimeFactory
from service.posture_recognition_service import PoseRecognition
from utils.logger import get_logger


logger = get_logger(__name__)


def load_streams_config(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Streams config not found: {p.resolve()}")

    streams = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(streams, list) or not streams:
        raise ValueError("stream_sources.json must contain a non-empty list")

    for i, s in enumerate(streams):
        if "source" not in s or "source_id" not in s:
            raise ValueError(f"Stream entry #{i} must contain 'source' and 'source_id'")

    return streams


def parse_zone(stream_entry: Dict[str, Any]):
    zone = stream_entry.get("normalized_zone_vertices")
    if not zone:
        return None

    if isinstance(zone, str):
        return ast.literal_eval(zone)

    return zone


async def run_all_streams(streams: List[Dict[str, Any]]):
    shared_backend, backend_name = InferenceRuntimeFactory.create_pose_backend(
        model_path=POSE_MODEL_PATH,
        use_tracking=False,
    )
    shared_human_detector = None
    try:
        shared_human_detector = InferenceRuntimeFactory.create_human_detector(
            preferred_model=HUMAN_DETECTOR_MODEL_PATH
        )
    except Exception as e:
        logger.warning(f"Shared human detector unavailable; fall alerts will be gated off: {e}")
    logger.info(f"Shared backend loaded once: {backend_name}")

    services: List[PoseRecognition] = []
    for s in streams:
        service = PoseRecognition(
            video_source=s["source"],
            source_id=s["source_id"],
            config=PostureConfig(),
            confidence_threshold=0.5,
            normalized_zone_vertices=parse_zone(s),
            backend=shared_backend,
            backend_name=backend_name,
            use_backend_track_ids=False,
            reconnect_interval_sec=5.0,
            person_detector=shared_human_detector,
            person_box_overlap_threshold=0.6,
        )
        services.append(service)

    try:
        for service in services:
            await service.start()

        logger.info(f"Started {len(services)} streams with one shared inference backend")

        while True:
            should_stop = False
            for service in services:
                keep_running = await service.process_once()
                if not keep_running:
                    should_stop = True
                    break

            if should_stop:
                logger.info("Stop signal received from UI, shutting down all streams")
                break

            # Small scheduler pacing prevents busy-spin CPU saturation over long runtimes.
            await asyncio.sleep(0.002)

    finally:
        for service in services:
            await service.stop()


def main():
    streams = load_streams_config(STREAM_SOURCES_FILE_PATH)
    logger.info(f"Starting {len(streams)} stream(s) in single-engine mode")

    try:
        asyncio.run(run_all_streams(streams))
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down...")


if __name__ == "__main__":
    main()
