"""
Main Entry Point for Posture Detection Service
Handles multiprocess stream processing
"""

import asyncio
import multiprocessing as mp
import json
from multiprocessing import Process
from typing import List, Dict, Any
from pathlib import Path

from config.posture_config import PostureConfig
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


async def run_posture_service(source: str, source_id: str):
    """
    Run posture detection service for a single stream
    """
    service = PoseRecognition(
        video_source=source,
        source_id=source_id,
        model_path="yolo26n-pose.pt",
        config=PostureConfig(),
        confidence_threshold=0.5
    )
    await service.run()


def process_entry(source: str, source_id: str):
    """
    Process entrypoint must be SYNC.
    It creates/runs the asyncio event loop inside the child process.
    """
    try:
        asyncio.run(run_posture_service(source, source_id))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Process error for source {source_id}: {e}")


def main():
    # IMPORTANT on macOS: spawn prevents many weird issues with async + ML libs
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # already set
        pass

    streams = load_streams_config("stream_sources.json")
    streams = streams[0:1]

    logger.info(f"Starting {len(streams)} posture detection service(s)")

    processes = []
    for s in streams:
        logger.info(f"Starting process for {s['source_id']}")
        p = Process(
            target=process_entry,   # ✅ sync wrapper, not async function
            args=(s["source"], s["source_id"]),
            name=f"PoseService-{s['source_id']}",
            daemon=False
        )
        p.start()
        processes.append(p)

    logger.info(f"All {len(processes)} processes started")

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Interrupt received, terminating all processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        for p in processes:
            if p.is_alive():
                p.kill()
        logger.info("All processes terminated")


if __name__ == "__main__":
    main()