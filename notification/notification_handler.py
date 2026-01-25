import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

from utils.logger import get_logger
from notification.fall_detection_alerts import FallDetectionAlerts

logger = get_logger(__name__)


@dataclass
class AlertJob:
    person_id: int
    source_id: str
    frame: np.ndarray
    timestamp: datetime
    confidence: float = 0.9


class NotificationHandler:
    def __init__(self, output_dir: str = "output/falling_detections", queue_size: int = 50, workers: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._notification_pusher = FallDetectionAlerts({})
        self._queue: asyncio.Queue[AlertJob] = asyncio.Queue(maxsize=queue_size)

        self._workers = []
        self._workers_count = workers
        self._running = False

    async def start(self):
        """Start background workers once per process."""
        if self._running:
            return
        self._running = True
        for i in range(self._workers_count):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))
        logger.info(f"Notification workers started: {self._workers_count}")

    async def stop(self):
        """Flush queue then stop workers and close resources."""
        if not self._running:
            return

        # ✅ Wait for all queued alerts to be processed
        await self._queue.join()

        self._running = False
        for _ in self._workers:
            await self._queue.put(None)

        await asyncio.gather(*self._workers, return_exceptions=True)

        # ✅ close aiohttp session
        await self._notification_pusher.close()

        logger.info("Notification workers stopped")

    async def notify_fall(self, person_id: int, frame: np.ndarray, source_id: str, confidence: float = 0.9):
        """
        Non-blocking: puts alert job in queue and returns immediately.
        Drops if queue is full (or you can wait with timeout).
        """
        if not self._running:
            await self.start()

        job = AlertJob(
            person_id=person_id,
            source_id=source_id,
            frame=frame.copy(),  # IMPORTANT: copy since stream loop will reuse frame buffer
            timestamp=datetime.now(),
            confidence=confidence
        )

        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning(f"Notification queue full, dropping alert: {source_id} person={person_id}")

    async def _worker_loop(self, worker_id: int):
        while True:
            job = await self._queue.get()
            if job is None:
                break
            try:
                await self._send(job)
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Failed to send alert: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    async def _send(self, job: AlertJob):
        # Save locally (optional)
        ts = job.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"falling_person{job.person_id}_{job.source_id}_{ts}.jpg"
        filepath = self.output_dir / filename

        # Writing can block; it’s OK here because it’s not in the stream loop anymore.
        # If you want, offload to thread too.
        # cv2.imwrite(str(filepath), job.frame)

        # Send push + sms (in background)
        ok = await self._notification_pusher.send_fall_alert(camera_name=job.source_id, alert_type="fall",
                                                             person_id=str(job.person_id), confidence=job.confidence,
                                                             image=job.frame)

        if ok:
            logger.warning(f"ALERT SENT: ...")
        else:
            logger.info(f"ALERT SKIPPED (cooldown/threshold): person={...}")
        return str(filepath)