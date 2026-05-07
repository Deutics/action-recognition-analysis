"""
Fall Detection Alert Service using Twilio + Webhook (async-safe, non-blocking best practices)

Key improvements vs old version:
- NO blocking Twilio SDK calls on the event loop: Twilio sends are offloaded via asyncio.to_thread()
- Reuse a single aiohttp.ClientSession (no per-request session creation)
- Proper JPEG encoding before base64 (do NOT base64 raw BGR bytes)
- Remove the 60-second await sleep (that was freezing your pipeline)
- Bounded concurrency with a semaphore (prevents overload when multiple alerts happen)
- Cleaner logging + safer env parsing + E.164 validation
- Proper resource cleanup via async close()

Drop-in usage:
    alerts = FallDetectionAlerts(config)
    await alerts.send_fall_alert(...)
    await alerts.close()   # at shutdown

Dependencies:
    pip install aiohttp twilio opencv-python numpy
"""

from __future__ import annotations

import os
import re
import json
import base64
import tempfile
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

import aiohttp
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from utils.logger import get_logger
from config.constants import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM_PHONE,
    TO_PHONE_NUMBERS,
    RIGGUARDIAN_WEBHOOK_URL,
    ENABLE_SMS,
    RUNTIME_ENV,
)

LOGGER = get_logger(__name__)


E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_phone_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _jpeg_base64(image_bgr: np.ndarray, quality: int = 80) -> Tuple[str, int]:
    """
    Convert BGR numpy image -> JPEG bytes -> base64 string.
    Returns (b64_string, jpg_bytes_len)
    """
    ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to JPEG-encode image")
    jpg_bytes = enc.tobytes()
    return base64.b64encode(jpg_bytes).decode("utf-8"), len(jpg_bytes)


def _json_safe(obj: Any):
    """Recursively convert non-JSON types (datetime, numpy, etc.) to JSON-safe."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalars
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


@dataclass
class AlertConfig:
    min_confidence: float = 0.7
    cooldown_seconds: int = 30
    webhook_timeout_sec: int = 10
    webhook_max_retries: int = 2
    webhook_backoff_base: float = 0.7  # seconds
    max_concurrent_sends: int = 3
    include_image_in_webhook: bool = True
    webhook_jpeg_quality: int = 80
    save_dir: str = "output/fall_notifications"
    location: str = "55CPW"


class FallDetectionAlerts:
    """Service for sending fall detection alerts via Twilio SMS + webhook push."""

    def __init__(self, config: dict):
        # --- Twilio credentials ---
        self.account_sid = TWILIO_ACCOUNT_SID
        self.auth_token = TWILIO_AUTH_TOKEN
        self.from_phone = TWILIO_FROM_PHONE
        self.to_phones = _parse_phone_list(TO_PHONE_NUMBERS)
        self.sms_enabled = ENABLE_SMS == "1"
        # --- webhook ---
        self.push_notification_url = RIGGUARDIAN_WEBHOOK_URL

        if self.sms_enabled:
            self._validate_phone_numbers()
        else:
            LOGGER.info(f"SMS notifications disabled for runtime env '{RUNTIME_ENV}'")

        # --- settings ---
        self.cfg = AlertConfig()

        self.last_alert_time: Dict[str, datetime] = {}  # per-camera cooldown

        # --- Twilio client ---
        self.client: Optional[Client] = None
        if self.sms_enabled and self.account_sid and self.auth_token:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                LOGGER.info("✅ Twilio client initialized")
            except Exception as e:
                LOGGER.error(f"❌ Failed to initialize Twilio client: {e}", exc_info=True)
                self.client = None
        else:
            if self.sms_enabled:
                LOGGER.warning("⚠️ Twilio credentials missing; SMS sending will be disabled")

        # --- HTTP session (reused) ---
        self._session: Optional[aiohttp.ClientSession] = None

        # bounded concurrency
        self._sem = asyncio.Semaphore(self.cfg.max_concurrent_sends)

        _ensure_dir(self.cfg.save_dir)

    # -------------------- lifecycle --------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.webhook_timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # -------------------- validation & cooldown --------------------

    def _validate_phone_numbers(self):
        if self.from_phone and not E164_RE.match(self.from_phone):
            LOGGER.warning(f"⚠️ From phone '{self.from_phone}' may not be E.164 (+123...) format")

        valid = []
        for p in self.to_phones:
            if E164_RE.match(p):
                valid.append(p)
            else:
                LOGGER.error(f"❌ Invalid recipient phone number: '{p}' (must be E.164: +1234567890)")

        if not valid:
            raise ValueError("No valid recipient phone numbers found. Use E.164 format: +1234567890")

        self.to_phones = valid
        LOGGER.info(f"📱 Configured {len(self.to_phones)} recipient number(s)")

    def should_send_alert(self, camera_name: str, confidence: float) -> bool:
        if confidence < self.cfg.min_confidence:
            return False

        now = _utcnow()
        last = self.last_alert_time.get(camera_name)
        if last:
            delta = (now - last).total_seconds()
            if delta < self.cfg.cooldown_seconds:
                LOGGER.info(
                    f"⏳ Cooldown active for {camera_name}: {delta:.1f}s < {self.cfg.cooldown_seconds}s"
                )
                return False
        return True

    # -------------------- message formatting --------------------

    def format_alert_message(
        self,
        camera_name: str,
        alert_type: str,
        person_id: str,
        confidence: float,
        timestamp: datetime,
        image_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        msg = [
            "🚨 FALL DETECTED 🚨",
            f"Location: {self.cfg.location}",
            f"Camera: {camera_name}",
            f"Confidence: {confidence:.1%}",
            f"Time: {timestamp.astimezone().strftime('%Y-%m-%d %I:%M%p')}",
        ]
        if image_path:
            msg.append("View camera: https://rigguardian.com/cameras")
        msg.append("")
        msg.append("Please check the location immediately.")
        return "\n".join(msg)

    # -------------------- disk: save image + metadata --------------------

    async def save_fall_image_to_file(
        self,
        camera_name: str,
        person_id: str,
        confidence: float,
        image_bgr: np.ndarray,
        keypoints: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save JPG + JSON metadata (async-friendly).
        Uses asyncio.to_thread for file I/O to avoid blocking the event loop.
        """
        ts = _utcnow()
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        filename = f"{ts_str}_{camera_name}_ReadImage.jpg"
        meta_filename = f"{ts_str}_{camera_name}_ReadImage.json"
        img_path = os.path.join(self.cfg.save_dir, filename)
        meta_path = os.path.join(self.cfg.save_dir, meta_filename)

        def _write_files():
            # write jpg
            ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise ValueError("Failed to encode image as JPEG")
            with open(img_path, "wb") as f:
                f.write(enc.tobytes())

            md = {
                "component_name": camera_name,
                "method_name": "ReadImage",
                "tags": ["Fall"],
                "timestamp": ts.isoformat(),
                "additional_metadata": {
                    "person_id": str(person_id),
                    "confidence": f"{confidence:.3f}",
                    "event_type": "fall",
                    "vision_service": "yolo-pose",
                    "keypoints": keypoints,
                },
            }
            if extra_metadata:
                md["additional_metadata"].update(extra_metadata)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(md, f, indent=2)

            return {"status": "success", "filename": filename, "path": img_path, "metadata_path": meta_path}

        try:
            return await asyncio.to_thread(_write_files)
        except Exception as e:
            LOGGER.error(f"❌ save_fall_image_to_file failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def save_image_locally_temp(self, image_bgr: np.ndarray, person_id: str) -> str:
        """
        Save a temp JPG path for SMS reference. (Twilio SMS cannot attach an image unless it's a public URL;
        keeping this for your message content / logging.)
        """
        ts = _utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"fall_detection_{person_id}_{ts}.jpg"
        out_path = os.path.join(tempfile.gettempdir(), filename)

        def _write():
            ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise ValueError("Failed to encode image as JPEG")
            with open(out_path, "wb") as f:
                f.write(enc.tobytes())
            return out_path

        try:
            return await asyncio.to_thread(_write)
        except Exception as e:
            LOGGER.error(f"❌ Failed to save temp image: {e}", exc_info=True)
            return ""

    # -------------------- Twilio SMS (blocking SDK -> thread) --------------------

    def _send_sms_blocking(self, body: str) -> int:
        """
        Runs in a worker thread. Returns count of successful sends.
        """
        if self.client is None:
            LOGGER.info("Twilio client not initialized; skipping SMS")
            return 0

        success = 0
        for phone in self.to_phones:
            try:
                LOGGER.info(f"📱 Sending SMS from {self.from_phone} to {phone}")
                msg = self.client.messages.create(body=body, from_=self.from_phone, to=phone)
                LOGGER.info(f"✅ SMS sent to {phone} SID={msg.sid}")
                success += 1
            except TwilioRestException as e:
                LOGGER.error(
                    f"❌ Twilio SMS failed to {phone}: code={getattr(e, 'code', None)} "
                    f"status={getattr(e, 'status', None)} msg={getattr(e, 'msg', str(e))}"
                )
            except Exception as e:
                LOGGER.error(f"❌ SMS failed to {phone}: {e}", exc_info=True)

        return success

    # -------------------- Webhook push (async) --------------------

    async def send_webhook_notification(
        self,
        camera_name: str,
        alert_type: str,
        person_id: str,
        confidence: float,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        image_bgr: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Sends webhook with retries and backoff. Reuses aiohttp session.
        """
        if not self.push_notification_url:
            LOGGER.warning("No webhook URL configured; skipping push")
            return False

        payload = {
            "alert_type": alert_type,
            "camera_name": camera_name,
            "person_id": str(person_id),
            "location": self.cfg.location,
            "confidence": confidence,
            "severity": "critical",
            "title": "Fall Alert Detected",
            "message": f"Fall detected at {self.cfg.location} on {camera_name} with {confidence:.1%} confidence",
            "requires_immediate_attention": True,
            "notification_type": "fall_detection",
            "timestamp": timestamp.isoformat(),
            "metadata": _json_safe(metadata or {}),
            "actions": [
                {"action": "view_camera", "title": "View Camera"},
                {"action": "acknowledge", "title": "Acknowledge"},
            ],
        }

        if self.cfg.include_image_in_webhook and image_bgr is not None:
            try:
                b64, jpg_len = _jpeg_base64(image_bgr, quality=self.cfg.webhook_jpeg_quality)
                payload["image"] = b64
                payload["image_filename"] = f"fall_alert_{camera_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                payload["image_encoding"] = "jpeg_base64"
                payload["image_bytes"] = jpg_len
            except Exception as e:
                LOGGER.warning(f"⚠️ Failed to attach image to webhook payload: {e}")

        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FallDetectionSystem/1.0",
            "X-Alert-Type": alert_type,
            "X-Severity": "critical",
            "Accept": "application/json",
        }

        for attempt in range(self.cfg.webhook_max_retries + 1):
            try:
                async with session.post(self.push_notification_url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        LOGGER.info(f"✅ Webhook delivered (status={resp.status})")
                        return True
                    LOGGER.warning(f"⚠️ Webhook failed status={resp.status} body={text[:500]}")
            except asyncio.TimeoutError:
                LOGGER.warning("⚠️ Webhook timeout")
            except aiohttp.ClientError as e:
                LOGGER.warning(f"⚠️ Webhook client error: {e}")
            except Exception as e:
                LOGGER.error(f"❌ Webhook exception: {e}", exc_info=True)

            if attempt < self.cfg.webhook_max_retries:
                backoff = self.cfg.webhook_backoff_base * (2 ** attempt)
                await asyncio.sleep(backoff)

        return False

    # -------------------- Public API: send alert --------------------

    async def send_fall_alert(
        self,
        camera_name: str,
        alert_type: str,
        person_id: str,
        confidence: float,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Main entry: saves files + sends SMS + sends webhook.
        Safe to call from your background notification worker.

        IMPORTANT:
        - `image` must be a BGR numpy array (OpenCV frame).
        """
        async with self._sem:
            try:
                if not self.should_send_alert(camera_name, confidence):
                    return False

                timestamp = _utcnow()
                self.last_alert_time[camera_name] = timestamp
                LOGGER.info(f"🕐 Alert time recorded for camera {camera_name}: {timestamp.isoformat()}")

                # Save file + metadata (non-blocking I/O)
                keypoints = (metadata or {}).get("keypoints") if isinstance(metadata, dict) else None
                await self.save_fall_image_to_file(
                    camera_name=camera_name,
                    person_id=person_id,
                    confidence=confidence,
                    image_bgr=image,
                    keypoints=keypoints,
                    extra_metadata={"source": "fall_detection"},
                )

                # Save temp image path (optional; SMS can't embed local file, but you keep it for logs)
                image_path = await self.save_image_locally_temp(image, person_id)

                # Build SMS text
                sms_body = self.format_alert_message(
                    camera_name=camera_name,
                    alert_type=alert_type,
                    person_id=person_id,
                    confidence=confidence,
                    timestamp=timestamp,
                    image_path=image_path,
                    metadata=metadata,
                )

                # Send webhook push (async)
                push_success = await self.send_webhook_notification(camera_name=camera_name,
                                                                    alert_type=alert_type,
                                                                    person_id=person_id,
                                                                    confidence=confidence,
                                                                    timestamp=timestamp,
                                                                    metadata=metadata,
                                                                    image_bgr=image)

                # Send SMS (Twilio is blocking -> thread)
                sms_success = 0
                if self.client is not None and self.from_phone:
                    sms_success = await asyncio.to_thread(self._send_sms_blocking, sms_body)
                    # pass

                if sms_success > 0:
                    LOGGER.info(f"✅ SMS sent to {sms_success}/{len(self.to_phones)} recipients")
                else:
                    LOGGER.warning("⚠️ No SMS recipients succeeded (or SMS disabled)")

                if push_success:
                    LOGGER.info("✅ Push webhook sent successfully")
                else:
                    LOGGER.warning("⚠️ Push webhook failed")

                # Consider alert successful if either channel succeeds
                return (sms_success > 0) or push_success

            except Exception as e:
                LOGGER.error(f"❌ Error sending fall alert: {e}", exc_info=True)
                return False
