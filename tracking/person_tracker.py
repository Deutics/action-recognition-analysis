# person_tracker.py

import time
from dataclasses import dataclass
from typing import Dict, Optional
from config.posture_config import PostureLabel
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PersonState:
    posture: str
    posture_start_time: float
    notified: bool = False
    last_trigger_time: Optional[float] = None
    last_seen_time: float = 0.0  # Track when person was last detected


class PersonTracker:
    def __init__(self, lying_threshold: float, cooldown_period: float = 10.0,
                 person_timeout: float = 3.0):
        self.lying_threshold = lying_threshold
        self.cooldown_period = cooldown_period
        self.person_timeout = person_timeout  # How long to keep person state after they disappear
        self.persons: Dict[int, PersonState] = {}

    def update(self, person_id: int, posture: str) -> bool:

        now = time.time()

        # Initialize state if new person
        if person_id not in self.persons:
            self.persons[person_id] = PersonState(
                posture=posture,
                posture_start_time=now,
                last_seen_time=now
            )
            # logger.info(f"[TRACKER] Person {person_id} initialized with posture={posture}")
            return False

        state = self.persons[person_id]
        state.last_seen_time = now  # Update last seen time

        # If posture changed, update posture and start time
        if posture != state.posture:
            # logger.info(
            #     f"[TRACKER] Person {person_id} posture changed: {state.posture} -> {posture}, resetting notified flag")
            state.posture = posture
            state.posture_start_time = now
            # Reset notified flag when person changes posture (e.g., gets up)
            state.notified = False
            return False

        # Only trigger notification for LYING persistence
        if posture == PostureLabel.LYING:
            lying_duration = now - state.posture_start_time

            # Has LYING persisted long enough?
            if lying_duration >= self.lying_threshold:
                # Check if already notified for this lying session
                if state.notified:
                    logger.debug(
                        f"[TRACKER] Person {person_id} already notified for this session (lying_duration={lying_duration:.2f}s)")
                    return False

                # Respect cooldown per person ID
                if state.last_trigger_time is not None:
                    cooldown_elapsed = now - state.last_trigger_time
                    if cooldown_elapsed < self.cooldown_period:
                        # logger.info(
                        #     f"[TRACKER] Person {person_id} in cooldown (elapsed={cooldown_elapsed:.2f}s, required={self.cooldown_period}s)")
                        return False

                # # TRIGGER!
                # logger.warning(
                #     f"[TRACKER] >>> TRIGGER for Person {person_id} <<< lying_duration={lying_duration:.2f}s, notified={state.notified}, last_trigger={(now - state.last_trigger_time) if state.last_trigger_time else 'None'}s ago")
                state.notified = True
                state.last_trigger_time = now
                return True
            else:
                logger.debug(
                    f"[TRACKER] Person {person_id} lying but threshold not met yet ({lying_duration:.2f}s / {self.lying_threshold}s)")

        # Otherwise, no trigger
        return False

    def cleanup(self, active_ids: set):
        """Remove persons that haven't been seen for person_timeout seconds"""
        now = time.time()
        self.persons = {
            pid: state for pid, state in self.persons.items()
            if pid in active_ids or (now - state.last_seen_time < self.person_timeout)
        }