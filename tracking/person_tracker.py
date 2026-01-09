# person_tracker.py

import time
from dataclasses import dataclass
from typing import Dict, Optional
from config.posture_config import PostureLabel


@dataclass
class PersonState:
    posture: str
    posture_start_time: float
    notified: bool = False


class PersonTracker:
    def __init__(self, lying_threshold: float):
        self.lying_threshold = lying_threshold
        self.persons: Dict[int, PersonState] = {}

    def update(self, person_id: int, posture: str) -> bool:
        """
        Returns True only once when a FALLING event is triggered
        """
        now = time.time()

        if person_id not in self.persons:
            self.persons[person_id] = PersonState(posture, now)
            return False

        state = self.persons[person_id]

        # posture changed → reset state
        if posture != state.posture:
            state.posture = posture
            state.posture_start_time = now
            state.notified = False
            return False

        # lying persistence check
        if posture == PostureLabel.LYING and not state.notified:
            if now - state.posture_start_time >= self.lying_threshold:
                state.notified = True
                return True

        return False

    def cleanup(self, active_ids: set):
        self.persons = {
            pid: state for pid, state in self.persons.items()
            if pid in active_ids
        }
