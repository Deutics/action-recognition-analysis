"""Person state tracking with ENHANCED DEBUG LOGGING."""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from config.detection_config import PostureLabel
from core.breakin.signal_history import SignalHistoryManager


@dataclass
class PersonState:
    """State tracking for individual person."""
    # Posture tracking
    posture: str
    posture_start_time: float
    notified: bool = False
    last_trigger_time: Optional[float] = None
    last_seen_time: float = 0.0
    
    # Force tracking
    force_score: float = 0.0
    force_frame_count: int = 0
    force_start_frame: Optional[int] = None
    force_notified: bool = False
    force_last_trigger_time: Optional[float] = None
    history_manager: SignalHistoryManager = field(default_factory=SignalHistoryManager)
    force_frames_seen: int = 0  # Total frames processed for this person


class PersonTracker:
    """
    Enhanced debug version - logs ALL persons, not just active ones.
    """
    
    def __init__(self, lying_threshold: float = 3.0, cooldown_period: float = 10.0,
                 person_timeout: float = 3.0, force_intent_cooldown: float = 5.0,
                 energy_threshold: float = 3.0, min_force_frames: int = 20,
                 decay_per_frame: float = 0.05, temporal_window: int = 15,
                 frame_score_threshold: float = 0.3):
        # Posture thresholds
        self.lying_threshold = lying_threshold
        self.cooldown_period = cooldown_period
        self.person_timeout = person_timeout
        
        # Force thresholds
        self.force_intent_cooldown = force_intent_cooldown
        self.energy_threshold = energy_threshold
        self.min_force_frames = min_force_frames
        self.decay_per_frame = decay_per_frame
        self.temporal_window = temporal_window
        self.frame_score_threshold = frame_score_threshold
        
        # State storage
        self.persons: Dict[int, PersonState] = {}
        self.global_frame_count = 0
        
        # DEBUG: Track all processed persons
        self.processed_persons_this_frame = set()
    
    def update(self, person_id: int, posture: str) -> bool:
        """Update posture state, returns True if fall alert triggered."""
        now = time.time()
        
        if person_id not in self.persons:
            self.persons[person_id] = PersonState(
                posture=posture, 
                posture_start_time=now, 
                last_seen_time=now
            )
            return False
        
        state = self.persons[person_id]
        state.last_seen_time = now
        
        if posture != state.posture:
            state.posture = posture
            state.posture_start_time = now
            state.notified = False
            return False
        
        if posture == PostureLabel.LYING:
            duration = now - state.posture_start_time
            
            if duration >= self.lying_threshold:
                if not state.notified and (
                    state.last_trigger_time is None or
                    now - state.last_trigger_time >= self.cooldown_period
                ):
                    state.notified = True
                    state.last_trigger_time = now
                    return True
        
        return False
    
    def get_signal_histories(self, person_id: int) -> Dict:
        """Get signal history deques for a person."""
        now = time.time()
        
        if person_id not in self.persons:
            self.persons[person_id] = PersonState(
                posture="Unknown",
                posture_start_time=now,
                last_seen_time=now
            )
        
        state = self.persons[person_id]
        return state.history_manager.get_histories_dict()
    
    def update_force(self, person_id: int, classification_result: Tuple[str, float]) -> bool:
        """
        Update force state with ENHANCED DEBUGGING.
        Logs ALL persons, shows why they're not triggering alerts.
        """
        now = time.time()
        self.global_frame_count += 1
        
        # Track this person was processed
        self.processed_persons_this_frame.add(person_id)
        
        # Initialize new person if needed
        if person_id not in self.persons:
            self.persons[person_id] = PersonState(
                posture="Unknown",
                posture_start_time=now,
                last_seen_time=now
            )
            print(f"[NEW PERSON] Person {person_id} initialized")
        
        state = self.persons[person_id]
        state.last_seen_time = now
        state.force_frames_seen += 1
        
        # Extract classification result
        label, score = classification_result
        
        # Update energy with decay
        state.force_score = self._accumulate_energy(state.force_score, score)
        
        # Update consecutive frame counter
        if score >= self.frame_score_threshold:
            state.force_frame_count += 1
            if state.force_start_frame is None:
                state.force_start_frame = self.global_frame_count
        else:
            if score < 0.1:
                state.force_frame_count = 0
                state.force_start_frame = None
                state.force_notified = False
        
        # Check cooldown
        cooldown_active = (
            state.force_last_trigger_time is not None and
            now - state.force_last_trigger_time < self.force_intent_cooldown
        )
        
        # Check if person has enough history
        has_sufficient_history = state.force_frames_seen >= self.temporal_window
        
        # Check alert conditions
        alert = self._should_trigger_force_alert(
            state, cooldown_active, has_sufficient_history
        )
        
        # # =====================================================================
        # # ENHANCED DEBUG OUTPUT - Shows ALL persons
        # # =====================================================================
        #
        # # Show basic info for everyone
        # print(f"\n[Person {person_id}] Label: {label}, Score: {score:.3f}, "
        #       f"Frames: {state.force_frame_count}/{self.min_force_frames}, "
        #       f"Energy: {state.force_score:.2f}/{self.energy_threshold}")
        #
        # # Show detailed info if person is building up OR has high score
        # if state.force_frame_count > 0 or state.force_score > 0.5 or score > 0.2:
        #     print(f"  └─ History: {state.force_frames_seen}/{self.temporal_window}")
        #
        #     # Show why no alert
        #     if not alert:
        #         reasons = []
        #         if cooldown_active:
        #             time_left = self.force_intent_cooldown - (now - state.force_last_trigger_time)
        #             reasons.append(f"COOLDOWN ({time_left:.1f}s left)")
        #         if state.force_score < self.energy_threshold:
        #             reasons.append(f"Energy too low ({state.force_score:.2f} < {self.energy_threshold})")
        #         if state.force_frame_count < self.min_force_frames:
        #             reasons.append(f"Frames too few ({state.force_frame_count} < {self.min_force_frames})")
        #         if not has_sufficient_history:
        #             reasons.append(f"Insufficient history ({state.force_frames_seen} < {self.temporal_window})")
        #
        #         if reasons:
        #             print(f"  └─ No Alert: {', '.join(reasons)}")
        
        # Alert triggered
        if alert:
            state.force_notified = True
            state.force_last_trigger_time = now
            state.force_frame_count = 0
            print(f"\n{'='*60}")
            print(f"🚨 FORCE ALERT TRIGGERED FOR PERSON {person_id}")
            print(f"   Energy: {state.force_score:.2f}, Consecutive Frames: {state.force_frame_count}")
            print(f"{'='*60}\n")
        
        return alert
    
    # def print_frame_summary(self):
    #     """Print summary of all tracked persons this frame."""
    #     if len(self.persons) > 1:
    #         print(f"\n[FRAME SUMMARY] Tracking {len(self.persons)} persons: {list(self.persons.keys())}")
    #         print(f"                Processed this frame: {list(self.processed_persons_this_frame)}")
    #
    #     # Reset for next frame
    #     self.processed_persons_this_frame.clear()
    
    def _accumulate_energy(self, current_energy: float, frame_score: float) -> float:
        """Update accumulated energy with decay."""
        energy = current_energy + frame_score - self.decay_per_frame
        return max(0.0, energy)
    
    def _should_trigger_force_alert(self, state: PersonState, cooldown_active: bool,
                                    has_sufficient_history: bool) -> bool:
        """Check if force alert conditions are met."""
        return (
            not cooldown_active and
            state.force_score >= self.energy_threshold and
            state.force_frame_count >= self.min_force_frames and
            has_sufficient_history
        )
    
    # def get_force_score(self, person_id: int) -> float:
    #     """Get accumulated force score for person."""
    #     if person_id in self.persons:
    #         return self.persons[person_id].force_score
    #     return 0.0
    
    def cleanup(self, active_ids: set):
        """Remove stale persons not seen recently."""
        now = time.time()
        
        # Show who's being removed
        removed = [pid for pid in self.persons.keys() if pid not in active_ids 
                   and (now - self.persons[pid].last_seen_time >= self.person_timeout)]
        if removed:
            print(f"[CLEANUP] Removing persons: {removed}")
        
        self.persons = {
            pid: state for pid, state in self.persons.items()
            if pid in active_ids or (now - state.last_seen_time < self.person_timeout)
        }
