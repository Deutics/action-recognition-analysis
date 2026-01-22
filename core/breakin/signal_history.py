"""Temporal signal history management."""
from collections import deque
from typing import Dict, Deque, Tuple, Union


class SignalHistoryManager:
    """Manages temporal histories for force detection signals."""
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.lean_history: Deque[float] = deque(maxlen=window_size)
        self.hip_history: Deque[float] = deque(maxlen=window_size)
        self.knee_history: Deque[float] = deque(maxlen=window_size)
        self.com_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.elbow_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self.compact_history: Deque[float] = deque(maxlen=window_size)
    
    def get_histories_dict(self) -> Dict:
        """Get all histories as dictionary."""
        return {
            'lean_history': self.lean_history,
            'hip_history': self.hip_history,
            'knee_history': self.knee_history,
            'com_history': self.com_history,
            'elbow_history': self.elbow_history,
            'compact_history': self.compact_history,
        }
    
    def get_history_lengths(self) -> Dict[str, int]:
        """Get lengths of all histories."""
        return {
            'lean': len(self.lean_history),
            'hip': len(self.hip_history),
            'knee': len(self.knee_history),
            'com': len(self.com_history),
            'elbow': len(self.elbow_history),
            'compact': len(self.compact_history),
        }
