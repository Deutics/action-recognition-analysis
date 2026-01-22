from typing import Dict, Tuple, Optional
import math
from config.detection_config import ForceIntentConfig
from .signal_calculator import SignalCalculator


class BreakinClassifier:
    """
    Classifies forceful door break-in attempts based on keypoints.
    
    Follows the same interface pattern as PostureClassifier:
    - Takes keypoints as input
    - Returns (label, confidence_score) tuple
    - Uses temporal signal histories for robust detection
    - INCLUDES DISTANCE GATING: Person must be near door to trigger force intent
    """
    
    # Classification labels
    FORCE_INTENT = "FORCE_INTENT"
    NORMAL = "NORMAL"
    INSUFFICIENT = "INSUFFICIENT"
    
    def __init__(self, config: Optional[ForceIntentConfig] = None, 
                 max_distance_to_door: float = 600.0):
        """
        Initialize break-in classifier.
        
        Args:
            config: Force detection configuration
            max_distance_to_door: Maximum distance (pixels) from door to consider force intent
                                  People farther than this are automatically classified as NORMAL
        """
        self.config = config or ForceIntentConfig()
        self.signal_calculator = SignalCalculator(
            temporal_window=self.config.thresholds.temporal_window
        )
        
        # Signal weights for aggregation
        self.weights = {
            'arm': self.config.weights.arm,
            'com': self.config.weights.com,
            'lean': self.config.weights.lean,
            'hip': self.config.weights.hip,
            'knee': self.config.weights.knee,
            'compact': self.config.weights.compact,
        }
        
        # Frame-level classification threshold
        self.frame_threshold = self.config.thresholds.frame_threshold
        
        # Distance gating threshold (CRITICAL for preventing false positives)
        self.max_distance_to_door = max_distance_to_door
    
    def classify(self, keypoints: Dict, door_center: Tuple[float, float],
                 signal_histories: Dict) -> Tuple[str, float]:
        """
        Classify force intent from keypoints.

        """
        # Check minimum keypoints
        if not self._has_minimum_keypoints(keypoints):
            return (self.INSUFFICIENT, 0.0)
        
        # CRITICAL: Check distance to door BEFORE calculating signals
        # This prevents false positives from people far from door
        distance_to_door = self._calculate_person_to_door_distance(keypoints, door_center)
        # print(f"[Distance] Person hip: {keypoints.get('hip_mid')}, "
        #       f"Door: {door_center}, Distance: {distance_to_door:.1f}px, "
        #       f"Max: {self.max_distance_to_door}px, "
        #       f"{'PASS' if distance_to_door <= self.max_distance_to_door else 'FILTERED'}")
        
        if distance_to_door > self.max_distance_to_door:
            # Person is too far from door - cannot be applying force
            # Return NORMAL with very low score
            return (self.NORMAL, 0.0)
        
        # Person is near door - proceed with signal calculation
        # Calculate all biomechanical signals
        signals = self.signal_calculator.calculate_all_signals(
            keypoints, door_center, signal_histories
        )
        
        # Aggregate signals into force score
        # score based purely on biomechanics (like original)
        force_score = self._aggregate_signals(signals)
        
        # Classify based on threshold
        if force_score >= self.frame_threshold:
            label = self.FORCE_INTENT
        else:
            label = self.NORMAL
        
        return (label, force_score)
    
    def _has_minimum_keypoints(self, keypoints: Dict) -> bool:
        """
        Check if sufficient keypoints exist for classification.
        
        Minimum requirements:
        - Shoulder and hip midpoints (for torso)
        - At least one leg point (hip, knee, or ankle)
        """
        if not (keypoints.get('shoulder_mid') and keypoints.get('hip_mid')):
            return False
        
        # Need at least some leg information
        has_leg_point = any([
            keypoints.get('knee_left'),
            keypoints.get('knee_right'),
            keypoints.get('ankle_left'),
            keypoints.get('ankle_right')
        ])
        
        return has_leg_point
    
    def _calculate_person_to_door_distance(self, keypoints: Dict, 
                                           door_center: Tuple[float, float]) -> float:
        """
        Calculate person's distance to door.
        
        Uses hip_mid as reference point (center of person's body).
        
        Args:
            keypoints: Person keypoints
            door_center: Door center coordinates
        
        Returns:
            Distance in pixels
        """
        hip_mid = keypoints.get('hip_mid')
        
        if not hip_mid:
            # Fallback to shoulder if hip not available
            hip_mid = keypoints.get('shoulder_mid')
        
        if not hip_mid:
            # No valid reference point - return large distance
            return float('inf')
        
        distance = math.hypot(
            door_center[0] - hip_mid[0],
            door_center[1] - hip_mid[1]
        )
        
        return distance
    
    def _aggregate_signals(self, signals: Dict[str, Optional[float]]) -> float:
        """
        Aggregate individual signal scores into overall force score.
        
        Uses weighted average of available signals.
        Normalizes by total available weight to handle missing signals gracefully.
        
        Args:
            signals: Dictionary mapping signal names to scores (0.0-1.0 or None)
        
        Returns:
            Aggregated force score (0.0-1.0)
        """
        weighted_sum = 0.0
        valid_weight_sum = 0.0
        
        for signal_name, score in signals.items():
            if score is not None:
                weight = self.weights.get(signal_name, 0.0)
                weighted_sum += score * weight
                valid_weight_sum += weight
        
        # Avoid division by zero
        if valid_weight_sum < 0.001:
            return 0.0
        
        # Normalize by available weights
        force_score = weighted_sum / valid_weight_sum
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, force_score))