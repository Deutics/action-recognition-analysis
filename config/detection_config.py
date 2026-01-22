"""Consolidated detection configuration."""
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DoorROIConfig:
    """Door region configuration."""
    x_center_ratio: float = 0.5
    y_center_ratio: float = 0.75
    width_ratio: float = 0.25
    height_ratio: float = 0.45
    approach_buffer: int = 50
    debug_visualize: bool = False


@dataclass
class PostureThresholds:
    """Posture detection thresholds."""
    vertical_deviation_threshold: float = 12.0
    alignment_threshold: float = 10.0
    bending_threshold: float = 20.0
    sitting_hip_height_ratio_max: float = 0.45
    sitting_knee_angle_max: float = 120.0
    standing_knee_angle_min: float = 135.0
    squatting_knee_angle_range: Tuple[float, float] = (60.0, 120.0)
    horizontal_deviation_threshold: float = 30.0
    min_keypoint_confidence: float = 0.3
    lying_duration_threshold: float = 0.5


@dataclass
class ForceSignalWeights:
    """Signal importance weights for force detection."""
    arm: float = 0.40
    com: float = 0.15
    lean: float = 0.15
    hip: float = 0.10
    knee: float = 0.10
    compact: float = 0.10


@dataclass
class ForceThresholds:
    """Force detection thresholds."""
    energy_threshold: float = 3.0
    min_force_frames: int = 20
    decay_per_frame: float = 0.05
    temporal_window: int = 15
    frame_threshold: float = 0.3
    cooldown_period: float = 10.0


@dataclass
class PostureConfig:
    """Posture detection configuration."""
    thresholds: PostureThresholds = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = PostureThresholds()
    
    @property
    def vertical_deviation_threshold(self):
        return self.thresholds.vertical_deviation_threshold
    
    @property
    def alignment_threshold(self):
        return self.thresholds.alignment_threshold
    
    @property
    def bending_threshold(self):
        return self.thresholds.bending_threshold
    
    @property
    def sitting_hip_height_ratio_max(self):
        return self.thresholds.sitting_hip_height_ratio_max
    
    @property
    def sitting_knee_angle_max(self):
        return self.thresholds.sitting_knee_angle_max
    
    @property
    def standing_knee_angle_min(self):
        return self.thresholds.standing_knee_angle_min
    
    @property
    def squatting_knee_angle_range(self):
        return self.thresholds.squatting_knee_angle_range
    
    @property
    def horizontal_deviation_threshold(self):
        return self.thresholds.horizontal_deviation_threshold
    
    @property
    def min_keypoint_confidence(self):
        return self.thresholds.min_keypoint_confidence
    
    @property
    def lying_duration_threshold(self):
        return self.thresholds.lying_duration_threshold


@dataclass
class ForceIntentConfig:
    """Force intent detection configuration."""
    enabled: bool = True
    door_roi: Optional[DoorROIConfig] = None
    weights: ForceSignalWeights = None
    thresholds: ForceThresholds = None
    
    def __post_init__(self):
        if self.door_roi is None:
            self.door_roi = DoorROIConfig()
        if self.weights is None:
            self.weights = ForceSignalWeights()
        if self.thresholds is None:
            self.thresholds = ForceThresholds()
    
    @property
    def accumulator(self):
        """Backward compatibility."""
        return type('obj', (), {'cooldown_period': self.thresholds.cooldown_period})()


class PostureLabel:
    """Posture label constants."""
    STANDING = "Standing"
    SITTING = "Sitting"
    BENDING = "Bending"
    SQUATTING = "Squatting"
    LYING = "Lying"
    UNKNOWN = "Unknown"
    INSUFFICIENT = "Insufficient_Keypoints"
    FALLING = "Falling"


class KeypointIndex:
    """YOLO pose keypoint indices."""
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
