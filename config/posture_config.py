"""
Posture Detection Configuration
Central configuration for all posture detection thresholds and parameters
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PostureConfig:
    """Configuration parameters for posture detection"""
    
    # Angle thresholds (degrees)
    vertical_deviation_threshold: float = 12.0
    alignment_threshold: float = 10.0
    bending_threshold: float = 20.0
    
    # Height ratio thresholds (for sitting detection)
    sitting_hip_height_ratio_max: float = 0.45
    
    # Knee angle thresholds (degrees from straight = 180)
    sitting_knee_angle_max: float = 120.0
    standing_knee_angle_min: float = 135.0
    squatting_knee_angle_range: Tuple[float, float] = (60.0, 120.0)
    
    # Horizontal orientation threshold (for lying detection)
    horizontal_deviation_threshold: float = 30.0

    # Confidence thresholds
    min_keypoint_confidence: float = 0.25
    lying_duration_threshold: float = 0.02

    # ---------------- NEW LYING FILTERS ----------------
    lying_min_points: int = 6  # was 4
    lying_torso_dev_horiz_max: float = 35.0  # was 22
    lying_body_axis_dev_horiz_max: float = 40.0  # was 25
    lying_aspect_ratio_min: float = 0.95  # was 1.25
    lying_height_ratio_max: float = 0.60  # was 0.40
    lying_floor_contact_min: float = 0.55  # was 0.78
    lying_gravity_proj_ratio_max: float = 0.55  # was 0.33


# Posture labels as string constants
class PostureLabel:
    STANDING = "Standing"
    SITTING = "Sitting"
    BENDING = "Bending"
    SQUATTING = "Squatting"
    LYING = "Lying"
    UNKNOWN = "Unknown"
    INSUFFICIENT = "Insufficient_Keypoints"
    FALLING = "Falling"


# YOLO pose keypoint indices for COCO format (17 keypoints)
class KeypointIndex:
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
