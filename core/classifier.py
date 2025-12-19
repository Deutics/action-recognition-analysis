"""
Posture Classifier
Core classification logic for detecting postures
"""

from typing import Dict, Tuple, Optional
from config.posture_config import PostureConfig, PostureLabel, KeypointIndex
from core.geometry import GeometryUtils


class PostureClassifier:
    """Main posture classification logic"""
    
    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.geo = GeometryUtils()
    
    def classify(self, kpts: Dict) -> Tuple[str, float]:
        """
        Classify posture based on keypoints with image tilt correction
        
        Args:
            kpts: Dictionary of extracted keypoints
            
        Returns:
            Tuple of (posture_label, confidence)
        """
        # Validate minimum required keypoints
        if not (kpts.get('shoulder_mid') and kpts.get('hip_mid')):
            return (PostureLabel.INSUFFICIENT, 0.0)
        
        if not (kpts.get('hip_mid') and (kpts.get('knee_mid') or kpts.get('ankle_mid'))):
            return (PostureLabel.INSUFFICIENT, 0.0)
        
        # STEP 1: Calculate θ₁ (torso angle from shoulder_mid to hip_mid)
        torso_angle = self.geo.calculate_angle_from_horizontal(kpts['hip_mid'], kpts['shoulder_mid'])
        torso_dev_vert = self.geo.calculate_deviation_from_vertical(torso_angle)
        
        # STEP 2: Calculate θ₂ (leg angle from hip_mid to ankle_mid)
        leg_endpoint = kpts.get('knee_mid') or kpts.get('ankle_mid')
        leg_angle = self.geo.calculate_angle_from_horizontal(kpts['hip_mid'], leg_endpoint)
        leg_dev_vert = self.geo.calculate_deviation_from_vertical(leg_angle)
        
        # STEP 3: Calculate segment alignment (θ₁ vs θ₂)
        segments_aligned = self.geo.are_segments_aligned(torso_angle, leg_angle, self.config.alignment_threshold)
        segments_diff = min(
            self.geo.calculate_angular_difference(torso_angle, leg_angle),
            self.geo.calculate_angular_difference(torso_angle, leg_angle + 180.0)
        )
        
        # STEP 4: IMAGE TILT DETECTION AND CORRECTION
        # If segments are aligned (θ₁ ≈ θ₂) but deviate from Y-axis, image is tilted
        tilt_offset = 0.0
        is_image_tilted = False
        
        if segments_aligned and torso_dev_vert > self.config.vertical_deviation_threshold:
            # Segments are parallel but not vertical → Image is tilted
            is_image_tilted = True
            tilt_offset = torso_angle - (-90.0)  # Offset from true vertical (-90°)
            
            # Apply tilt correction to angles
            torso_angle_corrected = torso_angle - tilt_offset
            leg_angle_corrected = leg_angle - tilt_offset
            torso_dev_vert = self.geo.calculate_deviation_from_vertical(torso_angle_corrected)
            leg_dev_vert = self.geo.calculate_deviation_from_vertical(leg_angle_corrected)
        else:
            # No tilt detected, use angles as-is
            torso_angle_corrected = torso_angle
            leg_angle_corrected = leg_angle
        
        # STEP 5: POSTURE CLASSIFICATION (using corrected angles)
        
        # RULE 1: LYING DOWN
        torso_dev_horiz = self.geo.calculate_deviation_from_horizontal(torso_angle_corrected)
        leg_dev_horiz = self.geo.calculate_deviation_from_horizontal(leg_angle_corrected)
        
        if (torso_dev_horiz < self.config.horizontal_deviation_threshold and
            leg_dev_horiz < self.config.horizontal_deviation_threshold):
            return (PostureLabel.LYING, 0.9)
        
        # RULE 2: STANDING
        if (torso_dev_vert < self.config.vertical_deviation_threshold and
            leg_dev_vert < self.config.vertical_deviation_threshold and
            segments_aligned):
            
            confidence = 0.85
            
            # Boost confidence if image was tilted (we corrected for it)
            if is_image_tilted:
                confidence += 0.05
            
            if kpts.get('hip_left') and kpts.get('knee_left') and kpts.get('ankle_left'):
                left_knee = self.geo.calculate_angle_between_three_points(
                    kpts['hip_left'], kpts['knee_left'], kpts['ankle_left']
                )
                if left_knee > self.config.standing_knee_angle_min:
                    confidence = 0.95
            
            if kpts.get('hip_right') and kpts.get('knee_right') and kpts.get('ankle_right'):
                right_knee = self.geo.calculate_angle_between_three_points(
                    kpts['hip_right'], kpts['knee_right'], kpts['ankle_right']
                )
                if right_knee > self.config.standing_knee_angle_min:
                    confidence = 0.95
            
            return (PostureLabel.STANDING, confidence)
        
        # RULE 3: SITTING
        sitting_count = 0
        
        if kpts.get('shoulder_mid') and kpts.get('ankle_mid'):
            body_height = abs(kpts['shoulder_mid'][1] - kpts['ankle_mid'][1])
            if body_height > 0:
                hip_height = abs(kpts['ankle_mid'][1] - kpts['hip_mid'][1])
                hip_ratio = hip_height / body_height
                if hip_ratio < self.config.sitting_hip_height_ratio_max:
                    sitting_count += 1
        
        avg_knee_angle = None
        knee_angles = []
        
        if kpts.get('hip_left') and kpts.get('knee_left') and kpts.get('ankle_left'):
            left_knee = self.geo.calculate_angle_between_three_points(
                kpts['hip_left'], kpts['knee_left'], kpts['ankle_left']
            )
            knee_angles.append(left_knee)
        
        if kpts.get('hip_right') and kpts.get('knee_right') and kpts.get('ankle_right'):
            right_knee = self.geo.calculate_angle_between_three_points(
                kpts['hip_right'], kpts['knee_right'], kpts['ankle_right']
            )
            knee_angles.append(right_knee)
        
        if knee_angles:
            avg_knee_angle = sum(knee_angles) / len(knee_angles)
            if avg_knee_angle < self.config.sitting_knee_angle_max:
                sitting_count += 1
        
        if (torso_dev_vert < self.config.vertical_deviation_threshold * 1.5 and
            leg_dev_vert > self.config.bending_threshold and
            not segments_aligned):
            sitting_count += 1
        
        if sitting_count >= 2:
            return (PostureLabel.SITTING, 0.8)
        
        # RULE 4: SQUATTING
        if avg_knee_angle is not None:
            if (self.config.squatting_knee_angle_range[0] <= avg_knee_angle <= 
                self.config.squatting_knee_angle_range[1]):
                
                if kpts.get('shoulder_mid') and kpts.get('ankle_mid'):
                    body_height = abs(kpts['shoulder_mid'][1] - kpts['ankle_mid'][1])
                    if body_height > 0:
                        hip_height = abs(kpts['ankle_mid'][1] - kpts['hip_mid'][1])
                        hip_ratio = hip_height / body_height
                        if hip_ratio > self.config.sitting_hip_height_ratio_max:
                            return (PostureLabel.SQUATTING, 0.75)
        
        # BENDING
        if (leg_dev_vert < self.config.vertical_deviation_threshold * 1.5 and
            segments_diff > self.config.bending_threshold):
            return (PostureLabel.BENDING, 0.8)
        
        # FALLBACK
        return (PostureLabel.UNKNOWN, 0.3)
