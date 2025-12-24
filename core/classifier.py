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
        tilt_offset = 0.0
        is_image_tilted = False

        if segments_aligned and torso_dev_vert > self.config.vertical_deviation_threshold:
            is_image_tilted = True
            tilt_offset = torso_angle - (-90.0)

            torso_angle_corrected = torso_angle - tilt_offset
            leg_angle_corrected = leg_angle - tilt_offset
            torso_dev_vert = self.geo.calculate_deviation_from_vertical(torso_angle_corrected)
            leg_dev_vert = self.geo.calculate_deviation_from_vertical(leg_angle_corrected)
        else:
            torso_angle_corrected = torso_angle
            leg_angle_corrected = leg_angle


        # PRIORITY CHECK - Calculate knee angles first
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
            max_knee_angle = max(knee_angles)
        else:
            max_knee_angle = None

        # Calculate hip height ratio (needed for multiple checks)
        hip_height_ratio = None
        if kpts.get('shoulder_mid') and kpts.get('ankle_mid'):
            body_height = abs(kpts['shoulder_mid'][1] - kpts['ankle_mid'][1])
            if body_height > 0:
                hip_height = abs(kpts['ankle_mid'][1] - kpts['hip_mid'][1])
                hip_height_ratio = hip_height / body_height

        # STEP 5: POSTURE CLASSIFICATION (using corrected angles)

        # RULE 1: LYING DOWN
        torso_dev_horiz = self.geo.calculate_deviation_from_horizontal(torso_angle_corrected)
        leg_dev_horiz = self.geo.calculate_deviation_from_horizontal(leg_angle_corrected)

        if (torso_dev_horiz < self.config.horizontal_deviation_threshold and
            leg_dev_horiz < self.config.horizontal_deviation_threshold):
            return (PostureLabel.LYING, 0.9)


        # sitting signal: bent knees + low hip
        if avg_knee_angle is not None and avg_knee_angle < self.config.sitting_knee_angle_max:
            if hip_height_ratio is not None and hip_height_ratio < 0.50:
                return (PostureLabel.SITTING, 0.85)

        # Backup sitting check (original logic - catches side view sitting)
        sitting_count = 0

        if hip_height_ratio is not None:
            if hip_height_ratio < self.config.sitting_hip_height_ratio_max:
                sitting_count += 1

        if avg_knee_angle is not None:
            if avg_knee_angle < self.config.sitting_knee_angle_max:
                sitting_count += 1

        if (torso_dev_vert < self.config.vertical_deviation_threshold * 1.5 and
            leg_dev_vert > self.config.bending_threshold and
            not segments_aligned):
            sitting_count += 1

        if sitting_count >= 2:
            return (PostureLabel.SITTING, 0.8)


        # standing check: torso + legs vertical + aligned
        if (torso_dev_vert < self.config.vertical_deviation_threshold and
            leg_dev_vert < self.config.vertical_deviation_threshold and
            segments_aligned):

            confidence = 0.85
            if is_image_tilted:
                confidence += 0.05

            if max_knee_angle is not None and max_knee_angle > self.config.standing_knee_angle_min:
                confidence = 0.95

            return (PostureLabel.STANDING, confidence)

        #Backup standing check - if torso vertical + at least one knee straight
        # (Catches standing with weight shifted or relaxed stance)
        if torso_dev_vert < self.config.vertical_deviation_threshold * 1.3:
            if max_knee_angle is not None and max_knee_angle > self.config.standing_knee_angle_min:
                # Additional check: hip should be high for standing
                if hip_height_ratio is not None and hip_height_ratio > 0.38:
                    # print(f"******* {hip_height_ratio} > 0.38")
                    return (PostureLabel.STANDING, 0.80)

            # print(f"******* {max_knee_angle} < {self.config.standing_knee_angle_min}")

        # RULE 4: SQUATTING
        if avg_knee_angle is not None:
            if (self.config.squatting_knee_angle_range[0] <= avg_knee_angle <=
                self.config.squatting_knee_angle_range[1]):

                if hip_height_ratio is not None:
                    if hip_height_ratio > self.config.sitting_hip_height_ratio_max:
                        return (PostureLabel.SQUATTING, 0.75)

        # RULE 5: BENDING
        if (leg_dev_vert < self.config.vertical_deviation_threshold * 1.5 and
            segments_diff > self.config.bending_threshold):
            return (PostureLabel.BENDING, 0.8)

        # FALLBACK
        return (PostureLabel.UNKNOWN, 0.3)