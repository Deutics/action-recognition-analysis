"""
Posture Classifier
Core classification logic with improved lying detection
"""

from typing import Dict, Tuple, Optional
from config.posture_config import PostureConfig, PostureLabel, KeypointIndex
from core.geometry import GeometryUtils


class PostureClassifier:
    """Main posture classification logic"""

    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.geo = GeometryUtils()
        self.frame_height = None

    def set_frame_dimensions(self, height: int, width: int):
        """
        Set actual frame dimensions (called from service)

        Args:
            height: Frame height in pixels
            width: Frame width in pixels
        """
        self.frame_height = height
        self.frame_width = width

    def classify(self, kpts: Dict, frame_height: Optional[int] = None) -> Tuple[str, float]:
        """
        Classify posture based on keypoints with image tilt correction

        Args:
            kpts: Dictionary of extracted keypoints
            frame_height: Actual frame height in pixels (optional, uses stored value if not provided)

        Returns:
            Tuple of (posture_label, confidence)
        """
        # Validate minimum required keypoints
        if not (kpts.get('shoulder_mid') and kpts.get('hip_mid')):
            return (PostureLabel.INSUFFICIENT, 0.0)

        if not (kpts.get('hip_mid') and (kpts.get('knee_mid') or kpts.get('ankle_mid'))):
            return (PostureLabel.INSUFFICIENT, 0.0)

        # Use provided frame_height or stored value
        if frame_height is not None:
            self.frame_height = frame_height

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

        body_compactness = self._calculate_body_compactness(kpts)

        # STEP 5: POSTURE CLASSIFICATION (using corrected angles)

        # RULE 1: LYING DOWN
        torso_dev_horiz = self.geo.calculate_deviation_from_horizontal(torso_angle_corrected)
        leg_dev_horiz = self.geo.calculate_deviation_from_horizontal(leg_angle_corrected)

        # PRIMARY lying check: Body horizontal (orientation-based, camera-angle independent)
        if (torso_dev_horiz < self.config.horizontal_deviation_threshold and
            leg_dev_horiz < self.config.horizontal_deviation_threshold):
            return (PostureLabel.LYING, 0.9)

        # SECONDARY lying check: Body is very compact vertically
        # (Lying person has small vertical extent relative to horizontal extent)
        if body_compactness is not None and body_compactness < 0.6:
            # Body is wider than it is tall → likely lying
            # Additional confirmation: check if in lower portion of frame
            if self.frame_height is not None:
                is_in_lower_frame = self._check_if_in_lower_frame(kpts, self.frame_height)
                if is_in_lower_frame:
                    return (PostureLabel.LYING, 0.80)

        # RULE 2: SITTING (with ground proximity check)
        if avg_knee_angle is not None and avg_knee_angle < self.config.sitting_knee_angle_max:
            if hip_height_ratio is not None and hip_height_ratio < 0.50:
                return (PostureLabel.SITTING, 0.85)

        # Backup sitting check
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

        # RULE 3: STANDING
        if (torso_dev_vert < self.config.vertical_deviation_threshold and
            leg_dev_vert < self.config.vertical_deviation_threshold and
            segments_aligned):

            confidence = 0.85
            if is_image_tilted:
                confidence += 0.05

            if max_knee_angle is not None and max_knee_angle > self.config.standing_knee_angle_min:
                confidence = 0.95

            return (PostureLabel.STANDING, confidence)

        # Backup standing check
        if torso_dev_vert < self.config.vertical_deviation_threshold * 1.3:
            if max_knee_angle is not None and max_knee_angle > self.config.standing_knee_angle_min:
                if hip_height_ratio is not None and hip_height_ratio > 0.38:
                    return (PostureLabel.STANDING, 0.80)

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

    def _calculate_body_compactness(self, kpts: Dict) -> Optional[float]:
        # Get all available keypoints
        all_x_coords = []
        all_y_coords = []

        for key in ['shoulder_left', 'shoulder_right', 'hip_left', 'hip_right',
                    'knee_left', 'knee_right', 'ankle_left', 'ankle_right']:
            if kpts.get(key):
                all_x_coords.append(kpts[key][0])
                all_y_coords.append(kpts[key][1])

        if len(all_x_coords) < 4 or len(all_y_coords) < 4:
            return None

        # Calculate body bounding box
        body_width = max(all_x_coords) - min(all_x_coords)
        body_height = max(all_y_coords) - min(all_y_coords)

        if body_width < 1:  # Avoid division by zero
            return None

        compactness = body_height / body_width
        return compactness

    def _check_if_in_lower_frame(self, kpts: Dict, frame_height: int) -> bool:

        if frame_height is None:
            return False

        # Calculate center of mass (average Y position of key body points)
        y_coords = []

        for key in ['shoulder_mid', 'hip_mid', 'knee_mid']:
            if kpts.get(key):
                y_coords.append(kpts[key][1])

        if not y_coords:
            return False

        center_of_mass_y = sum(y_coords) / len(y_coords)

        # Lower 40% of frame (more conservative than 30%)
        lower_frame_threshold = frame_height * 0.40

        return center_of_mass_y > lower_frame_threshold