"""
Posture Classifier
Geometry-first posture classification with gravity-aware lying correction
"""

from typing import Dict, Tuple, Optional
from config.posture_config import PostureConfig, PostureLabel
from core.geometry import GeometryUtils
import math


class PostureClassifier:

    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.geo = GeometryUtils()
        self.frame_height = None
        self.frame_width = None

    def set_frame_dimensions(self, height: int, width: int):
        self.frame_height = height
        self.frame_width = width

    def classify(self, kpts: Dict, frame_height: Optional[int] = None) -> Tuple[str, float]:

        # Minimum keypoints
        if not (kpts.get('shoulder_mid') and kpts.get('hip_mid')):
            return (PostureLabel.INSUFFICIENT, 0.0)

        if frame_height is not None:
            self.frame_height = frame_height

        # Torso & leg geometry
        torso_angle = self.geo.calculate_angle_from_horizontal(
            kpts['hip_mid'], kpts['shoulder_mid']
        )
        torso_dev_vert = self.geo.calculate_deviation_from_vertical(torso_angle)

        leg_endpoint = kpts.get('knee_mid') or kpts.get('ankle_mid')
        leg_angle = None
        leg_dev_vert = None

        if leg_endpoint:
            leg_angle = self.geo.calculate_angle_from_horizontal(
                kpts['hip_mid'], leg_endpoint
            )
            leg_dev_vert = self.geo.calculate_deviation_from_vertical(leg_angle)

        segments_aligned = (
            leg_angle is not None and
            self.geo.are_segments_aligned(
                torso_angle, leg_angle, self.config.alignment_threshold
            )
        )

        segments_diff = 0.0
        if leg_angle is not None:
            segments_diff = min(
                self.geo.calculate_angular_difference(torso_angle, leg_angle),
                self.geo.calculate_angular_difference(torso_angle, leg_angle + 180.0)
            )

        # Knee angles
        knee_angles = []
        for side in ['left', 'right']:
            hip_point = kpts.get(f'hip_{side}')
            knee_point = kpts.get(f'knee_{side}')
            ankle_point = kpts.get(f'ankle_{side}')
            if hip_point and knee_point and ankle_point:
                knee_angles.append(
                    self.geo.calculate_angle_between_three_points(hip_point, knee_point, ankle_point)
                )

        avg_knee_angle = sum(knee_angles) / len(knee_angles) if knee_angles else None
        max_knee_angle = max(knee_angles) if knee_angles else None

        # Hip height ratio
        hip_height_ratio = None
        if kpts.get('shoulder_mid') and kpts.get('ankle_mid'):
            body_height = abs(kpts['shoulder_mid'][1] - kpts['ankle_mid'][1])
            if body_height > 0:
                hip_height = abs(kpts['ankle_mid'][1] - kpts['hip_mid'][1])
                hip_height_ratio = hip_height / body_height

        # Body compactness
        body_compactness = self._calculate_body_compactness(kpts)

        # Gravity signals
        shoulder_x, shoulder_y = kpts['shoulder_mid']
        hip_x, hip_y = kpts['hip_mid']
        gravity_vector_x = hip_x - shoulder_x
        gravity_vector_y = hip_y - shoulder_y
        gravity_magnitude = math.hypot(gravity_vector_x, gravity_vector_y) + 1e-6
        gravity_proj_ratio = abs(gravity_vector_y) / gravity_magnitude
        is_inverted = hip_y < shoulder_y

        # Lying
        # torso_dev_horiz = self.geo.calculate_deviation_from_horizontal(torso_angle)
        # grounded_votes = 0
        #
        # if torso_dev_horiz < self.config.horizontal_deviation_threshold:
        #     grounded_votes += 1
        # if body_compactness is not None and body_compactness < 0.6:
        #     grounded_votes += 1
        # if gravity_proj_ratio < 0.45:
        #     grounded_votes += 1
        # if is_inverted:
        #     grounded_votes += 2
        #
        # if grounded_votes >= 2:
        #     return (PostureLabel.LYING, min(0.95, 0.7 + 0.1 * grounded_votes))

        # -------------------------
        # LYING (improved, floor-aware)
        # -------------------------

        # Count how many keypoints we actually have (after extractor filtering)
        present_kpt_keys = [
            'shoulder_left', 'shoulder_right', 'hip_left', 'hip_right',
            'knee_left', 'knee_right', 'ankle_left', 'ankle_right'
        ]
        present_count = sum(1 for k in present_kpt_keys if kpts.get(k))

        # Need frame height for floor/height ratios
        if self.frame_height is None and frame_height is not None:
            self.frame_height = frame_height

        # Compute spread + aspect ratio using available kpts
        xs, ys = [], []
        for k in present_kpt_keys + ['shoulder_mid', 'hip_mid']:
            if kpts.get(k):
                xs.append(kpts[k][0])
                ys.append(kpts[k][1])

        # Hard gate: not enough info => never claim LYING
        if len(xs) >= self.config.lying_min_points and self.frame_height:
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            aspect = (width / (height + 1e-6))
            height_ratio = height / float(self.frame_height)
            floor_contact = (max(ys) / float(self.frame_height))

            # torso horizontal check
            torso_dev_horiz = self.geo.calculate_deviation_from_horizontal(torso_angle)

            # body axis check (shoulder_mid -> ankle_mid is best)
            body_axis_dev_horiz = None
            if kpts.get('shoulder_mid') and kpts.get('ankle_mid'):
                body_axis_angle = self.geo.calculate_angle_from_horizontal(
                    kpts['shoulder_mid'], kpts['ankle_mid']
                )
                body_axis_dev_horiz = self.geo.calculate_deviation_from_horizontal(body_axis_angle)

            # Votes (but only after gates are satisfied)
            grounded_votes = 0

            # Strong gates first
            gates_ok = True

            if torso_dev_horiz > self.config.lying_torso_dev_horiz_max:
                gates_ok = False

            if body_axis_dev_horiz is None or body_axis_dev_horiz > self.config.lying_body_axis_dev_horiz_max:
                gates_ok = False

            if aspect < self.config.lying_aspect_ratio_min:
                gates_ok = False

            if height_ratio > self.config.lying_height_ratio_max:
                gates_ok = False

            if floor_contact < self.config.lying_floor_contact_min:
                gates_ok = False

            if gravity_proj_ratio > self.config.lying_gravity_proj_ratio_max:
                gates_ok = False

            if gates_ok:
                grounded_votes += 2  # if you passed all gates, it’s already strong

                # Optional extra softness: compactness check
                if body_compactness is not None and body_compactness < 0.55:
                    grounded_votes += 1

                # Optional: inverted skeleton is very suspicious (can be fall)
                if is_inverted:
                    grounded_votes += 1

                conf = min(0.98, 0.78 + 0.06 * grounded_votes)
                return (PostureLabel.LYING, conf)

        # Sitting
        if avg_knee_angle is not None and avg_knee_angle < self.config.sitting_knee_angle_max:
            if hip_height_ratio is not None and hip_height_ratio < 0.50:
                return (PostureLabel.SITTING, 0.85)

        sitting_count = 0
        if hip_height_ratio is not None and hip_height_ratio < self.config.sitting_hip_height_ratio_max:
            sitting_count += 1
        if avg_knee_angle is not None and avg_knee_angle < self.config.sitting_knee_angle_max:
            sitting_count += 1
        if (
            torso_dev_vert < self.config.vertical_deviation_threshold * 1.5 and
            leg_dev_vert is not None and
            leg_dev_vert > self.config.bending_threshold and
            not segments_aligned
        ):
            sitting_count += 1

        if sitting_count >= 2:
            return (PostureLabel.SITTING, 0.8)

        # Standing
        if (
            torso_dev_vert < self.config.vertical_deviation_threshold and
            leg_dev_vert is not None and
            leg_dev_vert < self.config.vertical_deviation_threshold and
            segments_aligned and
            not is_inverted
        ):
            confidence = 0.85
            if max_knee_angle is not None and max_knee_angle > self.config.standing_knee_angle_min:
                confidence = 0.95
            return (PostureLabel.STANDING, confidence)

        if (
            torso_dev_vert < self.config.vertical_deviation_threshold * 1.3 and
            max_knee_angle is not None and
            max_knee_angle > self.config.standing_knee_angle_min and
            hip_height_ratio is not None and
            hip_height_ratio > 0.38
        ):
            return (PostureLabel.STANDING, 0.8)

        # Bending
        if (
            leg_dev_vert is not None and
            leg_dev_vert < self.config.vertical_deviation_threshold * 1.5 and
            segments_diff > self.config.bending_threshold
        ):
            return (PostureLabel.BENDING, 0.8)

        return (PostureLabel.UNKNOWN, 0.3)

    def _calculate_body_compactness(self, kpts: Dict) -> Optional[float]:
        xs, ys = [], []
        for key in [
            'shoulder_left', 'shoulder_right',
            'hip_left', 'hip_right',
            'knee_left', 'knee_right',
            'ankle_left', 'ankle_right'
        ]:
            if kpts.get(key):
                xs.append(kpts[key][0])
                ys.append(kpts[key][1])

        if len(xs) < 4:
            return None

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if width < 1 or height < 1:
            return None

        return height / (width + height)