"""
Signal Calculator - Biomechanical signal calculations with MOVEMENT validation
CRITICAL FIX: Arm and COM signals now require ACTUAL MOVEMENT, not just proximity
"""

from typing import Dict, Optional, Tuple, Deque
from collections import deque
import math
from core.geometry import GeometryUtils


class SignalCalculator:
    """Pure signal calculation functions with temporal averaging."""
    
    def __init__(self, temporal_window: int = 15):
        self.geo = GeometryUtils()
        self.temporal_window = temporal_window
    
    def calculate_all_signals(self, keypoints: Dict, door_center: Tuple[float, float],
                              signal_histories: Dict) -> Dict[str, Optional[float]]:
        return {
            'lean': self.signal_torso_lean(
                keypoints, door_center, signal_histories['lean_history']),
            'hip': self.signal_hip_height(
                keypoints, signal_histories['hip_history']),
            'knee': self.signal_knee_flexion(
                keypoints, signal_histories['knee_history']),
            'com': self.signal_com_forward_shift(
                keypoints, door_center, signal_histories['com_history']),
            'arm': self.signal_arm_motion(
                keypoints, door_center, signal_histories['elbow_history']),
            'compact': self.signal_body_compactness(
                keypoints, signal_histories['compact_history']),
        }
    
    def signal_torso_lean(self, keypoints: Dict, door_center: Tuple[float, float],
                         lean_history: Deque[float], max_lean_angle: float = 35.0) -> Optional[float]:
        shoulder = keypoints.get('shoulder_mid')
        hip = keypoints.get('hip_mid')
        
        if not (shoulder and hip):
            return None
        
        torso_vec = (shoulder[0] - hip[0], shoulder[1] - hip[1])
        door_vec = (door_center[0] - hip[0], door_center[1] - hip[1])
        
        dot = torso_vec[0] * door_vec[0] + torso_vec[1] * door_vec[1]
        mag_torso = math.hypot(torso_vec[0], torso_vec[1])
        mag_door = math.hypot(door_vec[0], door_vec[1])
        
        if mag_torso < 1 or mag_door < 1:
            return None
        
        cos_angle = dot / (mag_torso * mag_door)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        
        deviation = 90.0 - angle_deg
        score = deviation / max_lean_angle
        score = max(0.0, min(1.0, score))
        
        lean_history.append(score)
        if len(lean_history) > self.temporal_window:
            lean_history.popleft()
        
        return sum(lean_history) / len(lean_history) if lean_history else score
    
    def signal_hip_height(self, keypoints: Dict, hip_history: Deque[float],
                         target_ratio: float = 0.5, sensitivity: float = 0.2) -> Optional[float]:
        hip = keypoints.get('hip_mid')
        ankle = keypoints.get('ankle_mid') or keypoints.get('knee_mid')
        shoulder = keypoints.get('shoulder_mid')
        
        if not (hip and ankle and shoulder):
            return None
        
        body_height = abs(shoulder[1] - ankle[1])
        if body_height < 1:
            return None
        
        hip_height = abs(ankle[1] - hip[1])
        hip_ratio = hip_height / body_height
        score = (target_ratio - hip_ratio) / sensitivity
        score = max(0.0, min(1.0, score))
        
        hip_history.append(score)
        if len(hip_history) > self.temporal_window:
            hip_history.popleft()
        
        return sum(hip_history) / len(hip_history) if hip_history else score
    
    def signal_knee_flexion(self, keypoints: Dict, knee_history: Deque[float],
                           max_flexion: float = 140.0, min_flexion: float = 100.0) -> Optional[float]:
        angles = []
        
        hip_l = keypoints.get('hip_left')
        knee_l = keypoints.get('knee_left')
        ankle_l = keypoints.get('ankle_left')
        if hip_l and knee_l and ankle_l:
            angles.append(self.geo.calculate_angle_between_three_points(hip_l, knee_l, ankle_l))
        
        hip_r = keypoints.get('hip_right')
        knee_r = keypoints.get('knee_right')
        ankle_r = keypoints.get('ankle_right')
        if hip_r and knee_r and ankle_r:
            angles.append(self.geo.calculate_angle_between_three_points(hip_r, knee_r, ankle_r))
        
        if not angles:
            return None
        
        avg_angle = sum(angles) / len(angles)
        score = (max_flexion - avg_angle) / (max_flexion - min_flexion)
        score = max(0.0, min(1.0, score))
        
        knee_history.append(score)
        if len(knee_history) > self.temporal_window:
            knee_history.popleft()
        
        return sum(knee_history) / len(knee_history) if knee_history else score
    
    def signal_com_forward_shift(self, keypoints: Dict, door_center: Tuple[float, float],
                                 com_history: Deque[Tuple[float, float]],
                                 min_movement_threshold: float = 10.0) -> Optional[float]:
        """COM shift - REQUIRES ACTUAL MOVEMENT toward door."""
        shoulder = keypoints.get('shoulder_mid')
        hip = keypoints.get('hip_mid')
        
        if not (shoulder and hip):
            return None
        
        com_curr = ((shoulder[0] + hip[0]) / 2.0, (shoulder[1] + hip[1]) / 2.0)
        
        com_history.append(com_curr)
        if len(com_history) > self.temporal_window:
            com_history.popleft()
        
        if len(com_history) < 5:  # Need minimum history
            return 0.0
        
        door_direction = self._calculate_direction_vector(com_curr, door_center)
        
        # Calculate total movement toward door
        total_movement = 0.0
        for i in range(1, len(com_history)):
            displacement = (com_history[i][0] - com_history[i-1][0],
                          com_history[i][1] - com_history[i-1][1])
            projection = self._project_vector_onto_direction(displacement, door_direction)
            if projection > 0:  # Only count toward door
                total_movement += projection
        
        # CRITICAL: Require minimum total movement
        if total_movement < min_movement_threshold:
            return 0.0  # No significant movement = no score
        
        # Score based on movement
        max_shift = 50.0
        score = total_movement / max_shift
        return max(0.0, min(1.0, score))
    
    def signal_arm_motion(self, keypoints: Dict, door_center: Tuple[float, float],
                         elbow_history: Deque[Tuple[float, float]],
                         min_oscillation: int = 5,
                         min_velocity: float = 5.0) -> Optional[float]:
        """Arm motion - REQUIRES OSCILLATION AND MOVEMENT, not just proximity."""
        elbow = keypoints.get('elbow_right') or keypoints.get('elbow_left')
        
        if not elbow:
            return None
        
        elbow_history.append(elbow)
        if len(elbow_history) > self.temporal_window:
            elbow_history.popleft()
        
        if len(elbow_history) < 8:  # Need more history for oscillation
            return 0.0
        
        door_direction = self._calculate_direction_vector(elbow, door_center)
        
        # Calculate velocities
        projected_velocities = []
        for i in range(1, len(elbow_history)):
            velocity = (elbow_history[i][0] - elbow_history[i-1][0],
                       elbow_history[i][1] - elbow_history[i-1][1])
            projection = self._project_vector_onto_direction(velocity, door_direction)
            projected_velocities.append(projection)
        
        if not projected_velocities:
            return 0.0
        
        # Count direction changes (oscillation)
        direction_changes = sum(1 for i in range(1, len(projected_velocities))
                               if (projected_velocities[i] > 0) != (projected_velocities[i-1] > 0))
        
        # CRITICAL: Require minimum oscillation
        if direction_changes < min_oscillation:
            return 0.0  # No oscillation = no score
        
        # Calculate average velocity magnitude
        velocities = [abs(v) for v in projected_velocities]
        avg_velocity = sum(velocities) / len(velocities)
        
        # CRITICAL: Require minimum velocity
        if avg_velocity < min_velocity:
            return 0.0  # No significant movement = no score
        
        # Score based on oscillation and velocity
        max_changes = 6
        oscillation_score = min(1.0, direction_changes / max_changes)
        
        max_velocity = 30.0
        velocity_score = min(1.0, avg_velocity / max_velocity)
        
        # Combined: 50% oscillation + 50% velocity (distance removed!)
        combined = 0.50 * oscillation_score + 0.50 * velocity_score
        return max(0.0, min(1.0, combined))
    
    def signal_body_compactness(self, keypoints: Dict, compact_history: Deque[float],
                                target_ratio: float = 0.55) -> Optional[float]:
        valid_kpts = [kpt for kpt in keypoints.values() if kpt and all(c > 0 for c in kpt)]
        
        if not valid_kpts:
            return None
        
        xs = [k[0] for k in valid_kpts]
        ys = [k[1] for k in valid_kpts]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        if (height + width) < 1e-3:
            return None
        
        ratio = height / (height + width)
        score = (target_ratio - ratio) / target_ratio
        score = max(0.0, min(1.0, score))
        
        compact_history.append(score)
        if len(compact_history) > self.temporal_window:
            compact_history.popleft()
        
        return sum(compact_history) / len(compact_history) if compact_history else score

    
    def _calculate_direction_vector(self, from_point: Tuple[float, float],
                                    to_point: Tuple[float, float]) -> Tuple[float, float]:
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)
    
    def _project_vector_onto_direction(self, vector: Tuple[float, float],
                                       direction: Tuple[float, float]) -> float:
        return vector[0] * direction[0] + vector[1] * direction[1]
