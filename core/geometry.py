"""
Geometry Utilities for Posture Detection
Handles all geometric calculations for angle, distance, and alignment
"""

import math
from typing import Tuple
import numpy as np


class GeometryUtils:
    """Utility functions for geometric calculations"""
    
    @staticmethod
    def calculate_midpoint(point_left: Tuple[float, float], 
                          point_right: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate midpoint between two points"""
        return (
            (point_left[0] + point_right[0]) / 2.0,
            (point_left[1] + point_right[1]) / 2.0
        )
    
    @staticmethod
    def calculate_angle_from_horizontal(point_start: Tuple[float, float], 
                                        point_end: Tuple[float, float]) -> float:
        """
        Calculate angle from horizontal x-axis (in degrees)
        Vertical up: -90°, Vertical down: +90°
        """
        delta_x = point_end[0] - point_start[0]
        delta_y = point_end[1] - point_start[1]
        return math.degrees(math.atan2(delta_y, delta_x))
    
    @staticmethod
    def calculate_angle_between_three_points(point_a: Tuple[float, float],
                                            point_b: Tuple[float, float],
                                            point_c: Tuple[float, float]) -> float:
        """
        Calculate angle at point_b formed by points a-b-c
        Returns angle in degrees (0-180)
        """
        ba = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])
        bc = np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return math.degrees(angle_radians)
    
    @staticmethod
    def normalize_angle_to_180(angle_degrees: float) -> float:
        """Normalize angle to range (-180, 180]"""
        return (angle_degrees + 180) % 360 - 180
    
    @staticmethod
    def calculate_angular_difference(angle_a: float, angle_b: float) -> float:
        """Calculate minimum absolute angular difference between two angles"""
        diff = GeometryUtils.normalize_angle_to_180(angle_a - angle_b)
        return abs(diff)
    
    @staticmethod
    def calculate_deviation_from_vertical(angle_from_horizontal: float) -> float:
        """Calculate minimum deviation from vertical"""
        deviation_from_up = GeometryUtils.calculate_angular_difference(angle_from_horizontal, -90.0)
        deviation_from_down = GeometryUtils.calculate_angular_difference(angle_from_horizontal, 90.0)
        return min(deviation_from_up, deviation_from_down)
    
    @staticmethod
    def calculate_deviation_from_horizontal(angle_from_horizontal: float) -> float:
        """Calculate minimum deviation from horizontal"""
        deviation_from_right = GeometryUtils.calculate_angular_difference(angle_from_horizontal, 0.0)
        deviation_from_left = GeometryUtils.calculate_angular_difference(angle_from_horizontal, 180.0)
        return min(deviation_from_right, deviation_from_left)
    
    @staticmethod
    def are_segments_aligned(angle_a: float, angle_b: float, 
                            threshold: float = 15.0) -> bool:
        """Check if two segments are aligned (considering bidirectional nature)"""
        diff_same_direction = GeometryUtils.calculate_angular_difference(angle_a, angle_b)
        diff_opposite_direction = GeometryUtils.calculate_angular_difference(angle_a, angle_b + 180.0)
        min_difference = min(diff_same_direction, diff_opposite_direction)
        return min_difference < threshold
