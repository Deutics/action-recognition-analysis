"""Keypoint extraction and validation."""
from typing import Dict, List, Tuple, Optional
from config.detection_config import PostureConfig, KeypointIndex
from core.geometry import GeometryUtils


class KeypointExtractor:
    """Extracts and validates keypoints from YOLO pose results."""
    
    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.geo = GeometryUtils()
    
    def extract(self, person_keypoints: List[List[float]], 
                confidence_scores: Optional[List[float]] = None) -> Dict:
        """Extract relevant keypoints with validation."""
        def get_kpt_if_valid(idx: int) -> Optional[Tuple[float, float]]:
            if idx >= len(person_keypoints):
                return None

            if confidence_scores is not None:
                if idx >= len(confidence_scores):
                    return None
                if confidence_scores[idx] < self.config.min_keypoint_confidence:
                    return None

            kpt = person_keypoints[idx]

            if kpt is None or len(kpt) < 2:
                return None
            if kpt[0] == 0 and kpt[1] == 0:
                return None

            return (float(kpt[0]), float(kpt[1]))

        kpts = {
            'shoulder_left': get_kpt_if_valid(KeypointIndex.LEFT_SHOULDER),
            'shoulder_right': get_kpt_if_valid(KeypointIndex.RIGHT_SHOULDER),
            'elbow_left': get_kpt_if_valid(KeypointIndex.LEFT_ELBOW),
            'elbow_right': get_kpt_if_valid(KeypointIndex.RIGHT_ELBOW),
            'hip_left': get_kpt_if_valid(KeypointIndex.LEFT_HIP),
            'hip_right': get_kpt_if_valid(KeypointIndex.RIGHT_HIP),
            'knee_left': get_kpt_if_valid(KeypointIndex.LEFT_KNEE),
            'knee_right': get_kpt_if_valid(KeypointIndex.RIGHT_KNEE),
            'ankle_left': get_kpt_if_valid(KeypointIndex.LEFT_ANKLE),
            'ankle_right': get_kpt_if_valid(KeypointIndex.RIGHT_ANKLE),
        }
        
        if kpts['shoulder_left'] and kpts['shoulder_right']:
            kpts['shoulder_mid'] = self.geo.calculate_midpoint(
                kpts['shoulder_left'], kpts['shoulder_right'])
        else:
            kpts['shoulder_mid'] = None
        
        if kpts['elbow_left'] and kpts['elbow_right']:
            kpts['elbow_mid'] = self.geo.calculate_midpoint(
                kpts['elbow_left'], kpts['elbow_right'])
        else:
            kpts['elbow_mid'] = None
            
        if kpts['hip_left'] and kpts['hip_right']:
            kpts['hip_mid'] = self.geo.calculate_midpoint(
                kpts['hip_left'], kpts['hip_right'])
        else:
            kpts['hip_mid'] = None
            
        if kpts['knee_left'] and kpts['knee_right']:
            kpts['knee_mid'] = self.geo.calculate_midpoint(
                kpts['knee_left'], kpts['knee_right'])
        else:
            kpts['knee_mid'] = None
            
        if kpts['ankle_left'] and kpts['ankle_right']:
            kpts['ankle_mid'] = self.geo.calculate_midpoint(
                kpts['ankle_left'], kpts['ankle_right'])
        else:
            kpts['ankle_mid'] = None
        
        return kpts
