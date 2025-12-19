"""
Keypoint Extractor
Extract and validate keypoints from YOLO pose results
"""

from typing import Dict, List, Tuple, Optional
from config.posture_config import PostureConfig, KeypointIndex
from core.geometry import GeometryUtils


class KeypointExtractor:
    """Extract and validate keypoints from YOLO pose results"""
    
    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.geo = GeometryUtils()
    
    def extract(self, person_keypoints: List[List[float]], 
                confidence_scores: Optional[List[float]] = None) -> Dict:
        """
        Extract relevant keypoints with validation
        
        Args:
            person_keypoints: List of [x, y] coordinates for 17 keypoints
            confidence_scores: Optional confidence scores for each keypoint
            
        Returns:
            Dictionary with extracted and validated keypoints
        """
        
        def get_kpt_if_valid(idx: int) -> Optional[Tuple[float, float]]:
            if confidence_scores is not None and confidence_scores[idx] < self.config.min_keypoint_confidence:
                return None
            kpt = person_keypoints[idx]
            if kpt[0] == 0 and kpt[1] == 0:
                return None
            return (kpt[0], kpt[1])
        
        # Extract individual keypoints
        kpts = {
            'shoulder_left': get_kpt_if_valid(KeypointIndex.LEFT_SHOULDER),
            'shoulder_right': get_kpt_if_valid(KeypointIndex.RIGHT_SHOULDER),
            'hip_left': get_kpt_if_valid(KeypointIndex.LEFT_HIP),
            'hip_right': get_kpt_if_valid(KeypointIndex.RIGHT_HIP),
            'knee_left': get_kpt_if_valid(KeypointIndex.LEFT_KNEE),
            'knee_right': get_kpt_if_valid(KeypointIndex.RIGHT_KNEE),
            'ankle_left': get_kpt_if_valid(KeypointIndex.LEFT_ANKLE),
            'ankle_right': get_kpt_if_valid(KeypointIndex.RIGHT_ANKLE),
        }
        
        # Calculate midpoints
        if kpts['shoulder_left'] and kpts['shoulder_right']:
            kpts['shoulder_mid'] = self.geo.calculate_midpoint(kpts['shoulder_left'], kpts['shoulder_right'])
        else:
            kpts['shoulder_mid'] = None
            
        if kpts['hip_left'] and kpts['hip_right']:
            kpts['hip_mid'] = self.geo.calculate_midpoint(kpts['hip_left'], kpts['hip_right'])
        else:
            kpts['hip_mid'] = None
            
        if kpts['knee_left'] and kpts['knee_right']:
            kpts['knee_mid'] = self.geo.calculate_midpoint(kpts['knee_left'], kpts['knee_right'])
        else:
            kpts['knee_mid'] = None
            
        if kpts['ankle_left'] and kpts['ankle_right']:
            kpts['ankle_mid'] = self.geo.calculate_midpoint(kpts['ankle_left'], kpts['ankle_right'])
        else:
            kpts['ankle_mid'] = None
        
        return kpts
