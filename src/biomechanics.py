import numpy as np
import math
from typing import Dict, List, Optional, Tuple

class BiomechanicsAnalyzer:
    def __init__(self, config: dict):
        self.thresholds = config['thresholds']
        
    def calculate_angle(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                       p3: Tuple[int, int]) -> float:
        """Calculate angle between three points"""
        try:
            # Vector from p2 to p1
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            # Vector from p2 to p3
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_angle))
            return angle
        except:
            return 0.0
    
    def calculate_front_elbow_angle(self, landmarks: Dict, frame_shape: Tuple, 
                                  pose_estimator) -> Optional[float]:
        """Calculate front elbow angle (shoulder-elbow-wrist)"""
        # Assuming right-handed batsman (left arm is front)
        shoulder = pose_estimator.get_joint_coordinates(landmarks, 11, frame_shape)  # Left shoulder
        elbow = pose_estimator.get_joint_coordinates(landmarks, 13, frame_shape)    # Left elbow
        wrist = pose_estimator.get_joint_coordinates(landmarks, 15, frame_shape)    # Left wrist
        
        if all([shoulder, elbow, wrist]):
            return self.calculate_angle(shoulder, elbow, wrist)
        return None
    
    def calculate_spine_lean(self, landmarks: Dict, frame_shape: Tuple, 
                           pose_estimator) -> Optional[float]:
        """Calculate spine lean angle"""
        hip = pose_estimator.get_joint_coordinates(landmarks, 23, frame_shape)      # Left hip
        shoulder = pose_estimator.get_joint_coordinates(landmarks, 11, frame_shape) # Left shoulder
        
        if all([hip, shoulder]):
            # Calculate angle from vertical
            vertical_vector = np.array([0, -1])  # Pointing up
            spine_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
            
            if np.linalg.norm(spine_vector) > 0:
                cos_angle = np.dot(vertical_vector, spine_vector) / np.linalg.norm(spine_vector)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = math.degrees(math.acos(cos_angle))
                return angle
        return None
    
    def calculate_head_knee_alignment(self, landmarks: Dict, frame_shape: Tuple, 
                                    pose_estimator) -> Optional[float]:
        """Calculate vertical distance between head and front knee"""
        nose = pose_estimator.get_joint_coordinates(landmarks, 0, frame_shape)     # Nose
        knee = pose_estimator.get_joint_coordinates(landmarks, 25, frame_shape)    # Left knee
        
        if all([nose, knee]):
            return abs(nose[0] - knee[0])  # Horizontal distance
        return None
    
    def calculate_foot_direction(self, landmarks: Dict, frame_shape: Tuple, 
                               pose_estimator) -> Optional[float]:
        """Calculate front foot direction angle"""
        ankle = pose_estimator.get_joint_coordinates(landmarks, 27, frame_shape)   # Left ankle
        foot = pose_estimator.get_joint_coordinates(landmarks, 31, frame_shape)    # Left foot index
        
        if all([ankle, foot]):
            foot_vector = np.array([foot[0] - ankle[0], foot[1] - ankle[1]])
            horizontal_vector = np.array([1, 0])  # Pointing right
            
            if np.linalg.norm(foot_vector) > 0:
                cos_angle = np.dot(horizontal_vector, foot_vector) / np.linalg.norm(foot_vector)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = math.degrees(math.acos(cos_angle))
                return angle
        return None
    
    def evaluate_metrics(self, metrics: Dict) -> Dict[str, str]:
        """Evaluate biomechanical metrics and return feedback"""
        feedback = {}
        
        # Elbow angle evaluation
        if metrics.get('elbow_angle'):
            angle = metrics['elbow_angle']
            if self.thresholds['elbow_angle']['good_min'] <= angle <= self.thresholds['elbow_angle']['good_max']:
                feedback['elbow'] = "✅ Good elbow elevation"
            else:
                feedback['elbow'] = "❌ Adjust elbow angle"
        
        # Spine lean evaluation
        if metrics.get('spine_lean'):
            lean = metrics['spine_lean']
            if self.thresholds['spine_lean']['good_min'] <= lean <= self.thresholds['spine_lean']['good_max']:
                feedback['spine'] = "✅ Good spine lean"
            else:
                feedback['spine'] = "❌ Adjust forward lean"
        
        # Head-knee alignment
        if metrics.get('head_knee_distance'):
            distance = metrics['head_knee_distance']
            if distance <= self.thresholds['head_knee_alignment']['max_distance']:
                feedback['head'] = "✅ Good head position"
            else:
                feedback['head'] = "❌ Head not over front knee"
        
        return feedback
