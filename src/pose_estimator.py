import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple

class PoseEstimator:
    def __init__(self, config: dict):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=config['pose']['model_complexity'],
            min_detection_confidence=config['pose']['min_detection_confidence'],
            min_tracking_confidence=config['pose']['min_tracking_confidence']
        )
        
    def extract_keypoints(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract pose keypoints from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            return {
                'landmarks': landmarks,
                'raw_results': results
            }
        return None
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose skeleton on frame"""
        if pose_data and 'raw_results' in pose_data:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_data['raw_results'].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
    
    def get_joint_coordinates(self, landmarks: Dict, joint_idx: int, 
                            frame_shape: Tuple) -> Optional[Tuple[int, int]]:
        """Convert normalized coordinates to pixel coordinates"""
        if joint_idx in landmarks:
            landmark = landmarks[joint_idx]
            if landmark['visibility'] > 0.5:
                x = int(landmark['x'] * frame_shape[1])
                y = int(landmark['y'] * frame_shape[0])
                return (x, y)
        return None
