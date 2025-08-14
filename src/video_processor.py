import cv2
import os
import yt_dlp
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

class VideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config['output']['directory']
        os.makedirs(self.output_dir, exist_ok=True)
    
    def download_video(self, url: str, output_path: str = './data/') -> str:
        """Download video from YouTube"""
        os.makedirs(output_path, exist_ok=True)
        
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'format': 'best[height<=720]',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename
    
    def setup_video_writer(self, input_video_path: str, output_path: str) -> Tuple[cv2.VideoWriter, Dict]:
        """Setup video writer with proper codec and settings"""
        cap = cv2.VideoCapture(input_video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use target dimensions if specified
        target_width = self.config['video'].get('target_width', width)
        target_height = self.config['video'].get('target_height', height)
        target_fps = self.config['video'].get('target_fps', fps)
        
        cap.release()
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))
        
        video_info = {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'target_fps': target_fps,
            'target_width': target_width,
            'target_height': target_height
        }
        
        return writer, video_info
    
    def add_text_overlay(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                        color: Tuple[int, int, int] = (255, 255, 255), 
                        font_scale: float = 0.6) -> np.ndarray:
        """Add text overlay to frame"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, 2, cv2.LINE_AA)
        return frame
    
    def add_metrics_overlay(self, frame: np.ndarray, metrics: Dict, 
                           feedback: Dict) -> np.ndarray:
        """Add biomechanical metrics overlay to frame"""
        y_offset = 30
        
        # Add metrics
        if metrics.get('elbow_angle'):
            text = f"Elbow: {metrics['elbow_angle']:.1f}°"
            frame = self.add_text_overlay(frame, text, (10, y_offset))
            y_offset += 25
        
        if metrics.get('spine_lean'):
            text = f"Spine Lean: {metrics['spine_lean']:.1f}°"
            frame = self.add_text_overlay(frame, text, (10, y_offset))
            y_offset += 25
        
        if metrics.get('head_knee_distance'):
            text = f"Head-Knee: {metrics['head_knee_distance']:.0f}px"
            frame = self.add_text_overlay(frame, text, (10, y_offset))
            y_offset += 25
        
        # Add feedback
        y_offset += 10
        for key, msg in feedback.items():
            color = (0, 255, 0) if "✅" in msg else (0, 0, 255)
            frame = self.add_text_overlay(frame, msg, (10, y_offset), color, 0.5)
            y_offset += 25
        
        return frame
