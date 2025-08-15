#!/usr/bin/env python3
"""
AthleteRise - Real-Time Cricket Cover Drive Analysis
Main script for processing cricket videos and analyzing biomechanics
"""

import cv2
import os
import time
import yaml
from typing import Dict, List
from tqdm import tqdm

from config_reader import ConfigReader
from src.pose_estimator import PoseEstimator
from src.biomechanics import BiomechanicsAnalyzer
from src.video_processor import VideoProcessor
from src.evaluator import ShotEvaluator

def analyze_video(video_path: str, config_path: str = "configs\\config.yaml") -> Dict:
    """
    Main function to analyze cricket cover drive video
    
    Args:
        video_path: Path to input video file
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing analysis results
    """
    
    # Load configuration
    config_reader = ConfigReader()
    config = config_reader.load_config(config_path)
    
    # Initialize components
    pose_estimator = PoseEstimator(config)
    biomechanics = BiomechanicsAnalyzer(config)
    video_processor = VideoProcessor(config)
    evaluator = ShotEvaluator(config)
    
    # Setup paths
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, config['output']['video_name'])
    evaluation_path = os.path.join(output_dir, config['output']['evaluation_file'])
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Setup video writer
    writer, video_info = video_processor.setup_video_writer(video_path, output_video_path)
    
    # Initialize tracking variables
    metrics_history = []
    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    
    print(f"Processing video: {video_info['total_frames']} frames at {video_info['fps']} FPS")
    
    # Process video frame by frame
    with tqdm(total=video_info['total_frames'], desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Resize frame if needed
            if (frame.shape[1] != video_info['target_width'] or 
                frame.shape[0] != video_info['target_height']):
                frame = cv2.resize(frame, (video_info['target_width'], 
                                         video_info['target_height']))
            
            # Extract pose
            pose_data = pose_estimator.extract_keypoints(frame)
            
            # Initialize frame metrics
            frame_metrics = {
                'frame_number': frame_count,
                'elbow_angle': None,
                'spine_lean': None,
                'head_knee_distance': None,
                'foot_direction': None
            }
            
            if pose_data:
                landmarks = pose_data['landmarks']
                frame_shape = frame.shape
                
                # Calculate biomechanical metrics
                elbow_angle = biomechanics.calculate_front_elbow_angle(
                    landmarks, frame_shape, pose_estimator)
                spine_lean = biomechanics.calculate_spine_lean(
                    landmarks, frame_shape, pose_estimator)
                head_knee_distance = biomechanics.calculate_head_knee_alignment(
                    landmarks, frame_shape, pose_estimator)
                foot_direction = biomechanics.calculate_foot_direction(
                    landmarks, frame_shape, pose_estimator)
                
                # Update frame metrics
                if elbow_angle: frame_metrics['elbow_angle'] = elbow_angle
                if spine_lean: frame_metrics['spine_lean'] = spine_lean
                if head_knee_distance: frame_metrics['head_knee_distance'] = head_knee_distance
                if foot_direction: frame_metrics['foot_direction'] = foot_direction
                
                # Draw pose skeleton
                frame = pose_estimator.draw_pose(frame, pose_data)
                
                # Get real-time feedback
                feedback = biomechanics.evaluate_metrics(frame_metrics)
                
                # Add overlays
                frame = video_processor.add_metrics_overlay(frame, frame_metrics, feedback)
            
            # Store metrics for final evaluation
            metrics_history.append(frame_metrics)
            
            # Write frame to output video
            writer.write(frame)
            
            # Update counters
            frame_count += 1
            fps_counter += 1
            
            # Calculate and display FPS every 30 frames
            if fps_counter % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - start_time)
                print(f"Processing FPS: {fps:.1f}")
                start_time = current_time
            
            pbar.update(1)
    
    # Clean up
    cap.release()
    writer.release()
    
    # Calculate final evaluation
    print("Calculating final evaluation...")
    category_scores = evaluator.calculate_category_scores(metrics_history)
    feedback = evaluator.generate_feedback(category_scores)
    
    # Save evaluation
    evaluator.save_evaluation(category_scores, feedback, evaluation_path)
    
    # Calculate average processing FPS
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    results = {
        'video_info': video_info,
        'processing_fps': avg_fps,
        'output_video': output_video_path,
        'evaluation_file': evaluation_path,
        'category_scores': category_scores,
        'overall_score': evaluator.calculate_overall_score(category_scores),
        'feedback': feedback
    }
    
    return results

def main():
    """Main execution function"""

    config_path = "configs\\config.yaml"
    config_reader = ConfigReader()
    config = config_reader.load_config(config_path)

    
    # youtube video URL for cricket cover drive analysis
    video_url = config['input_video']['url']
    if video_url == "NONE":
        print("No video URL provided in configuration. Please update the config.yaml file.")
        return
    
    print("AthleteRise - Cricket Cover Drive Analysis")
    print("=" * 50)
    
    # Initialize video processor for downloading
    config_reader = ConfigReader()
    config = config_reader.load_config("./configs/config.yaml")
    video_processor = VideoProcessor(config)
    
    # Download video
    print("Downloading video...")
    try:
        video_path = video_processor.download_video(video_url)
        print(f"Video downloaded: {video_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")
        print("Please ensure the video URL is accessible and try again.")
        return
    
    # Analyze video
    try:
        print("\nStarting analysis...")
        results = analyze_video(video_path)
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Overall Score: {results['overall_score']}/10")
        print(f"Processing Speed: {results['processing_fps']:.1f} FPS")
        print(f"Output Video: {results['output_video']}")
        print(f"Evaluation File: {results['evaluation_file']}")
        
        print("\nCategory Scores:")
        for category, score in results['category_scores'].items():
            print(f"  {category.replace('_', ' ').title()}: {score}/10")
        
        print("\nDetailed feedback saved to evaluation file.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
