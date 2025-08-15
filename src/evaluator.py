import json
import numpy as np
from typing import Dict, List

class ShotEvaluator:
    def __init__(self, config: dict):
        self.scoring_weights = config['scoring']
        self.thresholds = config['thresholds']
        
    def calculate_category_scores(self, metrics_history: List[Dict]) -> Dict[str, float]:
        """Calculate scores for each category based on metrics history"""
        scores = {}
        
        # Extract valid metrics
        elbow_angles = [m['elbow_angle'] for m in metrics_history if m.get('elbow_angle')]
        spine_leans = [m['spine_lean'] for m in metrics_history if m.get('spine_lean')]
        head_distances = [m['head_knee_distance'] for m in metrics_history if m.get('head_knee_distance')]
        
        # Footwork Score (based on consistency and foot positioning)
        footwork_score = self._evaluate_footwork(metrics_history)
        scores['footwork'] = footwork_score
        
        # Head Position Score
        head_score = self._evaluate_head_position(head_distances)
        scores['head_position'] = head_score
        
        # Swing Control Score (based on elbow angle consistency)
        swing_score = self._evaluate_swing_control(elbow_angles)
        scores['swing_control'] = swing_score
        
        # Balance Score (based on spine lean)
        balance_score = self._evaluate_balance(spine_leans)
        scores['balance'] = balance_score
        
        # Follow-through Score (based on final positions)
        followthrough_score = self._evaluate_followthrough(metrics_history)
        scores['follow_through'] = followthrough_score
        
        return scores
    
    def _evaluate_footwork(self, metrics_history: List[Dict]) -> float:
        """Evaluate footwork quality"""
        if not metrics_history:
            return 5.0
        
        # Simple heuristic based on movement consistency
        foot_angles = [m.get('foot_direction') for m in metrics_history if m.get('foot_direction')]
        if not foot_angles:
            return 6.0
        
        # Check consistency and proximity to ideal angle
        ideal_angle = self.thresholds['foot_direction']['ideal_angle']
        deviations = [abs(angle - ideal_angle) for angle in foot_angles]
        avg_deviation = np.mean(deviations)
        
        # Score based on deviation (lower is better)
        if avg_deviation <= 10:
            return 9.0
        elif avg_deviation <= 20:
            return 7.0
        elif avg_deviation <= 30:
            return 5.0
        else:
            return 3.0
    
    def _evaluate_head_position(self, head_distances: List[float]) -> float:
        """Evaluate head position consistency"""
        if not head_distances:
            return 5.0
        
        avg_distance = np.mean(head_distances)
        max_allowed = self.thresholds['head_knee_alignment']['max_distance']
        
        if avg_distance <= max_allowed * 0.5:
            return 9.0
        elif avg_distance <= max_allowed:
            return 7.0
        elif avg_distance <= max_allowed * 1.5:
            return 5.0
        else:
            return 3.0
    
    def _evaluate_swing_control(self, elbow_angles: List[float]) -> float:
        """Evaluate swing control based on elbow angle progression"""
        if not elbow_angles:
            return 5.0
        
        # Check if angles are within good range
        good_min = self.thresholds['elbow_angle']['good_min']
        good_max = self.thresholds['elbow_angle']['good_max']
        
        in_range_count = sum(1 for angle in elbow_angles 
                            if good_min <= angle <= good_max)
        percentage_in_range = in_range_count / len(elbow_angles)
        
        if percentage_in_range >= 0.8:
            return 9.0
        elif percentage_in_range >= 0.6:
            return 7.0
        elif percentage_in_range >= 0.4:
            return 5.0
        else:
            return 3.0
    
    def _evaluate_balance(self, spine_leans: List[float]) -> float:
        """Evaluate balance based on spine lean"""
        if not spine_leans:
            return 5.0
        
        good_min = self.thresholds['spine_lean']['good_min']
        good_max = self.thresholds['spine_lean']['good_max']
        
        in_range_count = sum(1 for lean in spine_leans 
                            if good_min <= lean <= good_max)
        percentage_in_range = in_range_count / len(spine_leans)
        
        if percentage_in_range >= 0.7:
            return 9.0
        elif percentage_in_range >= 0.5:
            return 7.0
        elif percentage_in_range >= 0.3:
            return 5.0
        else:
            return 3.0
    
    def _evaluate_followthrough(self, metrics_history: List[Dict]) -> float:
        """Evaluate follow-through quality"""
        if len(metrics_history) < 10:
            return 5.0
        
        # Look at final 20% of frames for follow-through
        final_frames = metrics_history[-len(metrics_history)//5:]
        
        # Simple heuristic: consistent elbow angles in final phase
        final_elbow_angles = [m.get('elbow_angle') for m in final_frames 
                             if m.get('elbow_angle')]
        
        if not final_elbow_angles:
            return 5.0
        
        # Check consistency in follow-through
        angle_variance = np.var(final_elbow_angles)
        
        if angle_variance <= 100:  # Low variance = good consistency
            return 8.0
        elif angle_variance <= 300:
            return 6.0
        else:
            return 4.0
    
    def generate_feedback(self, scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate actionable feedback for each category"""
        feedback = {}
        
        # Footwork feedback
        footwork_score = scores['footwork']
        if footwork_score >= 8:
            feedback['footwork'] = [
                "Excellent foot positioning and movement",
                "Maintain this stance consistency in future shots"
            ]
        elif footwork_score >= 6:
            feedback['footwork'] = [
                "Good footwork with minor adjustments needed",
                "Focus on foot direction alignment with the shot"
            ]
        else:
            feedback['footwork'] = [
                "Work on foot positioning and balance",
                "Practice getting your front foot pointing toward the target"
            ]
        
        # Head position feedback
        head_score = scores['head_position']
        if head_score >= 8:
            feedback['head_position'] = [
                "Excellent head stability over front knee",
                "Great visual alignment with the ball"
            ]
        elif head_score >= 6:
            feedback['head_position'] = [
                "Good head position with room for improvement",
                "Keep your head more directly over your front knee"
            ]
        else:
            feedback['head_position'] = [
                "Focus on keeping head still and over front knee",
                "Practice watching the ball closely through contact"
            ]
        
        # Swing control feedback
        swing_score = scores['swing_control']
        if swing_score >= 8:
            feedback['swing_control'] = [
                "Excellent elbow positioning and swing path",
                "Consistent technique throughout the shot"
            ]
        elif swing_score >= 6:
            feedback['swing_control'] = [
                "Good swing mechanics with minor refinements needed",
                "Work on maintaining consistent elbow height"
            ]
        else:
            feedback['swing_control'] = [
                "Focus on keeping front elbow up and extended",
                "Practice smooth, controlled swing movements"
            ]
        
        # Balance feedback
        balance_score = scores['balance']
        if balance_score >= 8:
            feedback['balance'] = [
                "Excellent balance and body positioning",
                "Great forward lean and weight transfer"
            ]
        elif balance_score >= 6:
            feedback['balance'] = [
                "Good balance with slight improvements possible",
                "Maintain consistent forward lean angle"
            ]
        else:
            feedback['balance'] = [
                "Work on balance and weight distribution",
                "Practice proper forward lean into the shot"
            ]
        
        # Follow-through feedback
        followthrough_score = scores['follow_through']
        if followthrough_score >= 8:
            feedback['follow_through'] = [
                "Excellent follow-through completion",
                "Good extension and finish position"
            ]
        elif followthrough_score >= 6:
            feedback['follow_through'] = [
                "Good follow-through with room for smoothness",
                "Focus on completing the swing motion"
            ]
        else:
            feedback['follow_through'] = [
                "Work on completing your follow-through",
                "Practice extending fully through the shot"
            ]
        
        return feedback
    
    def calculate_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        overall = 0.0
        for category, score in category_scores.items():
            weight = self.scoring_weights.get(category, 0.2)
            overall += score * weight
        return round(overall, 1)
    
    def save_evaluation(self, category_scores: Dict[str, float], 
                       feedback: Dict[str, List[str]], output_path: str):
        """Save evaluation results to JSON file"""
        overall_score = self.calculate_overall_score(category_scores)
        
        evaluation = {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'feedback': feedback,
            'grade': self._determine_grade(overall_score)
        }
        
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
    
    def _determine_grade(self, overall_score: float) -> str:
        """Determine skill grade based on overall score"""
        if overall_score >= 8.0:
            return "Advanced"
        elif overall_score >= 6.0:
            return "Intermediate"
        else:
            return "Beginner"
