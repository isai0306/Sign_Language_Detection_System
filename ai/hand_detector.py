"""
SignAI - Hand Detection Module
Uses MediaPipe for real-time hand landmark detection
"""

import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    """
    Hand detection using MediaPipe Hands
    Extracts hand landmarks from video frames
    """
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        import threading
        self._lock = threading.Lock()
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def find_hands(self, frame, draw=True):
        """
        Detect hands in frame and optionally draw landmarks
        
        Args:
            frame: Input image/frame (BGR format)
            draw: Whether to draw landmarks on frame
            
        Returns:
            frame: Frame with drawings (if draw=True)
            results: MediaPipe detection results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        with self._lock:
            results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, results
    
    def extract_landmarks(self, results):
        """
        Extract hand landmarks as numpy array
        
        Args:
            results: MediaPipe detection results
            
        Returns:
            landmarks_list: List of landmark arrays (one per hand)
        """
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(np.array(landmarks))
        
        return landmarks_list
    
    def get_hand_info(self, results):
        """
        Get detailed hand information
        
        Args:
            results: MediaPipe detection results
            
        Returns:
            hand_info: List of dictionaries with hand details
        """
        hand_info = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                info = {
                    'index': idx,
                    'label': handedness.classification[0].label,  # Left or Right
                    'score': handedness.classification[0].score,
                    'landmarks': self._landmarks_to_dict(hand_landmarks)
                }
                hand_info.append(info)
        
        return hand_info
    
    def _landmarks_to_dict(self, hand_landmarks):
        """Convert landmarks to dictionary format"""
        landmarks = {}
        landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        for idx, lm in enumerate(hand_landmarks.landmark):
            landmarks[landmark_names[idx]] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            }
        
        return landmarks
    
    def calculate_bounding_box(self, hand_landmarks, frame_width, frame_height):
        """
        Calculate bounding box around hand
        
        Args:
            hand_landmarks: Hand landmarks object
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            bbox: Tuple (x_min, y_min, x_max, y_max)
        """
        x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
        y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame_width, x_max + padding)
        y_max = min(frame_height, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def draw_signai_overlay(self, frame, results, hand_predictions=None):
        """
        Draw dynamic colored bounding boxes, red joint points, white bone lines, and labels
        (matches SignAI reference visualization style).
        """
        if not results.multi_hand_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        hand_predictions = hand_predictions or []
        
        # Color palette for gestures — dynamic per gesture
        colors = {
            'HELLO': (0, 200, 0),       # Green
            'HI': (0, 220, 0),          # Bright Green
            'THANK_YOU': (0, 255, 128), # Spring Green
            'YES': (0, 215, 255),       # Gold
            'NO': (0, 165, 255),        # Orange
            'HELP': (0, 0, 255),        # Red (priority)
            'WATER': (255, 100, 0),     # Blue-Orange
            'FOOD': (200, 100, 250),    # Violet
            'PLEASE': (255, 191, 0),    # Sky Blue
            'PEACE': (128, 255, 0),     # Lime
            'THUMBS_UP': (0, 215, 255), # Gold
            'GOOD': (0, 215, 255),      # Gold
            'CALL_ME': (255, 105, 180), # Hot Pink
            'LOVE': (147, 20, 255),     # Purple
            'STOP': (0, 0, 200),        # Dark Red
            # Reference image gestures
            'WRONG': (0, 0, 220),       # Red
            'NICE': (50, 200, 50),      # Mid Green
            'ACCIDENT': (0, 100, 255),  # Orange-Red
            'AEROPLANE': (255, 200, 0), # Cyan-Gold
            'BUSY': (80, 80, 80),       # Gray
            'AWESOME': (0, 200, 255),   # Gold-Cyan
            'TOGETHER': (200, 50, 200), # Magenta
            'CONFIDENT': (100, 255, 100), # Light Green
            # Day-to-day
            'GOOD_MORNING': (0, 220, 190), # Teal
            'WELCOME': (100, 200, 255),    # Light Blue
            'GOOD_NIGHT': (50, 50, 200),   # Navy
            'EXCUSE_ME': (200, 200, 0),    # Yellow
            'I_AM_FINE': (0, 200, 100),    # Green
            'HOW_ARE_YOU': (200, 100, 0),  # Orange
            'SEE_YOU_LATER': (150, 150, 255), # Lavender
            'UNKNOWN': (128, 128, 128)  # Gray
        }
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            bbox = self.calculate_bounding_box(hand_landmarks, w, h)
            x_min, y_min, x_max, y_max = bbox
            
            p = hand_predictions[idx] if idx < len(hand_predictions) and hand_predictions[idx] else {}
            g = p.get("gesture") or "UNKNOWN"
            c = p.get("confidence") or 0
            
            box_color = colors.get(g.upper(), (0, 255, 0)) # Default green
            
            # Thick bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            
            # Draw bones
            pts = []
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                pts.append((px, py))
            for connection in self.mp_hands.HAND_CONNECTIONS:
                a, b = connection
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw joints
            for pi in pts:
                cv2.circle(frame, pi, 4, (0, 0, 255), -1, cv2.LINE_AA)
                
            label = "Hand"
            if g and g != "UNKNOWN":
                # Convert 'HELLO' -> 'Hello'
                display_g = g.replace('_', ' ').title()
                label = f"{display_g}: {int(c * 100)}%"
            else:
                label = "..."
                
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ly = max(y_min - 8, th + 6)
            
            # Label background box
            cv2.rectangle(frame, (x_min, ly - th - 6), (x_min + tw + 10, ly + 4), box_color, -1)
            # Label text (black for contrast)
            cv2.putText(frame, label, (x_min + 5, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
        return frame

    def draw_bounding_box(self, frame, bbox, label="Hand"):
        """
        Draw bounding box on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box tuple (x_min, y_min, x_max, y_max)
            label: Label text
            
        Returns:
            frame: Frame with bounding box drawn
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(frame, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks to be scale and position invariant
        
        Args:
            landmarks: Raw landmark array
            
        Returns:
            normalized: Normalized landmarks
        """
        if len(landmarks) == 0:
            return landmarks
        
        # Reshape to (21, 3) - 21 landmarks, each with x, y, z
        landmarks = np.array(landmarks).reshape(21, 3)
        
        # Get wrist position (landmark 0)
        wrist = landmarks[0]
        
        # Translate so wrist is at origin
        landmarks = landmarks - wrist
        
        # Calculate max distance from wrist for scaling
        distances = np.linalg.norm(landmarks, axis=1)
        max_distance = np.max(distances)
        
        # Scale so max distance is 1
        if max_distance > 0:
            landmarks = landmarks / max_distance
        
        # Flatten back to 1D array
        return landmarks.flatten()
    
    def close(self):
        """Release resources"""
        self.hands.close()


# Utility function for standalone testing
def test_hand_detector():
    """Test hand detector with webcam"""
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    
    print("Hand Detector Test")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Detect hands
        frame, results = detector.find_hands(frame, draw=True)
        
        # Get hand info
        hand_info = detector.get_hand_info(results)
        
        # Display info on frame
        y_pos = 30
        for info in hand_info:
            text = f"{info['label']} Hand - Score: {info['score']:.2f}"
            cv2.putText(frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
        # Show frame
        cv2.imshow('Hand Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    test_hand_detector()