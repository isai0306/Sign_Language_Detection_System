"""
SignAI - Enhanced Gesture Recognition Module
Improved CNN architecture with better preprocessing and prediction
"""

import numpy as np
import pickle
import os
from collections import deque

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Model loading disabled.")


class GestureRecognizer:
    """
    Enhanced gesture recognition using CNN with improved architecture
    """
    
    def __init__(self, model_path=None, confidence_threshold=0.70):
        """
        Initialize gesture recognizer
        
        Args:
            model_path: Path to trained model file
            confidence_threshold: Minimum confidence for prediction (lowered to 0.70)
        """
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.gesture_classes = []
        self.prediction_history = deque(maxlen=5)  # Reduced for faster response
        self._model_kind = None  # 'cnn' | 'lstm'
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load trained model from file
        
        Args:
            model_path: Path to model file (.h5 or .keras)
        """
        if not TENSORFLOW_AVAILABLE:
            print("Error: TensorFlow is required to load models")
            return False
        
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            inp = self.model.input_shape
            if inp and len(inp) == 3 and (inp[1] is None or inp[1] > 3):
                self._model_kind = 'lstm'
            else:
                self._model_kind = 'cnn'
            print(f"   Model kind: {self._model_kind}")
            
            # Load gesture classes if available
            classes_path = model_path.replace('.h5', '_classes.pkl')
            if os.path.exists(classes_path):
                with open(classes_path, 'rb') as f:
                    self.gesture_classes = pickle.load(f)
                print(f"✅ Loaded {len(self.gesture_classes)} gesture classes")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._model_kind = None
            return False
    
    def preprocess_landmarks(self, landmarks):
        """
        Advanced preprocessing of landmarks for better recognition
        
        Args:
            landmarks: Raw landmarks (63 features)
            
        Returns:
            preprocessed: Processed landmarks ready for CNN
        """
        if len(landmarks) != 63:
            print(f"Warning: Expected 63 landmarks, got {len(landmarks)}")
            return None
        
        # Reshape to (21, 3)
        lm = np.array(landmarks).reshape(21, 3)
        
        # 1. Normalize to wrist (landmark 0)
        wrist = lm[0].copy()
        lm = lm - wrist
        
        # 2. Scale normalization
        distances = np.linalg.norm(lm, axis=1)
        max_distance = np.max(distances)
        if max_distance > 0:
            lm = lm / max_distance
        
        # 3. Rotation normalization (align to index finger direction)
        # Get vector from wrist to index finger MCP
        index_mcp = lm[5]  # Index finger MCP
        if np.linalg.norm(index_mcp[:2]) > 0:
            angle = np.arctan2(index_mcp[1], index_mcp[0])
            # Rotate to align with x-axis
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            lm[:, :2] = lm[:, :2] @ rotation_matrix.T
        
        return lm
    
    def predict_raw(self, landmarks, apply_threshold=False):
        """
        Single forward pass (no internal temporal deque). Used with external smoothing.
        """
        if self.model is None or self._model_kind != 'cnn':
            return self._dummy_prediction()
        try:
            processed_lm = self.preprocess_landmarks(landmarks)
            if processed_lm is None:
                return self._dummy_prediction()
            input_data = processed_lm.reshape(1, 21, 3, 1)
            predictions = self.model.predict(input_data, verbose=0)
            return self._scores_to_dict(predictions[0], apply_threshold=apply_threshold)
        except Exception as e:
            print(f"Error in predict_raw: {e}")
            return self._dummy_prediction()

    def predict_lstm(self, sequence_flat, apply_threshold=False):
        """
        sequence_flat: numpy (seq_len, 63) normalized landmark vectors per timestep.
        """
        if self.model is None or self._model_kind != 'lstm':
            return self._dummy_prediction()
        try:
            x = np.asarray(sequence_flat, dtype=np.float32).reshape(1, sequence_flat.shape[0], 63)
            predictions = self.model.predict(x, verbose=0)
            return self._scores_to_dict(predictions[0], apply_threshold=apply_threshold)
        except Exception as e:
            print(f"Error in predict_lstm: {e}")
            return self._dummy_prediction()

    def _scores_to_dict(self, scores, apply_threshold=True):
        confidence = float(np.max(scores))
        class_idx = int(np.argmax(scores))
        if self.gesture_classes:
            gesture_name = self.gesture_classes[class_idx]
        else:
            gesture_name = f"Gesture_{class_idx}"
        if apply_threshold and confidence < self.confidence_threshold:
            gesture_name = "UNKNOWN"
        return {
            'gesture': gesture_name,
            'confidence': confidence,
            'class_index': class_idx,
            'all_scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores),
        }

    def predict(self, landmarks, smooth=True):
        """
        Predict gesture from hand landmarks with enhanced processing
        
        Args:
            landmarks: Normalized hand landmarks (63 features)
            smooth: Apply temporal smoothing
            
        Returns:
            prediction: Dictionary with gesture, confidence, and raw scores
        """
        if self.model is None:
            return self._dummy_prediction()
        
        try:
            if self._model_kind == 'lstm':
                return self._dummy_prediction()

            # Preprocess landmarks
            processed_lm = self.preprocess_landmarks(landmarks)
            if processed_lm is None:
                return self._dummy_prediction()
            
            # Reshape for CNN input: (1, 21, 3, 1) - adding channel dimension
            input_data = processed_lm.reshape(1, 21, 3, 1)
            
            # Get prediction
            predictions = self.model.predict(input_data, verbose=0)
            
            if smooth and len(self.prediction_history) > 0:
                # Add to history
                self.prediction_history.append(predictions[0])
                # Weighted average (more weight to recent predictions)
                weights = np.linspace(0.5, 1.0, len(self.prediction_history))
                weights = weights / weights.sum()
                avg_predictions = np.average(
                    list(self.prediction_history), 
                    axis=0, 
                    weights=weights
                )
                predictions = [avg_predictions]
            else:
                self.prediction_history.append(predictions[0])
            
            d = self._scores_to_dict(predictions[0], apply_threshold=True)
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            if self.gesture_classes:
                top_3_gestures = [(self.gesture_classes[i], float(predictions[0][i]))
                                  for i in top_3_indices]
            else:
                top_3_gestures = [(f"Gesture_{i}", float(predictions[0][i]))
                                  for i in top_3_indices]
            d['top_3'] = top_3_gestures
            d['class_index'] = int(np.argmax(predictions[0]))
            print(f"Prediction: {d['gesture']} ({d['confidence']:.2%})")
            return d
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._dummy_prediction()
    
    def _dummy_prediction(self):
        """Return dummy prediction when model not available"""
        return {
            'gesture': 'UNKNOWN',
            'confidence': 0.0,
            'class_index': -1,
            'all_scores': []
        }
    
    def predict_batch(self, landmarks_batch):
        """
        Predict gestures for multiple landmark sets
        
        Args:
            landmarks_batch: List of landmark arrays
            
        Returns:
            predictions: List of prediction dictionaries
        """
        predictions = []
        for landmarks in landmarks_batch:
            pred = self.predict(landmarks, smooth=False)
            predictions.append(pred)
        return predictions
    
    def get_top_k_predictions(self, landmarks, k=3):
        """
        Get top K gesture predictions
        
        Args:
            landmarks: Hand landmarks
            k: Number of top predictions to return
            
        Returns:
            top_predictions: List of (gesture, confidence) tuples
        """
        if self.model is None:
            return []
        
        try:
            processed_lm = self.preprocess_landmarks(landmarks)
            if processed_lm is None:
                return []
            
            input_data = processed_lm.reshape(1, 21, 3, 1)
            predictions = self.model.predict(input_data, verbose=0)[0]
            
            # Get top K indices
            top_indices = np.argsort(predictions)[-k:][::-1]
            
            top_predictions = []
            for idx in top_indices:
                gesture = self.gesture_classes[idx] if self.gesture_classes else f"Gesture_{idx}"
                confidence = float(predictions[idx])
                top_predictions.append((gesture, confidence))
            
            return top_predictions
            
        except Exception as e:
            print(f"Error getting top predictions: {e}")
            return []
    
    def reset_history(self):
        """Clear prediction history"""
        self.prediction_history.clear()
    
    def get_model_info(self):
        """
        Get information about loaded model
        
        Returns:
            info: Dictionary with model details
        """
        if self.model is None:
            return {
                'loaded': False,
                'message': 'No model loaded'
            }
        
        return {
            'loaded': True,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.gesture_classes),
            'classes': self.gesture_classes,
            'confidence_threshold': self.confidence_threshold
        }


class SimpleGestureRecognizer:
    """
    Enhanced rule-based gesture recognizer
    Better finger detection and gesture rules
    """
    
    def __init__(self):
        self.gestures = {
            'HELLO': 'Open palm facing camera',
            'THANK_YOU': 'Hand moving from chin forward',
            'YES': 'Closed fist',
            'NO': 'Open hand shaking',
            'HELP': 'One finger pointing up',
            'PLEASE': 'Four fingers extended',
            'WATER': 'Three fingers (W shape)',
            'FOOD': 'Hand to mouth gesture'
        }
    
    def predict_from_landmarks(self, landmarks):
        """
        Enhanced rule-based prediction
        
        Args:
            landmarks: Hand landmarks array (63 values)
            
        Returns:
            prediction: Dictionary with gesture info
        """
        if len(landmarks) != 63:
            return {'gesture': 'UNKNOWN', 'confidence': 0.0}
        
        # Reshape to (21, 3)
        lm = np.array(landmarks).reshape(21, 3)
        
        # Normalize
        wrist = lm[0].copy()
        lm = lm - wrist
        
        # Get finger states
        fingers_extended = self._get_finger_states(lm)
        fingers_count = sum(fingers_extended)
        
        # Hand orientation
        palm_facing = self._is_palm_facing_camera(lm)
        
        # Gesture classification with better rules
        gesture, confidence = self._classify_gesture(
            fingers_extended, 
            fingers_count,
            palm_facing,
            lm
        )
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'fingers_extended': fingers_count,
            'finger_states': fingers_extended,
            'palm_facing': palm_facing
        }
    
    def _get_finger_states(self, landmarks):
        """
        Improved finger extension detection
        
        Returns:
            List of booleans [thumb, index, middle, ring, pinky]
        """
        fingers = []
        
        # Thumb (special case - check x-distance)
        thumb_extended = abs(landmarks[4][0] - landmarks[2][0]) > 0.08
        fingers.append(thumb_extended)
        
        # Other fingers - check if tip is above PIP joint
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            # Extended if tip is significantly above PIP
            extended = landmarks[tip_idx][1] < landmarks[pip_idx][1] - 0.05
            fingers.append(extended)
        
        return fingers
    
    def _is_palm_facing_camera(self, landmarks):
        """
        Check if palm is facing the camera
        
        Returns:
            Boolean indicating palm orientation
        """
        # Use z-coordinates to determine palm orientation
        # If fingertips have larger z than wrist, palm is facing camera
        avg_tip_z = np.mean([landmarks[i][2] for i in [4, 8, 12, 16, 20]])
        wrist_z = landmarks[0][2]
        
        return avg_tip_z > wrist_z
    
    def _classify_gesture(self, finger_states, fingers_count, palm_facing, landmarks):
        """
        Classify gesture based on finger states and orientation
        
        Returns:
            (gesture_name, confidence)
        """
        # All fingers extended + palm facing = HELLO
        if fingers_count == 5 and palm_facing:
            return 'HELLO', 0.90
        
        # Fist (no fingers) = YES
        if fingers_count == 0:
            return 'YES', 0.85
        
        # Only index finger = HELP
        if finger_states == [False, True, False, False, False]:
            return 'HELP', 0.88
        
        # Four fingers (no thumb) = PLEASE
        if fingers_count == 4 and not finger_states[0]:
            return 'PLEASE', 0.82
        
        # Three fingers (thumb, index, middle) = WATER
        if fingers_count == 3 and finger_states[0] and finger_states[1] and finger_states[2]:
            return 'WATER', 0.80
        
        # Two fingers (index, middle) = THANK YOU
        if fingers_count == 2 and finger_states[1] and finger_states[2]:
            return 'THANK_YOU', 0.75
        
        # Default based on finger count
        if fingers_count >= 3:
            return 'HELLO', 0.60
        elif fingers_count == 1:
            return 'HELP', 0.60
        else:
            return 'UNKNOWN', 0.50


# Test function
def test_gesture_recognizer():
    """Test gesture recognizer"""
    print("Testing Enhanced Gesture Recognizer")
    print("-" * 50)
    
    # Test with simple recognizer
    recognizer = SimpleGestureRecognizer()
    
    # Create test cases
    test_cases = [
        # All fingers extended (HELLO)
        np.concatenate([
            np.array([0, 0, 0]),  # Wrist
            *[np.array([i*0.1, -0.3, 0]) for i in range(1, 5)],  # Thumb
            *[np.array([0.2, -0.4 - i*0.1, 0]) for i in range(5)],  # Index
            *[np.array([0.1, -0.4 - i*0.1, 0]) for i in range(5)],  # Middle
            *[np.array([0, -0.4 - i*0.1, 0]) for i in range(5)],  # Ring
            *[np.array([-0.1, -0.4 - i*0.1, 0]) for i in range(5)]  # Pinky
        ]),
        # Fist (YES)
        np.random.rand(63) * 0.1
    ]
    
    for i, landmarks in enumerate(test_cases):
        prediction = recognizer.predict_from_landmarks(landmarks.tolist())
        print(f"\nTest {i+1}:")
        print(f"  Gesture: {prediction['gesture']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        print(f"  Fingers Extended: {prediction['fingers_extended']}")
    
    print("\n✅ Test complete")


if __name__ == "__main__":
    test_gesture_recognizer()