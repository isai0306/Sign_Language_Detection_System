"""
SignAI - Simplified API Routes
Core: Gesture Detection & Text-to-Sign
"""

from flask import Blueprint, request, jsonify, session
from datetime import datetime
import mysql.connector
import numpy as np
import base64
import cv2

from config import Config
from ai.hand_detector import HandDetector
from ai.gesture_recognizer import GestureRecognizer, SimpleGestureRecognizer
from utils.translator import Translator
from utils.tts import TextToSpeech

api_bp = Blueprint('api', __name__)

# Initialize AI modules
hand_detector = HandDetector(
    max_num_hands=Config.MEDIAPIPE_CONFIG['max_num_hands'],
    min_detection_confidence=Config.MEDIAPIPE_CONFIG['min_detection_confidence'],
    min_tracking_confidence=Config.MEDIAPIPE_CONFIG['min_tracking_confidence']
)

# Load gesture recognizer
try:
    gesture_recognizer = GestureRecognizer(
        model_path=Config.MODEL_PATH,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD
    )
    # Only use SimpleGestureRecognizer if no trained model is available
    if gesture_recognizer.model is None:
        gesture_recognizer = SimpleGestureRecognizer()
        USE_SIMPLE = True
        print("⚠️  Using SimpleGestureRecognizer (no trained model found)")
    else:
        USE_SIMPLE = False
        print("✅ Using trained CNN model for gesture recognition")
except Exception as e:
    print(f"❌ Error loading gesture recognizer: {e}")
    gesture_recognizer = SimpleGestureRecognizer()
    USE_SIMPLE = True

translator = Translator()
tts = TextToSpeech()

def get_db():
    return mysql.connector.connect(**Config.DB_CONFIG)

def reload_gesture_recognizer():
    """Reload gesture recognizer after training"""
    global gesture_recognizer, USE_SIMPLE
    
    try:
        print(f"DEBUG: Reloading model from {Config.MODEL_PATH}")
        gesture_recognizer = GestureRecognizer(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD
        )
        print(f"DEBUG: GestureRecognizer reloaded, model = {gesture_recognizer.model}")
        # Only use SimpleGestureRecognizer if no trained model is available
        if gesture_recognizer.model is None:
            gesture_recognizer = SimpleGestureRecognizer()
            USE_SIMPLE = True
            print("⚠️  Using SimpleGestureRecognizer (no trained model found)")
        else:
            USE_SIMPLE = False
            print("✅ Using trained CNN model for gesture recognition")
            print(f"DEBUG: USE_SIMPLE set to {USE_SIMPLE}")
    except Exception as e:
        print(f"❌ Error reloading gesture recognizer: {e}")
        gesture_recognizer = SimpleGestureRecognizer()
        USE_SIMPLE = True

# ============================================================
# CORE API 1: SIGN → TEXT (with audio)
# ============================================================

@api_bp.route('/detect-gesture', methods=['POST'])
def detect_gesture():
    """
    Detect gesture from video frame
    
    Input: {frame: base64_image, user_id: int}
    Output: {gesture, confidence, translations, audio_paths}
    """
    global USE_SIMPLE, gesture_recognizer
    
    try:
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'No frame'}), 400
        
        # Decode frame
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid frame'}), 400
        
        # Detect hands
        _, results = hand_detector.find_hands(frame, draw=False)
        
        if not results.multi_hand_landmarks:
            return jsonify({
                'success': True,
                'gesture': None,
                'message': 'No hands detected'
            })
        
        # Extract landmarks
        landmarks_list = hand_detector.extract_landmarks(results)
        if not landmarks_list:
            return jsonify({
                'success': True,
                'gesture': None,
                'message': 'No landmarks'
            })
        
        # Normalize and predict
        landmarks = hand_detector.normalize_landmarks(landmarks_list[0])
        
        # Check if we need to reload the model (in case it was trained while app was running)
        if USE_SIMPLE:
            try:
                temp_recognizer = GestureRecognizer(
                    model_path=Config.MODEL_PATH,
                    confidence_threshold=Config.CONFIDENCE_THRESHOLD
                )
                if temp_recognizer.model is not None:
                    gesture_recognizer = temp_recognizer
                    USE_SIMPLE = False
                    print("✅ Auto-detected trained model, switching to CNN")
            except:
                pass  # Keep using simple recognizer
        
        if USE_SIMPLE:
            prediction = gesture_recognizer.predict_from_landmarks(landmarks)
        else:
            prediction = gesture_recognizer.predict(landmarks)
        
        gesture_name = prediction['gesture']
        confidence = prediction['confidence']
        
        # Skip uncertain predictions
        if gesture_name == 'UNKNOWN' or confidence < 0.5:
            return jsonify({
                'success': True,
                'gesture': None,
                'confidence': confidence
            })
        
        # Translate
        translations = translator.translate_both(gesture_name)
        
        # Generate audio based on user preferences
        audio_paths = {}
        user_id = data.get('user_id') or session.get('user_id')
        
        if user_id:
            prefs = _get_user_preferences(user_id)
            if prefs and prefs['audio_enabled']:
                pref_lang = prefs['preferred_language'].lower()
                
                if pref_lang == 'both':
                    audio_paths = tts.generate_multilingual({
                        'tamil': translations['tamil'],
                        'hindi': translations['hindi']
                    })
                else:
                    audio_path = tts.generate_speech(translations[pref_lang], pref_lang)
                    if audio_path:
                        audio_paths[pref_lang] = audio_path
        
        # Log detection
        if user_id:
            _log_detection(user_id, gesture_name, confidence, translations)
        
        return jsonify({
            'success': True,
            'gesture': gesture_name,
            'confidence': float(confidence),
            'translations': translations,
            'audio_paths': audio_paths
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# CORE API 2: TEXT → SIGN (images)
# ============================================================

@api_bp.route('/text-to-sign', methods=['POST'])
def text_to_sign():
    """
    Convert text to sign language images
    
    Input: {text: string}
    Output: {words: [], sign_data: {word: {image}}}
    """
    try:
        data = request.get_json()
        input_text = data.get('text', '').strip().upper()
        
        if not input_text:
            return jsonify({'success': False, 'error': 'No text'}), 400
        
        # Split into words
        words = input_text.split()
        
        # Get gesture data for each word
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        sign_data = {}
        
        for word in words:
            cursor.execute("""
                SELECT gesture_name, image_path, description
                FROM gestures
                WHERE gesture_name = %s
            """, (word,))
            
            result = cursor.fetchone()
            
            if result:
                sign_data[word] = {
                    'image': result['image_path'],
                    'description': result['description'],
                    'found': True
                }
            else:
                # Word not in database
                sign_data[word] = {
                    'image': None,
                    'description': 'Sign not available',
                    'found': False
                }
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'input_text': input_text,
            'words': words,
            'sign_data': sign_data,
            'total_words': len(words),
            'found_signs': sum(1 for s in sign_data.values() if s['found'])
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# HELPER ENDPOINTS
# ============================================================

@api_bp.route('/gesture-history', methods=['GET'])
def gesture_history():
    """Get user's detection history"""
    try:
        user_id = request.args.get('user_id') or session.get('user_id')
        limit = request.args.get('limit', 20, type=int)
        
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID required'}), 400
        
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT gh.*, g.gesture_name
            FROM gesture_history gh
            JOIN gestures g ON gh.gesture_id = g.id
            WHERE gh.user_id = %s
            ORDER BY gh.detection_timestamp DESC
            LIMIT %s
        """, (user_id, limit))
        
        history = cursor.fetchall()
        
        # Convert datetime
        for item in history:
            item['detection_timestamp'] = item['detection_timestamp'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/available-gestures', methods=['GET'])
def available_gestures():
    """Get all available gestures"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT gesture_name, description, image_path, category
            FROM gestures
            ORDER BY category, gesture_name
        """)
        
        gestures = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'gestures': gestures,
            'total': len(gestures)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/system-info', methods=['GET'])
def system_info():
    """Get system information"""
    model_info = {
        'type': 'rule_based' if USE_SIMPLE else 'neural_network',
        'status': 'active'
    }
    
    return jsonify({
        'success': True,
        'model': model_info,
        'services': {
            'hand_detection': True,
            'translation': translator.is_translation_available(),
            'tts': tts.is_available()
        },
        'config': {
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            'supported_languages': list(Config.SUPPORTED_LANGUAGES.keys())
        }
    })


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_user_preferences(user_id):
    """Get user language preferences"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT preferred_language, audio_enabled FROM language_preferences WHERE user_id = %s",
            (user_id,)
        )
        prefs = cursor.fetchone()
        cursor.close()
        conn.close()
        return prefs
    except:
        return None


def _log_detection(user_id, gesture_name, confidence, translations):
    """Log gesture detection"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        # Get gesture ID
        cursor.execute("SELECT id FROM gestures WHERE gesture_name = %s", (gesture_name,))
        gesture = cursor.fetchone()
        
        if gesture:
            cursor.execute("""
                INSERT INTO gesture_history
                (user_id, gesture_id, detected_text, confidence_score,
                 translated_tamil, translated_hindi, detection_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, gesture['id'], gesture_name, confidence,
                translations['tamil'], translations['hindi'], datetime.now()
            ))
            conn.commit()
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Logging error: {e}")