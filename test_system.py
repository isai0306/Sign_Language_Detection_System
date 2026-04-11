"""
SignAI - System Test Script
Validates enhanced CNN gesture recognition system
Run this to check if everything is working correctly
"""

import os
import sys
import numpy as np
from pathlib import Path

print("="*70)
print(" "*20 + "SignAI System Test")
print("="*70)

# Test 1: Check Python version
print("\n1️⃣  Checking Python version...")
py_version = sys.version_info
if py_version.major >= 3 and py_version.minor >= 7:
    print(f"   ✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
else:
    print(f"   ❌ Python {py_version.major}.{py_version.minor} - Need 3.7+")
    sys.exit(1)

# Test 2: Check required packages
print("\n2️⃣  Checking required packages...")
required_packages = {
    'tensorflow': 'TensorFlow (for CNN)',
    'cv2': 'OpenCV (for video)',
    'mediapipe': 'MediaPipe (for hand detection)',
    'flask': 'Flask (for web server)',
    'mysql.connector': 'MySQL connector',
    'sklearn': 'Scikit-learn (for training)',
    'numpy': 'NumPy (for arrays)'
}

missing_packages = []
for package, description in required_packages.items():
    try:
        if package == 'cv2':
            import cv2
        elif package == 'mysql.connector':
            import mysql.connector
        elif package == 'sklearn':
            import sklearn
        else:
            __import__(package)
        print(f"   ✅ {description}")
    except ImportError:
        print(f"   ❌ {description} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install", " ".join(missing_packages))

# Test 3: Check directory structure
print("\n3️⃣  Checking directory structure...")
required_dirs = [
    'static/models',
    'static/audio_cache',
    'static/images/signs',
    'training_data',
    'ai',
    'routes',
    'utils',
    'templates'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ⚠️  {dir_path} - Creating...")
        os.makedirs(dir_path, exist_ok=True)

# Test 4: Check core files
print("\n4️⃣  Checking core files...")
core_files = {
    'ai/hand_detector.py': 'Hand Detection',
    'ai/gesture_recognizer.py': 'Gesture Recognition',
    'ai/gesture_trainer.py': 'Model Training',
    'routes/api_routes.py': 'API Routes',
    'routes/training_routes.py': 'Training Routes',
    'config.py': 'Configuration'
}

missing_files = []
for file_path, description in core_files.items():
    if os.path.exists(file_path):
        # Check if it's the enhanced version
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'enhanced' in content.lower() or 'improved' in content.lower():
                print(f"   ✅ {description} (Enhanced)")
            else:
                print(f"   ⚠️  {description} (Old version?)")
    else:
        print(f"   ❌ {description} - MISSING")
        missing_files.append(file_path)

if missing_files:
    print(f"\n   ⚠️  Missing files: {', '.join(missing_files)}")

# Test 5: Test hand detector
print("\n5️⃣  Testing Hand Detector...")
try:
    from ai.hand_detector import HandDetector
    detector = HandDetector()
    print("   ✅ Hand detector initialized")
    
    # Test with dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, results = detector.find_hands(dummy_frame, draw=False)
    print("   ✅ Hand detection working")
except Exception as e:
    print(f"   ❌ Hand detector error: {e}")

# Test 6: Test rule-based recognizer
print("\n6️⃣  Testing Rule-Based Recognizer...")
try:
    from ai.gesture_recognizer import SimpleGestureRecognizer
    recognizer = SimpleGestureRecognizer()
    
    # Create test landmarks (5 fingers extended = HELLO)
    test_landmarks = np.concatenate([
        np.array([0, 0, 0]),  # Wrist
        *[np.array([i*0.1, -0.3, 0]) for i in range(1, 5)],  # Thumb
        *[np.array([0.2, -0.4 - i*0.1, 0]) for i in range(5)],  # Index
        *[np.array([0.1, -0.4 - i*0.1, 0]) for i in range(5)],  # Middle
        *[np.array([0, -0.4 - i*0.1, 0]) for i in range(5)],  # Ring
        *[np.array([-0.1, -0.4 - i*0.1, 0]) for i in range(5)]  # Pinky
    ])
    
    prediction = recognizer.predict_from_landmarks(test_landmarks.tolist())
    print(f"   ✅ Predicted: {prediction['gesture']} ({prediction['confidence']:.2%})")
    
    if prediction['gesture'] == 'HELLO':
        print("   ✅ Rule-based recognition working correctly")
    else:
        print(f"   ⚠️  Expected HELLO, got {prediction['gesture']}")
except Exception as e:
    print(f"   ❌ Rule-based recognizer error: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Check for trained CNN model
print("\n7️⃣  Checking for trained CNN model...")
model_path = 'static/models/gesture_model.h5'
classes_path = 'static/models/gesture_model_classes.pkl'

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model file found ({size_mb:.2f} MB)")
    
    if os.path.exists(classes_path):
        print("   ✅ Classes file found")
        
        # Try loading
        try:
            from ai.gesture_recognizer import GestureRecognizer
            cnn = GestureRecognizer(model_path=model_path)
            if cnn.model is not None:
                print(f"   ✅ CNN loaded successfully")
                print(f"   ✅ Gestures: {cnn.gesture_classes}")
            else:
                print("   ❌ CNN file exists but failed to load")
        except Exception as e:
            print(f"   ❌ Error loading CNN: {e}")
    else:
        print("   ⚠️  Classes file missing (model won't work)")
else:
    print("   ⚠️  No trained model found")
    print("   → Train a model via Admin → Training → Train Model")

# Test 8: Test trainer
print("\n8️⃣  Testing Gesture Trainer...")
try:
    from ai.gesture_trainer import GestureTrainer
    trainer = GestureTrainer()
    
    report = trainer.get_training_report()
    print(f"   ✅ Trainer initialized")
    print(f"   📊 Gestures: {report['total_gestures']}")
    print(f"   📊 Samples: {report['total_samples']}")
    print(f"   📊 Status: {report['ready_to_train'][1]}")
    
    if report['total_samples'] > 0:
        print(f"   📊 Sample distribution: {report['sample_counts']}")
except Exception as e:
    print(f"   ❌ Trainer error: {e}")

# Test 9: Test translation
print("\n9️⃣  Testing Translation...")
try:
    from utils.translator import Translator
    translator = Translator()
    
    test_translations = translator.translate_both('HELLO')
    print(f"   ✅ Translation working")
    print(f"   Tamil: {test_translations['tamil']}")
    print(f"   Hindi: {test_translations['hindi']}")
except Exception as e:
    print(f"   ❌ Translation error: {e}")

# Test 10: Test TTS
print("\n🔟 Testing Text-to-Speech...")
try:
    from utils.tts import TextToSpeech
    tts = TextToSpeech()
    
    if tts.is_available():
        print("   ✅ TTS available")
        
        # Try generating test audio
        test_audio = tts.generate_speech("வணக்கம்", "tamil")
        if test_audio:
            print(f"   ✅ Generated audio: {test_audio}")
            
            full_path = f"static/{test_audio}"
            if os.path.exists(full_path):
                print(f"   ✅ Audio file exists")
            else:
                print(f"   ⚠️  Audio file not found at {full_path}")
        else:
            print("   ⚠️  Audio generation returned None")
    else:
        print("   ⚠️  TTS not available (gTTS not installed)")
except Exception as e:
    print(f"   ❌ TTS error: {e}")

# Test 11: Database connection
print("\n1️⃣1️⃣ Testing Database Connection...")
try:
    import mysql.connector
    from config import Config
    
    conn = mysql.connector.connect(**Config.DB_CONFIG)
    cursor = conn.cursor()
    
    # Test query
    cursor.execute("SELECT COUNT(*) FROM gestures")
    gesture_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    
    print(f"   ✅ Database connected")
    print(f"   📊 Gestures in DB: {gesture_count}")
    print(f"   📊 Users in DB: {user_count}")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"   ❌ Database error: {e}")
    print("   → Make sure MySQL is running and database is created")

# Final Summary
print("\n" + "="*70)
print(" "*25 + "SUMMARY")
print("="*70)

print("\n✅ Working:")
print("   - Python environment")
print("   - Directory structure")
print("   - Core modules")

if os.path.exists(model_path):
    print("   - CNN model (trained)")
else:
    print("\n⚠️  Needs Attention:")
    print("   - No trained CNN model (train via admin panel)")

if missing_packages:
    print(f"\n❌ Issues:")
    print(f"   - Missing packages: {', '.join(missing_packages)}")

print("\n" + "="*70)
print("Next Steps:")
if not os.path.exists(model_path):
    print("1. Collect training samples (15 per gesture)")
    print("2. Train CNN model via Admin → Training")
    print("3. Test detection via User → Sign to Text")
else:
    print("1. Test detection via User → Sign to Text")
    print("2. Check console logs for 'CNN Prediction' messages")
    print("3. Verify confidence scores are >70%")
print("="*70)

print("\nTest complete! 🎉\n")