"""
SignAI Training Routes - Updated for Enhanced Trainer
Integrates with improved CNN architecture and training process
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime
import mysql.connector
import numpy as np
import base64

from config import Config
from functools import wraps

training_bp = Blueprint('training', __name__)

hand_detector = None
trainer = None
_training_ready = False


def _bootstrap_training():
    global hand_detector, trainer, _training_ready
    if _training_ready:
        return
    from ai.hand_detector import HandDetector
    from ai.gesture_trainer import GestureTrainer

    hand_detector = HandDetector()
    trainer = GestureTrainer()
    _training_ready = True


@training_bp.before_request
def _training_lazy_load():
    _bootstrap_training()

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin access required', 'error')
            return redirect(url_for('auth.admin_login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================
# TRAINING DASHBOARD
# ============================

@training_bp.route('/dashboard')
@admin_required
def training_dashboard():
    """Training dashboard with enhanced reporting"""
    report = trainer.get_training_report()
    
    # Add recommendation based on sample counts
    min_recommended = 15
    gestures_needing_more = []
    
    for gesture, count in report['sample_counts'].items():
        if count < min_recommended:
            gestures_needing_more.append({
                'name': gesture,
                'current': count,
                'needed': min_recommended - count
            })
    
    report['recommendations'] = gestures_needing_more
    
    return render_template('admin/training_dashboard.html', report=report)


# ============================
# COLLECT TRAINING DATA
# ============================

@training_bp.route('/collect/<gesture_name>')
@admin_required
def collect_data(gesture_name):
    """Data collection page for specific gesture"""
    sample_count = len(trainer.samples.get(gesture_name, []))
    
    # Get gesture info from database
    try:
        conn = mysql.connector.connect(**Config.DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM gestures WHERE gesture_name = %s", (gesture_name,))
        gesture_info = cursor.fetchone()
        cursor.close()
        conn.close()
    except:
        gesture_info = None
    
    return render_template('admin/collect_data.html', 
                         gesture_name=gesture_name,
                         sample_count=sample_count,
                         gesture_info=gesture_info,
                         recommended_samples=15)


@training_bp.route('/api/add-sample', methods=['POST'])
@admin_required
def add_sample():
    """
    Add training sample with enhanced validation
    
    Expected JSON:
    {
        "gesture_name": "HELLO",
        "frame": "base64_encoded_image"
    }
    """
    try:
        import cv2

        data = request.get_json()
        gesture_name = data.get('gesture_name')
        frame_data = data.get('frame')
        
        if not gesture_name or not frame_data:
            return jsonify({'success': False, 'error': 'Missing data'}), 400
        
        # Decode frame
        frame_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect hands
        _, results = hand_detector.find_hands(frame, draw=False)
        
        if not results.multi_hand_landmarks:
            return jsonify({
                'success': False,
                'error': 'No hands detected. Please ensure your hand is clearly visible.'
            })
        
        # Extract landmarks
        landmarks_list = hand_detector.extract_landmarks(results)
        if not landmarks_list:
            return jsonify({
                'success': False,
                'error': 'Could not extract landmarks. Try repositioning your hand.'
            })
        
        # Normalize
        landmarks = landmarks_list[0]
        normalized = hand_detector.normalize_landmarks(landmarks)
        
        # Validate landmarks quality
        if np.sum(np.abs(normalized)) < 0.01:
            return jsonify({
                'success': False,
                'error': 'Invalid landmarks detected. Please try again.'
            })
        
        # Add sample
        success = trainer.add_sample(gesture_name, normalized.tolist())
        
        if success:
            sample_count = len(trainer.samples[gesture_name])
            recommended = 15
            progress = min(100, int((sample_count / recommended) * 100))
            
            return jsonify({
                'success': True,
                'sample_count': sample_count,
                'progress': progress,
                'recommended': recommended,
                'message': f'Sample {sample_count} added successfully! ✅'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add sample'
            })
        
    except Exception as e:
        print(f"Error adding sample: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================
# TRAIN MODEL
# ============================

@training_bp.route('/train', methods=['GET', 'POST'])
@admin_required
def train_model():
    """Train gesture recognition model with enhanced CNN"""
    if request.method == 'POST':
        model_type = request.form.get('model_type', 'neural_network')
        if model_type == 'random_forest':
            model_type = 'neural_network'
        use_augmentation = request.form.get('use_augmentation', 'true').lower() == 'true'
        epochs = int(request.form.get('epochs', '150'))
        seq_len = int(request.form.get('sequence_length', Config.SEQUENCE_LENGTH))
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"{'='*60}")
        print(f"Model Type: {model_type}")
        print(f"Data Augmentation: {use_augmentation}")
        print(f"Max Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        try:
            # Train model with enhanced trainer
            accuracy = trainer.train_model(
                model_type=model_type,
                model_path='static/models/gesture_model.h5',
                use_augmentation=use_augmentation,
                epochs=epochs,
                sequence_length=seq_len,
            )
            
            if accuracy:
                accuracy_percent = accuracy * 100
                
                if accuracy_percent >= 85:
                    message = f'🎉 Excellent! Model trained with {accuracy_percent:.2f}% accuracy!'
                    category = 'success'
                elif accuracy_percent >= 70:
                    message = f'✅ Good! Model trained with {accuracy_percent:.2f}% accuracy.'
                    category = 'success'
                else:
                    message = f'⚠️  Model trained with {accuracy_percent:.2f}% accuracy. Consider collecting more samples.'
                    category = 'warning'
                
                flash(message, category)
                
                # Log training event
                _log_training_event(model_type, accuracy, use_augmentation, epochs)
                
                # Reload the model in API routes
                try:
                    from routes.api_routes import reload_gesture_recognizer
                    if reload_gesture_recognizer():
                        flash('✅ Model loaded and ready to use!', 'success')
                    else:
                        flash('⚠️  Model saved but reload failed. Restart server to use new model.', 'warning')
                except Exception as e:
                    print(f"Reload error: {e}")
                    flash('⚠️  Please restart the server to use the new model.', 'warning')
            else:
                flash('❌ Training failed. Check console for errors.', 'error')
        
        except Exception as e:
            print(f"\n❌ Training Error: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Training error: {str(e)}', 'error')
        
        return redirect(url_for('training.training_dashboard'))
    
    # GET - show training form
    report = trainer.get_training_report()
    
    # Calculate estimated training time
    total_samples = report['total_samples']
    if total_samples > 0:
        # Rough estimate: 100 samples ≈ 2 minutes
        estimated_minutes = max(1, int(total_samples / 50))
        report['estimated_time'] = f"{estimated_minutes} minutes"
    else:
        report['estimated_time'] = "N/A"
    
    return render_template('admin/train_model.html', report=report)


# ============================
# MANAGE SAMPLES
# ============================

@training_bp.route('/samples')
@admin_required
def view_samples():
    """View all collected samples with quality metrics"""
    report = trainer.get_training_report()
    
    # Add quality assessment
    for gesture, count in report['sample_counts'].items():
        if count >= 25:
            quality = "Excellent"
        elif count >= 15:
            quality = "Good"
        elif count >= 10:
            quality = "Fair"
        else:
            quality = "Insufficient"
        
        if gesture not in report:
            report[gesture] = {}
        report['sample_counts'][gesture] = {
            'count': count,
            'quality': quality
        }
    
    return render_template('admin/view_samples.html', report=report)


@training_bp.route('/samples/clear/<gesture_name>', methods=['POST'])
@admin_required
def clear_samples(gesture_name):
    """Clear samples for a gesture"""
    trainer.clear_samples(gesture_name)
    flash(f'🗑️  Cleared samples for {gesture_name}', 'success')
    return redirect(url_for('training.view_samples'))


@training_bp.route('/samples/clear-all', methods=['POST'])
@admin_required
def clear_all_samples():
    """Clear all samples"""
    if request.form.get('confirm') == 'yes':
        trainer.clear_samples()
        flash('🗑️  Cleared all training samples', 'success')
    return redirect(url_for('training.view_samples'))


@training_bp.route('/samples/save', methods=['POST'])
@admin_required
def save_samples():
    """Save current samples to disk"""
    filename = trainer.save_samples()
    if filename:
        flash(f'💾 Samples saved to {filename}', 'success')
    else:
        flash('❌ Failed to save samples', 'error')
    return redirect(url_for('training.training_dashboard'))


@training_bp.route('/samples/export', methods=['GET'])
@admin_required
def export_samples():
    """Export samples as JSON for backup"""
    import json
    from flask import Response
    
    try:
        # Convert samples to JSON-serializable format
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'gesture_count': len(trainer.samples),
            'total_samples': sum(len(s) for s in trainer.samples.values()),
            'samples': {
                gesture: [s.tolist() if isinstance(s, np.ndarray) else s 
                         for s in samples]
                for gesture, samples in trainer.samples.items()
            }
        }
        
        json_data = json.dumps(export_data, indent=2)
        
        return Response(
            json_data,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=gesture_samples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            }
        )
    except Exception as e:
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('training.view_samples'))


# ============================
# HELPER FUNCTIONS
# ============================

def _log_training_event(model_type, accuracy, use_augmentation, epochs):
    """Log model training to database"""
    try:
        conn = mysql.connector.connect(**Config.DB_CONFIG)
        cursor = conn.cursor()
        
        # Log to model_versions table
        cursor.execute("""
            INSERT INTO model_versions
            (model_name, version, model_path, accuracy, is_active, 
             trained_by, trained_at, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            model_type,
            datetime.now().strftime('%Y%m%d_%H%M%S'),
            'static/models/gesture_model.h5',
            accuracy * 100,
            True,
            session.get('admin_id'),
            datetime.now(),
            f"Augmentation: {use_augmentation}, Epochs: {epochs}"
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Training event logged to database")
    except Exception as e:
        print(f"Warning: Could not log training event: {e}")


# ============================
# API ENDPOINTS FOR TRAINING
# ============================

@training_bp.route('/api/training-status', methods=['GET'])
@admin_required
def training_status():
    """Get current training status"""
    report = trainer.get_training_report()
    
    return jsonify({
        'success': True,
        'total_gestures': report['total_gestures'],
        'total_samples': report['total_samples'],
        'sample_counts': report['sample_counts'],
        'ready_to_train': report['ready_to_train'][0],
        'message': report['ready_to_train'][1]
    })


@training_bp.route('/api/sample-quality/<gesture_name>', methods=['GET'])
@admin_required
def sample_quality(gesture_name):
    """Check quality of samples for a gesture"""
    if gesture_name not in trainer.samples:
        return jsonify({'success': False, 'error': 'Gesture not found'}), 404
    
    samples = trainer.samples[gesture_name]
    count = len(samples)
    
    # Assess quality
    if count >= 25:
        quality = "excellent"
        message = "More than enough samples! ✅"
    elif count >= 15:
        quality = "good"
        message = "Good amount of samples ✅"
    elif count >= 10:
        quality = "fair"
        message = f"Minimum met. Add {15-count} more for better results"
    else:
        quality = "insufficient"
        message = f"Need {10-count} more samples to train"
    
    return jsonify({
        'success': True,
        'gesture': gesture_name,
        'count': count,
        'quality': quality,
        'message': message,
        'recommended': 15
    })