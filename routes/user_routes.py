"""
SignAI - Simplified User Routes
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory, abort
from datetime import datetime
import mysql.connector
from config import Config
from functools import wraps
import os
import re

user_bp = Blueprint('user', __name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_FOLDER = os.path.join(PROJECT_ROOT, "INDIAN SIGN LANGUAGE ANIMATED VIDEOS")
_DATASET_INDEX = None

def get_db():
    return mysql.connector.connect(**Config.DB_CONFIG)


def _normalize_sign_key(text):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dataset_index():
    global _DATASET_INDEX
    if _DATASET_INDEX is not None:
        return _DATASET_INDEX

    index = {}
    if not os.path.isdir(DATASET_FOLDER):
        _DATASET_INDEX = index
        return index

    for name in os.listdir(DATASET_FOLDER):
        if not name.lower().endswith(".mp4"):
            continue
        stem = os.path.splitext(name)[0]
        key = _normalize_sign_key(stem.replace("_", " ").replace("-", " "))
        if key and key not in index:
            index[key] = name

    _DATASET_INDEX = index
    return index


def _lookup_gesture_image(cursor, token):
    word = token.upper()
    cursor.execute(
        """
        SELECT gesture_name, image_path, description
        FROM gestures WHERE UPPER(gesture_name) = %s
        """,
        (word,),
    )
    result = cursor.fetchone()
    if result:
        return result

    cursor.execute(
        """
        SELECT gesture_name, image_path, description
        FROM gestures WHERE UPPER(REPLACE(gesture_name,' ','_')) = %s
        """,
        (word.replace(" ", "_"),),
    )
    return cursor.fetchone()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login', 'error')
            return redirect(url_for('auth.user_login'))
        return f(*args, **kwargs)
    return decorated_function

# Dashboard
@user_bp.route('/dashboard')
@login_required
def dashboard():
    print(f"Dashboard accessed by user {session.get('user_id')}")
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        user_id = session['user_id']
        
        # Statistics
        cursor.execute("""
            SELECT COUNT(*) as total_gestures
            FROM gesture_history WHERE user_id = %s
        """, (user_id,))
        gesture_stats = cursor.fetchone()
        
        # Recent activity
        cursor.execute("""
            SELECT gh.*, g.gesture_name
            FROM gesture_history gh
            JOIN gestures g ON gh.gesture_id = g.id
            WHERE gh.user_id = %s
            ORDER BY gh.detection_timestamp DESC
            LIMIT 10
        """, (user_id,))
        recent_activity = cursor.fetchall()
        
        # Language preference
        cursor.execute("""
            SELECT * FROM language_preferences WHERE user_id = %s
        """, (user_id,))
        language_pref = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        # Dummy text stats since no table exists
        text_stats = {'text_to_sign_count': 0}
        
        return render_template('user/dashboard.html',
            gesture_stats=gesture_stats,
            recent_activity=recent_activity,
            language_pref=language_pref,
            text_stats=text_stats
        )
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('home'))

# Sign to Text
@user_bp.route('/sign-to-text')
@login_required
def sign_to_text():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT preferred_language, audio_enabled
            FROM language_preferences WHERE user_id = %s
        """, (session['user_id'],))
        
        language_pref = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return render_template('user/sign_to_text.html',
            language_pref=language_pref
        )
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('user.dashboard'))

# Text to Sign
@user_bp.route('/sign-video/<path:filename>')
@login_required
def sign_video(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DATASET_FOLDER, safe_name)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(DATASET_FOLDER, safe_name)


@user_bp.route('/text-to-sign', methods=['GET', 'POST'])
@login_required
def text_to_sign():
    if request.method == 'POST':
        input_text = request.form.get('input_text', '').strip()
        
        if not input_text:
            flash('Please enter text', 'error')
            return redirect(url_for('user.text_to_sign'))
        
        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)

            dataset = _dataset_index()
            tokens = re.findall(r"[A-Za-z0-9']+", input_text)
            sign_units = []
            i = 0

            while i < len(tokens):
                matched = False

                # Prefer full phrase videos first (e.g., "thank you", "do not")
                for size in range(min(4, len(tokens) - i), 0, -1):
                    chunk = " ".join(tokens[i:i + size])
                    key = _normalize_sign_key(chunk)
                    if key in dataset:
                        sign_units.append({
                            "label": chunk.upper(),
                            "video_filename": dataset[key],
                            "video_url": url_for("user.sign_video", filename=dataset[key]),
                            "image_path": None,
                        })
                        i += size
                        matched = True
                        break

                if matched:
                    continue

                token = tokens[i]
                gesture = _lookup_gesture_image(cursor, token)

                if gesture and gesture.get("image_path"):
                    sign_units.append({
                        "label": token.upper(),
                        "video_filename": None,
                        "video_url": None,
                        "image_path": gesture["image_path"],
                    })
                    i += 1
                    continue

                # Word-wise only: do not split unknown words into characters.
                sign_units.append({
                    "label": token.upper(),
                    "video_filename": None,
                    "video_url": None,
                    "image_path": None,
                })

                i += 1

            cursor.close()
            conn.close()
            
            return render_template('user/text_to_sign.html',
                input_text=input_text,
                sign_units=sign_units,
                show_result=True
            )
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('user/text_to_sign.html', show_result=False, sign_units=[])

# History
@user_bp.route('/history')
@login_required
def history():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        page = request.args.get('page', 1, type=int)
        per_page = 20
        offset = (page - 1) * per_page
        
        cursor.execute("""
            SELECT COUNT(*) as total FROM gesture_history
            WHERE user_id = %s
        """, (session['user_id'],))
        total = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT gh.*, g.gesture_name
            FROM gesture_history gh
            JOIN gestures g ON gh.gesture_id = g.id
            WHERE gh.user_id = %s
            ORDER BY gh.detection_timestamp DESC
            LIMIT %s OFFSET %s
        """, (session['user_id'], per_page, offset))
        
        history_items = cursor.fetchall()
        cursor.close()
        conn.close()
        
        total_pages = (total + per_page - 1) // per_page
        
        return render_template('user/history.html',
            history_items=history_items,
            page=page,
            total_pages=total_pages
        )
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('user.dashboard'))

# Profile
@user_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        preferred_language = request.form.get('preferred_language')
        audio_enabled = request.form.get('audio_enabled') == 'on'
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE users SET name = %s, phone = %s WHERE id = %s
            """, (name, phone, session['user_id']))
            
            cursor.execute("""
                UPDATE language_preferences
                SET preferred_language = %s, audio_enabled = %s, updated_at = %s
                WHERE user_id = %s
            """, (preferred_language, audio_enabled, datetime.now(), session['user_id']))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            session['user_name'] = name
            flash('Profile updated!', 'success')
            return redirect(url_for('user.profile'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    # GET
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT u.*, lp.preferred_language, lp.audio_enabled
            FROM users u
            LEFT JOIN language_preferences lp ON u.id = lp.user_id
            WHERE u.id = %s
        """, (session['user_id'],))
        
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return render_template('user/profile.html', user=user_data)
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('user.dashboard'))