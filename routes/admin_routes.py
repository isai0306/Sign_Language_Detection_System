"""
SignAI - Simplified Admin Routes
Focus: Dashboard, Gesture Management, Model Training
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from datetime import datetime
import mysql.connector
from config import Config
from functools import wraps

admin_bp = Blueprint('admin', __name__)

def get_db():
    return mysql.connector.connect(**Config.DB_CONFIG)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Admin access required', 'error')
            return redirect(url_for('auth.admin_login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# DASHBOARD
# ============================================================

@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """Main admin dashboard"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        # Total users
        cursor.execute("SELECT COUNT(*) as total FROM users WHERE status = 'ACTIVE'")
        total_users = cursor.fetchone()['total']
        
        # Total gestures
        cursor.execute("SELECT COUNT(*) as total FROM gestures")
        total_gestures = cursor.fetchone()['total']
        
        # Total detections (last 30 days)
        cursor.execute("""
            SELECT COUNT(*) as total FROM gesture_history
            WHERE detection_timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        """)
        total_detections = cursor.fetchone()['total']
        
        # Top gestures (last 30 days)
        cursor.execute("""
            SELECT g.gesture_name, COUNT(*) as usage_count
            FROM gesture_history gh
            JOIN gestures g ON gh.gesture_id = g.id
            WHERE gh.detection_timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY g.gesture_name
            ORDER BY usage_count DESC
            LIMIT 10
        """)
        gesture_stats = cursor.fetchall()
        
        # Recent users
        cursor.execute("""
            SELECT * FROM users
            ORDER BY created_at DESC
            LIMIT 10
        """)
        recent_users = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return render_template('admin/dashboard.html',
            total_users=total_users,
            total_gestures=total_gestures,
            total_detections=total_detections,
            gesture_stats=gesture_stats,
            recent_users=recent_users
        )
        
    except mysql.connector.Error as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('home'))


# ============================================================
# USER MANAGEMENT
# ============================================================

@admin_bp.route('/users')
@admin_required
def users():
    """User management"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        page = request.args.get('page', 1, type=int)
        per_page = 20
        offset = (page - 1) * per_page
        
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT u.*,
                   (SELECT COUNT(*) FROM gesture_history WHERE user_id = u.id) as gesture_count
            FROM users u
            ORDER BY u.created_at DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        
        users_list = cursor.fetchall()
        cursor.close()
        conn.close()
        
        total_pages = (total + per_page - 1) // per_page
        
        return render_template('admin/users.html',
            users=users_list,
            page=page,
            total_pages=total_pages
        )
        
    except mysql.connector.Error as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route('/users/<int:user_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Toggle user status"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT status FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        if user:
            new_status = 'INACTIVE' if user['status'] == 'ACTIVE' else 'ACTIVE'
            cursor.execute("UPDATE users SET status = %s WHERE id = %s", (new_status, user_id))
            conn.commit()
            flash(f'User status: {new_status}', 'success')
        
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('admin.users'))


# ============================================================
# GESTURE MANAGEMENT
# ============================================================

@admin_bp.route('/gestures')
@admin_required
def gestures():
    """Gesture library"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT g.*,
                   (SELECT COUNT(*) FROM gesture_history WHERE gesture_id = g.id) as usage_count
            FROM gestures g
            ORDER BY g.category, g.gesture_name
        """)
        
        gestures_list = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Categorize gestures
        basic_gestures = [g for g in gestures_list if g['category'] == 'BASIC']
        alphabet_gestures = [g for g in gestures_list if g['category'] == 'ALPHABET']
        number_gestures = [g for g in gestures_list if g['category'] == 'NUMBER']
        custom_gestures = [g for g in gestures_list if g['category'] == 'CUSTOM']
        
        return render_template('admin/gestures.html', 
                             gestures=gestures_list,
                             basic_gestures=basic_gestures,
                             alphabet_gestures=alphabet_gestures,
                             number_gestures=number_gestures,
                             custom_gestures=custom_gestures)
        
    except mysql.connector.Error as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))


@admin_bp.route('/gestures/add', methods=['GET', 'POST'])
@admin_required
def add_gesture():
    """Add new gesture"""
    if request.method == 'POST':
        gesture_name = request.form.get('gesture_name', '').strip().upper()
        description = request.form.get('description', '').strip()
        category = request.form.get('category', 'BASIC')
        image_path = request.form.get('image_path', '').strip()
        
        if not gesture_name or not description:
            flash('Name and description required', 'error')
            return redirect(url_for('admin.add_gesture'))
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM gestures WHERE gesture_name = %s", (gesture_name,))
            if cursor.fetchone():
                flash(f'"{gesture_name}" already exists', 'error')
                return redirect(url_for('admin.add_gesture'))
            
            cursor.execute("""
                INSERT INTO gestures 
                (gesture_name, description, image_path, category, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (gesture_name, description, image_path, category, datetime.now()))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            flash(f'Gesture "{gesture_name}" added!', 'success')
            return redirect(url_for('admin.gestures'))
            
        except mysql.connector.Error as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('admin/add_gesture.html')


@admin_bp.route('/gestures/<int:gesture_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_gesture(gesture_id):
    """Edit gesture"""
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    
    if request.method == 'POST':
        gesture_name = request.form.get('gesture_name', '').strip().upper()
        description = request.form.get('description', '').strip()
        category = request.form.get('category', 'BASIC')
        image_path = request.form.get('image_path', '').strip()
        
        try:
            is_emergency = 1 if request.form.get("is_emergency") == "on" else 0
            cursor.execute("""
                UPDATE gestures
                SET gesture_name = %s, description = %s,
                    image_path = %s, category = %s, is_emergency = %s
                WHERE id = %s
            """, (gesture_name, description, image_path, category, is_emergency, gesture_id))
            
            conn.commit()
            flash(f'Gesture "{gesture_name}" updated!', 'success')
            cursor.close()
            conn.close()
            return redirect(url_for('admin.gestures'))
            
        except mysql.connector.Error as e:
            flash(f'Error: {str(e)}', 'error')
    
    # GET - load data
    cursor.execute("SELECT * FROM gestures WHERE id = %s", (gesture_id,))
    gesture = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not gesture:
        flash('Gesture not found', 'error')
        return redirect(url_for('admin.gestures'))
    
    return render_template('admin/edit_gesture.html', gesture=gesture)


@admin_bp.route('/gestures/<int:gesture_id>/delete', methods=['POST'])
@admin_required
def delete_gesture(gesture_id):
    """Delete gesture"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT gesture_name FROM gestures WHERE id = %s", (gesture_id,))
        gesture = cursor.fetchone()
        
        if gesture:
            cursor.execute("DELETE FROM gestures WHERE id = %s", (gesture_id,))
            conn.commit()
            flash(f'Gesture "{gesture["gesture_name"]}" deleted', 'success')
        
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('admin.gestures'))


# ============================================================
# MODEL TRAINING (Link to training routes)
# ============================================================

@admin_bp.route('/training')
@admin_required
def training_redirect():
    """Redirect to training dashboard"""
    return redirect(url_for('training.training_dashboard'))


# ============================================================
# ANALYTICS
# ============================================================

@admin_bp.route('/analytics')
@admin_required
def analytics():
    """System analytics"""
    usage_chart = []
    emergency_trends = []
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        
        # Daily usage (last 30 days)
        cursor.execute("""
            SELECT DATE(detection_timestamp) as date, COUNT(*) as count
            FROM gesture_history
            WHERE detection_timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(detection_timestamp)
            ORDER BY date
        """)
        daily_usage = cursor.fetchall()
        usage_chart = []
        for row in daily_usage:
            dt = row["date"]
            label = dt.strftime("%m/%d") if hasattr(dt, "strftime") else str(dt)
            usage_chart.append({"day": label, "count": int(row["count"])})
        
        # Category distribution
        cursor.execute("""
            SELECT g.category, COUNT(*) as count
            FROM gesture_history gh
            JOIN gestures g ON gh.gesture_id = g.id
            GROUP BY g.category
        """)
        category_stats = cursor.fetchall()
        
        # Language preferences
        cursor.execute("""
            SELECT preferred_language, COUNT(*) as count
            FROM language_preferences
            GROUP BY preferred_language
        """)
        language_stats = cursor.fetchall()

        try:
            cursor.execute("""
                SELECT DATE(created_at) AS date, COUNT(*) AS count
                FROM emergency_logs
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            for row in cursor.fetchall():
                emergency_trends.append(
                    {"date": row["date"], "count": row["count"]}
                )
        except mysql.connector.Error:
            emergency_trends = []
        
        cursor.close()
        conn.close()
        
        return render_template('admin/analytics.html',
            daily_usage=daily_usage,
            usage_chart=usage_chart,
            category_stats=category_stats,
            language_stats=language_stats,
            emergency_trends=emergency_trends,
        )
        
    except mysql.connector.Error as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('admin.dashboard'))