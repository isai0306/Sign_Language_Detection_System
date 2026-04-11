"""
SignAI Authentication Routes
User and Admin Login/Register/Logout
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import mysql.connector
from config import Config

# Create Blueprint
auth_bp = Blueprint('auth', __name__)

# Database helper
def get_db():
    """Get database connection"""
    return mysql.connector.connect(**Config.DB_CONFIG)


def _normalize_hash(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value).strip()


def _password_matches(stored_hash, plain_password: str) -> bool:
    """Compare plain password to Werkzeug hash (handles bytes from MySQL)."""
    stored_hash = _normalize_hash(stored_hash)
    if not stored_hash or plain_password is None:
        return False
    try:
        return check_password_hash(stored_hash, plain_password)
    except (ValueError, TypeError, AttributeError):
        return False

# ============================
# USER ROUTES
# ============================

@auth_bp.route('/user/login', methods=['GET', 'POST'])
def user_login():
    """User login page"""
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('user_login.html')

        conn = None
        cursor = None
        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT * FROM users
                WHERE LOWER(TRIM(email)) = %s AND status = 'ACTIVE'
                """,
                (email,),
            )
            user = cursor.fetchone()

            if user and _password_matches(user.get('password'), password):
                cursor.execute(
                    "UPDATE users SET last_login = %s WHERE id = %s",
                    (datetime.now(), user['id']),
                )
                conn.commit()

                session.clear()
                session.permanent = True
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                session['user_email'] = user['email']
                session['register_number'] = user.get('register_number') or ''

                flash(f'Welcome back, {user["name"]}!', 'success')
                return redirect(url_for('user.dashboard'))

            flash(
                'Invalid email or password. If you never registered, use Register first. '
                'Fresh database: run python create_db.py then try demo@signai.local / signai123',
                'error',
            )
        except mysql.connector.Error as e:
            errno = getattr(e, 'errno', None)
            if errno == 1045:
                flash(
                    'Database refused login (wrong MySQL user/password). '
                    'Set SIGNAI_DB_PASSWORD in the environment or fix config.py.',
                    'error',
                )
            elif errno in (2003, 2002):
                flash(
                    'Cannot reach MySQL. Start MySQL/WAMP and ensure host is correct in config.py.',
                    'error',
                )
            elif errno == 1049:
                flash(
                    f"Database '{Config.DB_CONFIG.get('database', 'signai_db')}' does not exist. Run: python create_db.py",
                    'error',
                )
            else:
                flash(f'Database error ({errno}): {str(e)}', 'error')
        except Exception as e:
            flash(f'Login failed: {str(e)}', 'error')
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    return render_template('user_login.html')


@auth_bp.route('/user/register', methods=['GET', 'POST'])
def user_register():
    """User registration page"""
    if request.method == 'POST':
        register_number = request.form.get('register_number')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([register_number, name, email, password, confirm_password]):
            flash('All fields are required', 'error')
            return redirect(url_for('auth.user_register'))

        email = email.strip().lower()
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('auth.user_register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('auth.user_register'))
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute(
                "SELECT id FROM users WHERE LOWER(TRIM(email)) = %s OR register_number = %s",
                (email, register_number)
            )
            
            if cursor.fetchone():
                flash('Email or Register Number already exists', 'error')
                return redirect(url_for('auth.user_register'))
            
            # Hash password and insert user
            hashed_password = generate_password_hash(password)
            
            cursor.execute("""
                INSERT INTO users 
                (register_number, name, email, phone, password, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (register_number, name, email, phone, hashed_password, datetime.now()))
            
            user_id = cursor.lastrowid
            
            # Create default language preference
            cursor.execute("""
                INSERT INTO language_preferences 
                (user_id, preferred_language, audio_enabled, updated_at)
                VALUES (%s, 'TAMIL', TRUE, %s)
            """, (user_id, datetime.now()))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.user_login'))
            
        except mysql.connector.Error as e:
            flash(f'Registration failed: {str(e)}', 'error')

    return render_template('register.html',
                           is_logged_in=False)


# ============================
# ADMIN ROUTES
# ============================

@auth_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        admin_id = (request.form.get('admin_id') or '').strip()
        password = request.form.get('password') or ''

        if not admin_id or not password:
            flash('Please enter both Admin ID and password', 'error')
            return render_template('admin_login.html')

        conn = None
        cursor = None
        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                "SELECT * FROM admins WHERE admin_id = %s",
                (admin_id,),
            )
            admin = cursor.fetchone()

            if admin and _password_matches(admin.get('password'), password):
                cursor.execute(
                    "UPDATE admins SET last_login = %s WHERE id = %s",
                    (datetime.now(), admin['id']),
                )
                conn.commit()

                session.clear()
                session.permanent = True
                session['admin_id'] = admin['id']
                session['admin_name'] = admin['name']
                session['admin_email'] = admin['email']
                session['admin_role'] = admin.get('role', 'admin')

                flash(f'Welcome, {admin["name"]}!', 'success')
                return redirect(url_for('admin.dashboard'))

            flash(
                'Invalid Admin ID or password. Default after create_db: admin / admin123',
                'error',
            )
        except mysql.connector.Error as e:
            errno = getattr(e, 'errno', None)
            if errno in (2003, 2002, 1045, 1049):
                flash(
                    'Cannot connect to MySQL or wrong credentials/database. '
                    'Start MySQL, run create_db.py, set SIGNAI_DB_PASSWORD if needed.',
                    'error',
                )
            else:
                flash(f'Database error: {str(e)}', 'error')
        except Exception as e:
            flash(f'Login failed: {str(e)}', 'error')
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    return render_template('admin_login.html')


# ============================
# LOGOUT ROUTES
# ============================

@auth_bp.route('/logout')
def logout():
    """Logout for both user and admin"""
    user_type = 'admin' if 'admin_id' in session else 'user'
    session.clear()
    flash('You have been logged out successfully', 'success')
    
    if user_type == 'admin':
        return redirect(url_for('auth.admin_login'))
    else:
        return redirect(url_for('auth.user_login'))


# ============================
# PASSWORD CHANGE
# ============================

@auth_bp.route('/change-password', methods=['GET', 'POST'])
def change_password():
    """Change password for logged-in users"""
    if 'user_id' not in session and 'admin_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([current_password, new_password, confirm_password]):
            flash('All fields are required', 'error')
            return redirect(url_for('auth.change_password'))
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return redirect(url_for('auth.change_password'))
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('auth.change_password'))
        
        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)
            
            if 'admin_id' in session:
                cursor.execute(
                    "SELECT password FROM admins WHERE id = %s",
                    (session['admin_id'],)
                )
                user = cursor.fetchone()
                table = 'admins'
                user_id = session['admin_id']
            else:
                cursor.execute(
                    "SELECT password FROM users WHERE id = %s",
                    (session['user_id'],)
                )
                user = cursor.fetchone()
                table = 'users'
                user_id = session['user_id']
            
            if user and check_password_hash(user['password'], current_password):
                hashed_password = generate_password_hash(new_password)
                cursor.execute(
                    f"UPDATE {table} SET password = %s WHERE id = %s",
                    (hashed_password, user_id)
                )
                conn.commit()
                flash('Password changed successfully!', 'success')
                return redirect(url_for('auth.logout'))
            else:
                flash('Current password is incorrect', 'error')
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('change_password.html')