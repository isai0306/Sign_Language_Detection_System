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

# ============================
# USER ROUTES
# ============================

@auth_bp.route('/user/login', methods=['GET', 'POST'])
def user_login():
    """User login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return redirect(url_for('auth.user_login'))
        
        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(
                "SELECT * FROM users WHERE email = %s AND status = 'ACTIVE'",
                (email,)
            )
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password'], password):
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = %s WHERE id = %s",
                    (datetime.now(), user['id'])
                )
                conn.commit()
                
                # Clear any admin session and set user session
                session.clear()
                session.permanent = True
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                session['user_email'] = user['email']
                session['register_number'] = user['register_number']
                
                flash(f'Welcome back, {user["name"]}!', 'success')
                print(f"Login successful for user {user['id']}, redirecting to dashboard")
                return redirect(url_for('user.dashboard'))
            else:
                flash('Invalid email or password', 'error')
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            flash(f'Database error: {str(e)}', 'error')
    
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
                "SELECT id FROM users WHERE email = %s OR register_number = %s",
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
        admin_id = request.form.get('admin_id')
        password = request.form.get('password')
        
        if not admin_id or not password:
            flash('Please enter both Admin ID and password', 'error')
            return redirect(url_for('auth.admin_login'))
        
        try:
            conn = get_db()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(
                "SELECT * FROM admins WHERE admin_id = %s",
                (admin_id,)
            )
            admin = cursor.fetchone()
            
            if admin and check_password_hash(admin['password'], password):
                # Update last login
                cursor.execute(
                    "UPDATE admins SET last_login = %s WHERE id = %s",
                    (datetime.now(), admin['id'])
                )
                conn.commit()
                
                # Clear any user session and set admin session
                session.clear()
                session.permanent = True
                session['admin_id'] = admin['id']
                session['admin_name'] = admin['name']
                session['admin_email'] = admin['email']
                session['admin_role'] = admin['role']
                
                flash(f'Welcome, {admin["name"]}!', 'success')
                return redirect(url_for('admin.dashboard'))
            else:
                flash('Invalid Admin ID or password', 'error')
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            flash(f'Database error: {str(e)}', 'error')
    
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