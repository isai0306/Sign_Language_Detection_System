from flask import Flask, render_template, session
from datetime import timedelta
import os

from config import Config
from routes.auth import auth_bp
from routes.user_routes import user_bp
from routes.admin_routes import admin_bp
from routes.api_routes import api_bp
from routes.training_routes import training_bp

app = Flask(__name__)
app.config.from_object(Config)
app.permanent_session_lifetime = timedelta(hours=2)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(user_bp, url_prefix='/user')
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(training_bp, url_prefix='/admin/training')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/modern-sign')
def modern_sign():
    # This page will host the React-based Advanced Visualization component
    return render_template('modern_sign.html')

@app.context_processor
def inject_user():
    return dict(
        is_logged_in='user_id' in session or 'admin_id' in session,
        user_name=session.get('user_name') or session.get('admin_name'),
        is_admin='admin_id' in session
    )

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    os.makedirs('static/images/signs', exist_ok=True)
    os.makedirs('static/audio_cache', exist_ok=True)
    
    print("SignAI Server Starting...")
    print("URL: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')