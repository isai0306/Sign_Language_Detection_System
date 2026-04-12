"""
SignAI Configuration File
Application settings and constants
"""

import os
from datetime import timedelta

class Config:
    """Flask application configuration"""
    
    # ============================
    # FLASK SETTINGS
    # ============================
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'signai-super-secret-key-change-in-production'
    DEBUG = True
    TESTING = False
    
    # ============================
    # DATABASE SETTINGS (MySQL)
    # ============================
    # Override with env if MySQL uses a password: set SIGNAI_DB_PASSWORD
    DB_CONFIG = {
        'host': os.environ.get('SIGNAI_DB_HOST', 'localhost'),
        'user': os.environ.get('SIGNAI_DB_USER', 'root'),
        'password': os.environ.get('SIGNAI_DB_PASSWORD', os.environ.get('MYSQL_PASSWORD', '')),
        'database': os.environ.get('SIGNAI_DB_NAME', 'signai_db'),
        'charset': 'utf8',
        'autocommit': True,
    }
    
    # ============================
    # SESSION SETTINGS
    # ============================
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # ============================
    # FILE UPLOAD SETTINGS
    # ============================
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
    
    # ============================
    # AI MODEL SETTINGS
    # ============================
    MODEL_PATH = 'static/models/gesture_model.h5'
    CONFIDENCE_THRESHOLD = 0.60  # Lower for real-time responsiveness
    SEQUENCE_LENGTH = 25
    VOTE_WINDOW = 7   # Faster response with smaller window
    FRAME_PROCESS_INTERVAL = 1  # Process EVERY frame for better detection
    EMERGENCY_COOLDOWN_SEC = 45.0
    TRANSLATION_CACHE_MAX = 512
    LSTM_EPOCHS_DEFAULT = 80
    
    # ============================
    # LANGUAGE SETTINGS
    # ============================
    SUPPORTED_LANGUAGES = {
        'tamil': {
            'code': 'ta',
            'name': 'Tamil',
            'display': 'தமிழ்'
        },
        'hindi': {
            'code': 'hi',
            'name': 'Hindi',
            'display': 'हिंदी'
        },
        'english': {
            'code': 'en',
            'name': 'English',
            'display': 'English'
        }
    }

    
    # ============================
    # MEDIAPIPE SETTINGS
    # ============================
    MEDIAPIPE_CONFIG = {
        'max_num_hands': 2,
        'min_detection_confidence': 0.5,  # Lower for better detection at angles
        'min_tracking_confidence': 0.4    # Lower for continuous tracking
    }
    
    # ============================
    # PAGINATION
    # ============================
    ITEMS_PER_PAGE = 20
    
    # ============================
    # STATIC PATHS
    # ============================
    STATIC_FOLDER = 'static'
    TEMPLATE_FOLDER = 'templates'
    SIGN_IMAGES_PATH = 'static/images/signs'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True  # Require HTTPS
    
    # Override with environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DB_CONFIG = {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'user': os.environ.get('DB_USER', 'root'),
        'password': os.environ.get('DB_PASSWORD', ''),
        'database': os.environ.get('DB_NAME', 'signai_db'),
        'charset': 'utf8',
        'autocommit': True
    }


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}