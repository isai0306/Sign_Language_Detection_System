"""
SignAI - Simplified Database Schema
Core functionalities: Sign→Text, Text→Sign, Model Training
"""

import os
import mysql.connector
from werkzeug.security import generate_password_hash
from datetime import datetime

# Database Configuration (match config.py / set SIGNAI_DB_PASSWORD if MySQL has a password)
DB_HOST = os.environ.get("SIGNAI_DB_HOST", "localhost")
DB_USER = os.environ.get("SIGNAI_DB_USER", "root")
DB_PASSWORD = os.environ.get(
    "SIGNAI_DB_PASSWORD", os.environ.get("MYSQL_PASSWORD", "")
)
DB_NAME = os.environ.get("SIGNAI_DB_NAME", "signai_db")

def create_database():
    """Create database"""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            charset='utf8'
        )
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
            "CHARACTER SET utf8 COLLATE utf8_general_ci"
        )
        print(f"✅ Database '{DB_NAME}' created")
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return False

def create_tables():
    """Create simplified table structure"""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8'
        )
        cursor = conn.cursor()

        # USERS TABLE
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            register_number VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            phone VARCHAR(20),
            status ENUM('ACTIVE','INACTIVE') DEFAULT 'ACTIVE',
            created_at DATETIME NOT NULL,
            last_login DATETIME
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # ADMINS TABLE
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            admin_id VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at DATETIME NOT NULL,
            last_login DATETIME
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # GESTURES LIBRARY
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS gestures (
            id INT AUTO_INCREMENT PRIMARY KEY,
            gesture_name VARCHAR(50) UNIQUE NOT NULL,
            description TEXT,
            image_path VARCHAR(255),
            category ENUM('BASIC','ALPHABET','NUMBER','CUSTOM') DEFAULT 'BASIC',
            created_at DATETIME NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)
        
        # Drop emoji column if it exists (for backward compatibility)
        try:
            cursor.execute("ALTER TABLE gestures DROP COLUMN emoji")
            print("✅ Dropped emoji column from gestures table")
        except mysql.connector.Error:
            pass  # Column might not exist or already dropped

        # GESTURE DETECTION HISTORY
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS gesture_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            gesture_id INT NOT NULL,
            detected_text VARCHAR(100),
            confidence_score DECIMAL(5,2),
            translated_tamil TEXT,
            translated_hindi TEXT,
            detection_timestamp DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (gesture_id) REFERENCES gestures(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # LANGUAGE PREFERENCES
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS language_preferences (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT UNIQUE NOT NULL,
            preferred_language ENUM('TAMIL','HINDI','BOTH') DEFAULT 'TAMIL',
            audio_enabled BOOLEAN DEFAULT TRUE,
            updated_at DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # TRAINING DATASETS
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_datasets (
            id INT AUTO_INCREMENT PRIMARY KEY,
            dataset_name VARCHAR(100) NOT NULL,
            gesture_label VARCHAR(50) NOT NULL,
            file_path VARCHAR(255) NOT NULL,
            file_type ENUM('IMAGE','VIDEO') DEFAULT 'IMAGE',
            uploaded_by INT,
            upload_date DATETIME NOT NULL,
            is_processed BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (uploaded_by) REFERENCES admins(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        # MODEL VERSIONS
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            version VARCHAR(20) NOT NULL,
            model_path VARCHAR(255) NOT NULL,
            accuracy DECIMAL(5,2),
            is_active BOOLEAN DEFAULT FALSE,
            trained_by INT,
            trained_at DATETIME NOT NULL,
            notes TEXT,
            FOREIGN KEY (trained_by) REFERENCES admins(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ All tables created")
        _run_migrations()
        return True

    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return False


def _run_migrations():
    """Apply additive schema updates for newer SignAI features."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8',
        )
        cur = conn.cursor()
        statements = [
            "ALTER TABLE admins ADD COLUMN role VARCHAR(40) DEFAULT 'admin'",
            "ALTER TABLE gestures ADD COLUMN is_emergency TINYINT(1) NOT NULL DEFAULT 0",
            """ALTER TABLE gesture_history ADD COLUMN is_emergency TINYINT(1) NOT NULL DEFAULT 0""",
            "ALTER TABLE gesture_history ADD COLUMN sentence_text VARCHAR(500) NULL",
        ]
        for sql in statements:
            try:
                cur.execute(sql)
                conn.commit()
            except mysql.connector.Error:
                pass
        cur.execute("""
            CREATE TABLE IF NOT EXISTS emergency_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NULL,
                gesture_name VARCHAR(80) NOT NULL,
                confidence_score DECIMAL(6,4) NULL,
                detail VARCHAR(500) NULL,
                created_at DATETIME NOT NULL,
                INDEX idx_emergency_created (created_at),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Schema migrations applied")
    except mysql.connector.Error as e:
        print(f"⚠️ Migration note: {e}")

def insert_default_data():
    """Insert default admin and gestures"""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8'
        )
        cursor = conn.cursor()

        # Default Admin
        cursor.execute("SELECT * FROM admins WHERE admin_id = %s", ("admin",))
        if not cursor.fetchone():
            hashed_password = generate_password_hash("admin123")
            cursor.execute("""
                INSERT INTO admins (admin_id, name, email, password, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, ("admin", "System Admin", "admin@signai.com", hashed_password, datetime.now()))
            print("✅ Default admin created (admin/admin123)")

        # Demo user (only when users table is empty — for first-run login testing)
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            demo_pw = generate_password_hash("signai123")
            cursor.execute(
                """
                INSERT INTO users
                (register_number, name, email, phone, password, created_at, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE')
                """,
                (
                    "SIGNAI-DEMO-1",
                    "Demo User",
                    "demo@signai.local",
                    "",
                    demo_pw,
                    datetime.now(),
                ),
            )
            print("✅ Demo user: demo@signai.local / signai123")

        # Basic Gestures
        cursor.execute("SELECT COUNT(*) FROM gestures")
        if cursor.fetchone()[0] == 0:
            gestures = [
                ('HELLO', 'Greeting sign', 'images/signs/hello.jpg', 'BASIC'),
                ('HELP', 'Request for assistance', 'images/signs/help.jpg', 'BASIC'),
                ('YES', 'Affirmative response', 'images/signs/yes.jpg', 'BASIC'),
                ('THANK YOU', 'Expression of gratitude', 'signs/thankyou.png', 'BASIC'),
                ('NO', 'Negative response', 'signs/no.png', 'BASIC'),
                ('PLEASE', 'Polite request', 'signs/please.png', 'BASIC'),
                ('SORRY', 'Apology', 'signs/sorry.png', 'BASIC'),
                ('WATER', 'Request water', 'signs/water.png', 'BASIC'),
                ('FOOD', 'Request food', 'signs/food.png', 'BASIC'),
            ]
            cursor.executemany("""
                INSERT INTO gestures 
                (gesture_name, description, image_path, category, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, gestures)
            print(f"✅ {len(gestures)} default gestures created")
        
        # Update existing gestures with correct image paths (for existing databases)
        updates = [
            ('images/signs/hello.jpg', 'HELLO'),
            ('images/signs/help.jpg', 'HELP'),
            ('images/signs/yes.jpg', 'YES'),
        ]
        for image_path, gesture_name in updates:
            cursor.execute("""
                UPDATE gestures 
                SET image_path = %s 
                WHERE gesture_name = %s
            """, (image_path, gesture_name))
        print("✅ Updated image paths for existing gestures")

        conn.commit()
        cursor.close()
        conn.close()
        return True

    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main setup"""
    print("=" * 70)
    print(" " * 20 + "SIGNAI SIMPLIFIED SETUP")
    print("=" * 70)
    
    print("\n📌 Step 1: Creating database...")
    if not create_database():
        return
    
    print("\n📌 Step 2: Creating tables...")
    if not create_tables():
        return
    
    print("\n📌 Step 3: Inserting default data...")
    if not insert_default_data():
        return
    
    print("\n" + "=" * 70)
    print("✅ DATABASE SETUP COMPLETED")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   Tables: 7 core tables")
    print("   Admin: admin / admin123")
    print("   Gestures: 8 basic gestures")
    print("\n⚠️  Change admin password after first login!")
    print("=" * 70)

if __name__ == "__main__":
    main()