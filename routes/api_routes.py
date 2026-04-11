"""
SignAI API routes — thin controllers; logic in services/.
"""

import os
import re
import tempfile
import difflib
from datetime import datetime

import mysql.connector
from flask import Blueprint, request, jsonify, session

from config import Config
from utils.translator import Translator
from utils.tts import TextToSpeech
from utils.app_logger import get_logger
from utils.db import get_db
from utils.validation import safe_gesture_name
from services import detection_service, emergency_service

api_bp = Blueprint("api", __name__)
_log = get_logger("signai.api")

hand_detector = None
gesture_recognizer = None
USE_SIMPLE = True
translator = Translator()
tts = TextToSpeech()
_api_bootstrapped = False


def _bootstrap_api():
    global _api_bootstrapped, hand_detector, gesture_recognizer, USE_SIMPLE
    if _api_bootstrapped:
        return
    from ai.hand_detector import HandDetector
    from ai.gesture_recognizer import GestureRecognizer, SimpleGestureRecognizer

    hand_detector = HandDetector(
        max_num_hands=Config.MEDIAPIPE_CONFIG["max_num_hands"],
        min_detection_confidence=Config.MEDIAPIPE_CONFIG["min_detection_confidence"],
        min_tracking_confidence=Config.MEDIAPIPE_CONFIG["min_tracking_confidence"],
    )
    _init_recognizer()
    _api_bootstrapped = True


@api_bp.before_request
def _api_lazy_load():
    _bootstrap_api()


def _init_recognizer():
    global gesture_recognizer, USE_SIMPLE
    from ai.gesture_recognizer import GestureRecognizer, SimpleGestureRecognizer

    simple = SimpleGestureRecognizer()
    try:
        neural = GestureRecognizer(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        )
        if neural.model is None:
            gesture_recognizer = simple
            USE_SIMPLE = True
            _log.warning("Using SimpleGestureRecognizer (no trained model)")
            detection_service.configure_runtime(
                hand_detector, None, simple, True, translator
            )
        else:
            gesture_recognizer = neural
            USE_SIMPLE = False
            _log.info("Loaded neural gesture model (%s)", neural._model_kind)
            detection_service.configure_runtime(
                hand_detector, neural, simple, False, translator
            )
    except Exception as e:
        _log.error("Recognizer init: %s", e)
        gesture_recognizer = simple
        USE_SIMPLE = True
        detection_service.configure_runtime(
            hand_detector, None, simple, True, translator
        )


def reload_gesture_recognizer():
    """Reload model after training (returns True on success)."""
    global gesture_recognizer, USE_SIMPLE
    _bootstrap_api()
    from ai.gesture_recognizer import GestureRecognizer, SimpleGestureRecognizer

    simple = SimpleGestureRecognizer()
    try:
        neural = GestureRecognizer(
            model_path=Config.MODEL_PATH,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        )
        if neural.model is None:
            gesture_recognizer = simple
            USE_SIMPLE = True
            detection_service.configure_runtime(
                hand_detector, None, simple, True, translator
            )
        else:
            gesture_recognizer = neural
            USE_SIMPLE = False
            detection_service.configure_runtime(
                hand_detector, neural, simple, False, translator
            )
        _log.info("Recognizer reloaded simple=%s", USE_SIMPLE)
        return True
    except Exception as e:
        _log.error("reload_gesture_recognizer: %s", e)
        return False


@api_bp.route("/detect-gesture", methods=["POST"])
def detect_gesture():
    try:
        from ai.gesture_recognizer import GestureRecognizer

        data = request.get_json()
        if not data or "frame" not in data:
            return jsonify({"success": False, "error": "No frame"}), 400

        if USE_SIMPLE:
            try:
                temp = GestureRecognizer(
                    model_path=Config.MODEL_PATH,
                    confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                )
                if temp.model is not None:
                    reload_gesture_recognizer()
            except Exception:
                pass

        uid = data.get("user_id") or session.get("user_id")
        out = detection_service.detect_gesture_payload(
            data["frame"], uid, min_confidence=0.45
        )
        if not out.get("success", True) and out.get("error"):
            return jsonify(out), 400
        return jsonify(out)
    except Exception as e:
        _log.exception("detect-gesture: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


def _normalize_word(w: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "", w.upper())


@api_bp.route("/text-to-sign", methods=["POST"])
def text_to_sign():
    try:
        data = request.get_json() or {}
        raw = (data.get("text") or "").strip()
        if not raw:
            return jsonify({"success": False, "error": "No text"}), 400

        sentence = re.split(r"[\s,;.!?]+", raw)
        words = [w for w in sentence if w]
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT gesture_name, image_path, description FROM gestures"
        )
        all_rows = cursor.fetchall()
        known = [r["gesture_name"].upper() for r in all_rows]
        name_to_row = {r["gesture_name"].upper(): r for r in all_rows}

        sign_data = {}
        for w in words:
            key = _normalize_word(w)
            if not key:
                continue
            match = name_to_row.get(key)
            if not match:
                close = difflib.get_close_matches(key, known, n=1, cutoff=0.75)
                if close:
                    match = name_to_row.get(close[0])
            if match:
                sign_data[key] = {
                    "image": match["image_path"],
                    "description": match["description"],
                    "found": True,
                    "matched_as": match["gesture_name"],
                }
            else:
                sign_data[key] = {
                    "image": None,
                    "description": "Sign not in library — showing text fallback",
                    "found": False,
                    "matched_as": None,
                }

        cursor.close()
        conn.close()

        return jsonify(
            {
                "success": True,
                "input_text": raw,
                "words": list(sign_data.keys()),
                "sign_data": sign_data,
                "total_words": len(sign_data),
                "found_signs": sum(1 for s in sign_data.values() if s["found"]),
            }
        )
    except Exception as e:
        _log.exception("text-to-sign: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/gesture-history", methods=["GET"])
def gesture_history():
    try:
        user_id = request.args.get("user_id") or session.get("user_id")
        limit = request.args.get("limit", 20, type=int)
        if not user_id:
            return jsonify({"success": False, "error": "User ID required"}), 400

        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT gh.*, g.gesture_name
            FROM gesture_history gh
            JOIN gestures g ON gh.gesture_id = g.id
            WHERE gh.user_id = %s
            ORDER BY gh.detection_timestamp DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        history = cursor.fetchall()
        for item in history:
            ts = item["detection_timestamp"]
            item["detection_timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        cursor.close()
        conn.close()
        return jsonify({"success": True, "history": history, "count": len(history)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/available-gestures", methods=["GET"])
def available_gestures():
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT gesture_name, description, image_path, category
            FROM gestures
            ORDER BY category, gesture_name
            """
        )
        gestures = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify({"success": True, "gestures": gestures, "total": len(gestures)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/system-info", methods=["GET"])
def system_info():
    kind = "simple"
    if not USE_SIMPLE and gesture_recognizer and getattr(
        gesture_recognizer, "model", None
    ):
        kind = getattr(gesture_recognizer, "_model_kind", "cnn")
    return jsonify(
        {
            "success": True,
            "model": {"type": kind, "status": "active"},
            "services": {
                "hand_detection": True,
                "translation": translator.is_translation_available(),
                "tts": tts.is_available(),
            },
            "config": {
                "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                "sequence_length": Config.SEQUENCE_LENGTH,
                "supported_languages": list(Config.SUPPORTED_LANGUAGES.keys()),
            },
        }
    )


@api_bp.route("/emergency-alert", methods=["POST"])
def emergency_alert():
    try:
        data = request.get_json() or {}
        uid = data.get("user_id") or session.get("user_id")
        gname = safe_gesture_name(data.get("gesture_name", ""))
        if not gname:
            return jsonify({"success": False, "error": "Invalid gesture"}), 400
        emergency_service.log_emergency_event(
            uid, gname, float(data.get("confidence", 0)), data.get("location_info")
        )
        return jsonify({"success": True, "message": "Logged"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    try:
        import speech_recognition as sr
    except ImportError:
        return jsonify(
            {
                "success": False,
                "error": "Install SpeechRecognition and PyAudio for speech input",
            }
        ), 501

    if "audio" not in request.files:
        return jsonify({"success": False, "error": "Missing audio file"}), 400

    f = request.files["audio"]
    if not f.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("wav", "webm", "flac", "ogg"):
        return jsonify({"success": False, "error": "Unsupported audio type"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + ext)
    try:
        f.save(tmp.name)
        r = sr.Recognizer()
        with sr.AudioFile(tmp.name) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return jsonify({"success": True, "text": text})
    except sr.UnknownValueError:
        return jsonify({"success": False, "error": "Could not understand audio"}), 400
    except Exception as e:
        _log.warning("stt: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@api_bp.route("/test-audio", methods=["GET"])
def test_audio():
    path = tts.generate_speech("SignAI audio check", "english")
    if path:
        return jsonify({"success": True, "audio_path": "/static/" + path})
    return jsonify({"success": False, "error": "TTS unavailable"}), 500
