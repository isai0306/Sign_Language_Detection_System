"""Sign → text pipeline: MediaPipe, sequence / vote smoothing, overlay, NLP context."""
from __future__ import annotations

import base64
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import Config
from services import emergency_service, nlp_postprocess
from utils.app_logger import get_logger
from utils.db import get_db
from utils.translator import Translator

_log = get_logger("signai.detect")

_recognizer: Optional[Any] = None
_simple_recognizer: Optional[Any] = None
_use_simple: bool = True
_hand_detector: Optional[Any] = None
_translator: Optional[Translator] = None
_emergency_names_cache: Optional[set] = None
_emergency_cache_ts: float = 0.0


def configure_runtime(
    hand_detector: Any,
    recognizer: Optional[Any],
    simple_recognizer: Any,
    use_simple: bool,
    translator: Translator,
) -> None:
    global _hand_detector, _recognizer, _simple_recognizer, _use_simple, _translator
    _hand_detector = hand_detector
    _recognizer = recognizer
    _simple_recognizer = simple_recognizer
    _use_simple = use_simple
    _translator = translator


def _user_key(user_id: Optional[Any]) -> str:
    return str(user_id or "anon")


def _get_state(uid: Optional[Any]) -> dict:
    if not hasattr(_get_state, "_states"):
        _get_state._states = {}
    key = _user_key(uid)
    if key not in _get_state._states:
        _get_state._states[key] = {
            "seq": deque(maxlen=Config.SEQUENCE_LENGTH),
            "votes": deque(maxlen=Config.VOTE_WINDOW),
            "sentence_tokens": deque(maxlen=12),
            "tick": 0,
            "last_emergency": None,
            "last_payload": None,
            "last_logged_gesture": None,
        }
    return _get_state._states[key]


def _emergency_set() -> set:
    global _emergency_names_cache, _emergency_cache_ts
    if _emergency_names_cache is None or (time.time() - _emergency_cache_ts) > 60:
        _emergency_names_cache = emergency_service.load_emergency_gesture_names()
        _emergency_cache_ts = time.time()
    return _emergency_names_cache


def _decode_frame(frame_b64: str) -> Optional[np.ndarray]:
    try:
        import cv2

        raw = frame_b64.split(",", 1)[1] if "," in frame_b64 else frame_b64
        buf = base64.b64decode(raw)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        _log.warning("frame decode failed: %s", e)
        return None


def _majority_label(votes: deque) -> Tuple[Optional[str], float]:
    if not votes:
        return None, 0.0
    filtered = [v for v in votes if v and v != "UNKNOWN"]
    if len(filtered) < max(2, len(votes) // 3):
        return None, 0.0
    c = Counter(filtered)
    label, count = c.most_common(1)[0]
    return label, count / len(filtered)


def _update_sequence_and_predict(primary_flat: np.ndarray, user_id: Optional[int]) -> Dict[str, Any]:
    st = _get_state(user_id)
    st["seq"].append(primary_flat.astype(np.float32))
    if _use_simple or _recognizer is None or _recognizer.model is None:
        assert _simple_recognizer is not None
        return _simple_recognizer.predict_from_landmarks(primary_flat.tolist())
    kind = getattr(_recognizer, "_model_kind", "cnn")
    if kind == "lstm":
        if len(st["seq"]) < 5:
            return {"gesture": "UNKNOWN", "confidence": 0.0}
        arr = np.stack(list(st["seq"]), axis=0)
        if arr.shape[0] < Config.SEQUENCE_LENGTH:
            pad = Config.SEQUENCE_LENGTH - arr.shape[0]
            arr = np.vstack([np.tile(arr[0], (pad, 1)), arr])
        arr = arr[-Config.SEQUENCE_LENGTH :]
        return _recognizer.predict_lstm(arr, apply_threshold=False)
    return _recognizer.predict_raw(primary_flat.tolist(), apply_threshold=False)


def _per_hand_quick_pred(landmarks_list: List[np.ndarray], shared: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use shared primary prediction on every box (matches dual-hand demo UI)."""
    return [dict(shared) for _ in landmarks_list]


def detect_gesture_payload(
    frame_b64: str,
    user_id: Optional[int],
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    frame = _decode_frame(frame_b64)
    if frame is None:
        return {"success": False, "error": "Invalid frame"}
    return detect_gesture_from_bgr(frame, user_id, min_confidence, return_annotated_bgr=False)


def detect_gesture_from_bgr(
    frame: np.ndarray,
    user_id: Optional[int],
    min_confidence: float = 0.5,
    return_annotated_bgr: bool = False,
) -> Dict[str, Any]:
    """
    Same pipeline as HTTP API but accepts a BGR OpenCV image (desktop / native camera).
    If return_annotated_bgr is True, adds key 'annotated_rgb' (H,W,3) uint8 for UI toolkits.
    """
    assert _hand_detector is not None and _translator is not None
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return {"success": False, "error": "Invalid frame shape"}

    st = _get_state(user_id)
    st["tick"] += 1
    if st["tick"] % Config.FRAME_PROCESS_INTERVAL != 0 and st["last_payload"]:
        p = dict(st["last_payload"])
        p["skipped_frame"] = True
        return p

    _, results = _hand_detector.find_hands(frame, draw=False)
    if not results.multi_hand_landmarks:
        out = {
            "success": True,
            "gesture": None,
            "message": "No hands detected",
            "annotated_frame": None,
            "hands": [],
        }
        st["last_payload"] = out
        return out

    landmarks_list = _hand_detector.extract_landmarks(results)
    if not landmarks_list:
        out = {
            "success": True,
            "gesture": None,
            "message": "No landmarks",
            "annotated_frame": None,
            "hands": [],
        }
        st["last_payload"] = out
        return out

    primary = _hand_detector.normalize_landmarks(landmarks_list[0])
    raw = _update_sequence_and_predict(primary, user_id)

    if raw.get("confidence", 0) >= min_confidence and raw.get("gesture") not in (
        None,
        "UNKNOWN",
    ):
        st["votes"].append(raw["gesture"])
    else:
        st["votes"].append("UNKNOWN")

    voted_gesture, vote_ratio = _majority_label(st["votes"])
    if not voted_gesture:
        final_gesture, final_conf = None, float(raw.get("confidence", 0))
    else:
        final_gesture = voted_gesture
        final_conf = float(raw.get("confidence", 0)) * float(vote_ratio)

    overlay_label = {
        "gesture": raw.get("gesture") or "UNKNOWN",
        "confidence": float(raw.get("confidence", 0)),
    }
    if final_gesture:
        overlay_label = {"gesture": final_gesture, "confidence": float(final_conf)}
    overlay_preds = _per_hand_quick_pred(landmarks_list, overlay_label)
    import cv2

    vis = frame.copy()
    _hand_detector.draw_signai_overlay(vis, results, overlay_preds)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    annotated_b64 = None
    if ok:
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode(
            "ascii"
        )

    sentence = ""
    is_emergency_gesture = False
    emit_emergency_alert = False
    if final_gesture and final_conf >= Config.CONFIDENCE_THRESHOLD:
        if not st["sentence_tokens"] or st["sentence_tokens"][-1] != final_gesture:
            st["sentence_tokens"].append(final_gesture)
        sentence = nlp_postprocess.gesture_tokens_to_sentence(list(st["sentence_tokens"]))
        is_emergency_gesture = emergency_service.is_emergency_gesture(
            final_gesture, _emergency_set()
        )
        if is_emergency_gesture and not emergency_service.cooldown_active(
            st["last_emergency"], Config.EMERGENCY_COOLDOWN_SEC
        ):
            emergency_service.log_emergency_event(
                user_id, final_gesture, final_conf, "live_detection"
            )
            st["last_emergency"] = time.time()
            emit_emergency_alert = True

    translations = {"english": "", "tamil": "", "hindi": ""}
    audio_paths: Dict[str, str] = {}
    display_text = ""
    if final_gesture and final_conf >= Config.CONFIDENCE_THRESHOLD:
        base = sentence or final_gesture.replace("_", " ").title()
        translations = _translator.translate_both(base)
        display_text = base
        audio_paths = _maybe_tts(user_id, translations)

    hands_meta = []
    for i, lm in enumerate(landmarks_list):
        h, w = frame.shape[:2]
        bbox = _hand_detector.calculate_bounding_box(
            results.multi_hand_landmarks[i], w, h
        )
        pr = overlay_preds[i] if i < len(overlay_preds) else {}
        hands_meta.append(
            {
                "bbox": bbox,
                "gesture": pr.get("gesture"),
                "confidence": float(pr.get("confidence", 0)),
            }
        )

    if (
        final_gesture
        and final_conf >= Config.CONFIDENCE_THRESHOLD
        and st["last_logged_gesture"] != final_gesture
    ):
        log_prediction_db(
            user_id,
            final_gesture,
            final_conf,
            translations,
            sentence,
            bool(is_emergency_gesture),
        )
        st["last_logged_gesture"] = final_gesture

    out = {
        "success": True,
        "gesture": final_gesture,
        "confidence": float(final_conf),
        "raw_gesture": raw.get("gesture"),
        "vote_ratio": float(vote_ratio) if voted_gesture else 0.0,
        "sentence": sentence,
        "display_text": display_text or (final_gesture or ""),
        "translations": translations,
        "audio_paths": audio_paths,
        "is_emergency": bool(
            emit_emergency_alert
            and final_gesture
            and final_conf >= Config.CONFIDENCE_THRESHOLD
        ),
        "annotated_frame": annotated_b64,
        "hands": hands_meta,
        "model_kind": "simple"
        if _use_simple
        else getattr(_recognizer, "_model_kind", "cnn"),
    }
    if return_annotated_bgr:
        import cv2

        out["annotated_rgb"] = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    st["last_payload"] = {k: v for k, v in out.items() if k != "annotated_rgb"}
    return out


def _maybe_tts(user_id: Optional[int], translations: Dict[str, str]) -> Dict[str, str]:
    from utils.tts import TextToSpeech

    paths: Dict[str, str] = {}
    if not user_id:
        return paths
    try:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT preferred_language, audio_enabled FROM language_preferences WHERE user_id = %s",
            (user_id,),
        )
        prefs = cur.fetchone()
        cur.close()
        conn.close()
        if not prefs or not prefs.get("audio_enabled"):
            return paths
        tts = TextToSpeech()
        pref = (prefs.get("preferred_language") or "TAMIL").lower()
        if pref == "both":
            return tts.generate_multilingual(
                {"tamil": translations["tamil"], "hindi": translations["hindi"]}
            )
        lang = "tamil" if pref == "tamil" else "hindi"
        p = tts.generate_speech(translations[lang], lang)
        if p:
            paths[lang] = p
    except Exception as e:
        _log.warning("tts skip: %s", e)
    return paths


def log_prediction_db(
    user_id: Optional[int],
    gesture_name: str,
    confidence: float,
    translations: Dict[str, str],
    sentence: str,
    is_emergency: bool,
) -> None:
    if not user_id or not gesture_name:
        return
    try:
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id FROM gestures WHERE gesture_name = %s", (gesture_name,))
        row = cur.fetchone()
        if not row:
            alt = gesture_name.replace("_", " ")
            cur.execute("SELECT id FROM gestures WHERE gesture_name = %s", (alt,))
            row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return
        params_ext = (
            user_id,
            row["id"],
            gesture_name,
            confidence,
            translations.get("tamil"),
            translations.get("hindi"),
            1 if is_emergency else 0,
            (sentence or "")[:500],
        )
        params_basic = (
            user_id,
            row["id"],
            gesture_name,
            confidence,
            translations.get("tamil"),
            translations.get("hindi"),
        )
        try:
            cur.execute(
                """
                INSERT INTO gesture_history
                (user_id, gesture_id, detected_text, confidence_score,
                 translated_tamil, translated_hindi, detection_timestamp, is_emergency, sentence_text)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
                """,
                params_ext,
            )
        except Exception:
            cur.execute(
                """
                INSERT INTO gesture_history
                (user_id, gesture_id, detected_text, confidence_score,
                 translated_tamil, translated_hindi, detection_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                params_basic,
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        _log.warning("gesture log: %s", e)
