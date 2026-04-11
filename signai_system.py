"""
SignAI — optional native desktop window (Tkinter). The main product is the web app: python app.py

Run:  python signai_system.py
"""
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

from config import Config
from utils.app_logger import init_app_logging


def _ensure_dirs():
    for d in (
        "uploads",
        "static/models",
        "static/images/signs",
        "static/audio_cache",
        "logs",
    ):
        os.makedirs(d, exist_ok=True)


def _bootstrap_ai():
    from routes.api_routes import _bootstrap_api

    _bootstrap_api()


def _try_user_login(email: str, password: str):
    from werkzeug.security import check_password_hash

    from utils.db import get_db

    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, name, password FROM users WHERE email = %s AND status = 'ACTIVE'",
        (email.strip(),),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row and check_password_hash(row["password"], password):
        return int(row["id"]), row["name"]
    return None, None


class SignAISystemApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SignAI — Sign Language System")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self._user_id: Optional[int] = None
        self._user_name = "Guest"
        self._cam_running = False
        self._frame_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_q: queue.Queue = queue.Queue(maxsize=2)
        self._worker: threading.Thread | None = None
        self._photo = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.tab_live = ttk.Frame(nb)
        self.tab_tts = ttk.Frame(nb)
        self.tab_account = ttk.Frame(nb)
        nb.add(self.tab_live, text="Sign to text")
        nb.add(self.tab_tts, text="Text to sign")
        nb.add(self.tab_account, text="Account")

        self._build_live_tab()
        self._build_text_to_sign_tab()
        self._build_account_tab()

    def _build_live_tab(self):
        top = ttk.Frame(self.tab_live)
        top.pack(fill=tk.X, pady=(0, 6))
        self.lbl_user = ttk.Label(top, text="Signed in as: Guest (local)")
        self.lbl_user.pack(side=tk.LEFT)
        self.btn_start = ttk.Button(top, text="Start camera", command=self._start_camera)
        self.btn_start.pack(side=tk.RIGHT, padx=4)
        self.btn_stop = ttk.Button(top, text="Stop", command=self._stop_camera, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT)

        self.emergency_banner = ttk.Label(
            self.tab_live,
            text="",
            foreground="#b91c1c",
            font=("Segoe UI", 11, "bold"),
        )
        self.emergency_banner.pack(fill=tk.X, pady=4)

        mid = ttk.Frame(self.tab_live)
        mid.pack(fill=tk.BOTH, expand=True)
        self.video = ttk.Label(mid, relief=tk.SUNKEN, anchor=tk.CENTER)
        self.video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        side = ttk.Frame(mid, width=280)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(side, text="Gesture", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.var_gesture = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.var_gesture, font=("Segoe UI", 16, "bold")).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(side, text="Confidence", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.var_conf = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.var_conf, font=("Segoe UI", 12)).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(side, text="Sentence", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.var_sentence = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.var_sentence, wraplength=260, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(side, text="Tamil", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.var_ta = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.var_ta, wraplength=260, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(side, text="Hindi", font=("Segoe UI", 9)).pack(anchor=tk.W)
        self.var_hi = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.var_hi, wraplength=260, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))
        self.btn_play_audio = ttk.Button(side, text="Play last TTS", command=self._play_last_audio, state=tk.DISABLED)
        self.btn_play_audio.pack(anchor=tk.W, pady=4)
        self._last_audio_path: Optional[str] = None

    def _build_text_to_sign_tab(self):
        row = ttk.Frame(self.tab_tts)
        row.pack(fill=tk.X, pady=6)
        ttk.Label(row, text="Text:").pack(side=tk.LEFT)
        self.ent_tts = ttk.Entry(row, width=50)
        self.ent_tts.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(row, text="Show signs", command=self._do_text_to_sign).pack(side=tk.LEFT)

        canvas = tk.Canvas(self.tab_tts, highlightthickness=0)
        scroll = ttk.Scrollbar(self.tab_tts, orient=tk.VERTICAL, command=canvas.yview)
        self.tts_inner = ttk.Frame(canvas)
        self.tts_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.tts_inner, anchor=tk.NW)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=8)
        self.tts_canvas = canvas
        self._tts_images: List[object] = []

    def _build_account_tab(self):
        f = ttk.LabelFrame(self.tab_account, text="Sign in (optional — uses database)")
        f.pack(fill=tk.X, padx=12, pady=12)
        ttk.Label(f, text="Email").grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        self.ent_email = ttk.Entry(f, width=40)
        self.ent_email.grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(f, text="Password").grid(row=1, column=0, sticky=tk.W, padx=6, pady=4)
        self.ent_pass = ttk.Entry(f, width=40, show="*")
        self.ent_pass.grid(row=1, column=1, padx=6, pady=4)
        ttk.Button(f, text="Login", command=self._do_login).grid(row=2, column=1, sticky=tk.W, padx=6, pady=8)
        ttk.Button(f, text="Continue as Guest", command=self._do_guest).grid(row=2, column=1, sticky=tk.E, padx=6, pady=8)

        ttk.Label(
            self.tab_account,
            text="Guest mode runs fully on this PC. Login enables history and TTS preferences from the server DB.",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=12, pady=8)

    def _do_guest(self):
        self._user_id = None
        self._user_name = "Guest"
        self.lbl_user.config(text="Signed in as: Guest (local)")
        messagebox.showinfo("SignAI", "Using local guest profile. Gestures work; DB logging is limited.")

    def _do_login(self):
        email = self.ent_email.get().strip()
        pw = self.ent_pass.get()
        if not email or not pw:
            messagebox.showwarning("SignAI", "Enter email and password.")
            return
        try:
            uid, name = _try_user_login(email, pw)
        except Exception as e:
            messagebox.showerror("SignAI", f"Database error: {e}\nIs MySQL running and create_db done?")
            return
        if uid is None:
            messagebox.showerror("SignAI", "Invalid credentials.")
            return
        self._user_id = uid
        self._user_name = name or "User"
        self.lbl_user.config(text=f"Signed in as: {self._user_name}")
        messagebox.showinfo("SignAI", f"Welcome, {self._user_name}.")

    def _do_text_to_sign(self):
        import re
        import difflib

        from utils.db import get_db

        raw = self.ent_tts.get().strip()
        if not raw:
            return
        for w in self.tts_inner.winfo_children():
            w.destroy()
        self._tts_images.clear()

        words = [w for w in re.split(r"[\s,;.!?]+", raw) if w]
        try:
            conn = get_db()
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT gesture_name, image_path, description FROM gestures")
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            messagebox.showerror("SignAI", f"Database error: {e}")
            return

        known = [r["gesture_name"].upper() for r in rows]
        name_to = {r["gesture_name"].upper(): r for r in rows}

        def norm(w):
            return re.sub(r"[^A-Z0-9_]", "", w.upper())

        r = 0
        for w in words:
            key = norm(w)
            if not key:
                continue
            m = name_to.get(key)
            if not m:
                c = difflib.get_close_matches(key, known, n=1, cutoff=0.75)
                if c:
                    m = name_to.get(c[0])
            fr = ttk.Frame(self.tts_inner)
            fr.grid(row=r, column=0, sticky=tk.W, pady=6, padx=4)
            r += 1
            ttk.Label(fr, text=key, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
            if m and m.get("image_path"):
                path = os.path.join("static", m["image_path"].replace("\\", "/").lstrip("/"))
                if os.path.isfile(path):
                    try:
                        from PIL import Image, ImageTk

                        im = Image.open(path)
                        im.thumbnail((220, 220))
                        ph = ImageTk.PhotoImage(im)
                        self._tts_images.append(ph)
                        ttk.Label(fr, image=ph).pack(anchor=tk.W)
                    except Exception:
                        ttk.Label(fr, text="(image load error)").pack(anchor=tk.W)
                else:
                    ttk.Label(fr, text=f"Missing file: {path}").pack(anchor=tk.W)
            else:
                ttk.Label(fr, text="No sign in library for this word.", foreground="#64748b").pack(anchor=tk.W)
            if m and m.get("description"):
                ttk.Label(fr, text=m["description"], wraplength=400).pack(anchor=tk.W)

        self.tts_canvas.update_idletasks()
        self.tts_canvas.configure(scrollregion=self.tts_canvas.bbox("all"))

    def _play_last_audio(self):
        if not self._last_audio_path or not os.path.isfile(self._last_audio_path):
            return
        if sys.platform == "win32":
            os.startfile(self._last_audio_path)  # type: ignore[attr-defined]
        else:
            import subprocess

            subprocess.Popen(["xdg-open", self._last_audio_path], stderr=subprocess.DEVNULL)

    def _start_camera(self):
        if self._cam_running:
            return
        try:
            _bootstrap_ai()
        except Exception as e:
            messagebox.showerror(
                "SignAI",
                f"Could not load AI (OpenCV / MediaPipe / TensorFlow).\n{e}",
            )
            return
        self._cam_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._worker = threading.Thread(target=self._camera_worker, daemon=True)
        self._worker.start()
        self.after(50, self._pump_results)

    def _stop_camera(self):
        self._cam_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        try:
            while True:
                self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                self._result_q.get_nowait()
        except queue.Empty:
            pass

    def _camera_worker(self):
        import cv2

        from services.detection_service import detect_gesture_from_bgr

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._result_q.put({"error": "Camera not available"})
            self._cam_running = False
            return
        while self._cam_running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            try:
                out = detect_gesture_from_bgr(
                    frame, self._user_id, min_confidence=0.45, return_annotated_bgr=True
                )
            except Exception as e:
                out = {"success": False, "error": str(e), "annotated_rgb": None}
            rgb = out.get("annotated_rgb")
            if rgb is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                self._result_q.put_nowait({"rgb": rgb, "out": out})
            except queue.Full:
                try:
                    self._result_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._result_q.put_nowait({"rgb": rgb, "out": out})
                except queue.Full:
                    pass
        cap.release()

    def _pump_results(self):
        if not self._cam_running and self._worker and not self._worker.is_alive():
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            return
        try:
            while True:
                item = self._result_q.get_nowait()
                if item.get("error"):
                    messagebox.showerror("SignAI", item["error"])
                    self._stop_camera()
                    return
                rgb = item.get("rgb")
                out = item.get("out") or {}
                if rgb is not None:
                    self._show_frame(rgb)
                if out.get("success"):
                    g = out.get("gesture")
                    self.var_gesture.set(g.replace("_", " ") if g else "—")
                    c = out.get("confidence") or 0
                    self.var_conf.set(f"{int(c * 100)}%" if c else "—")
                    self.var_sentence.set(out.get("sentence") or out.get("display_text") or "—")
                    tr = out.get("translations") or {}
                    self.var_ta.set(tr.get("tamil") or "—")
                    self.var_hi.set(tr.get("hindi") or "—")
                    if out.get("is_emergency"):
                        self.emergency_banner.config(
                            text="EMERGENCY GESTURE — logged for responders"
                        )
                        self.bell()
                    else:
                        self.emergency_banner.config(text="")
                    paths = out.get("audio_paths") or {}
                    if paths:
                        rel = next(iter(paths.values()))
                        full = os.path.join("static", rel.replace("\\", "/"))
                        if os.path.isfile(full):
                            self._last_audio_path = os.path.abspath(full)
                            self.btn_play_audio.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        if self._cam_running:
            self.after(50, self._pump_results)

    def _show_frame(self, rgb):
        from PIL import Image, ImageTk

        h, w = rgb.shape[:2]
        max_w, max_h = 640, 480
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        im = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image=im)
        self.video.config(image=self._photo)

    def _on_close(self):
        self._stop_camera()
        time.sleep(0.15)
        self.destroy()


def main():
    _ensure_dirs()
    init_app_logging()
    app = SignAISystemApp()
    app.mainloop()


if __name__ == "__main__":
    main()
