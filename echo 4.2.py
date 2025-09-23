#!/usr/bin/env python3
# Echo 4.2 – Live & Batch GUI
# - FaceMesh dots removed (connections only)
# - Hi-Res Strokes: optional for Live; forced 3× for Batch
# - Glow separated from sharp strokes
# - High-quality Batch codecs (FFV1 default with fallbacks), no audio track
# - Eyes / Irises / Lips layers (independent), Irises toggles refine=True

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import threading, sys, os, time, csv, platform

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face = mp.solutions.face_mesh

# ---------- Utils ----------
def hex_to_bgr(hx: str):
    hx = (hx or "").lstrip('#')
    if len(hx) != 6:
        return (0, 255, 0)
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return (b, g, r)

def safe_color_pick(start="#FFFFFF"):
    c = colorchooser.askcolor(color=start)
    return c[1] if (isinstance(c, tuple) and c[1]) else None

def backend_sequence():
    sysname = platform.system().lower()
    seq = []
    if sysname == "windows":
        seq += [getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400)]
    elif sysname == "darwin":
        seq += [getattr(cv2, "CAP_AVFOUNDATION", 1200)]
    else:
        seq += [getattr(cv2, "CAP_V4L2", 200)]
    seq += [0]
    # de-dupe
    seen, out = set(), []
    for b in seq:
        if b not in seen:
            seen.add(b); out.append(b)
    return out

# ---------- App ----------
class HolisticApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Echo 4.2 – Live & Batch GUI")

        # ===== Live controls =====
        self.pose_on  = tk.BooleanVar(self.root, True)
        self.face_on  = tk.BooleanVar(self.root, True)    # FaceMesh tessellation (no dots)
        self.hands_on = tk.BooleanVar(self.root, True)

        self.video_opacity   = tk.DoubleVar(self.root, 1.0)
        self.overlay_opacity = tk.DoubleVar(self.root, 1.0)
        self.transparent_bg  = tk.BooleanVar(self.root, False)
        self.show_fps        = tk.BooleanVar(self.root, False)

        self.pose_color = tk.StringVar(self.root, "#00FF00")
        self.pose_thick = tk.IntVar(self.root, 2)

        self.face_color = tk.StringVar(self.root, "#0000FF")
        self.face_thick = tk.IntVar(self.root, 1)

        self.hand_color  = tk.StringVar(self.root, "#FFFF00")
        self.hand_rad    = tk.IntVar(self.root, 3)  # landmark thickness
        self.hand_thick  = tk.IntVar(self.root, 2)  # connection thickness

        # Glow (applies to Pose/Face/Hands only; not to Eyes/Irises/Lips)
        self.glow_on        = tk.BooleanVar(self.root, True)
        self.glow_radius    = tk.IntVar(self.root, 10)
        self.glow_intensity = tk.DoubleVar(self.root, 0.5)

        # New: Eyes / Irises / Lips (independent of FaceMesh), render on top (no glow)
        self.eyes_on    = tk.BooleanVar(self.root, True)
        self.irises_on  = tk.BooleanVar(self.root, False)  # toggles refine=True
        self.lips_on    = tk.BooleanVar(self.root, True)

        self.eyes_color   = tk.StringVar(self.root, "#00FFFF")
        self.eyes_thick   = tk.IntVar(self.root, 2)
        self.eyes_fill_on = tk.BooleanVar(self.root, False)

        self.irises_color   = tk.StringVar(self.root, "#FF00FF")
        self.irises_thick   = tk.IntVar(self.root, 2)
        self.irises_fill_on = tk.BooleanVar(self.root, False)

        self.lips_color   = tk.StringVar(self.root, "#FF3A3A")
        self.lips_thick   = tk.IntVar(self.root, 3)
        self.lips_fill_on = tk.BooleanVar(self.root, False)

        self.fill_opacity = tk.DoubleVar(self.root, 0.35)  # shared fill alpha

        # Hi-Res Strokes (Live only; Batch forces 3× regardless)
        self.hi_res_strokes_live = tk.BooleanVar(self.root, False)
        self.live_scale_factor = 2  # 2× for live when enabled

        # ===== Batch =====
        self.input_path   = tk.StringVar(self.root, "")
        self.output_path  = tk.StringVar(self.root, "")
        self.batch_progress = tk.DoubleVar(self.root, 0.0)
        self.batch_status   = tk.StringVar(self.root, "Idle")
        self.batch_running  = False
        self.batch_cancel   = False

        # Codec selection (OpenCV only)
        self.codec_choice = tk.StringVar(self.root, "FFV1 (Lossless AVI)")  # default per spec

        # ===== Live runtime =====
        self.running = True
        self.live_paused = False
        self.last_time = time.time()
        self.available_cams = []
        self.cam_idx_var = tk.StringVar(self.root, "0")
        self.backends = backend_sequence()
        self.cap = None

        # Mediapipe holistic – live (refine set by irises toggle)
        self.refine_enabled = False
        self.holistic = self._make_holistic(refine=False)

        self._build_ui()
        self._refresh_cameras()
        self._open_selected_camera()
        self._update_live()

        # Recreate Holistic when irises toggle changes (for Live)
        self.irises_on.trace_add('write', lambda *a: self._maybe_rebuild_holistic())

        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)
        self.root.mainloop()

    # ---------- Mediapipe ----------
    def _make_holistic(self, refine=False, static=False):
        return mp_holistic.Holistic(
            static_image_mode=static,
            model_complexity=1,
            refine_face_landmarks=bool(refine),
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _maybe_rebuild_holistic(self):
        desired = bool(self.irises_on.get())
        if desired != self.refine_enabled and not self.batch_running:
            try:
                self.holistic.close()
            except: pass
            self.holistic = self._make_holistic(refine=desired, static=False)
            self.refine_enabled = desired

    # ---------- UI ----------
    def _build_ui(self):
        nb = ttk.Notebook(self.root); nb.pack(fill="both", expand=True)

        # Live Tab
        live = ttk.Frame(nb); nb.add(live, text="Live")
        live.columnconfigure(0, weight=1)

        self.cam_lbl = tk.Label(live, text="Opening camera…")
        self.cam_lbl.grid(row=0, column=0, rowspan=80, padx=6, pady=6, sticky="nsew")

        # Camera controls
        ttk.Label(live, text="Camera Index").grid(row=0, column=1, sticky="w", padx=4, pady=(6,0))
        self.cam_combo = ttk.Combobox(live, textvariable=self.cam_idx_var, values=("0","1","2","3","4","5"), width=6, state="readonly")
        self.cam_combo.grid(row=0, column=2, sticky="w", padx=4, pady=(6,0))
        ttk.Button(live, text="Reopen", command=self._open_selected_camera).grid(row=0, column=3, sticky="we", padx=4, pady=(6,0))
        ttk.Button(live, text="Refresh", command=self._refresh_cameras).grid(row=0, column=4, sticky="we", padx=4, pady=(6,0))

        row = 1
        for label, var in [("Pose", self.pose_on), ("FaceMesh", self.face_on), ("Hands", self.hands_on),
                           ("Eyes", self.eyes_on), ("Irises", self.irises_on), ("Lips", self.lips_on)]:
            ttk.Checkbutton(live, text=label, variable=var).grid(row=row, column=1, columnspan=3, sticky="w"); row+=1

        # Opacity / misc
        ttk.Label(live, text="Video Opacity").grid(row=row, column=1, sticky="w"); row+=1
        tk.Scale(live, from_=0, to=1, resolution=0.1, orient="horizontal", variable=self.video_opacity).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1
        ttk.Label(live, text="Overlay Opacity").grid(row=row, column=1, sticky="w"); row+=1
        tk.Scale(live, from_=0, to=1, resolution=0.1, orient="horizontal", variable=self.overlay_opacity).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1
        ttk.Checkbutton(live, text="Transparent Background", variable=self.transparent_bg).grid(row=row, column=1, columnspan=3, sticky="w"); row+=1
        ttk.Checkbutton(live, text="Show FPS", variable=self.show_fps).grid(row=row, column=1, columnspan=3, sticky="w"); row+=1

        # Style controls
        def color_row(text, getter, setter, scale_var=None, smin=None, smax=None):
            nonlocal row
            def pick():
                newc = safe_color_pick(getter())
                if newc: setter(newc)
            ttk.Button(live, text=text, command=pick).grid(row=row, column=1, sticky="we"); row+=1
            if scale_var is not None:
                tk.Scale(live, from_=smin, to=smax, orient="horizontal", variable=scale_var).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1

        color_row("Pick Pose Color…", self.pose_color.get, self.pose_color.set, self.pose_thick, 1, 10)
        color_row("Pick Face Color…", self.face_color.get, self.face_color.set, self.face_thick, 1, 10)
        color_row("Pick Hand Color…", self.hand_color.get, self.hand_color.set)
        tk.Scale(live, from_=1, to=10, orient="horizontal", variable=self.hand_rad).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1
        tk.Scale(live, from_=1, to=10, orient="horizontal", variable=self.hand_thick).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1

        ttk.Checkbutton(live, text="Enable Glow (Pose/Face/Hands)", variable=self.glow_on).grid(row=row, column=1, sticky="w"); row+=1
        tk.Scale(live, from_=1, to=30, orient="horizontal", variable=self.glow_radius).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1
        tk.Scale(live, from_=0, to=1, resolution=0.05, orient="horizontal", variable=self.glow_intensity).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1

        ttk.Separator(live).grid(row=row, column=1, columnspan=3, sticky="we", pady=6); row+=1
        ttk.Label(live, text="Eyes / Irises / Lips (on top, no glow)").grid(row=row, column=1, columnspan=3, sticky="w"); row+=1

        color_row("Pick Eyes Color…", self.eyes_color.get, self.eyes_color.set, self.eyes_thick, 1, 10)
        ttk.Checkbutton(live, text="Fill Eyes", variable=self.eyes_fill_on).grid(row=row, column=1, sticky="w"); row+=1

        color_row("Pick Irises Color…", self.irises_color.get, self.irises_color.set, self.irises_thick, 1, 10)
        ttk.Checkbutton(live, text="Fill Irises", variable=self.irises_fill_on).grid(row=row, column=1, sticky="w"); row+=1

        color_row("Pick Lips Color…", self.lips_color.get, self.lips_color.set, self.lips_thick, 1, 10)
        ttk.Checkbutton(live, text="Fill Lips", variable=self.lips_fill_on).grid(row=row, column=1, sticky="w"); row+=1

        ttk.Label(live, text="Filled Shapes Opacity").grid(row=row, column=1, sticky="w"); row+=1
        tk.Scale(live, from_=0, to=1, resolution=0.05, orient="horizontal", variable=self.fill_opacity).grid(row=row, column=1, columnspan=3, sticky="we"); row+=1

        ttk.Separator(live).grid(row=row, column=1, columnspan=3, sticky="we", pady=6); row+=1
        ttk.Checkbutton(live, text="Hi-Res Strokes (2×) – Live only", variable=self.hi_res_strokes_live).grid(row=row, column=1, columnspan=3, sticky="w"); row+=1

        # Batch Tab
        batch = ttk.Frame(nb); nb.add(batch, text="Batch")
        batch.columnconfigure(1, weight=1)

        ttk.Label(batch, text="Input Video:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        self.in_entry = ttk.Entry(batch, textvariable=self.input_path); self.in_entry.grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(batch, text="Browse…", command=self._browse_in).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(batch, text="Output File:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        self.out_entry = ttk.Entry(batch, textvariable=self.output_path); self.out_entry.grid(row=1, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(batch, text="Browse…", command=self._browse_out).grid(row=1, column=2, padx=6, pady=6)

        ttk.Label(batch, text="Codec:").grid(row=2, column=0, padx=6, pady=6, sticky="e")
        ttk.Combobox(batch, textvariable=self.codec_choice,
                     values=("FFV1 (Lossless AVI)", "XVID (AVI)", "MJPG (AVI)", "MP4V (MP4)"),
                     state="readonly", width=20).grid(row=2, column=1, sticky="w")

        self.start_btn  = ttk.Button(batch, text="Start Batch", command=self._on_start)
        self.cancel_btn = ttk.Button(batch, text="Cancel Batch", command=self._on_cancel, state="disabled")
        self.start_btn.grid(row=3, column=0, columnspan=2, padx=6, pady=10, sticky="we")
        self.cancel_btn.grid(row=3, column=2, padx=6, pady=10, sticky="we")

        ttk.Progressbar(batch, variable=self.batch_progress, maximum=100).grid(row=4, column=0, columnspan=3, sticky="we", padx=6)
        ttk.Label(batch, textvariable=self.batch_status).grid(row=5, column=0, columnspan=3, sticky="w", padx=6, pady=(4,10))

        for var in (self.input_path, self.output_path):
            var.trace_add('write', lambda *a: self._update_start_state())
        self._update_start_state()

    # ---------- Camera handling ----------
    def _refresh_cameras(self):
        found = []
        for idx in range(0, 10):
            if self._probe_index(idx):
                found.append(str(idx))
        if not found:
            found = ["0","1","2","3","4","5"]
        self.available_cams = found
        self.cam_combo.configure(values=tuple(found))
        if self.cam_idx_var.get() not in found:
            self.cam_idx_var.set(found[0])

    def _probe_index(self, idx: int) -> bool:
        for be in self.backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    return True
        return False

    def _open_selected_camera(self):
        sel = self.cam_idx_var.get()
        try: idx = int(sel)
        except: idx = 0
        try:
            if self.cap: self.cap.release()
        except: pass

        self.cap = self._open_camera(idx)
        if self.cap and self.cap.isOpened():
            self.cam_lbl.config(text="")
            self.root.title(f"Echo 4.2 – Live & Batch GUI  |  Camera idx {idx}")
        else:
            self.cam_lbl.config(text="No camera frame (choose another index and click Reopen)")

    def _open_camera(self, preferred_idx=None):
        indices = [preferred_idx] if preferred_idx is not None else list(range(0,10))
        for idx in indices:
            for be in self.backends:
                cap = cv2.VideoCapture(idx, be)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        return cap
                cap.release()
        return cv2.VideoCapture()

    # ---------- Live loop ----------
    def _update_live(self):
        if not self.running:
            return
        self._maybe_rebuild_holistic()  # align refine with irises toggle

        if not self.live_paused and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)  # mirror
                out = self._draw(frame, self.holistic, draw_fps=self.show_fps.get(),
                                 supersample=(self.live_scale_factor if self.hi_res_strokes_live.get() else 1))
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
                self.cam_lbl.imgtk = img
                self.cam_lbl.configure(image=img, text="")
            else:
                self.cam_lbl.configure(text="No camera frame (try Reopen)", image="")
        else:
            if self.live_paused:
                self.cam_lbl.configure(text="Live paused during Batch…", image="")
        self.root.after(15, self._update_live)

    # ---------- Region utils for fills ----------
    def _region_points_from_conns(self, face_landmarks, shape, connections):
        if not face_landmarks: return None
        h, w = shape[0], shape[1]
        idxs = set()
        for a, b in connections: idxs.add(a); idxs.add(b)
        pts, lms = [], face_landmarks.landmark
        for i in idxs:
            if 0 <= i < len(lms):
                x = int(lms[i].x * w); y = int(lms[i].y * h)
                pts.append([x, y])
        if not pts: return None
        return np.array(pts, dtype=np.int32).reshape(-1,1,2)

    def _fill_region(self, target, points, color_bgr, alpha):
        if points is None or len(points) < 3: return
        hull = cv2.convexHull(points)
        layer = np.zeros_like(target)
        cv2.fillPoly(layer, [hull], color_bgr)
        cv2.addWeighted(layer, float(alpha), target, 1.0, 0, dst=target)

    # ---------- Drawing & compositing ----------
    def _draw(self, frame, holistic, draw_fps=False, supersample=1):
        # supersample overlays only (sharp downscale for silky lines)
        h, w = frame.shape[:2]
        scale = max(1, int(supersample))
        work_w, work_h = w*scale, h*scale

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        overlay_glow_src = np.zeros((work_h, work_w, 3), dtype=np.uint8)
        overlay_top      = np.zeros((work_h, work_w, 3), dtype=np.uint8)

        # Pose (glow-eligible)
        if self.pose_on.get() and res.pose_landmarks:
            ds = mp_drawing.DrawingSpec(color=hex_to_bgr(self.pose_color.get()), thickness=self.pose_thick.get())
            # draw only connections? Pose looks fine with both; keep joints per user's wish
            mp_drawing.draw_landmarks(
                overlay_glow_src, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=hex_to_bgr(self.hand_color.get()),
                                                             thickness=self.hand_rad.get(),
                                                             circle_radius=max(1, self.hand_rad.get())),
                connection_drawing_spec=ds
            )

        # Face tessellation (glow-eligible) — NO landmark dots
        if self.face_on.get() and res.face_landmarks:
            ds = mp_drawing.DrawingSpec(color=hex_to_bgr(self.face_color.get()), thickness=self.face_thick.get())
            mp_drawing.draw_landmarks(
                overlay_glow_src, res.face_landmarks, mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,  # remove white dots entirely
                connection_drawing_spec=ds
            )

        # Hands (glow-eligible) – keep joint dots per your preference
        if self.hands_on.get():
            for hnd in (res.left_hand_landmarks, res.right_hand_landmarks):
                if hnd:
                    lds = mp_drawing.DrawingSpec(color=hex_to_bgr(self.hand_color.get()),
                                                 thickness=self.hand_rad.get(),
                                                 circle_radius=max(1, self.hand_rad.get()))
                    cds = mp_drawing.DrawingSpec(color=hex_to_bgr(self.hand_color.get()),
                                                 thickness=self.hand_thick.get())
                    mp_drawing.draw_landmarks(overlay_glow_src, hnd, mp_holistic.HAND_CONNECTIONS,
                                              landmark_drawing_spec=lds, connection_drawing_spec=cds)

        # ----- Top layers (no glow), independent of FaceMesh toggle -----
        if res.face_landmarks:
            # Eyes
            if self.eyes_on.get():
                col = hex_to_bgr(self.eyes_color.get())
                ds = mp_drawing.DrawingSpec(color=col, thickness=self.eyes_thick.get())
                for conns in (mp_face.FACEMESH_LEFT_EYE, mp_face.FACEMESH_RIGHT_EYE):
                    if self.eyes_fill_on.get():
                        pts = self._region_points_from_conns(res.face_landmarks, (work_h, work_w), conns)
                        self._fill_region(overlay_top, pts, col, self.fill_opacity.get())
                    mp_drawing.draw_landmarks(overlay_top, res.face_landmarks, conns,
                                              landmark_drawing_spec=None, connection_drawing_spec=ds)

            # Irises (requires refine=True)
            if self.irises_on.get():
                col = hex_to_bgr(self.irises_color.get())
                ds = mp_drawing.DrawingSpec(color=col, thickness=self.irises_thick.get())
                conns = mp_face.FACEMESH_IRISES
                if self.irises_fill_on.get():
                    pts = self._region_points_from_conns(res.face_landmarks, (work_h, work_w), conns)
                    self._fill_region(overlay_top, pts, col, self.fill_opacity.get())
                mp_drawing.draw_landmarks(overlay_top, res.face_landmarks, conns,
                                          landmark_drawing_spec=None, connection_drawing_spec=ds)

            # Lips
            if self.lips_on.get():
                col = hex_to_bgr(self.lips_color.get())
                ds = mp_drawing.DrawingSpec(color=col, thickness=self.lips_thick.get())
                conns = mp_face.FACEMESH_LIPS
                if self.lips_fill_on.get():
                    pts = self._region_points_from_conns(res.face_landmarks, (work_h, work_w), conns)
                    self._fill_region(overlay_top, pts, col, self.fill_opacity.get())
                mp_drawing.draw_landmarks(overlay_top, res.face_landmarks, conns,
                                          landmark_drawing_spec=None, connection_drawing_spec=ds)

        # Downscale overlays back to frame size for compositing
        if scale != 1:
            overlay_glow_src = cv2.resize(overlay_glow_src, (w, h), interpolation=cv2.INTER_AREA)
            overlay_top      = cv2.resize(overlay_top,      (w, h), interpolation=cv2.INTER_AREA)

        # Glow only from glow-source
        if self.glow_on.get():
            r_out = max(1, int(self.glow_radius.get()))
            r_in  = max(1, r_out // 2)
            bi = cv2.GaussianBlur(overlay_glow_src, (2*r_in+1, 2*r_in+1), 0)
            bo = cv2.GaussianBlur(overlay_glow_src, (2*r_out+1, 2*r_out+1), 0)
            glow = cv2.addWeighted(bi, 0.6*self.glow_intensity.get(), bo, 0.4*self.glow_intensity.get(), 0)
        else:
            glow = np.zeros_like(frame)

        # Composite (base video → add glow → add sharp overlays)
        base = (frame.astype(np.float32) * float(self.video_opacity.get())).astype(np.uint8)
        lit  = cv2.add(base, glow)

        overlay_combined = cv2.add(overlay_glow_src, overlay_top)
        alpha = float(self.overlay_opacity.get())

        if self.transparent_bg.get():
            final = cv2.addWeighted(overlay_combined, alpha, lit, 1.0, 0.0)
        else:
            final = cv2.addWeighted(overlay_combined, alpha, lit, 1.0 - alpha, 0.0)

        if draw_fps:
            now = time.time()
            fps = 1.0 / max(now - self.last_time, 1e-3)
            self.last_time = now
            cv2.putText(final, f"{int(fps)} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return final

    # ---------- Batch ----------
    def _update_start_state(self):
        ok = bool(self.input_path.get()) and bool(self.output_path.get()) and not self.batch_running
        self.start_btn.config(state="normal" if ok else "disabled")

    def _browse_in(self):
        p = filedialog.askopenfilename(filetypes=[("Video","*.mp4 *.avi *.mov *.mkv")])
        if p: self.input_path.set(p)

    def _browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".avi",
                                         filetypes=[("AVI / MP4","*.avi *.mp4")])
        if p: self.output_path.set(p)

    def _on_start(self):
        if not os.path.isfile(self.input_path.get()) or not self.output_path.get():
            messagebox.showwarning("Invalid Paths", "Please select valid input/output.")
            return
        self.batch_running = True
        self.batch_cancel  = False
        self.batch_status.set("Starting…")
        self.cancel_btn.config(state="normal")
        self.start_btn.config(state="disabled")

        # Pause live & free camera
        self.live_paused = True
        try:
            if self.cap: self.cap.release()
        except: pass

        threading.Thread(target=self._start_batch, daemon=True).start()

    def _on_cancel(self):
        self.batch_cancel = True
        self.batch_status.set("Cancelling…")

    def _start_batch(self):
        cap = cv2.VideoCapture(self.input_path.get())
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Holistic for batch – refine matches Irises toggle; static=True
        hol = self._make_holistic(refine=bool(self.irises_on.get()), static=True)

        # Decide codec chain
        choice = self.codec_choice.get()
        chain = []
        if "FFV1" in choice:
            chain = [("FFV1", "AVI")]
        elif "XVID" in choice:
            chain = [("XVID", "AVI")]
        elif "MJPG" in choice:
            chain = [("MJPG", "AVI")]
        else:
            chain = [("mp4v", "MP4")]

        # Always fall back to ensure we can write something
        fallback = [("FFV1","AVI"), ("XVID","AVI"), ("MJPG","AVI"), ("mp4v","MP4")]
        for item in fallback:
            if item not in chain:
                chain.append(item)

        # Open writer with first working codec; adjust extension if needed
        out_path_user = self.output_path.get()
        writer, used_codec, used_path = None, None, out_path_user
        for fourcc_name, container in chain:
            ext = ".avi" if container == "AVI" else ".mp4"
            candidate_path = out_path_user
            # If user picked a mismatched extension, silently adapt the actual file we write
            if not candidate_path.lower().endswith(ext):
                base, _ = os.path.splitext(out_path_user)
                candidate_path = base + ext

            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            test_writer = cv2.VideoWriter(candidate_path, fourcc, fps, (w, h))
            if test_writer.isOpened():
                writer, used_codec, used_path = test_writer, fourcc_name, candidate_path
                break

        if writer is None:
            self.batch_running = False
            self.root.after(0, lambda: messagebox.showerror(
                "Writer Error",
                "Could not open a VideoWriter for any of the available codecs."
            ))
            cap.release(); hol.close()
            self.root.after(0, self._batch_done)
            return

        if used_path != out_path_user:
            self.root.after(0, lambda:
                self.batch_status.set(f"Codec {used_codec} selected → writing {os.path.basename(used_path)}")
            )
        else:
            self.root.after(0, lambda:
                self.batch_status.set(f"Codec {used_codec} selected")
            )

        processed = 0
        faces_accum = 0.0
        hands_accum = 0.0
        ms_accum = 0.0

        # Force supersampling 3× for batch
        batch_scale = 3

        try:
            while True:
                if self.batch_cancel: break
                ret, frame = cap.read()
                if not ret: break

                t0 = time.perf_counter()
                out = self._draw(frame, hol, draw_fps=False, supersample=batch_scale)
                dt = (time.perf_counter() - t0) * 1000.0
                ms_accum += dt

                # Simple metrics
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hol.process(rgb)
                faces_accum += 1.0 if res.face_landmarks else 0.0
                hands_accum += (1.0 if res.left_hand_landmarks else 0.0) + (1.0 if res.right_hand_landmarks else 0.0)

                writer.write(out)
                processed += 1

                prog = (processed/total)*100 if total else 100
                self.root.after(0, lambda p=prog, n=processed, t=total, d=dt: (
                    self.batch_progress.set(p),
                    self.batch_status.set(f"Processing {n}/{t} | {int(d)} ms/frame | {used_codec}")
                ))
        finally:
            cap.release(); writer.release(); hol.close()

        # Write CSV metrics
        if processed > 0 and not self.batch_cancel:
            avg_faces = faces_accum / processed
            avg_hands = hands_accum / processed
            avg_ms    = ms_accum / processed
            base, _ = os.path.splitext(used_path)
            csv_path  = base + "_metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                wcsv = csv.writer(f)
                wcsv.writerow(["total_frames","avg_faces_per_frame","avg_hands_per_frame","avg_ms_per_frame"])
                wcsv.writerow([processed, f"{avg_faces:.4f}", f"{avg_hands:.4f}", f"{avg_ms:.2f}"])

        self.batch_running = False
        self.root.after(0, self._batch_done)

    def _batch_done(self):
        self.cancel_btn.config(state="disabled")
        self._update_start_state()
        self.batch_status.set("Batch cancelled" if self.batch_cancel else "Batch complete")
        # Resume live; rebuild holistic if needed and reopen camera
        self.live_paused = False
        self._maybe_rebuild_holistic()
        self._open_selected_camera()
        if not self.batch_cancel:
            messagebox.showinfo("Batch Complete", "Rendering finished.")

    # ---------- Shutdown ----------
    def _shutdown(self):
        self.running = False
        try:
            if self.cap: self.cap.release()
        except: pass
        try:
            if self.holistic: self.holistic.close()
        except: pass
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    HolisticApp()
