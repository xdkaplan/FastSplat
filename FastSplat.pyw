"""FastSplat — one-window launcher for the photos -> splat pipeline.

Pick a photos folder, get a 3D Gaussian Splat. Runs:
    photos -> COLMAP sparse reconstruction -> LichtFeld training -> viewer

See README.md for expected environment and customization.
Edit the path constants below if you move COLMAP or LichtFeld.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# --- Paths (edit if you move things) ----------------------------------------
# Call colmap.exe directly, NOT colmap.bat. The bundled .bat does
# `set ARGUMENTS=%*` then `if "%ARGUMENTS%"==""` which mis-parses any path
# containing spaces (fails with e.g. "Alex was unexpected at this time").
# We replicate the .bat's env setup ourselves in colmap_env().
COLMAP_ROOT = r"C:\Repos\colmap\colmap"
COLMAP_EXE = COLMAP_ROOT + r"\bin\colmap.exe"

# GLOMAP is an optional drop-in replacement for COLMAP's mapper step (global
# SfM vs COLMAP's incremental). 5-10x faster on large photo sets. Same output
# format, so LichtFeld reads it unchanged.
# Install: grab the Windows binary zip from https://github.com/colmap/glomap/releases
# and extract so glomap.exe lands at the path below. If the file isn't present,
# FastSplat falls back to plain COLMAP automatically — no error.
GLOMAP_ROOT = r"C:\Repos\colmap\glomap"
GLOMAP_EXE = GLOMAP_ROOT + r"\bin\glomap.exe"

LICHTFELD_EXE = r"C:\Repos\MrNeRF\LichtFeld-Studio\build\LichtFeld-Studio.exe"
LICHTFELD_DLL_PATHS = [
    r"C:\Repos\MrNeRF\LichtFeld-Studio\build",
    r"C:\Repos\MrNeRF\LichtFeld-Studio\build\vcpkg_installed\x64-windows\bin",
]

# Suppress console windows for any child process we spawn on Windows.
# 0 elsewhere (flag is a no-op on non-Windows).
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Persisted UI settings live in %APPDATA%\FastSplat\settings.json
SETTINGS_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "FastSplat" / "settings.json"

# Per-run logs go next to the script in logs/. Gitignored.
LOG_DIR = Path(__file__).parent / "logs"

# Your GPU's total VRAM (GB). Drives the pre-flight warning.
# Quadro RTX 5000 = 16. Edit if you move to a different card.
GPU_VRAM_GB = 16

# Named setting presets. Selecting one in the UI snaps every listed field to
# its value; fields not listed are left alone. Add your own by appending entries
# here (or, eventually, externalising to a JSON file).
PRESETS: dict[str, dict[str, object]] = {
    "(custom)": {},  # placeholder; selecting it doesn't change anything
    "Pixel 6a multi-room": {
        "iter":              20_000,
        "max_cap":           1_200_000,
        "matcher":           "sequential",   # ordered walk-through; ±30 neighbors plenty
        "use_gut":           True,
        "use_bilateral_grid": True,
        "use_mip":           True,
        "sh_degree":         2,
        "use_ppisp":         False,   # locked exposure -> PPISP would over-correct real lighting
        "mask_subject":      False,
        "open_after":        True,
    },
    "Outdoor monument (DSLR, manual)": {
        "iter":              30_000,
        "max_cap":           1_500_000,
        "matcher":           "exhaustive",   # orbit captures often aren't strictly ordered
        "use_gut":           True,
        "use_bilateral_grid": True,
        "use_mip":           True,
        "sh_degree":         3,
        "use_ppisp":         False,
        "mask_subject":      False,
        "open_after":        True,
    },
    "Draft (fastest, ~2-5 min)": {
        # Goal: "did the pipeline run + is the rough shape recognizable" in
        # the minimum possible time. Trades quality aggressively for speed.
        "iter":              3_000,
        "max_cap":           400_000,
        "matcher":           "sequential",   # fastest SfM for typical walk-through captures
        "use_gut":           True,
        "use_bilateral_grid": False,
        "use_mip":           False,
        "sh_degree":         1,        # flat-ish color; fastest per-iter
        "use_ppisp":         False,
        "mask_subject":      False,
        "open_after":        True,
    },
    "Quick test (7k iter)": {
        # Middle ground between draft and full quality. Recognizable splat,
        # not production grade.
        "iter":              7_000,
        "max_cap":           800_000,
        "matcher":           "sequential",
        "use_gut":           True,
        "use_bilateral_grid": False,
        "use_mip":           False,
        "sh_degree":         2,
        "use_ppisp":         False,
        "mask_subject":      False,
        "open_after":        True,
    },
}

# Empirical VRAM cost during training.
# - Per-million-gaussians is dominated by Adam optimizer state + backprop
#   intermediates, which scale with the number of SH coefficients (i.e., SH degree).
# - Fixed overhead covers CUDA runtime + the Vulkan viewer's interop buffer.
# Numbers are conservative estimates calibrated to a 16 GB card with the default
# strategy/options. Within ~30% of reality; meant for "are you in danger" warnings.
def estimate_vram_gb(max_cap: int, sh_degree: int = 2,
                     viewer_running: bool = True) -> float:
    per_million_gb = {0: 1.5, 1: 2.2, 2: 3.0, 3: 4.2}.get(sh_degree, 3.0)
    gaussian_gb = (max_cap / 1_000_000.0) * per_million_gb
    overhead_gb = 3.5 + (1.5 if viewer_running else 0.0)
    return gaussian_gb + overhead_gb
# ---------------------------------------------------------------------------


class ToolTip:
    """Lightweight tooltip helper: ToolTip(widget, "text") attaches a hover tip."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: object = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, justify="left",
                 bg="#fff7d4", fg="#222222", relief="solid", borderwidth=1,
                 font=("Segoe UI", 9), wraplength=420).pack(ipadx=6, ipady=4)

    def _hide(self, _event: object = None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def lichtfeld_env() -> dict[str, str]:
    """Return an env dict with LichtFeld's DLL paths prepended to PATH."""
    env = os.environ.copy()
    env["PATH"] = ";".join(LICHTFELD_DLL_PATHS + [env.get("PATH", "")])
    return env


def colmap_env() -> dict[str, str]:
    """Return an env dict matching what colmap.bat sets up: bin/ on PATH for
    DLLs, plugins/ on QT_PLUGIN_PATH."""
    env = os.environ.copy()
    env["PATH"] = f"{COLMAP_ROOT}\\bin;{env.get('PATH', '')}"
    env["QT_PLUGIN_PATH"] = f"{COLMAP_ROOT}\\plugins;{env.get('QT_PLUGIN_PATH', '')}"
    return env


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FastSplat")
        self.geometry("860x720")
        self.minsize(740, 540)

        # State
        self.photos_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.iter_var = tk.IntVar(value=7000)
        self.run_colmap = tk.BooleanVar(value=True)
        self.run_train = tk.BooleanVar(value=True)
        self.use_gut = tk.BooleanVar(value=True)
        self.open_after = tk.BooleanVar(value=True)
        self.mask_subject = tk.BooleanVar(value=False)
        self.mask_model = tk.StringVar(value="u2net")
        self.sky_only = tk.BooleanVar(value=False)
        self.force_remask = tk.BooleanVar(value=False)
        self.output_name_var = tk.StringVar(value="")
        self.max_cap_var = tk.IntVar(value=1_000_000)
        # SfM matcher choice (exhaustive vs sequential)
        self.matcher_var = tk.StringVar(value="exhaustive")
        # Advanced training flags (passed through to LichtFeld)
        self.use_bilateral_grid = tk.BooleanVar(value=False)
        self.use_mip = tk.BooleanVar(value=False)
        self.use_ppisp = tk.BooleanVar(value=False)
        self.sh_degree_var = tk.IntVar(value=2)
        # Preset selector (UI-only convenience; doesn't persist as such)
        self.preset_var = tk.StringVar(value="(custom)")

        # Worker -> UI message queue
        self._msg_q: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        # File handle for the current run's saved log (None when not running).
        # Written to via _log(); opened in _on_go, closed when worker finishes.
        self._log_file = None  # type: ignore[var-annotated]

        # Suppress trace callbacks during settings load so we don't
        # repeatedly fire the judger.
        self._loading_settings = False

        self._build_ui()
        self._load_settings()
        # Save settings on window close and re-judge any restored photos path.
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._drain_queue)
        self.after(100, self._refresh_judger)
        self.after(100, self._refresh_vram_estimate)

        # Auto-refresh the judger whenever the photos path changes.
        self.photos_var.trace_add("write", lambda *_: self._refresh_judger())
        # Update the VRAM estimate when the cap, SH degree, or viewer toggle changes.
        self.max_cap_var.trace_add("write", lambda *_: self._refresh_vram_estimate())
        self.sh_degree_var.trace_add("write", lambda *_: self._refresh_vram_estimate())
        self.open_after.trace_add("write", lambda *_: self._refresh_vram_estimate())

    # -- UI ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pad = dict(padx=8, pady=4)

        # Preset selector row (very top)
        preset_row = ttk.Frame(self)
        preset_row.grid(row=0, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(preset_row, text="Preset").pack(side="left", padx=(0, 4))
        preset_combo = ttk.Combobox(preset_row, textvariable=self.preset_var,
                                    values=list(PRESETS.keys()),
                                    width=36, state="readonly")
        preset_combo.pack(side="left", padx=(0, 8))
        preset_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())
        ToolTip(preset_combo,
                "Snap all training flags to a curated configuration for a "
                "specific kind of shoot. Doesn't touch your photos / output / "
                "tag fields. Select '(custom)' to leave settings alone.")
        ttk.Label(preset_row,
                  text="(snaps flags to curated values; doesn't change folders/tag)",
                  foreground="#888888").pack(side="left", padx=(0, 0))

        # Photos row
        ttk.Label(self, text="Photos folder").grid(row=1, column=0, sticky="w", **pad)
        photos_entry = ttk.Entry(self, textvariable=self.photos_var)
        photos_entry.grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse...", command=self._pick_photos).grid(row=1, column=2, **pad)
        ToolTip(photos_entry,
                "Folder containing your raw photos. Subfolders are ignored. "
                "Common formats: .jpg .png .tif .heic.")

        # Judger / pre-flight status row (auto-updated when photos folder changes)
        self.judger_var = tk.StringVar(value="")
        self.judger_label = tk.Label(self, textvariable=self.judger_var,
                                     justify="left", anchor="w",
                                     bg=self.cget("bg"), fg="#bbbbbb",
                                     font=("Segoe UI", 9), wraplength=720)
        self.judger_label.grid(row=2, column=0, columnspan=3, sticky="we",
                               padx=8, pady=(0, 4))

        # Output row
        ttk.Label(self, text="Output folder").grid(row=3, column=0, sticky="w", **pad)
        output_entry = ttk.Entry(self, textvariable=self.output_var)
        output_entry.grid(row=3, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse...", command=self._pick_output).grid(row=3, column=2, **pad)
        ToolTip(output_entry,
                "Where trained .ply files land. Auto-filled to a sibling of the "
                "photos folder when you pick one.")

        # Output tag/name row
        ttk.Label(self, text="Output tag/name").grid(row=4, column=0, sticky="w", **pad)
        tag_frame = ttk.Frame(self)
        tag_frame.grid(row=4, column=1, columnspan=2, sticky="we", **pad)
        tag_entry = ttk.Entry(tag_frame, textvariable=self.output_name_var)
        tag_entry.pack(side="left", fill="x", expand=True)
        ttk.Label(tag_frame, text="(leave blank for default 'splat_<iter>.ply')",
                  foreground="#888888").pack(side="left", padx=(8, 0))
        ToolTip(tag_entry,
                "If set, the final .ply is named <tag>.ply instead of splat_<iter>.ply. "
                "Useful for stamping runs like 'monument-pixel-v1', "
                "'monument-dslr-masked' so multiple attempts coexist in one output dir.")

        # Options row
        opts = ttk.Frame(self)
        opts.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(opts, text="Iterations").pack(side="left", padx=(0, 4))
        iter_spin = ttk.Spinbox(opts, from_=1000, to=60000, increment=1000,
                                textvariable=self.iter_var, width=8)
        iter_spin.pack(side="left", padx=(0, 12))
        ToolTip(iter_spin,
                "Training iterations. 7000 = quick smoke test (~5-15 min on RTX 5000). "
                "15000 = solid result. 30000 = full production quality. Diminishing "
                "returns past ~30k.")
        ttk.Label(opts, text="Max gaussians").pack(side="left", padx=(0, 4))
        cap_spin = ttk.Spinbox(opts, from_=100_000, to=5_000_000, increment=100_000,
                               textvariable=self.max_cap_var, width=10)
        cap_spin.pack(side="left", padx=(0, 4))
        ToolTip(cap_spin,
                "Ceiling on total gaussian count during densification. Higher = "
                "potentially sharper splat but more VRAM. The training will stop "
                "spawning new gaussians once it hits this cap. For a 16 GB card, "
                "1,000,000 is safe; 2,000,000 is borderline; 3,000,000+ risks OOM.")
        self.vram_label = ttk.Label(opts, text="", foreground="#888888")
        self.vram_label.pack(side="left", padx=(0, 16))
        ToolTip(self.vram_label,
                f"Rough VRAM estimate at the max-cap. Your GPU has "
                f"{GPU_VRAM_GB} GB. Green = comfortable. Amber = close to budget. "
                "Red = will likely OOM during densification. Tune GPU_VRAM_GB "
                "at the top of fast_splat.py if you're not on a 16 GB card.")

        cb_colmap = ttk.Checkbutton(opts, text="SfM", variable=self.run_colmap)
        cb_colmap.pack(side="left", padx=4)
        ToolTip(cb_colmap,
                "Run structure-from-motion (COLMAP front-end + GLOMAP mapper if "
                "GLOMAP is installed, else COLMAP-only). Uncheck only if a sparse "
                "model already exists for this scene and you want to skip straight "
                "to training.")
        matcher_combo = ttk.Combobox(opts, textvariable=self.matcher_var, width=12,
                                     state="readonly",
                                     values=("exhaustive", "sequential"))
        matcher_combo.pack(side="left", padx=(4, 4))
        ToolTip(matcher_combo,
                "Feature matcher for SfM. 'exhaustive' = every photo against every "
                "other photo (thorough, scales O(N²), slow on big sets). 'sequential' "
                "= each photo matched against its ±30 neighbors in capture order "
                "(massively faster, ideal for ordered walk-throughs and orbit "
                "captures). Use 'sequential' when photos were taken in a natural "
                "sequence; use 'exhaustive' when photo order is arbitrary.")
        cb_train = ttk.Checkbutton(opts, text="Train", variable=self.run_train)
        cb_train.pack(side="left", padx=4)
        ToolTip(cb_train,
                "Run LichtFeld training. Uncheck to stop after COLMAP (e.g. you only "
                "wanted the sparse reconstruction).")
        cb_gut = ttk.Checkbutton(opts, text="--gut", variable=self.use_gut)
        cb_gut.pack(side="left", padx=4)
        ToolTip(cb_gut,
                "Pass --gut to LichtFeld (enables 3DGUT mode). Required when COLMAP "
                "picks a distorted camera model (SIMPLE_RADIAL, fisheye, spherical). "
                "Safe to leave on for pinhole sets too; it's a no-op there.")
        cb_view = ttk.Checkbutton(opts, text="Open viewer after",
                                  variable=self.open_after)
        cb_view.pack(side="left", padx=4)
        ToolTip(cb_view,
                "After training completes, auto-launch LichtFeld in viewer mode (-v) "
                "on the latest .ply so you can crop / inspect / export immediately.")

        # Advanced flags row (LichtFeld training quality knobs)
        adv = ttk.Frame(self)
        adv.grid(row=6, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(adv, text="SH degree").pack(side="left", padx=(0, 4))
        sh_spin = ttk.Spinbox(adv, from_=0, to=3, increment=1,
                              textvariable=self.sh_degree_var, width=4)
        sh_spin.pack(side="left", padx=(0, 12))
        ToolTip(sh_spin,
                "Spherical harmonic degree for per-gaussian view-dependent color. "
                "0 = flat color (smallest, fastest). 2 = default (good balance). "
                "3 = max quality, ~40% more VRAM per gaussian. Lower if you're "
                "hitting OOM; raise for shoots with rich lighting variation.")
        cb_bg = ttk.Checkbutton(adv, text="--bilateral-grid",
                                variable=self.use_bilateral_grid)
        cb_bg.pack(side="left", padx=4)
        ToolTip(cb_bg,
                "Per-region color refinement. Especially helps multi-room scenes "
                "with different ambient lighting per area, or any scene where "
                "color varies by location. Modest VRAM cost; usually worth on.")
        cb_mip = ttk.Checkbutton(adv, text="--enable-mip",
                                 variable=self.use_mip)
        cb_mip.pack(side="left", padx=4)
        ToolTip(cb_mip,
                "Anti-aliasing / mip filter. Reduces shimmer when the viewer "
                "zooms / moves the camera. Negligible cost; usually worth on.")
        cb_ppisp = ttk.Checkbutton(adv, text="--ppisp",
                                   variable=self.use_ppisp)
        cb_ppisp.pack(side="left", padx=4)
        ToolTip(cb_ppisp,
                "Per-camera appearance correction. Helps when WB/exposure varied "
                "between shots (e.g. Pixel HDR+ drift, mixed-camera capture). "
                "AVOID when exposure/WB was locked across deliberately different "
                "lighting — PPISP will erase the real lighting signal.")

        # Masking row
        mask_row = ttk.Frame(self)
        mask_row.grid(row=7, column=0, columnspan=3, sticky="we", **pad)
        cb_mask = ttk.Checkbutton(mask_row, text="Mask subject (rembg)",
                                  variable=self.mask_subject)
        cb_mask.pack(side="left", padx=(0, 12))
        ToolTip(cb_mask,
                "Run rembg on every photo before COLMAP to extract the subject. "
                "Eliminates the 50-100m halo of background gaussians on outdoor "
                "subjects (monuments, statues, vehicles). Adds ~30s-2min of preprocessing.")
        ttk.Label(mask_row, text="Model").pack(side="left", padx=(0, 4))
        model_combo = ttk.Combobox(mask_row, textvariable=self.mask_model, width=22,
                                   values=("u2net", "u2netp", "silueta",
                                           "isnet-general-use", "birefnet-general"))
        model_combo.pack(side="left", padx=(0, 8))
        ToolTip(model_combo,
                "rembg model choice. birefnet-general = best edges, slowest. "
                "isnet-general-use = goldilocks (almost as good, ~2x faster). "
                "u2net = fast default but tends to chop statue bases. "
                "First time using a model triggers a one-time download.")
        self.preview_btn = ttk.Button(mask_row, text="Preview sample",
                                      command=self._on_preview_sample)
        self.preview_btn.pack(side="left", padx=(8, 8))
        ToolTip(self.preview_btn,
                "Mask ~1/10 of your photos with the chosen model and pop up a "
                "side-by-side review window (original | cutout-on-magenta). Quick "
                "way to check model quality before committing to the full batch.")
        cb_sky = ttk.Checkbutton(mask_row, text="Sky-only",
                                 variable=self.sky_only)
        cb_sky.pack(side="left", padx=4)
        ToolTip(cb_sky,
                "Color-based sky removal (top 50% of frame). Uses HSV thresholds "
                "(bright + low-saturation, or clearly blue) — NOT rembg, so vertical "
                "subjects like columns/towers/spires reaching the top of frame are "
                "preserved as long as they're not actually sky-colored. The Model "
                "dropdown is ignored when this is on. Much faster than rembg too.")
        cb_force = ttk.Checkbutton(mask_row, text="Force re-mask",
                                   variable=self.force_remask)
        cb_force.pack(side="left", padx=4)
        ToolTip(cb_force,
                "Discard cached masks and reprocess on the next Go. Auto-triggered "
                "when you change the model dropdown, so usually only needed to "
                "redo masks with the SAME model (e.g. interrupted run).")

        # Go button
        self.go_btn = ttk.Button(self, text="Go", command=self._on_go)
        self.go_btn.grid(row=8, column=0, columnspan=3, sticky="we", **pad)
        ToolTip(self.go_btn,
                "Run the configured pipeline (mask -> COLMAP -> train -> open viewer). "
                "Safe to re-click: completed stages auto-skip on the next run.")

        # Log
        log_frame = ttk.Frame(self)
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", **pad)
        self.log = tk.Text(log_frame, wrap="word", height=20, bg="#1e2832", fg="#dddddd",
                           insertbackground="#ffffff", font=("Consolas", 10))
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_scroll.pack(side="right", fill="y")
        self.log["yscrollcommand"] = log_scroll.set

        self.grid_rowconfigure(9, weight=1)
        self.grid_columnconfigure(1, weight=1)

    # -- Pickers -------------------------------------------------------------
    def _pick_photos(self) -> None:
        path = filedialog.askdirectory(title="Pick photos folder")
        if not path:
            return
        self.photos_var.set(path)
        # Auto-derive output dir as sibling of photos, if user hasn't set one
        if not self.output_var.get():
            sibling = Path(path).parent / f"{Path(path).name}-output"
            self.output_var.set(str(sibling))

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Pick output folder")
        if path:
            self.output_var.set(path)

    # -- Logging from any thread --------------------------------------------
    def _log(self, msg: str) -> None:
        self._msg_q.put(msg)
        # Mirror to the per-run log file if one is open. CPython file writes
        # are atomic per line, so this is safe from the worker thread.
        if self._log_file is not None:
            try:
                self._log_file.write(msg)
                self._log_file.flush()
            except Exception:
                pass

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self._msg_q.get_nowait()
                if msg is None:  # sentinel: worker done
                    self.go_btn.config(state="normal", text="Go")
                    self._close_log_file()
                    continue
                self.log.insert("end", msg)
                self.log.see("end")
        except queue.Empty:
            pass
        self.after(50, self._drain_queue)

    def _open_log_file(self) -> None:
        """Open a per-run log file in logs/. Filename includes the timestamp and
        the photos-folder basename so runs are easy to identify after the fact."""
        try:
            import datetime
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            photos_dir = self.photos_var.get().strip()
            scene_name = Path(photos_dir).name if photos_dir else "scene"
            # Sanitize for filesystem (spaces are fine on NTFS, but strip slashes)
            safe = "".join(c if c not in r'\/:*?"<>|' else "_" for c in scene_name)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"{ts}__{safe}.log"
            self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
            # Header with run context so the file is self-describing
            tag = self.output_name_var.get().strip()
            self._log_file.write(f"# FastSplat run log\n")
            self._log_file.write(f"# Timestamp: {ts}\n")
            self._log_file.write(f"# Photos:    {photos_dir}\n")
            self._log_file.write(f"# Output:    {self.output_var.get().strip()}\n")
            self._log_file.write(f"# Tag:       {tag or '(none)'}\n")
            self._log_file.write(f"# Preset:    {self.preset_var.get()}\n")
            self._log_file.write(f"# Matcher:   {self.matcher_var.get()}\n")
            self._log_file.write(f"# Iter:      {self.iter_var.get()}\n")
            self._log_file.write(f"# Max-cap:   {self.max_cap_var.get():,}\n")
            self._log_file.write(f"# SH degree: {self.sh_degree_var.get()}\n")
            self._log_file.write(f"# Flags:     --gut={self.use_gut.get()} "
                                 f"--bilateral-grid={self.use_bilateral_grid.get()} "
                                 f"--enable-mip={self.use_mip.get()} "
                                 f"--ppisp={self.use_ppisp.get()}\n")
            self._log_file.write(f"# Mask:      subject={self.mask_subject.get()} "
                                 f"model={self.mask_model.get()} "
                                 f"sky_only={self.sky_only.get()}\n")
            self._log_file.write("# -" * 30 + "\n\n")
            self._log_file.flush()
            self._log(f"Logging to: {log_path}\n")
        except Exception as exc:  # noqa: BLE001
            self._log_file = None
            # Don't block the run if log file can't open
            self._msg_q.put(f"WARN: could not open log file: {exc!r}\n")

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    # -- Run -----------------------------------------------------------------
    def _on_go(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        photos = self.photos_var.get().strip()
        if not photos or not Path(photos).is_dir():
            messagebox.showerror("FastSplat", "Pick a valid photos folder first.")
            return
        if not self.output_var.get().strip():
            messagebox.showerror("FastSplat", "Pick or type an output folder.")
            return

        # Pre-flight VRAM warning if estimated peak is over budget
        if self.run_train.get():
            est = estimate_vram_gb(int(self.max_cap_var.get()),
                                   sh_degree=int(self.sh_degree_var.get()),
                                   viewer_running=bool(self.open_after.get()))
            if est > GPU_VRAM_GB * 0.90:
                proceed = messagebox.askyesno(
                    "FastSplat — VRAM warning",
                    f"Estimated peak VRAM is ≈ {est:.1f} GB on a "
                    f"{GPU_VRAM_GB} GB card. Training is likely to hit OOM "
                    f"during densification.\n\n"
                    "Suggested fixes:\n"
                    f"  • Lower 'Max gaussians' (currently "
                    f"{self.max_cap_var.get():,})\n"
                    "  • Uncheck 'Open viewer after' (the viewer's interop "
                    "buffer takes ~1.5 GB)\n\n"
                    "Run anyway?")
                if not proceed:
                    return

        self.log.delete("1.0", "end")
        self.go_btn.config(state="disabled", text="Working...")
        self._save_settings()  # remember this run's config even if the app crashes mid-pipeline
        self._open_log_file()
        self._worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self._worker.start()

    def _run_pipeline(self) -> None:
        try:
            photos = Path(self.photos_var.get())
            output = Path(self.output_var.get())
            # Scene dir: sibling of photos, named after the photo dir
            scene = photos.parent / f"{photos.name}-scene"

            tag = self.output_name_var.get().strip()
            self._log(f"Photos:  {photos}\n")
            self._log(f"Scene:   {scene}\n")
            self._log(f"Output:  {output}\n")
            if tag:
                self._log(f"Tag:     {tag}  (output will be {tag}.ply)\n")
            self._log("\n")

            # Sanity check binaries
            if not Path(COLMAP_EXE).exists():
                self._log(f"ERROR: COLMAP not found at {COLMAP_EXE}\n")
                return
            if not Path(LICHTFELD_EXE).exists():
                self._log(f"ERROR: LichtFeld not found at {LICHTFELD_EXE}\n")
                return

            # 1. Make scene dir + junction images/ -> photos
            scene.mkdir(parents=True, exist_ok=True)
            img_link = scene / "images"
            if not img_link.exists():
                self._log(f"Junction: {img_link} -> {photos}\n")
                rc = subprocess.call(["cmd", "/c", "mklink", "/J", str(img_link), str(photos)],
                                     creationflags=NO_WINDOW)
                if rc != 0:
                    self._log(f"ERROR: mklink failed (exit {rc})\n")
                    return
            else:
                self._log("Junction already exists, reusing.\n")

            # 1b. Optional: mask subject with rembg, write RGBA PNGs to images_masked/.
            # COLMAP + LichtFeld then read from images_masked/ instead of the junction.
            # Background RGB is replaced with neutral gray so COLMAP doesn't extract
            # features from it; alpha=0 there so LichtFeld's alpha-as-mask drops it.
            colmap_image_path = img_link
            images_subdir = "images"   # what LichtFeld's --images should be
            if self.mask_subject.get():
                masked_dir = scene / "images_masked"
                meta_path = masked_dir / ".rembg_meta.json"
                current_model = self.mask_model.get()
                current_sky_only = bool(self.sky_only.get())

                # Decide whether to invalidate the existing masked cache:
                #  - "Force re-mask" checkbox is on, OR
                #  - .rembg_meta.json records a different model or sky_only flag
                must_clear = self.force_remask.get()
                if not must_clear and meta_path.exists():
                    try:
                        prev = json.loads(meta_path.read_text())
                        prev_model = prev.get("model")
                        prev_sky_only = bool(prev.get("sky_only", False))
                        if prev_model and prev_model != current_model:
                            self._log(f"\nExisting masks were generated with '{prev_model}'; "
                                      f"clearing for re-mask with '{current_model}'.\n")
                            must_clear = True
                        elif prev_sky_only != current_sky_only:
                            self._log(f"\nSky-only mode changed (was {prev_sky_only}, now "
                                      f"{current_sky_only}); clearing for re-mask.\n")
                            must_clear = True
                    except Exception:
                        pass  # corrupt meta -> just leave the cache alone
                if must_clear and masked_dir.exists():
                    shutil.rmtree(masked_dir)
                    self._log("Cleared masked cache; COLMAP cache will refresh below.\n")

                rc = self._mask_with_rembg(img_link, masked_dir,
                                           model=current_model,
                                           sky_only=current_sky_only)
                if rc != 0:
                    self._log("ERROR: rembg masking failed; aborting.\n")
                    return
                # Stamp the cache with the settings that produced it
                try:
                    meta_path.write_text(json.dumps({
                        "model": current_model,
                        "sky_only": current_sky_only,
                    }))
                except Exception as exc:  # noqa: BLE001
                    self._log(f"WARN: could not write {meta_path.name}: {exc!r}\n")
                colmap_image_path = masked_dir
                images_subdir = "images_masked"

            # 1c. Detect pipeline-mode changes vs the last run on this scene.
            # If the active images_subdir differs from what COLMAP was built against
            # last time (masked vs unmasked, or model switch above), the COLMAP
            # database + sparse model are stale (wrong filename extensions/contents).
            # Invalidate them so SfM re-runs.
            state_path = scene / ".fastsplat_state.json"
            prev_subdir: str | None = None
            if state_path.exists():
                try:
                    prev_subdir = json.loads(state_path.read_text()).get("images_subdir")
                except Exception:
                    pass
            if prev_subdir and prev_subdir != images_subdir:
                self._log(f"\nMode changed: previous run used '{prev_subdir}', "
                          f"this run uses '{images_subdir}'. Invalidating COLMAP cache.\n")
                self._invalidate_colmap_cache(scene)

            # 1d. Belt-and-suspenders: read COLMAP's database (if present) and check
            # whether any file it references actually exists in the current images
            # dir. Handles scenes set up by older FastSplat versions (no state file)
            # or any other path where the DB is stale relative to the image dir.
            db_path = scene / "database.db"
            if db_path.exists() and (scene / "sparse").exists():
                ref_name = self._colmap_db_sample_image(db_path)
                if ref_name:
                    expected = scene / images_subdir / ref_name
                    if not expected.exists():
                        self._log(f"\nCOLMAP DB references '{ref_name}' which is not "
                                  f"present in {images_subdir}/. Invalidating COLMAP cache.\n")
                        self._invalidate_colmap_cache(scene)

            # Stamp the current mode for next run
            try:
                state_path.write_text(json.dumps({"images_subdir": images_subdir}))
            except Exception as exc:  # noqa: BLE001
                self._log(f"WARN: could not write {state_path.name}: {exc!r}\n")

            # 2. SfM — COLMAP front-end + GLOMAP mapper if available, else
            # plain COLMAP automatic_reconstructor.
            sparse = scene / "sparse" / "0"
            if self.run_colmap.get():
                if sparse.exists():
                    self._log("\nSparse model already present; skipping SfM.\n")
                else:
                    rc = self._run_sfm(scene, Path(colmap_image_path))
                    if rc != 0:
                        self._log(f"ERROR: SfM failed (exit {rc})\n")
                        return
                    if not sparse.exists():
                        self._log(f"ERROR: SfM completed but no model at {sparse}\n")
                        return

            # 3. Train
            if self.run_train.get():
                if not sparse.exists():
                    self._log(f"ERROR: cannot train, no sparse model at {sparse}\n")
                    return
                output.mkdir(parents=True, exist_ok=True)
                self._log("\n--- LichtFeld training ---\n")
                args = [LICHTFELD_EXE,
                        "-d", str(scene),
                        "-o", str(output),
                        "--images", images_subdir,
                        "--iter", str(self.iter_var.get()),
                        "--max-cap", str(self.max_cap_var.get()),
                        "--sh-degree", str(self.sh_degree_var.get()),
                        "--train",       # auto-start training on launch
                        "--no-splash"]   # skip the splash screen
                if self.use_gut.get():
                    args.append("--gut")
                if self.use_bilateral_grid.get():
                    args.append("--bilateral-grid")
                if self.use_mip.get():
                    args.append("--enable-mip")
                if self.use_ppisp.get():
                    args.append("--ppisp")
                tag = self.output_name_var.get().strip()
                if tag:
                    args.extend(["--output-name", tag])
                rc = self._stream(args, env=lichtfeld_env())
                if rc != 0:
                    self._log(f"ERROR: LichtFeld training failed (exit {rc})\n")
                    return

            # 4. Open viewer on latest .ply
            if self.open_after.get():
                tag = self.output_name_var.get().strip()
                glob_pattern = f"{tag}*.ply" if tag else "splat_*.ply"
                plys = sorted(output.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
                if plys:
                    target = plys[-1]
                    self._log(f"\nOpening viewer: {target.name}\n")
                    subprocess.Popen([LICHTFELD_EXE, "-v", str(target)],
                                     env=lichtfeld_env(), creationflags=NO_WINDOW)
                else:
                    self._log(f"\nNo {glob_pattern} in output; skipping viewer.\n")

            self._log("\nDone.\n")
        except Exception as exc:  # noqa: BLE001 - top-level UI catch
            self._log(f"\nERROR: {exc!r}\n")
        finally:
            self._msg_q.put(None)  # sentinel: re-enable Go button

    def _run_sfm(self, scene: Path, image_path: Path) -> int:
        """Run the SfM pipeline.

        Prefers COLMAP front-end (feature extraction + matching) + GLOMAP mapper
        (global SfM) when GLOMAP is available — 5-10x faster on large sets, same
        output format. Falls back to COLMAP's automatic_reconstructor if GLOMAP
        isn't installed at GLOMAP_EXE.
        """
        sparse = scene / "sparse"
        sparse.mkdir(parents=True, exist_ok=True)
        db = scene / "database.db"
        use_glomap = Path(GLOMAP_EXE).is_file()

        if not use_glomap:
            # Path A: plain COLMAP automatic_reconstructor (current default).
            self._log("\n--- SfM: COLMAP automatic_reconstructor ---\n")
            self._log("(GLOMAP not found at " + GLOMAP_EXE + " — using plain COLMAP. "
                      "Install GLOMAP for 5-10x faster SfM on large sets.)\n")
            return self._stream([
                COLMAP_EXE, "automatic_reconstructor",
                "--workspace_path", str(scene),
                "--image_path", str(image_path),
                "--sparse", "1", "--dense", "0", "--use_gpu", "1",
            ], env=colmap_env())

        # Path B: COLMAP front-end + GLOMAP mapper.
        self._log("\n--- SfM: COLMAP features + GLOMAP global mapper ---\n")

        # 2a. Feature extraction (GPU SIFT — `--FeatureExtraction.use_gpu`
        # in COLMAP 4.1+; the older `--SiftExtraction.use_gpu` was renamed.
        # GPU is the default but being explicit makes intent clear.)
        self._log("Step 1/3: feature extraction...\n")
        rc = self._stream([
            COLMAP_EXE, "feature_extractor",
            "--database_path", str(db),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "0",
            "--FeatureExtraction.use_gpu", "1",
        ], env=colmap_env())
        if rc != 0:
            self._log(f"feature_extractor failed (exit {rc})\n")
            return rc

        # 2b. Feature matching — exhaustive or sequential based on UI choice.
        # Sequential is much faster on ordered walk-throughs (±30 neighbors only).
        matcher_choice = self.matcher_var.get()
        if matcher_choice == "sequential":
            self._log("Step 2/3: feature matching (sequential, ±30 neighbors, GPU)...\n")
            rc = self._stream([
                COLMAP_EXE, "sequential_matcher",
                "--database_path", str(db),
                "--FeatureMatching.use_gpu", "1",
                "--SequentialMatching.overlap", "30",
            ], env=colmap_env())
            matcher_cmd_name = "sequential_matcher"
        else:
            self._log("Step 2/3: feature matching (exhaustive, GPU)...\n")
            rc = self._stream([
                COLMAP_EXE, "exhaustive_matcher",
                "--database_path", str(db),
                "--FeatureMatching.use_gpu", "1",
            ], env=colmap_env())
            matcher_cmd_name = "exhaustive_matcher"
        if rc != 0:
            self._log(f"{matcher_cmd_name} failed (exit {rc})\n")
            return rc

        # 2c. GLOMAP global mapper — the speedup-from-GLOMAP step.
        self._log("Step 3/3: GLOMAP global mapper...\n")
        rc = self._stream([
            GLOMAP_EXE, "mapper",
            "--database_path", str(db),
            "--image_path", str(image_path),
            "--output_path", str(sparse),
        ], env=colmap_env())
        if rc != 0:
            self._log(f"glomap mapper failed (exit {rc})\n")
        return rc

    def _invalidate_colmap_cache(self, scene: Path) -> None:
        """Delete the COLMAP database and sparse model so the next run rebuilds.
        Used when the inputs feeding COLMAP have changed (masks regenerated,
        or mask mode flipped on/off)."""
        for stale in [scene / "database.db",
                      scene / "database.db-shm",
                      scene / "database.db-wal",
                      scene / "sparse"]:
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            elif stale.exists():
                try:
                    stale.unlink()
                except Exception:
                    pass

    # -- Presets ----------------------------------------------------------
    def _apply_preset(self) -> None:
        """Snap settings to the values defined in PRESETS for the selected name."""
        name = self.preset_var.get()
        cfg = PRESETS.get(name)
        if not cfg:
            return  # "(custom)" or unknown -> leave settings alone
        # Map config keys to tk.Variable references on self
        mapping: dict[str, tk.Variable] = {
            "iter":              self.iter_var,
            "max_cap":           self.max_cap_var,
            "matcher":           self.matcher_var,
            "use_gut":           self.use_gut,
            "use_bilateral_grid": self.use_bilateral_grid,
            "use_mip":           self.use_mip,
            "use_ppisp":         self.use_ppisp,
            "sh_degree":         self.sh_degree_var,
            "mask_subject":      self.mask_subject,
            "open_after":        self.open_after,
        }
        applied = []
        for key, value in cfg.items():
            var = mapping.get(key)
            if var is None:
                continue
            try:
                var.set(value)
                applied.append(key)
            except Exception:
                pass
        if applied:
            self._log(f"Applied preset '{name}': {', '.join(applied)}\n")
        # Recompute VRAM estimate now that max_cap may have changed
        self._refresh_vram_estimate()

    # -- VRAM estimate ----------------------------------------------------
    def _refresh_vram_estimate(self) -> None:
        if self._loading_settings:
            return
        try:
            cap = int(self.max_cap_var.get())
        except (tk.TclError, ValueError):
            self.vram_label.config(text="")
            return
        est = estimate_vram_gb(cap,
                               sh_degree=int(self.sh_degree_var.get()),
                               viewer_running=bool(self.open_after.get()))
        # Color thresholds: < 70% of card = green, 70-90% = amber, > 90% = red
        ratio = est / GPU_VRAM_GB
        if ratio < 0.70:
            color = "#7fd57f"
        elif ratio < 0.90:
            color = "#d5a370"
        else:
            color = "#d57070"
        self.vram_label.config(
            text=f"≈ {est:.1f} / {GPU_VRAM_GB} GB peak", foreground=color)

    # -- Photo set judger / pre-flight ------------------------------------
    def _refresh_judger(self) -> None:
        """Kick off a background scan of the photos folder to update the judger."""
        if self._loading_settings:
            return
        path = self.photos_var.get().strip()
        if not path:
            self.judger_var.set("")
            self.judger_label.config(fg="#888888")
            return
        if not Path(path).is_dir():
            self.judger_var.set(f"⚠ '{path}' is not a folder.")
            self.judger_label.config(fg="#d57070")
            return
        # Quick stats can run on the UI thread; scanning images for EXIF
        # is where it could get slow, so push to a worker.
        threading.Thread(target=self._scan_photo_folder, args=(path,),
                         daemon=True).start()

    def _scan_photo_folder(self, path: str) -> None:
        folder = Path(path)
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".webp", ".bmp"}
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in exts)
        n = len(files)
        if n == 0:
            self.after(0, lambda: (self.judger_var.set("⚠ No images found in this folder."),
                                   self.judger_label.config(fg="#d5a370")))
            return

        total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
        # Sample up to 5 evenly-spaced photos for resolution + EXIF
        step = max(1, n // 5)
        sample = files[::step][:5]

        widths, heights = [], []
        cameras: set[str] = set()
        try:
            from PIL import Image  # type: ignore
            from PIL.ExifTags import TAGS  # type: ignore
            for p in sample:
                try:
                    with Image.open(p) as img:
                        widths.append(img.size[0])
                        heights.append(img.size[1])
                        exif = getattr(img, "_getexif", lambda: None)()
                        if exif:
                            for tag_id, val in exif.items():
                                name = TAGS.get(tag_id, "")
                                if name == "Model":
                                    cameras.add(str(val).strip("\x00 "))
                except Exception:
                    pass
        except ImportError:
            pass

        avg_mp = (sum(w * h for w, h in zip(widths, heights)) / len(widths) / 1e6
                  if widths else 0.0)
        # Build verdict lines
        lines: list[tuple[str, str]] = []  # (level, message)
        lines.append(("info", f"{n} photos · "
                              f"{f'{avg_mp:.1f} MP avg · ' if avg_mp else ''}"
                              f"{total_mb:.0f} MB total"
                              + (f" · {', '.join(sorted(cameras))}" if cameras else "")))

        # Count verdict
        if n < 20:
            lines.append(("warn", f"Photo count low ({n}). 30+ recommended; "
                                  "expect sparse / incomplete reconstruction."))
        elif n < 30:
            lines.append(("info", f"{n} photos is workable but on the lighter side. "
                                  "More coverage = sharper splat."))
        elif n > 400:
            glomap_msg = ("" if Path(GLOMAP_EXE).is_file()
                          else " Consider installing GLOMAP (see README) "
                               "to cut SfM time by 5-10x.")
            lines.append(("warn", f"{n} photos — SfM runtime will be long."
                                  + glomap_msg))
        else:
            lines.append(("ok", "Photo count looks good."))

        # Resolution verdict
        if avg_mp:
            if avg_mp < 2:
                lines.append(("warn", f"{avg_mp:.1f} MP avg — low resolution; "
                                      "splat will be visibly soft. "
                                      "Use higher-res originals if available."))
            elif avg_mp < 4:
                lines.append(("info", f"{avg_mp:.1f} MP avg — adequate but not crisp."))
            elif avg_mp > 24:
                lines.append(("info", f"{avg_mp:.1f} MP avg — LichtFeld will downsample "
                                      "to its 3840px default; pre-resizing won't hurt."))
            else:
                lines.append(("ok", f"Resolution looks good ({avg_mp:.1f} MP avg)."))

        # Mixed-orientation warning if the sample is mixed portrait/landscape
        if widths and heights:
            ratios = [w / h for w, h in zip(widths, heights)]
            portrait = sum(1 for r in ratios if r < 1)
            landscape = len(ratios) - portrait
            if portrait and landscape:
                lines.append(("info",
                              f"Mixed orientations (sample: {portrait}P + {landscape}L). "
                              "Fine for SfM; just unusual."))

        # Multi-camera warning
        if len(cameras) > 1:
            lines.append(("warn", f"Multiple cameras detected ({len(cameras)}). "
                                  "Lighting / WB may differ; consider --ppisp."))

        # Choose color: red if any warn, yellow if any info-only, green if all ok
        levels = {l for l, _ in lines}
        if "warn" in levels:
            color = "#d5a370"
        elif "ok" in levels and "warn" not in levels:
            color = "#7fd57f"
        else:
            color = "#bbbbbb"

        prefix_map = {"ok": "✓", "info": "·", "warn": "⚠"}
        text = "\n".join(f"{prefix_map.get(level, '·')} {msg}" for level, msg in lines)

        self.after(0, lambda: (self.judger_var.set(text),
                               self.judger_label.config(fg=color)))

    # -- Settings persistence ----------------------------------------------
    def _persisted_vars(self) -> dict[str, tk.Variable]:
        return {
            "photos":        self.photos_var,
            "output":        self.output_var,
            "tag":           self.output_name_var,
            "iter":          self.iter_var,
            "run_colmap":    self.run_colmap,
            "run_train":     self.run_train,
            "use_gut":       self.use_gut,
            "open_after":    self.open_after,
            "mask_subject":  self.mask_subject,
            "mask_model":    self.mask_model,
            "sky_only":      self.sky_only,
            "force_remask":  self.force_remask,
            "max_cap":       self.max_cap_var,
            "matcher":       self.matcher_var,
            "use_bilateral_grid": self.use_bilateral_grid,
            "use_mip":       self.use_mip,
            "use_ppisp":     self.use_ppisp,
            "sh_degree":     self.sh_degree_var,
        }

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text())
        except Exception:
            return
        self._loading_settings = True
        try:
            for key, var in self._persisted_vars().items():
                if key in data:
                    try:
                        var.set(data[key])
                    except Exception:
                        pass
        finally:
            self._loading_settings = False

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.get() for k, v in self._persisted_vars().items()}
            SETTINGS_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # don't crash on shutdown

    def _on_close(self) -> None:
        self._save_settings()
        self._close_log_file()
        self.destroy()

    @staticmethod
    def _colmap_db_sample_image(db_path: Path) -> str | None:
        """Return one filename from COLMAP's images table, or None if unreadable.
        Used to sanity-check whether the DB matches the current images dir."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.execute("SELECT name FROM images LIMIT 1")
                row = cur.fetchone()
            finally:
                conn.close()
            return row[0] if row else None
        except Exception:
            return None

    @staticmethod
    def _quote_for_display(args: list[str]) -> str:
        """Format a list of subprocess args for human-readable logging.

        Note: the actual subprocess.Popen call receives the list as-is — Python's
        subprocess.list2cmdline on Windows handles all the per-arg escaping. This
        function exists ONLY so the log line looks like something you could
        copy-paste into a shell, with quotes around any arg containing spaces or
        cmd-special characters."""
        special = set(' \t()&|<>^"')
        out = []
        for a in args:
            s = str(a)
            if not s or any(ch in special for ch in s):
                # double up embedded quotes the cmd way
                out.append('"' + s.replace('"', '""') + '"')
            else:
                out.append(s)
        return " ".join(out)

    def _stream(self, args: list[str], env: dict[str, str] | None = None) -> int:
        """Run a subprocess and stream stdout+stderr to the log."""
        self._log(f"$ {self._quote_for_display(args)}\n")
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, bufsize=1, encoding="utf-8",
                                errors="replace", creationflags=NO_WINDOW)
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log(line)
        proc.wait()
        return proc.returncode

    # Fraction of image height that's considered "sky" when sky_only mode is on.
    # 0.50 = top 50% is where sky-colored pixels get masked; bottom 50% is always kept.
    SKY_ONLY_TOP_FRACTION = 0.50

    @staticmethod
    def _compute_sky_mask_hsv(rgb_arr, top_fraction: float):
        """HSV-style sky detection: mask bright low-saturation OR blue pixels in
        the top region. Doesn't rely on rembg, so vertical subjects (columns,
        towers, spires) that reach the top of frame are preserved as long as
        their actual color isn't bright sky-like.

        rgb_arr: (H, W, 3) uint8 numpy array
        Returns boolean (H, W) array — True where pixel should be masked.
        """
        import numpy as np
        H = rgb_arr.shape[0]
        r = rgb_arr[..., 0].astype(np.int16)
        g = rgb_arr[..., 1].astype(np.int16)
        b = rgb_arr[..., 2].astype(np.int16)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat_proxy = max_c - min_c                    # 0-255, higher = more saturated
        brightness = max_c                            # HSV V channel approx

        # Sky if: bright AND (low-saturated OR clearly blue-dominant)
        is_bright = brightness > 160
        is_low_sat = sat_proxy < 50
        is_blue = (b > r + 10) & (b > g + 10) & (b > 100)
        sky_color = is_bright & (is_low_sat | is_blue)

        # Restrict to the top region (subject below the horizon line stays
        # regardless of color)
        region = np.zeros_like(sky_color, dtype=bool)
        horizon = int(H * top_fraction)
        region[:horizon] = True
        return sky_color & region

    def _mask_with_rembg(self, in_dir: Path, out_dir: Path, model: str,
                         sky_only: bool = False) -> int:
        """Batch-mask photos in in_dir, write subject-only RGBA PNGs to out_dir.

        Background pixels get RGB = (128, 128, 128) and alpha = 0 — gray so COLMAP
        finds no features there, alpha=0 so LichtFeld's alpha-as-mask ignores them.

        If sky_only=True, this method **skips rembg entirely** and uses a color-
        based sky filter (bright + low-saturation, OR clearly blue, in the top
        SKY_ONLY_TOP_FRACTION of the frame). This correctly preserves vertical
        subjects (columns, towers, spires) that reach the top of the frame —
        rembg's subject segmentation often fails for those compositions.

        Skips images already present in out_dir (safe to re-run).
        """
        try:
            from PIL import Image  # type: ignore
            import numpy as np
        except ImportError:
            self._log("ERROR: Pillow and numpy required. pip install Pillow numpy\n")
            return 1

        out_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        images = sorted(p for p in in_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in exts)
        if not images:
            self._log(f"ERROR: no images found in {in_dir}\n")
            return 1

        if sky_only:
            return self._mask_sky_hsv(out_dir, images)

        # Full-subject path: rembg-based segmentation.
        try:
            from rembg import new_session, remove  # type: ignore
        except ImportError:
            self._log("ERROR: rembg not installed.\n")
            self._log('  Install with:  pip install "rembg[gpu]"   (or `rembg` for CPU)\n')
            return 1

        self._log(f"\n--- rembg masking ({model}, full subject) ---\n")
        self._log(f"Loading model... (first run will download ~180 MB)\n")
        import time as _time
        t0 = _time.time()
        session = new_session(model)
        self._log(f"Model loaded in {_time.time() - t0:.1f}s\n")

        done = 0
        skipped = 0
        t_batch = _time.time()
        for i, img_path in enumerate(images, 1):
            out_path = out_dir / (img_path.stem + ".png")
            if out_path.exists():
                skipped += 1
                continue
            with Image.open(img_path) as src:
                src_rgb = src.convert("RGB")
                rgba = remove(src_rgb, session=session)  # type: ignore[arg-type]
            # Replace background RGB with mid-gray (so COLMAP sees no features there)
            arr = np.array(rgba.convert("RGBA"))
            bg_mask = arr[..., 3] < 128
            arr[bg_mask] = (128, 128, 128, 0)
            Image.fromarray(arr, mode="RGBA").save(out_path, optimize=False)
            done += 1
            if i % 5 == 0 or i == len(images):
                elapsed = _time.time() - t_batch
                rate = done / elapsed if elapsed > 0 and done > 0 else 0
                remaining = (len(images) - i) / rate if rate > 0 else 0
                self._log(f"  {i}/{len(images)}  "
                          f"({rate:.1f} img/s, eta {remaining:.0f}s)\n")
        total = _time.time() - t_batch
        self._log(f"Masking done: {done} new, {skipped} reused, {total:.1f}s total\n")
        return 0

    def _mask_sky_hsv(self, out_dir: Path, images: list) -> int:
        """Sky-only masking via color thresholds. No rembg involved.
        Much faster than rembg AND robust to "subject extends to top of frame"
        cases that confuse subject-segmentation models."""
        try:
            from PIL import Image  # type: ignore
            import numpy as np
        except ImportError:
            self._log("ERROR: Pillow and numpy required.\n")
            return 1

        pct = int(self.SKY_ONLY_TOP_FRACTION * 100)
        self._log(f"\n--- HSV sky masking (top {pct}% of each frame) ---\n")
        self._log("No model needed — color-based; rembg dropdown is ignored.\n")

        import time as _time
        t_batch = _time.time()
        done = skipped = 0
        for i, img_path in enumerate(images, 1):
            out_path = out_dir / (img_path.stem + ".png")
            if out_path.exists():
                skipped += 1
                continue
            with Image.open(img_path) as src:
                src_rgb = src.convert("RGB")
            src_arr = np.asarray(src_rgb, dtype=np.uint8)
            H, W = src_arr.shape[:2]

            sky_mask = self._compute_sky_mask_hsv(src_arr, self.SKY_ONLY_TOP_FRACTION)

            # Build RGBA output: original everywhere, sky-detected pixels set to
            # neutral gray with alpha=0
            out_arr = np.empty((H, W, 4), dtype=np.uint8)
            out_arr[..., :3] = src_arr
            out_arr[..., 3] = 255
            out_arr[sky_mask] = (128, 128, 128, 0)

            Image.fromarray(out_arr, mode="RGBA").save(out_path, optimize=False)
            done += 1
            if i % 5 == 0 or i == len(images):
                elapsed = _time.time() - t_batch
                rate = done / elapsed if elapsed > 0 and done > 0 else 0
                remaining = (len(images) - i) / rate if rate > 0 else 0
                self._log(f"  {i}/{len(images)}  "
                          f"({rate:.1f} img/s, eta {remaining:.0f}s)\n")
        total = _time.time() - t_batch
        self._log(f"Sky masking done: {done} new, {skipped} reused, {total:.1f}s total\n")
        return 0

    # -- Mask preview (sample 1/10) ----------------------------------------
    def _on_preview_sample(self) -> None:
        """Mask a ~1/10 sample of the input photos, then pop up a review grid."""
        photos = self.photos_var.get().strip()
        if not photos or not Path(photos).is_dir():
            messagebox.showerror("FastSplat", "Pick a valid photos folder first.")
            return
        self.preview_btn.config(state="disabled", text="Sampling...")
        threading.Thread(target=self._run_preview_sample, daemon=True).start()

    def _run_preview_sample(self) -> None:
        try:
            from rembg import new_session, remove  # type: ignore
            from PIL import Image  # type: ignore
            import numpy as np
        except ImportError as e:
            self._log(f"ERROR: missing dep ({e}). Install: pip install rembg Pillow numpy\n")
            self.after(0, lambda: self.preview_btn.config(state="normal", text="Preview sample"))
            return

        photos_dir = Path(self.photos_var.get())
        model = self.mask_model.get()
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        images = sorted(p for p in photos_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in exts)
        if not images:
            self._log("No images found.\n")
            self.after(0, lambda: self.preview_btn.config(state="normal", text="Preview sample"))
            return

        # Sample ~1/10 of the set, capped between 3 and 12 for sanity.
        sample_count = max(3, min(12, len(images) // 10 or 1))
        step = max(1, len(images) // sample_count)
        sampled = images[::step][:sample_count]

        sky_only = bool(self.sky_only.get())
        mode_desc = "sky-only (HSV)" if sky_only else f"full subject ({model})"
        self._log(f"\n--- Mask preview ({mode_desc}) ---\n")
        self._log(f"Sampling {len(sampled)}/{len(images)} photos...\n")

        import time as _t
        t0 = _t.time()
        session = None
        if not sky_only:
            try:
                session = new_session(model)
            except Exception as exc:  # noqa: BLE001
                self._log(f"ERROR loading model: {exc!r}\n")
                self.after(0, lambda: self.preview_btn.config(state="normal", text="Preview sample"))
                return
            self._log(f"Model loaded in {_t.time() - t0:.1f}s\n")

        pairs = []  # (filename, orig_thumb, cutout_thumb)
        thumb_size = (340, 260)
        for i, p in enumerate(sampled, 1):
            with Image.open(p) as src:
                src_rgb = src.convert("RGB")
            if sky_only:
                # Color-based sky filter (same code path the full pipeline uses)
                src_arr = np.asarray(src_rgb, dtype=np.uint8)
                H, W = src_arr.shape[:2]
                sky_mask = self._compute_sky_mask_hsv(src_arr, self.SKY_ONLY_TOP_FRACTION)
                arr = np.empty((H, W, 4), dtype=np.uint8)
                arr[..., :3] = src_arr
                arr[..., 3] = 255
                arr[sky_mask] = (128, 128, 128, 0)
                rgba = Image.fromarray(arr, mode="RGBA")
            else:
                rgba = remove(src_rgb, session=session)  # type: ignore[arg-type]
            # Composite the cutout over magenta so transparent areas are unmistakable
            magenta = Image.new("RGB", rgba.size, (255, 0, 255))
            magenta.paste(rgba, (0, 0), rgba if rgba.mode == "RGBA" else None)
            # Make thumbnails
            orig_t = src_rgb.copy()
            orig_t.thumbnail(thumb_size)
            cut_t = magenta
            cut_t.thumbnail(thumb_size)
            # Coverage stat
            arr = np.array(rgba.convert("RGBA"))[..., 3]
            cov = (arr > 128).mean() * 100
            pairs.append((p.name, orig_t, cut_t, cov))
            self._log(f"  {i}/{len(sampled)}  {p.name}  ({cov:.1f}% subject)\n")

        elapsed = _t.time() - t0
        self._log(f"Sample done in {elapsed:.1f}s. Opening preview window...\n")
        self.after(0, lambda: self._show_preview_window(pairs, model))

    def _show_preview_window(self, pairs: list, model: str) -> None:
        try:
            from PIL import ImageTk  # type: ignore
        except ImportError:
            self._log("ERROR: Pillow ImageTk not available.\n")
            self.preview_btn.config(state="normal", text="Preview sample")
            return

        win = tk.Toplevel(self)
        win.title(f"Mask preview — {model} — {len(pairs)} samples")
        win.geometry("780x720")
        win.configure(bg="#1e2832")

        header = ttk.Frame(win)
        header.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(header,
                  text="Magenta = background (will be excluded from training).  "
                       "Verify edges are tight and the base is included.").pack(side="left")
        ttk.Button(header, text="Close", command=win.destroy).pack(side="right")

        # Scrollable canvas hosting the grid
        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        canvas = tk.Canvas(canvas_frame, bg="#1e2832", highlightthickness=0)
        scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        inner = tk.Frame(canvas, bg="#1e2832")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        photo_refs = []  # keep PhotoImage refs alive
        for name, orig, cutout, cov in pairs:
            row = tk.Frame(inner, bg="#1e2832")
            row.pack(fill="x", pady=6)
            tk.Label(row, text=f"{name}  —  subject {cov:.1f}% of frame",
                     bg="#1e2832", fg="#dddddd",
                     font=("Consolas", 9)).pack(anchor="w")
            img_row = tk.Frame(row, bg="#1e2832")
            img_row.pack(anchor="w")
            o_ph = ImageTk.PhotoImage(orig)
            c_ph = ImageTk.PhotoImage(cutout)
            photo_refs.extend([o_ph, c_ph])
            tk.Label(img_row, image=o_ph, bg="#1e2832").pack(side="left", padx=4)
            tk.Label(img_row, image=c_ph, bg="#1e2832").pack(side="left", padx=4)

        # Bind mousewheel for natural scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        win.protocol("WM_DELETE_WINDOW",
                     lambda: (canvas.unbind_all("<MouseWheel>"), win.destroy()))

        win.photo_refs = photo_refs  # type: ignore[attr-defined]  # anti-GC
        inner.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        self.preview_btn.config(state="normal", text="Preview sample")


if __name__ == "__main__":
    Launcher().mainloop()
