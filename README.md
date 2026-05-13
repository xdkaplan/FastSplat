# FastSplat

A small Windows GUI that runs the photos → 3D Gaussian Splat pipeline end-to-end. Pick a photo folder, click Go, get a splat. Calls **COLMAP** to do Structure-from-Motion, then **LichtFeld Studio** to train the splat, then opens the viewer on the result.

Single Python file, stdlib only, tkinter for the UI. Edit the path constants at the top of `FastSplat.pyw` if you move things.

## Expected environment

This is built for a specific Windows machine setup. Other configurations will need the paths in `FastSplat.pyw` adjusted.

### Hardware

- **NVIDIA GPU**, Turing (SM 75) or newer. Tested on a Quadro RTX 5000.
- A recent NVIDIA driver. CUDA 13.2 and COLMAP 4.1's CUDA build both work fine on driver 595.79.
- Reasonable VRAM. 8 GB is the practical floor; 16 GB+ is comfortable. Large scenes push past 16 GB.
- ~10 GB free disk for COLMAP intermediates + LichtFeld output per scene.

### Software

| Tool | Expected location | Notes |
|------|-------------------|-------|
| **Python 3.11+** | on PATH (`python`, `pythonw`) | stdlib only — no `pip install` needed |
| **COLMAP 4.1+ (CUDA build)** | `C:\Repos\colmap\colmap\bin\colmap.exe` | The bundled `colmap.bat` wrapper has a `cmd` parsing bug that fails on paths with spaces — FastSplat calls `bin\colmap.exe` directly and recreates the env (PATH, QT_PLUGIN_PATH) itself |
| **GLOMAP (optional, recommended)** | `C:\Repos\colmap\glomap\bin\glomap.exe` | Drop-in faster replacement for COLMAP's mapper step (global SfM vs incremental). 5-10× faster on large photo sets, same output format. FastSplat auto-detects and uses if present, falls back to plain COLMAP otherwise. Install: grab the Windows zip from [github.com/colmap/glomap/releases](https://github.com/colmap/glomap/releases) and extract to `C:\Repos\colmap\glomap\` |
| **LichtFeld Studio (built from source)** | `C:\Repos\MrNeRF\LichtFeld-Studio\build\LichtFeld-Studio.exe` | Built with vcpkg deps next to it |
| **vcpkg-installed DLLs** | `C:\Repos\MrNeRF\LichtFeld-Studio\build\vcpkg_installed\x64-windows\bin\` | LichtFeld's runtime depends on these; PATH-prepended automatically |

If any of those paths differ on your machine, edit the `*_ROOT` / `*_EXE` / `LICHTFELD_DLL_PATHS` constants near the top of `FastSplat.pyw`. Nothing else needs to change.

### Highly recommended: install GLOMAP

For any photo set above ~100 photos, COLMAP's default incremental SfM becomes a bottleneck — feature matching on 300+ photos can take 1-2 hours. **GLOMAP** is a drop-in replacement for COLMAP's mapper step that uses global SfM instead, **5-10× faster on large sets with no quality loss**. Same output format, so LichtFeld reads it unchanged.

FastSplat auto-detects GLOMAP at `C:\Repos\colmap\glomap\bin\glomap.exe`. If present → uses it. If absent → falls back to plain COLMAP automatically (no errors, no config change needed).

**To install:**

1. Go to https://github.com/colmap/glomap/releases
2. Download the latest **Windows + CUDA** zip (e.g. `glomap-1.x.x-windows-cuda.zip`). Works alongside CUDA 13 even if built against 12.x — the binary bundles its own runtime DLLs.
3. Extract so the layout is:
   ```
   C:\Repos\colmap\glomap\
   └── bin\
       ├── glomap.exe          ← FastSplat looks here
       └── (DLLs)
   ```
4. Next FastSplat run will log `SfM: COLMAP features + GLOMAP global mapper` instead of `automatic_reconstructor`. If something goes wrong with a specific scene, rename `glomap.exe` and FastSplat reverts to plain COLMAP automatically.

**Time difference, real numbers from a 371-photo Pixel 6a interior:**

| Step | COLMAP only | COLMAP + GLOMAP |
|---|---|---|
| Feature extraction | ~5 min | ~5 min |
| Feature matching | ~75 min (exhaustive, 137k pairs) | ~15 min (GPU exhaustive) |
| Mapper | ~30 min (incremental) | ~5 min (global) |
| **Total SfM** | **~110 min** | **~25 min** |

Training time after SfM is unchanged. So GLOMAP cuts roughly an hour off every multi-room or large-subject run.

### Input photos

- 30–200 JPEG/PNG photos in a single folder, no subfolders.
- High overlap (~70%), shot in an orbit around the subject.
- Constant exposure if possible (manual mode helps).
- 4–8 MP per image is ideal; lower resolutions work but reconstructions get softer.

## Usage

### Double-click

Open `FastSplat.pyw` in Explorer. The window opens; pick photos; click Go.

**First-launch heads-up**: if Python was installed via the Microsoft Store, `.pyw` may not be associated with `pythonw.exe` by default. Right-click `FastSplat.pyw` → Open with → Choose another app → browse to `pythonw.exe` → check "Always use". After that, double-clicks just work.

### From a shell

```powershell
pythonw C:\Repos\xdkaplan\FastSplat\FastSplat.pyw
```

### Pinning to taskbar (recommended)

Right-click `FastSplat.pyw` → **Send to → Desktop (create shortcut)** → drag the shortcut to your taskbar or Start. The shortcut bypasses any file-association weirdness and lets you set a custom icon.

## What the pipeline does

For a photos folder at `<parent>\<name>`:

1. **Junction images**: creates `<parent>\<name>-scene\images\` as an NTFS junction pointing to the photos folder, so COLMAP can read them without copying.
2. **COLMAP**: runs `colmap automatic_reconstructor --use_gpu 1 --sparse 1 --dense 0`. Output lands at `<parent>\<name>-scene\sparse\0\`. Skipped if a sparse model already exists (re-running is safe and cheap).
3. **LichtFeld training**: calls `LichtFeld-Studio.exe -d <scene> -o <output> --iter N [--gut]`. The DLL search path is set automatically for the child process — no need to do the `$env:PATH = ...` dance yourself.
4. **Viewer**: opens LichtFeld on the latest `splat_*.ply` in the output dir.

The output folder defaults to `<parent>\<name>-output\` (sibling of the photos folder) but is editable in the UI.

## UI options

| Control | Default | What it does |
|---------|---------|--------------|
| Output tag/name | _(blank)_ | If set, the final .ply is named `<tag>.ply` instead of `splat_<iter>.ply`. Useful for stamping runs like `monument-pixel-v1`, `monument-dslr-masked`. Lets multiple runs coexist in one output folder without overwrite. |
| Iterations | 7000 | Training iterations. 7k = quick smoke test (~5–15 min on a Quadro RTX 5000). 30k = production quality. |
| COLMAP | on | Run COLMAP. Turn off to skip if you've already done it for this scene. |
| Train | on | Run LichtFeld training. Turn off to only do the COLMAP step. |
| --gut | on | Pass `--gut` to LichtFeld. Safe default — handles distorted-camera datasets (COLMAP's `SIMPLE_RADIAL` model). Turn off only if you've explicitly forced a pinhole model. |
| Open viewer after | on | Launch LichtFeld in viewer mode (`-v`) on the latest .ply after training completes. |
| Max gaussians | 1,000,000 | Ceiling on densification. Higher = sharper splat potential but more VRAM. The estimate label next to the field updates live: green = safe, amber = borderline, red = likely OOM. Pre-flight dialog blocks Go if you're in the red zone. For 16 GB cards: 1M is safe, 2M is borderline, 3M+ risks OOM. Tune `GPU_VRAM_GB` at the top of `FastSplat.pyw` for other cards. |
| Mask subject (rembg) | off | Run [rembg](https://github.com/danielgatis/rembg) on every input photo to extract the subject before COLMAP. Background pixels become alpha=0 + RGB neutral gray, so COLMAP finds no features in the background and LichtFeld's alpha-as-mask drops those pixels from the loss. Eliminates the 100m halo of distant gaussians for outdoor subjects. |
| Model (rembg) | `u2net` | Which rembg model to use. `u2net` (default, fast) is good for clear figure/ground scenes; `birefnet-general` is sharper for complex backgrounds but ~5× slower. Smoke-test on one photo with `rembg_smoke.py` before picking. |

## Output layout

```
E:\path\to\photos              <- you point FastSplat here
E:\path\to\photos-scene\       <- created by FastSplat
  images\                      <- junction back to the photos folder
  sparse\0\                    <- COLMAP output: cameras.bin, images.bin, points3D.bin
  database.db                  <- COLMAP feature database
  colmap.log                   <- COLMAP stdout (handy if something fails)
E:\path\to\photos-output\      <- created by FastSplat
  splat_<iter>.ply             <- one or more training checkpoints; latest is the final
```

## After training: cropping and sharing

FastSplat stops once the viewer is open with the trained splat. The rest is manual GUI work:

1. **Crop** floaters and background junk inside the LichtFeld viewer (Select tool, marquee-drag, Delete).
2. **Export PLY** from LichtFeld's File menu.
3. **(Optional) Adjust exposure / color** in [SuperSplat](https://superspl.at/editor) — LichtFeld's render-time exposure controls don't bake into export. SuperSplat's do.
4. **Export HTML viewer** from LichtFeld for sharing — single `.html` with the splat embedded.
5. **Host on [Netlify Drop](https://app.netlify.com/drop)** — drag the `.html`, get a shareable URL.

## Troubleshooting

- **"COLMAP not found at …"** — fix the `COLMAP_ROOT` and `COLMAP_EXE` constants in `FastSplat.pyw`.
- **"LichtFeld not found at …"** — fix `LICHTFELD_EXE` similarly.
- **LichtFeld crashes immediately on launch with a missing-DLL dialog** — the `LICHTFELD_DLL_PATHS` list is wrong; should point at LichtFeld's `build\` and `build\vcpkg_installed\x64-windows\bin\` directories.
- **COLMAP registers very few images** — input photos likely too low-res, too dark, or with insufficient overlap. Check `colmap.log` for the per-image SIFT feature counts; under ~500 features means trouble.
- **LichtFeld trains but the splat is dark / wrong colors** — render-time exposure won't fix this; bake brightness in SuperSplat (see "After training" above).

## Subject masking with rembg

Optional preprocessing step that solves the **outdoor halo problem** in 3DGS: when you photograph a monument or statue with sky/distant scenery behind it, COLMAP triangulates the distant features at "approximately infinity," and 3DGS dutifully spawns gaussians out to ~100 m radius to fit them. The subject ends up surrounded by a cloud of fuzzy garbage gaussians.

### Setup (one-time)

```powershell
pip install "rembg[gpu]"          # uses your CUDA 13.2; ~25s/74 photos
# or: pip install rembg            # CPU-only fallback; ~5-8 min/74 photos
```

### Preview the mask quality before committing

In FastSplat, pick your photos folder, choose a model, then click **Preview sample**. It masks ~1/10 of your photos (3–12 samples) with the chosen model and pops up a window showing each original + its cutout side by side. Magenta = excluded background.

Verify across the sample:
- Subject edges are tight, not blobby
- The base / plinth of the subject is included (not chopped off)
- Sky / distant background is fully magenta
- Variation across the set is acceptable (no photos with chunks missing)

If `u2net` looks bad, switch the **Model** dropdown to `birefnet-general` (higher quality, ~5× slower) and click Preview sample again. Repeat until happy, then check **Mask subject** + click Go for the full pipeline.

### Run the full pipeline with masking

In FastSplat, check **Mask subject (rembg)** before clicking Go. Pipeline becomes:

```
photos -> rembg masking (RGBA PNGs in images_masked/) -> COLMAP -> LichtFeld
```

Masked PNGs go in `<scene>\images_masked\`, alongside a `.rembg_meta.json` file recording which model produced them. Subsequent re-runs skip already-masked images (safe to interrupt and resume). Background RGB is replaced with neutral gray and alpha=0, so:
- COLMAP sees no features in the background → no halo triangulation
- LichtFeld's default alpha-as-mask drops background pixels from training loss → no halo gaussians

### Iterating on the model choice

If you change the **Model** dropdown to a different value and click Go again, FastSplat auto-detects the mismatch (via `.rembg_meta.json`), clears the old masked PNGs, **and** invalidates the COLMAP database + sparse model (since they were built from stale features). The next run re-masks with the new model and re-runs COLMAP. No manual cleanup needed.

If you want to re-mask with the **same** model (e.g. you suspect cache corruption or interrupted a prior run), check **Force re-mask** before clicking Go. Same effect: clears masks + COLMAP cache, starts fresh.

### When to use it

- **Use** for outdoor subjects with empty sky / distant landscape behind them (monuments, statues, vehicles in open lots)
- **Use** for indoor objects on a uniform background (turntable subject capture)
- **Skip** for full scene captures where everything is part of the splat (rooms, factories, landscapes)
- **Skip** if you want a splat *with* the surroundings (e.g., monument in context with its plaza)

### Sky-only mode — color-based, model-free

Full rembg masking is aggressive: it kills the ground / plinth / nearby context along with the sky, and often fails on tall narrow subjects that don't read as "the main thing" in the frame (columns, statue tops). **Sky-only mode** sidesteps both problems by ignoring rembg entirely and using **HSV color thresholds** to find sky pixels directly:

A pixel is masked if **(bright AND low-saturation) OR (clearly blue)**, restricted to the top half of the frame.

That means:
- ✓ A dark gray stone column reaching to the top of frame is **kept** (it's not bright sky-colored)
- ✓ A white overcast sky is **removed** (bright + low-saturation)
- ✓ A blue sky is **removed** (blue-dominant pixels)
- ✓ Trees, ground, building façades below the horizon are always **kept**

The **Model** dropdown is ignored when Sky-only is on. No rembg session, no model download, no GPU inference — it's all per-pixel arithmetic via numpy. **Much faster than rembg**: ~30-60 sec for a full 74-photo batch on CPU, no DirectML needed.

Best for:
- Outdoor monuments / statues with sky behind them
- Vertical subjects that confuse rembg's subject segmentation
- Any time rembg full-mask was "removing too much"

Less useful for:
- Indoor scenes (no sky to mask)
- Scenes where bright building features dominate the top of frame (they may get incorrectly classified as sky)
- Scenes shot at sunset/dawn where sky color is unusual

The 50% horizon line and the brightness/saturation thresholds are constants (`SKY_ONLY_TOP_FRACTION`, plus inline thresholds in `_compute_sky_mask_hsv`) — edit if your typical shots need a different bias.

## Presets

The **Preset** dropdown at the top of the window snaps every training flag to a curated configuration for a specific kind of shoot. Doesn't touch your photos / output / tag fields, just the flags. Selecting `(custom)` leaves settings alone.

Built-in presets (defined in `PRESETS` at the top of `FastSplat.pyw` — add your own there):

| Preset | What it sets |
|---|---|
| **Draft (fastest, ~2-5 min)** | 3k iter, max-cap 400k, SH 1, sequential matcher, all heavy flags off |
| **Quick test (7k iter)** | 7k iter, max-cap 800k, SH 2, sequential matcher, heavy flags off |
| **Pixel 6a multi-room** | 20k iter, max-cap 1.2M, SH 2, **sequential matcher**, bilateral-grid on, mip on, **ppisp off** (locked-exposure shoots across rooms with different ambient lighting — PPISP would over-correct real signal) |
| **Outdoor monument (DSLR, manual)** | 30k iter, max-cap 1.5M, SH 3, **exhaustive matcher** (orbit captures aren't strictly ordered), bilateral-grid on, mip on, ppisp off |

## Matchers

The **Matcher** dropdown next to the SfM checkbox controls how COLMAP pairs photos for feature matching:

| Matcher | Behavior | When to use |
|---|---|---|
| **`exhaustive`** | Every photo against every other photo. O(N²) pairs. | Orbit captures, photos in arbitrary order, small sets (<150 photos) |
| **`sequential`** | Each photo against its ±30 neighbors in capture order. Linear in N. | Walk-through captures, video frame extracts, any case where photos are taken in a natural sequence |

For ordered walk-throughs of 200+ photos, **sequential is 5-20× faster** with no quality loss — the matcher is only avoiding distant pairs that wouldn't have matched anyway (room A photos vs room D photos far away in capture). For unordered captures (e.g. orbiting a monument with random gaps), use exhaustive.

To add your own, edit the `PRESETS` dict near the top of `FastSplat.pyw`. Keys map directly to flag names (`use_gut`, `use_bilateral_grid`, etc. — see the dict for the full set).

## Advanced training flags

Beyond the basic options, FastSplat now exposes LichtFeld's quality knobs:

| Flag | What it does | When to enable |
|---|---|---|
| **SH degree** (0-3) | Spherical harmonic degree for view-dependent color | 2 = default, 3 for rich lighting variation, 0-1 for memory-constrained |
| **`--bilateral-grid`** | Per-region color refinement | Usually on; especially helps multi-room scenes with varied lighting |
| **`--enable-mip`** | Anti-aliasing / mip filter | Usually on; negligible cost |
| **`--ppisp`** | Per-camera appearance correction | When auto-WB / auto-exposure drifted between shots. **Off** when you locked exposure across deliberately-different lighting — would erase real signal. |

## Progress bar

A live progress bar + status line sits between the Go button and the log pane. It parses log output in real time to detect the current pipeline stage and intra-stage progress.

Currently understood patterns:
- **Masking** — `image N/M`
- **Feature extraction** — `Processed file [N/M]`
- **Feature matching (exhaustive)** — `Processing block [I/8, J/8]`
- **COLMAP mapper** — `num_reg_frames=N` against total photo count
- **LichtFeld training** — `iteration N` against total iter from settings

Stages with unknown log formats (sequential matching, GLOMAP mapper output, some training paths) show "in progress" without a percentage until I see real logs to parse. The bar still advances correctly when those stages complete — they just don't have intra-stage granularity yet.

ETA (next phase): in-stage extrapolation from observed rate. Coming once I have a few full log files in `logs/` to calibrate against.

## Per-run logs

Every Go click writes a complete log of the run to `logs/<timestamp>__<scene-name>.log` next to the script. The file mirrors what the in-app log pane shows, with a header capturing the run's config (preset, matcher, iter, max-cap, all the flags).

Useful for:
- Debugging a run that failed hours ago without remembering exactly what setting you picked
- Sharing a log when asking for help — the header tells someone else exactly how you ran it
- Building a progress bar (parse the per-stage progress lines after the fact)

The `logs/` folder is gitignored — no risk of committing.

## QOL features

- **Settings persistence**: Your last-used folders, tag, iter count, and all checkboxes are saved to `%APPDATA%\FastSplat\settings.json` on app close and on every Go click. Next launch reopens with the same config.
- **Tooltips**: Hover any field, checkbox, or button for a sentence or two of context. Especially helpful for the cryptic flags (`--gut`, model dropdown choices).
- **Judger / pre-flight**: When you pick a photos folder, FastSplat scans the set and shows a verdict line right below — photo count, average MP, camera model from EXIF, and warnings like "low resolution" / "multiple cameras detected" / "too few photos." Colored:
  - ✓ green = good shape
  - · gray = informational
  - ⚠ amber = something worth knowing before clicking Go

## Why this exists

Without FastSplat, every new scene was: open admin PowerShell, run Enter-VsDevShell, prepend PATH manually, mkdir scene, mklink junction, type the COLMAP invocation, watch the log, type the LichtFeld invocation with `-d -o --iter --gut`, then relaunch in `-v` mode. ~12 commands to babysit. FastSplat collapses that to "pick folder, click Go."
