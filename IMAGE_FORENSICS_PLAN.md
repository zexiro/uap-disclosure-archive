# Image Forensics — Feature Plan

The corpus is ~140 stills (`raw/images/*.jpg`). Every analysis below either runs
in the visitor's browser on the already-loaded image, or runs once at ingest
and is served as a static JSON/PNG sidecar. Server-side cost stays at zero;
end-user cost stays at zero.

Framing principle: every tool exposes how it works and lets the visitor make
their own call. No opaque "real / fake" verdicts — those age badly and invite
mockery. We ship math, not judgement.

## Wave 1 — client-side forensics toolbar (lightbox)

Each filter runs on a Canvas2D pipeline inside the existing lightbox. No
network, no model download, no build step. Toggle on/off per filter.

| Feature                 | What it does                                                                 | Notes                                                       |
| ----------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Invert / negative       | Per-pixel `255 - v`                                                          | Reveals detail in deep shadows; pseudo-IR feel              |
| Channel split           | Show R / G / B in isolation                                                  | Overexposed sky photos often have all signal in one channel |
| CLAHE stretch           | Contrast-limited adaptive histogram equalization (tile-based)                | Astronomy-style "pull faint detail out"                     |
| Edge overlay            | Sobel gradient magnitude, transparent red overlay                            | Outlines geometry against blown-out backgrounds             |
| Unsharp mask            | Gaussian blur subtraction, no hallucination                                  | Slider for strength                                         |
| Sigma anomaly heatmap   | Pixels >Nσ from local mean rendered as heatmap                               | Highlights real bright objects vs sensor noise              |
| FFT magnitude view      | 2D log-magnitude spectrum, centered DC                                       | Surfaces scan lines, half-tone screens, double-compression  |
| ELA (Error Level)       | Re-encode to JPEG q=90 client-side, diff, amplify                            | Classic image-forensics splice detector                     |

Implementation:
- Add a forensics toolbar to `.lb-toolbar` in `ui/index.html` (lightbox header)
- Add `<canvas id="lb-canvas">` as a sibling of `#lb-img`; toggle visibility
- Filters compose: apply in order of toolbar selection, "reset" returns to source
- Each filter is a pure function `(ImageData) → ImageData`, no shared state

Acceptance: every filter works on the existing 139 JPEGs, applies in <500ms on
a mid-range laptop, keyboard shortcuts don't collide with existing lightbox
bindings (zoom/rotate/navigate).

## Wave 2 — build-time metadata sidecars

One-shot extraction at ingest, stored as `ui/image_forensics.json` keyed by
filename. Surfaced in the lightbox toolbar as a "Metadata" pane.

| Feature                 | What it does                                                                 | Notes                                                       |
| ----------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------- |
| EXIF dump               | Camera, software, GPS, timestamps, software signatures                       | `piexif` or `pyexiv2`; <1s for the corpus                   |
| C2PA manifest reader    | Read tamper-evident provenance if present                                    | Mostly modern Adobe / iPhone / Microsoft images             |
| JPEG generation hint    | Detect double-quantization via DCT histogram                                 | Flags "this has been re-saved many times"                   |
| File hashes             | SHA-256 + pHash, already partly present                                      | Consolidate into the sidecar                                |
| Sun/moon position       | Given EXIF GPS+time, compute solar/lunar altitude/azimuth                    | Resolves a lot of "mysterious light" cases; `skyfield`      |

Implementation:
- New script `scripts/forensics.py` — iterates `raw/images/`, writes
  `ui/image_forensics.json`
- Wire into the existing Makefile / pipeline alongside `build_thumbs.py`
- Lightbox loads the JSON lazily on first forensic toolbar open

## Wave 3 — model-based (each is a separable PR, build-time only)

These add real value but each requires a one-time model download
(~6–100 MB onnx). They run on the same nightly pipeline as OCR.

| Feature                 | Model                                                       | Cost                                |
| ----------------------- | ----------------------------------------------------------- | ----------------------------------- |
| Real-ESRGAN 4× enhance  | `realesr-general-x4v3` ONNX (~17 MB)                        | Stored as `raw/images_enhanced/*`   |
| Depth-Anything map      | `depth-anything-small` ONNX (~80 MB)                        | Stored as `raw/images_depth/*`      |
| YOLO object boxes       | `yolov8n` ONNX (~6 MB)                                      | Boxes JSON, drawn as overlay        |
| Foreground cutout       | u2net (already in stack via hyperframes-media)              | Transparent PNG sibling             |
| AI-generated score      | `Organika/sdxl-detector` ONNX (~100 MB)                     | Score with confidence band, NOT verdict |

Framing: enhanced + depth are labelled as plausible reconstructions, never as
new evidence. The AI score ships with model name + training date so visitors
can judge it themselves.

## Wave 4 — cross-corpus (later)

- **Redaction-diff between tranches** — pixel-align same-document images across
  releases, surface blacked-out boxes that got unredacted. Highest editorial
  impact when Release 2 drops.
- **Wikimedia Commons reverse pHash lookup** — catches recycled stock images.
  Heavy lift to build the index, near-zero to query.

## Delivery order

1. Wave 1 ships first — pure client-side, no pipeline changes, lands in one PR.
2. Wave 2 ships next — touches the build pipeline but stays small.
3. Wave 3 ships per-feature, each as its own PR with its own model download.
4. Wave 4 waits for Release 2 (redaction-diff) or a free afternoon (Commons).
