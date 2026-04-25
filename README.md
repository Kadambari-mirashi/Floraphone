# Floraphone

Floraphone - every flower has a voice.

Software starter for your hackathon architecture:
- Mobile webpage captures flower photo from phone camera
- Ollama (default) or Claude generates accessibility description + music JSON
- Browser (Tone.js) plays melody + phone vibration per note
- Optional forwarding of beat timings to Pico endpoint (if available)
- S-compatible endpoints for capture/result/beats/register are included
- Supports both scan modes: mobile camera upload and USB webcam capture

## Quick start

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Copy environment config:

```bash
cp .env.example .env
```

3) Export variables from `.env` (or use your own env loader):

```bash
export $(grep -v '^#' .env | xargs)
```

4) Run app:

```bash
python app.py
```

5) Open:

`http://127.0.0.1:5001`

## Endpoints

- `POST /analyze-photo`: mobile upload endpoint (`multipart/form-data` with `photo`) for camera images
- `POST /trigger`: capture + Claude + melody + haptic forward
- `POST /button-pressed`: Pico calls this when physical button is pressed
- `POST /haptic`: optional relay endpoint that forwards `{"beats":[...]}` to Pico
- `GET /health`: pre-demo readiness check for model provider + camera + haptic URL
- `POST /capture`: S-compatible capture endpoint; returns `{id, image_url, melody_json, result}`
- `GET /result/<id>`: returns cached result for a prior capture id
- `POST /beats/<id>`: forwards provided beats (or cached beats) to Pico endpoint
- `POST /register`: Pico registers its IP and optional path/port for haptic forwarding

## Health Check

Use this before live demos:

```bash
curl -s http://127.0.0.1:5001/health
```

Behavior:
- Returns `200` when provider + camera are ready (`ok: true`)
- Returns `503` when a required dependency is not ready
- Reports component details for:
  - `provider` (mock/ollama/anthropic status)
  - `camera` (camera index and frame capture status)
  - `haptics` (whether a Pico endpoint is configured/registered)
  - `cache_size` (number of cached captures in memory)

## S Integration Contract (Current)

- `POST /capture`
  - Trigger capture + model pipeline.
  - Returns capture `id`, `image_url`, `melody_json` (raw model schema), and `result`.

- `GET /result/<id>`
  - Returns cached result object for that `id` (includes normalized `music` and `sensory_description`).

- `POST /register`
  - Pico boot registration.
  - Request JSON:
    - `{"ip":"192.168.1.91","port":8080,"path":"/buzz"}`
  - Server stores URL and uses it for all haptic forwards.

- `POST /beats/<id>`
  - Request JSON:
    - `{"beats":[0,500,1000]}`
    - or `{"beats_ms":[...]}`
  - If beats missing, server uses cached beats from `<id>`.
  - Forwards to currently configured/registered Pico endpoint.

## 2-Minute Contract Test (curl)

Run these after `python app.py` is up on port `5001`.

1) Health check:

```bash
curl -s http://127.0.0.1:5001/health | python -m json.tool
```

2) Capture once and save response:

```bash
curl -s -X POST http://127.0.0.1:5001/capture | tee /tmp/flora_capture.json
```

3) Extract capture id:

```bash
python - <<'PY'
import json
with open('/tmp/flora_capture.json') as f:
    data = json.load(f)
print(data['id'])
PY
```

4) Read cached result (replace `<ID>`):

```bash
curl -s http://127.0.0.1:5001/result/<ID> | python -m json.tool
```

5) Register Pico endpoint (replace IP/path/port as needed):

```bash
curl -s -X POST http://127.0.0.1:5001/register \
  -H "Content-Type: application/json" \
  -d '{"ip":"192.168.1.91","port":8080,"path":"/buzz"}' | python -m json.tool
```

6) Forward beats for capture id (replace `<ID>`):

```bash
curl -s -X POST http://127.0.0.1:5001/beats/<ID> \
  -H "Content-Type: application/json" \
  -d '{"beats":[0,500,1000,1500]}' | python -m json.tool
```

Tip: if Pico is not ready yet, step 6 should still return JSON with a clear forwarding error instead of crashing.

## Deterministic Music Mapping (Implemented)

To keep the same flower sounding the same across runs, generation now has two layers:

1) **Flower cache by image fingerprint**
- The server computes a perceptual fingerprint from each captured frame (`16x16` grayscale hash).
- If a fingerprint has been seen before, it reuses the cached `sensory_description` and deterministic `melody_json`.
- Response includes:
  - `cache_hit` (boolean)
  - `flower_fingerprint` (hash id)

2) **Deterministic mapper (no randomness)**
- Model output is treated as feature extraction (`shape_descriptor`, `texture`, `symmetry`, `petal_count_estimate`, `dominant_colors`, `key`).
- Server converts those attributes into melody using fixed rules:
  - **Shape -> contour + tempo + duration**
    - `tall`: ascending contour
    - `drooping`: descending contour, slower tempo, longer notes
    - `spiky`: jagged contour, faster tempo, shorter notes
    - `round`: arc contour
  - **Petal count -> note count**
    - clamped to `5..16`
  - **Texture -> voice**
    - `waxy->pluck`, `velvety->sine`, `papery->triangle`, `fuzzy->am`
  - **Symmetry -> motif structure**
    - `radial`: repeating motif
    - `bilateral`: mirrored call/response
    - `asymmetric`: through-composed default
  - **Color saturation -> velocity**
    - more saturated colors increase note velocity

This means even before cache reuse, the same extracted attributes map to the same tune every time.

## UI and Accessibility

- Spring-themed interface with high-contrast text and clear button states
- Chic pixel-art spring backdrop (flowers, grass, butterflies, bees)
- Keyboard-friendly focus rings and semantic structure (`main`, `section`, headings)
- Live status announcements via `aria-live` for capture and health updates
- Auto health polling in UI every 8 seconds to show readiness before interaction
- Description is read aloud via browser speech synthesis after each capture
- Manual voice controls are available in UI (`Read Aloud`, `Stop Voice`)
- Animation toggle is available (`Pause Animations`) and reduced-motion settings are respected

## Notes

- For no-API dry run, set `USE_MOCK=true`.
- For mobile-only demos, set `REQUIRE_USB_CAMERA=false` (default).
- For USB webcam demos, keep `/trigger` flow and set `REQUIRE_USB_CAMERA=true`.
- Default model provider is local Ollama (`LLM_PROVIDER=ollama`).
- For Ollama, run local server and choose a vision-capable model like `llava:7b`.
- To switch later: set `LLM_PROVIDER=anthropic` and add `ANTHROPIC_API_KEY`.
- Set `PICO_HAPTIC_URL` to a default Pico route, e.g. `http://<pico-ip>/haptic`.
- `POST /register` can override `PICO_HAPTIC_URL` dynamically during runtime.
- If camera does not open, change `CAMERA_INDEX` to `1` or `2`.
