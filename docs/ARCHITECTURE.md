# Architecture

## 1. Pipeline: Enrolment

```
Browser webcam
   │  captures a frame every ~400ms while "Start Capturing" is active
   ▼
POST /api/employees/<id>/enroll   { image: base64 JPEG }
   │
   ▼
FaceEngine.detect_faces(frame)
   │  Haar Cascade over CLAHE-equalized grayscale
   │  → reject frame if 0 or >1 faces found (keeps the dataset clean)
   ▼
DatasetManager.save_sample(employee_id, normalized_face)
   │  saved to database/dataset/<employee_id>/NNNN.png
   ▼
... repeats until SAMPLES_PER_EMPLOYEE (default 30) reached ...
   ▼
POST /api/employees/<id>/train
   │
   ▼
DatasetManager.load_all()  → { employee_id: [face_images...] }
   ▼
FaceEngine.train(dataset)
   │  cv2.face.LBPHFaceRecognizer.train(images, labels)
   │  writes database/lbph_model.yml + labels.json
```

## 2. Pipeline: Live Recognition & Attendance

```
Browser webcam (Live Capture page)
   │  captures + POSTs a frame every 1s
   ▼
POST /api/recognize   { image: base64 JPEG, source: "camera-1" }
   │
   ▼
FaceEngine.recognize(frame)
   │  1. detect_faces() → list of all faces in frame (handles crowds)
   │  2. for each face: LBPH predict() → (label, distance)
   │  3. distance <= LBPH_CONFIDENCE_THRESHOLD ? known : Unknown
   ▼
for each detected face:
   snapshot saved to static/captures/{recognized|unknown}/
   ▼
AttendanceEngine.process_event(employee_id, confidence, ...)
   │
   ├─ employee_id is None            → log "unknown", stop
   ├─ employee inactive               → log "inactive_employee", stop
   ├─ no attendance row for today     → CREATE row, check_in_time=now,
   │                                     status = Late if past grace period
   ├─ checked in, no checkout,
   │  < MIN_MINUTES_BEFORE_CHECKOUT   → "duplicate_ignored"
   ├─ checked in, no checkout,
   │  >= MIN_MINUTES_BEFORE_CHECKOUT  → SET check_out_time=now
   └─ already checked in AND out      → "duplicate_ignored"
   ▼
JSON response → browser draws bounding boxes + live feed list
```

## 3. Why Haar Cascade + LBPH (and not a deep model)

| Requirement | Haar + LBPH | Deep embeddings (dlib/FaceNet/ArcFace) |
|---|---|---|
| CPU-only real-time inference | ✅ trivial | ⚠️ needs optimized runtime (ONNX/TensorRT) for real-time on CPU |
| Zero native build dependencies | ✅ ships in `opencv-contrib-python` | ❌ `dlib` requires a CMake/C++ build; large model downloads |
| Accuracy on large rosters / hard poses | ⚠️ moderate | ✅ significantly better |
| Training speed on enroll | ✅ sub-second retrain | ✅ no retrain needed (just store new embedding) |

For an internal office deployment (dozens to low hundreds of employees,
controlled entry-point camera angle), Haar+LBPH is an appropriate
accuracy/complexity trade-off and keeps the whole stack `pip install`-able
with no GPU. `FaceEngine` isolates all CV logic behind `detect_faces()` /
`train()` / `recognize()`, so a deep embedding backend (e.g. an ONNX-exported
ArcFace model with cosine-similarity matching instead of LBPH labels) can be
substituted later without changing the API layer, the database schema, or
the attendance business rules.

## 4. Data Model

```
Employee 1───* Attendance          (one row per employee per calendar day)
Employee 1───* RecognitionLog      (every recognition attempt, incl. unknown/duplicate)
```

`Attendance` has a unique constraint on `(employee_id, date)` — the
business logic in `AttendanceEngine` is the only writer, so this constraint
is a safety net against ever double-inserting a day's row from a race
condition (e.g. two camera frames processed concurrently).

## 5. Handling Real-World Conditions — Design Detail

**Multiple faces in frame.** `FaceEngine.detect_faces()` returns every face
Haar Cascade finds (capped at `MAX_FACES_PER_FRAME` to bound worst-case
latency on a crowded frame). `/api/recognize` loops over all of them and
runs each through the full attendance pipeline independently — one frame
with three people produces three independent check-in/duplicate/unknown
outcomes in a single response.

**Unknown persons.** LBPH's `predict()` always returns *some* label — it
never natively says "I don't know this person." The system treats the
returned distance as a confidence signal: anything worse (numerically
higher) than `LBPH_CONFIDENCE_THRESHOLD` is overridden to `Unknown`
regardless of which label LBPH nominally picked. Unknown snapshots are
saved separately (`static/captures/unknown/`) for a security review
workflow.

**Duplicate attendance.** Rather than trying to detect "is this the exact
same recognition event," the system tracks explicit per-day state
(`check_in_time`, `check_out_time`) and applies a minimum-gap rule before a
second recognition is allowed to count as a checkout. This is robust to a
person standing in front of the camera for several seconds (many frames,
one event) as well as to someone walking past twice.

**Lighting variation.** CLAHE (Contrast Limited Adaptive Histogram
Equalization) is applied to every frame before both detection and
recognition, at both enrolment and inference time. This locally normalizes
contrast so a shadow across half a face, or an overall dim/bright room,
doesn't shift the LBP texture patterns as much as raw grayscale would.
