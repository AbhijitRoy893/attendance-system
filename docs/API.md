# API Reference

Base URL: `http://localhost:5001/api`
All bodies are JSON. All responses are JSON.

**No authentication is required in this build** — every endpoint below is
open to any client that can reach the server. See the README's "Security
Note" section before deploying this beyond a local demo.

---

## Employees

### `POST /employees`
Create an employee record.

**Body**
```json
{ "employee_code": "EMP-1042", "name": "Ananya Sharma", "department": "Engineering",
  "designation": "Software Engineer II", "email": "ananya@company.com", "phone": "+91..." }
```
`employee_code` and `name` are required. Returns `201` with the employee object,
or `409` if `employee_code` already exists.

### `GET /employees`
List employees. Query params: `department`, `active_only=true|false`.

### `GET /employees/<id>` · `PUT /employees/<id>` · `DELETE /employees/<id>`
Get / update / soft-delete (deactivates employee, removes face data, retrains model,
frees the `employee_code` for reuse). Soft-delete is reversible.

### `POST /employees/<id>/reactivate`
Restores a deactivated employee (`is_active` → `true`). Face samples were
removed on deactivation, so the employee must be re-enrolled via `/register`
afterward. Returns `400` if the employee is already active.

### `DELETE /employees/<id>/purge`
**Permanently and irreversibly** deletes the employee record, including
their attendance history (their recognition-log entries are anonymized
rather than deleted, to preserve the audit trail's row count). Only
allowed if the employee is already deactivated (`is_active=false`) —
returns `400` otherwise. This safeguard exists so a single accidental
click can't destroy a record; the UI requires deactivating first, then a
separate, clearly-labeled "Delete Permanently" action with a confirmation
prompt.

---

## Enrolment

### `POST /employees/<id>/enroll`
Submit one webcam frame as a training sample.

**Body**: `{ "image": "data:image/jpeg;base64,..." }`

**Responses**
- `200` → `{ message, sample_path, samples_collected, samples_required, ready_to_train }`
- `422` → no face, or more than one face, detected in the frame

### `POST /employees/<id>/train`
Retrain the LBPH recognizer on the full on-disk dataset (call after enrolling
samples for a new or updated employee).

**Response**: `{ message, employees_in_model, total_samples }`

### `GET /employees/<id>/enroll/status`
Check enrolment progress without submitting a new sample.

---

## Recognition (attendance capture)

### `POST /recognize`
The core endpoint. Submit one webcam frame; detects every face present,
classifies each as known/unknown, and applies attendance business rules.

**Body**: `{ "image": "data:image/jpeg;base64,...", "source": "camera-1" }`

**Response**
```json
{
  "faces_detected": 2,
  "timestamp": "2026-08-25T09:31:02",
  "results": [
    {
      "box": [210, 198, 183, 183],
      "employee_id": 7,
      "is_known": true,
      "confidence": 42.3,
      "event": "check_in",
      "detail": { "attendance": { "...": "..." }, "status": "Present" }
    },
    {
      "box": [450, 200, 170, 170],
      "employee_id": null,
      "is_known": false,
      "confidence": 999.0,
      "event": "unknown",
      "detail": { "message": "Face not recognized" }
    }
  ]
}
```

`event` is one of: `check_in`, `check_out`, `duplicate_ignored`, `unknown`,
`inactive_employee`. `confidence` is the LBPH distance — **lower is a
better match**; this is the opposite of a typical 0–1 similarity score.

---

## Attendance

### `GET /attendance`
Query params: `date` (`YYYY-MM-DD`), `employee_id`, `department`.

### `GET /attendance/today`
Dashboard summary: `{ date, total_employees, present, absent, late, records[] }`.

---

## Reports & Audit

### `GET /reports/summary`
Query params: `start`, `end` (`YYYY-MM-DD`). Returns per-employee present/late
day counts over the range.

### `GET /logs`
Full recognition audit trail (including unknown faces and ignored
duplicates). Query params: `limit` (default 100), `event_type`.

### `GET /admin-logs`
Audit trail of admin actions — who created, edited, deactivated,
reactivated, or permanently deleted an employee record, and when. Query
params: `limit` (default 100).

### `GET /health`
`{ status, model_trained, enrolled_identities }` — used by the frontend
sidebar status pill and suitable for uptime/monitoring checks.

---

## Error format

Non-2xx responses return `{ "error": "human-readable message" }` with an
appropriate HTTP status code (`400` bad request, `404` not found, `409`
conflict, `422` unprocessable frame).
