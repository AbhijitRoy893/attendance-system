"""
REST API
========
    Employees
        POST   /api/employees                  create employee record
        GET    /api/employees                  list employees (include_inactive=true to see deactivated)
        GET    /api/employees/<id>              employee detail
        PUT    /api/employees/<id>              update employee
        DELETE /api/employees/<id>              deactivate + remove face data (reversible)
        POST   /api/employees/<id>/reactivate   restore a deactivated employee
        DELETE /api/employees/<id>/purge        permanently erase the record (irreversible)

    Enrolment (face capture / training)
        POST   /api/employees/<id>/enroll       submit one webcam frame as a training sample
        POST   /api/employees/<id>/train        (re)train the recognizer on the saved samples
        GET    /api/employees/<id>/enroll/status

    Recognition (attendance capture)
        POST   /api/recognize                   submit one webcam frame; detects + marks attendance

    Attendance
        GET    /api/attendance                  list, filterable by date/employee/department
        GET    /api/attendance/today

    Reports / logs
        GET    /api/reports/summary
        GET    /api/logs                        recognition audit trail (camera events)
        GET    /api/admin-logs                  admin action audit trail (who changed what)

    GET    /api/health
"""

from datetime import datetime, date

from flask import Blueprint, jsonify, request

from app.config import Config
from app.core.attendance_engine import AttendanceEngine
from app.core.dataset_manager import DatasetManager
from app.core.face_engine import FaceEngine
from app.models.db_models import db, Employee, Attendance, RecognitionLog, AdminActionLog
from app.utils.image_utils import decode_base64_image, save_snapshot

api_bp = Blueprint("api", __name__)

# Singletons shared across requests (model + dataset live on disk anyway)
face_engine = FaceEngine()
dataset_manager = DatasetManager()
attendance_engine = AttendanceEngine()


def _log_admin_action(action: str, employee=None, employee_id=None, employee_code=None, details=None):
    # No login system in this build, so admin actions are attributed to a
    # single shared "admin" identity. Swap in a real username here if a
    # login/user system is reintroduced later.
    db.session.add(AdminActionLog(
        admin_username="admin",
        action=action,
        employee_id=employee.id if employee else employee_id,
        employee_code=employee.employee_code if employee else employee_code,
        details=details,
    ))


# ========================================================================
# Employees
# ========================================================================
@api_bp.route("/employees", methods=["POST"])
def create_employee():
    data = request.get_json(force=True)
    required = ["employee_code", "name"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    if Employee.query.filter_by(employee_code=data["employee_code"]).first():
        return jsonify({"error": "employee_code already exists"}), 409

    employee = Employee(
        employee_code=data["employee_code"],
        name=data["name"],
        department=data.get("department"),
        designation=data.get("designation"),
        email=data.get("email"),
        phone=data.get("phone"),
    )
    db.session.add(employee)
    db.session.flush()  # get employee.id before logging
    _log_admin_action("created", employee=employee, details=f"Registered {employee.name}")
    db.session.commit()
    return jsonify(employee.to_dict()), 201


@api_bp.route("/employees", methods=["GET"])
def list_employees():
    query = Employee.query
    department = request.args.get("department")
    active_only = request.args.get("active_only", "false").lower() == "true"
    include_inactive = request.args.get("include_inactive", "true").lower() == "true"
    if department:
        query = query.filter_by(department=department)
    if active_only:
        query = query.filter_by(is_active=True)
    elif not include_inactive:
        query = query.filter_by(is_active=True)
    employees = query.order_by(Employee.is_active.desc(), Employee.name).all()
    return jsonify([e.to_dict() for e in employees])


@api_bp.route("/employees/<int:employee_id>", methods=["GET"])
def get_employee(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    return jsonify(employee.to_dict())


@api_bp.route("/employees/<int:employee_id>", methods=["PUT"])
def update_employee(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    data = request.get_json(force=True)
    changed_fields = []
    for field in ["name", "department", "designation", "email", "phone"]:
        if field in data and data[field] != getattr(employee, field):
            setattr(employee, field, data[field])
            changed_fields.append(field)
    db.session.flush()
    if changed_fields:
        _log_admin_action("updated", employee=employee, details=f"Changed: {', '.join(changed_fields)}")
    db.session.commit()
    return jsonify(employee.to_dict())


@api_bp.route("/employees/<int:employee_id>", methods=["DELETE"])
def delete_employee(employee_id):
    """Soft delete: deactivates the employee and removes their face data
    (so they stop being matched), but keeps the DB row and attendance
    history intact. Reversible via POST /employees/<id>/reactivate."""
    employee = Employee.query.get_or_404(employee_id)
    employee.is_active = False

    # Free up the employee_code so it can be reused for a new registration,
    # while keeping the original record (and its attendance history) intact.
    if not employee.employee_code.endswith("-deleted"):
        employee.employee_code = f"{employee.employee_code}-{employee.id}-deleted"

    dataset_manager.delete_employee(employee_id)
    _log_admin_action("deactivated", employee=employee, details="Face data removed, code freed for reuse")
    db.session.commit()

    # Retrain without this employee's samples so recognition no longer
    # matches against a deactivated identity.
    dataset = dataset_manager.load_all()
    if dataset:
        face_engine.train(dataset)
    return jsonify({"message": "Employee deactivated, face data removed, and employee_code freed for reuse"})


@api_bp.route("/employees/<int:employee_id>/reactivate", methods=["POST"])
def reactivate_employee(employee_id):
    """Restores a deactivated employee. Face samples were removed on
    deactivation, so re-enrolment via /register is required afterward."""
    employee = Employee.query.get_or_404(employee_id)
    if employee.is_active:
        return jsonify({"error": "Employee is already active"}), 400

    employee.is_active = True
    employee.face_samples_count = 0
    _log_admin_action("reactivated", employee=employee, details="Requires re-enrolment of face samples")
    db.session.commit()
    return jsonify({
        "message": "Employee reactivated. Face samples were removed on deactivation — re-enroll via Register Employee.",
        "employee": employee.to_dict(),
    })


@api_bp.route("/employees/<int:employee_id>/purge", methods=["DELETE"])
def purge_employee(employee_id):
    """Permanently erases the employee record. Only allowed once the
    employee has already been deactivated, as a safeguard against
    accidental irreversible deletion. This also deletes their attendance
    history — the UI must warn the admin clearly before calling this."""
    employee = Employee.query.get_or_404(employee_id)
    if employee.is_active:
        return jsonify({"error": "Deactivate the employee before permanently deleting."}), 400

    employee_code = employee.employee_code
    dataset_manager.delete_employee(employee_id)

    Attendance.query.filter_by(employee_id=employee_id).delete()
    RecognitionLog.query.filter_by(employee_id=employee_id).update({"employee_id": None})

    _log_admin_action("purged", employee_id=employee_id, employee_code=employee_code,
                       details="Permanently deleted, including attendance history")
    db.session.delete(employee)
    db.session.commit()
    return jsonify({"message": "Employee permanently deleted"})


# ========================================================================
# Enrolment
# ========================================================================
@api_bp.route("/employees/<int:employee_id>/enroll", methods=["POST"])
def enroll_face_sample(employee_id):
    """Accepts one webcam frame (base64 JPEG/PNG), detects exactly one
    face, and stores it as a training sample. Call repeatedly (e.g. while
    the user turns their head slightly) to build a robust sample set."""
    employee = Employee.query.get_or_404(employee_id)
    data = request.get_json(force=True)
    if "image" not in data:
        return jsonify({"error": "Missing 'image' (base64 data URL)"}), 400

    try:
        frame = decode_base64_image(data["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    faces = face_engine.detect_faces(frame)
    if len(faces) == 0:
        return jsonify({"error": "No face detected in frame. Ensure good lighting and face the camera."}), 422
    if len(faces) > 1:
        return jsonify({"error": "Multiple faces detected. Only the enrolling employee should be in frame."}), 422

    path = dataset_manager.save_sample(employee_id, faces[0].aligned)
    employee.face_samples_count = dataset_manager.count_samples(employee_id)
    db.session.commit()

    return jsonify({
        "message": "Sample captured",
        "sample_path": path,
        "samples_collected": employee.face_samples_count,
        "samples_required": Config.SAMPLES_PER_EMPLOYEE,
        "ready_to_train": employee.face_samples_count >= Config.SAMPLES_PER_EMPLOYEE,
    })


@api_bp.route("/employees/<int:employee_id>/train", methods=["POST"])
def train_employee(employee_id):
    """Retrain the recognizer on the full dataset (must be done after
    enrolling a new employee, or after collecting more samples)."""
    employee = Employee.query.get_or_404(employee_id)
    dataset = dataset_manager.load_all()
    if employee_id not in dataset:
        return jsonify({"error": "No face samples found for this employee. Enroll samples first."}), 400

    face_engine.train(dataset)
    return jsonify({
        "message": "Model retrained",
        "employees_in_model": len(dataset),
        "total_samples": sum(len(v) for v in dataset.values()),
    })


@api_bp.route("/employees/<int:employee_id>/enroll/status", methods=["GET"])
def enroll_status(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    count = dataset_manager.count_samples(employee_id)
    return jsonify({
        "samples_collected": count,
        "samples_required": Config.SAMPLES_PER_EMPLOYEE,
        "ready_to_train": count >= Config.SAMPLES_PER_EMPLOYEE,
        "model_trained_with_this_employee": str(employee_id) in face_engine.labels,
    })


# ========================================================================
# Recognition / attendance capture
# ========================================================================
@api_bp.route("/recognize", methods=["POST"])
def recognize():
    """
    Core attendance-capture endpoint. Accepts one webcam frame and:
      1. Detects ALL faces present (handles multiple people in frame).
      2. Classifies each as a known employee or Unknown.
      3. Applies attendance business rules (check-in / check-out / duplicate).
      4. Logs every event for audit purposes.
    """
    data = request.get_json(force=True)
    if "image" not in data:
        return jsonify({"error": "Missing 'image' (base64 data URL)"}), 400

    try:
        frame = decode_base64_image(data["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    source = data.get("source", "camera-1")
    now = datetime.now()
    results = face_engine.recognize(frame)

    if not results:
        return jsonify({"faces_detected": 0, "results": []})

    output = []
    for r in results:
        if r.is_known:
            snapshot_dir = Config.RECOGNIZED_LOG_DIR
        else:
            snapshot_dir = Config.UNKNOWN_LOG_DIR
        snapshot_path = save_snapshot(frame, snapshot_dir)

        event_type, payload = attendance_engine.process_event(
            employee_id=r.employee_id,
            confidence=r.confidence,
            source=source,
            image_path=snapshot_path,
            now=now,
        )
        output.append({
            "box": r.box,
            "employee_id": r.employee_id,
            "is_known": r.is_known,
            "confidence": round(r.confidence, 2),
            "event": event_type,
            "detail": payload,
        })

    return jsonify({"faces_detected": len(results), "timestamp": now.isoformat(), "results": output})


# ========================================================================
# Attendance records
# ========================================================================
@api_bp.route("/attendance", methods=["GET"])
def list_attendance():
    query = Attendance.query
    date_str = request.args.get("date")
    employee_id = request.args.get("employee_id")
    department = request.args.get("department")

    if date_str:
        query = query.filter_by(date=datetime.strptime(date_str, "%Y-%m-%d").date())
    if employee_id:
        query = query.filter_by(employee_id=int(employee_id))
    if department:
        query = query.join(Employee).filter(Employee.department == department)

    records = query.order_by(Attendance.date.desc(), Attendance.check_in_time.desc()).all()
    return jsonify([r.to_dict() for r in records])


@api_bp.route("/attendance/today", methods=["GET"])
def attendance_today():
    records = Attendance.query.filter_by(date=date.today()).all()
    total_active = Employee.query.filter_by(is_active=True).count()
    present = len(records)
    late = sum(1 for r in records if r.status == "Late")
    return jsonify({
        "date": date.today().isoformat(),
        "total_employees": total_active,
        "present": present,
        "absent": max(total_active - present, 0),
        "late": late,
        "records": [r.to_dict() for r in records],
    })


# ========================================================================
# Reports / logs
# ========================================================================
@api_bp.route("/reports/summary", methods=["GET"])
def reports_summary():
    start = request.args.get("start")
    end = request.args.get("end")
    query = Attendance.query
    if start:
        query = query.filter(Attendance.date >= datetime.strptime(start, "%Y-%m-%d").date())
    if end:
        query = query.filter(Attendance.date <= datetime.strptime(end, "%Y-%m-%d").date())
    records = query.all()

    by_employee = {}
    for r in records:
        key = r.employee_id
        by_employee.setdefault(key, {
            "employee_id": key,
            "employee_name": r.employee.name if r.employee else None,
            "present_days": 0,
            "late_days": 0,
        })
        by_employee[key]["present_days"] += 1
        if r.status == "Late":
            by_employee[key]["late_days"] += 1

    return jsonify(list(by_employee.values()))


@api_bp.route("/logs", methods=["GET"])
def recognition_logs():
    limit = int(request.args.get("limit", 100))
    event_type = request.args.get("event_type")
    query = RecognitionLog.query
    if event_type:
        query = query.filter_by(event_type=event_type)
    logs = query.order_by(RecognitionLog.timestamp.desc()).limit(limit).all()
    return jsonify([l.to_dict() for l in logs])


@api_bp.route("/admin-logs", methods=["GET"])
def admin_logs():
    """Audit trail of admin actions: who created/edited/deactivated/
    reactivated/permanently-deleted an employee record, and when."""
    limit = int(request.args.get("limit", 100))
    logs = AdminActionLog.query.order_by(AdminActionLog.timestamp.desc()).limit(limit).all()
    return jsonify([l.to_dict() for l in logs])


@api_bp.route("/health", methods=["GET"])
def health():
    # /api/health is used by the sidebar status pill on every page,
    # and is suitable for uptime/monitoring checks.
    return jsonify({
        "status": "ok",
        "model_trained": face_engine.is_trained,
        "enrolled_identities": len(face_engine.labels),
    })
