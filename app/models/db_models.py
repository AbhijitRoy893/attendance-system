"""
Database models.

Employee            -> one row per enrolled person
Attendance           -> one row per employee per day (check-in / check-out)
RecognitionLog       -> audit trail of every recognition event (incl. unknowns
                        and duplicates that were deliberately ignored)
"""

from datetime import datetime, date

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Employee(db.Model):
    __tablename__ = "employees"

    id = db.Column(db.Integer, primary_key=True)
    employee_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    department = db.Column(db.String(80), nullable=True)
    designation = db.Column(db.String(80), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    face_samples_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)

    attendances = db.relationship("Attendance", backref="employee", lazy="dynamic")
    logs = db.relationship("RecognitionLog", backref="employee", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id,
            "employee_code": self.employee_code,
            "name": self.name,
            "department": self.department,
            "designation": self.designation,
            "email": self.email,
            "phone": self.phone,
            "is_active": self.is_active,
            "face_samples_count": self.face_samples_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Attendance(db.Model):
    __tablename__ = "attendance"
    __table_args__ = (
        db.UniqueConstraint("employee_id", "date", name="uq_employee_date"),
    )

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    date = db.Column(db.Date, default=date.today, nullable=False, index=True)
    check_in_time = db.Column(db.DateTime, nullable=True)
    check_out_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default="Present")   # Present / Late / Absent
    check_in_confidence = db.Column(db.Float, nullable=True)
    check_out_confidence = db.Column(db.Float, nullable=True)
    source = db.Column(db.String(50), default="camera-1")

    def to_dict(self):
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "employee_name": self.employee.name if self.employee else None,
            "employee_code": self.employee.employee_code if self.employee else None,
            "department": self.employee.department if self.employee else None,
            "date": self.date.isoformat(),
            "check_in_time": self.check_in_time.isoformat() if self.check_in_time else None,
            "check_out_time": self.check_out_time.isoformat() if self.check_out_time else None,
            "status": self.status,
            "check_in_confidence": self.check_in_confidence,
            "check_out_confidence": self.check_out_confidence,
            "source": self.source,
        }


class AdminActionLog(db.Model):
    """Audit trail of admin-initiated changes (create/edit/deactivate/
    reactivate/permanently-delete an employee). Separate from
    RecognitionLog, which tracks camera events rather than admin actions."""
    __tablename__ = "admin_action_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now, index=True)
    admin_username = db.Column(db.String(80), nullable=False)
    action = db.Column(db.String(30), nullable=False)  # created / updated / deactivated / reactivated / purged
    employee_id = db.Column(db.Integer, nullable=True)  # not a FK: row may have been purged
    employee_code = db.Column(db.String(20), nullable=True)
    details = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "admin_username": self.admin_username,
            "action": self.action,
            "employee_id": self.employee_id,
            "employee_code": self.employee_code,
            "details": self.details,
        }


class RecognitionLog(db.Model):
    """Every recognition attempt is logged for audit / debugging, including
    unknown faces and duplicates that were intentionally not re-marked."""
    __tablename__ = "recognition_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now, index=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=True)
    event_type = db.Column(db.String(30), nullable=False)  # recognized / unknown / duplicate_ignored
    confidence = db.Column(db.Float, nullable=True)
    image_path = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "employee_id": self.employee_id,
            "employee_name": self.employee.name if self.employee else "Unknown",
            "event_type": self.event_type,
            "confidence": self.confidence,
            "image_path": self.image_path,
        }
