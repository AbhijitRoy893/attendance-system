"""
Unit tests for the attendance business rules (app/core/attendance_engine.py).

Run with:  pytest tests/ -v
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.config import Config
from app.core.attendance_engine import AttendanceEngine
from app.models.db_models import db, Employee


class TestConfig(Config):
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    TESTING = True


@pytest.fixture()
def app():
    application = create_app(TestConfig)
    with application.app_context():
        db.create_all()
        employee = Employee(employee_code="EMP-100", name="Jane Doe", department="QA")
        db.session.add(employee)
        db.session.commit()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def engine():
    return AttendanceEngine()


def test_first_recognition_of_day_checks_in(app, engine):
    with app.app_context():
        event, payload = engine.process_event(employee_id=1, confidence=40.0)
        assert event == "check_in"
        assert payload["attendance"]["status"] in ("Present", "Late")


def test_immediate_repeat_recognition_is_ignored_as_duplicate(app, engine):
    with app.app_context():
        now = datetime.now()
        engine.process_event(employee_id=1, confidence=40.0, now=now)
        event, payload = engine.process_event(employee_id=1, confidence=41.0, now=now + timedelta(seconds=5))
        assert event == "duplicate_ignored"


def test_checkout_allowed_after_minimum_gap(app, engine):
    with app.app_context():
        now = datetime.now()
        engine.process_event(employee_id=1, confidence=40.0, now=now)
        event, payload = engine.process_event(
            employee_id=1, confidence=38.0,
            now=now + timedelta(minutes=Config.MIN_MINUTES_BEFORE_CHECKOUT + 5),
        )
        assert event == "check_out"
        assert payload["attendance"]["check_out_time"] is not None


def test_recognition_after_full_day_completed_is_duplicate(app, engine):
    with app.app_context():
        now = datetime.now()
        engine.process_event(employee_id=1, confidence=40.0, now=now)
        engine.process_event(employee_id=1, confidence=38.0,
                              now=now + timedelta(minutes=Config.MIN_MINUTES_BEFORE_CHECKOUT + 5))
        event, _ = engine.process_event(
            employee_id=1, confidence=39.0,
            now=now + timedelta(minutes=Config.MIN_MINUTES_BEFORE_CHECKOUT + 20),
        )
        assert event == "duplicate_ignored"


def test_unknown_face_never_touches_attendance(app, engine):
    with app.app_context():
        event, payload = engine.process_event(employee_id=None, confidence=999.0)
        assert event == "unknown"
        from app.models.db_models import Attendance
        assert Attendance.query.count() == 0


def test_inactive_employee_is_rejected(app, engine):
    with app.app_context():
        emp = Employee.query.get(1)
        emp.is_active = False
        db.session.commit()
        event, _ = engine.process_event(employee_id=1, confidence=40.0)
        assert event == "inactive_employee"


def test_late_status_after_grace_period(app, engine):
    with app.app_context():
        today = datetime.now().date()
        hh, mm = map(int, Config.SHIFT_START_TIME.split(":"))
        late_time = datetime.combine(today, datetime.min.time()) + timedelta(
            hours=hh, minutes=mm + Config.LATE_GRACE_MINUTES + 1
        )
        assert engine._late_status(late_time, today) == "Late"


def test_on_time_status_within_grace_period(app, engine):
    with app.app_context():
        today = datetime.now().date()
        hh, mm = map(int, Config.SHIFT_START_TIME.split(":"))
        on_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=hh, minutes=mm)
        assert engine._late_status(on_time, today) == "Present"
