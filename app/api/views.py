"""Server-rendered page routes (the pages themselves talk to /api/* via JS)."""

from flask import Blueprint, render_template

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def dashboard():
    return render_template("dashboard.html")


@views_bp.route("/register")
def register_page():
    return render_template("register.html")


@views_bp.route("/live")
def live_page():
    return render_template("live.html")


@views_bp.route("/reports")
def reports_page():
    return render_template("reports.html")


@views_bp.route("/employees")
def employees_page():
    return render_template("employees.html")
