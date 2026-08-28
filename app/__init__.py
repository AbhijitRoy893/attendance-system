"""Application factory."""

from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.models.db_models import db


def create_app(config_class=Config):
    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config.from_object(config_class)

    db.init_app(app)
    CORS(app)

    with app.app_context():
        db.create_all()

    from app.api.routes import api_bp
    from app.api.views import views_bp

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)

    return app
