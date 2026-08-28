"""
Entry point.

    $ python run.py

Starts the Flask dev server on http://localhost:5001 (override with the
PORT environment variable; 5000 is skipped by default since macOS often
occupies it with AirPlay Receiver).

  /            dashboard (today's attendance overview)
  /register    enroll a new employee's face
  /live        live camera attendance capture
  /employees   manage employees (edit / deactivate / reactivate / delete)
  /reports     attendance history & reports
  /api/...     REST API (see app/api/routes.py)
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
