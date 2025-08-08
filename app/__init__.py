import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from .encoders import NumpyEncoder
from .services.engine import StorySearchEngine
from .services.feedback import FeedbackDB
from .services.voice import VoiceHandler
from .routes import bp as api_bp


def create_app() -> Flask:
    """Application factory that wires up services, blueprints, and config."""
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    app = Flask(__name__)
    CORS(app)
    app.json_encoder = NumpyEncoder

    # Initialize core services (singletons)
    search_engine = StorySearchEngine()
    # Feedback/Voice are optional in local dev; avoid crashing when env vars are missing
    try:
        feedback_db = FeedbackDB()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("FeedbackDB not initialized: %s", exc)
        feedback_db = None
    try:
        voice_handler = VoiceHandler()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("VoiceHandler not initialized: %s", exc)
        voice_handler = None

    # Expose services via app config so routes can access them
    app.config["SEARCH_ENGINE"] = search_engine
    app.config["FEEDBACK_DB"] = feedback_db
    app.config["VOICE_HANDLER"] = voice_handler

    # Register API routes
    app.register_blueprint(api_bp)

    return app


