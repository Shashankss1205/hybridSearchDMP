import os
from typing import Optional

from sarvamai import SarvamAI


class VoiceHandler:
    """Handle speech-to-text using SarvamAI Saarika (v2.5)."""

    def __init__(self):
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("Please set SARVAM_API_KEY env var")
        self.client = SarvamAI(api_subscription_key=api_key)

    def speech_to_text(self, audio_path: str, model: str = "saarika:v2.5", language_code: str = "unknown") -> str:
        with open(audio_path, "rb") as f:
            resp = self.client.speech_to_text.transcribe(file=f, model=model, language_code=language_code)
        return resp.transcript


