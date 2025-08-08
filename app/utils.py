import logging
import os

import google.generativeai as generative_ai

logger = logging.getLogger(__name__)


def translate_to_english(text: str) -> str:
    try:
        prompt = f"""
        Translate the following text to English if from a different language, or return the original text if already in english. Only return the translation, no additional text:

        Text to translate: {text}
        """
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Translation error: %s", exc)
        return text


def translate_and_refine(text: str) -> str:
    try:
        prompt = f"""
You are a query cleaning assistant for a story search engine.

Instructions:
- Translate the input to English if it is in a different language.
- Then, remove common filler or generic phrases related to stories or requests. This includes:
  "story of", "stories about", "a tale on", "book of", "novel about",
  as well as conversational prefixes like "tell me", "give me", "show me", "I want a story about", etc.
- Only return the essential topic or subject of the story in a short phrase.
- Do not repeat or rephrase the original structure.
- Do not include extra words like "a", "the", "story", "book", etc.
- DO NOT explain or describe anything — just return the final cleaned phrase.

Now process this:
{text}
"""
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini API error: %s", exc)
        return text.strip()


def get_font(text: str) -> str:
    try:
        prompt = f"""
        Write the text in the script it's spoken, like text in devnagri script for Hindi, text in Arabic script for Arabic, etc. If the text is already in English, return 'as it is. Only return the scripted text, no additional text:
        Text to translate: {text}
        """
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Font/Script conversion error: %s", exc)
        return text


