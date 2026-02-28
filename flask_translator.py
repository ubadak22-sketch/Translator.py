#!/usr/bin/env python3
"""
Flask Translator API
====================
A complete translation API with JWT auth, TTS, STT, OCR, and 100+ languages.
Compatible with Termux (Android) and standard Python environments.

Install dependencies:
    pip install flask flask-cors pyjwt requests Pillow pytesseract SpeechRecognition gtts

For Termux:
    pkg install tesseract
    pip install flask flask-cors pyjwt requests Pillow pytesseract SpeechRecognition gtts
"""

import base64
import datetime
import io
import json
import logging
import os
import tempfile
import urllib.parse
import urllib.request
from functools import wraps

# ---------------------------------------------------------------------------
# Third-party imports (with graceful degradation if not installed)
# ---------------------------------------------------------------------------
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("[WARN] pyjwt not installed. JWT auth disabled. Run: pip install pyjwt")

try:
    import requests as req_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests not installed. Run: pip install requests")

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] Pillow/pytesseract not installed. OCR disabled.")

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("[WARN] SpeechRecognition not installed. STT disabled.")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("[WARN] gTTS not installed. TTS disabled. Run: pip install gtts")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("JWT_SECRET", "super-secret-dev-key-change-in-production")
TOKEN_EXPIRY_MINUTES = 30
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
PORT = int(os.environ.get("PORT", 5000))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flask_translator")

# ---------------------------------------------------------------------------
# Complete language list (100+ languages)
# ---------------------------------------------------------------------------
LANGUAGES = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "ny": "Chichewa",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "co": "Corsican",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Frisian",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "iw": "Hebrew",
    "hi": "Hindi",
    "hmn": "Hmong",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "rw": "Kinyarwanda",
    "ko": "Korean",
    "ku": "Kurdish (Kurmanji)",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mn": "Mongolian",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "no": "Norwegian",
    "or": "Odia (Oriya)",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sm": "Samoan",
    "gd": "Scots Gaelic",
    "sr": "Serbian",
    "st": "Sesotho",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "tg": "Tajik",
    "ta": "Tamil",
    "tt": "Tatar",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "tk": "Turkmen",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu",
    # Additional languages
    "ak": "Akan",
    "bho": "Bhojpuri",
    "doi": "Dogri",
    "dv": "Dhivehi",
    "ee": "Ewe",
    "gn": "Guaraní",
    "ilo": "Ilocano",
    "kri": "Krio",
    "lus": "Mizo",
    "mai": "Maithili",
    "mni-Mtei": "Meitei (Manipuri)",
    "nso": "Northern Sotho (Sepedi)",
    "om": "Oromo",
    "qu": "Quechua",
    "sa": "Sanskrit",
    "ts": "Tsonga",
    "ti": "Tigrinya",
}

# ---------------------------------------------------------------------------
# Flask App Initialization
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# JWT Authentication Decorator
# ---------------------------------------------------------------------------
def require_jwt(f):
    """Decorator to protect routes with JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not JWT_AVAILABLE:
            return jsonify({"error": "JWT library not available"}), 503

        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]

        if not token:
            return jsonify({"error": "Missing token", "hint": "Add Authorization: Bearer <token>"}), 401

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.jwt_payload = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401

        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Translation Service
# ---------------------------------------------------------------------------
class TranslationService:
    """Handles text translation and language detection using Google Translate."""

    TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
    DETECT_URL = "https://translate.googleapis.com/translate_a/single"

    @staticmethod
    def translate(text: str, target_lang: str, source_lang: str = "auto") -> dict:
        """Translate text using Google Translate free API."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        target_lang = target_lang.lower()
        if target_lang not in LANGUAGES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        params = {
            "client": "gtx",
            "sl": source_lang if source_lang != "auto" else "auto",
            "tl": target_lang,
            "dt": ["t", "ld"],
            "q": text,
        }

        if REQUESTS_AVAILABLE:
            resp = req_lib.get(TranslationService.TRANSLATE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        else:
            # Fallback using urllib
            query = urllib.parse.urlencode(
                [(k, v) if not isinstance(v, list) else (k, vi) for k, v in params.items() for vi in (v if isinstance(v, list) else [v])]
            )
            url = f"{TranslationService.TRANSLATE_URL}?{query}"
            with urllib.request.urlopen(url, timeout=10) as r:
                data = json.loads(r.read().decode())

        translated_parts = [part[0] for part in data[0] if part[0]]
        translated_text = "".join(translated_parts)
        detected_lang = data[2] if len(data) > 2 else source_lang

        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": {
                "code": detected_lang,
                "name": LANGUAGES.get(detected_lang, detected_lang),
            },
            "target_language": {
                "code": target_lang,
                "name": LANGUAGES.get(target_lang, target_lang),
            },
        }

    @staticmethod
    def detect(text: str) -> dict:
        """Detect the language of a given text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": "en",
            "dt": "t",
            "q": text,
        }

        if REQUESTS_AVAILABLE:
            resp = req_lib.get(TranslationService.DETECT_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        else:
            query = urllib.parse.urlencode([(k, v) for k, v in params.items()])
            url = f"{TranslationService.DETECT_URL}?{query}"
            with urllib.request.urlopen(url, timeout=10) as r:
                data = json.loads(r.read().decode())

        detected_lang = data[2] if len(data) > 2 else "unknown"
        return {
            "text": text,
            "detected_language": {
                "code": detected_lang,
                "name": LANGUAGES.get(detected_lang, detected_lang),
            },
        }


# ---------------------------------------------------------------------------
# Speech Service
# ---------------------------------------------------------------------------
class SpeechService:
    """Handles Text-to-Speech and Speech-to-Text."""

    @staticmethod
    def text_to_speech(text: str, lang: str = "en") -> str:
        """Convert text to speech. Returns base64-encoded MP3 audio."""
        if not GTTS_AVAILABLE:
            raise RuntimeError("gTTS not installed. Run: pip install gtts")
        if lang not in LANGUAGES:
            raise ValueError(f"Unsupported language: {lang}")

        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def speech_to_text(audio_b64: str, lang: str = "en-US") -> dict:
        """Convert base64-encoded WAV audio to text."""
        if not STT_AVAILABLE:
            raise RuntimeError("SpeechRecognition not installed. Run: pip install SpeechRecognition")

        audio_bytes = base64.b64decode(audio_b64)
        recognizer = sr.Recognizer()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=lang)
            return {"text": text, "language": lang}
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Image Service
# ---------------------------------------------------------------------------
class ImageService:
    """Handles OCR text extraction from images."""

    @staticmethod
    def extract_text(image_b64: str, lang: str = "eng") -> str:
        """Extract text from a base64-encoded image using Tesseract OCR."""
        if not OCR_AVAILABLE:
            raise RuntimeError("Pillow/pytesseract not installed. Run: pip install Pillow pytesseract")

        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()

    @staticmethod
    def translate_image(image_b64: str, target_lang: str, ocr_lang: str = "eng") -> dict:
        """Extract text from image and translate it."""
        extracted_text = ImageService.extract_text(image_b64, ocr_lang)
        if not extracted_text:
            return {
                "extracted_text": "",
                "translation": None,
                "message": "No text found in image",
            }
        translation = TranslationService.translate(extracted_text, target_lang)
        return {
            "extracted_text": extracted_text,
            "translation": translation,
        }


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------
def success(data: dict, status: int = 200):
    return jsonify({"success": True, **data}), status


def error(msg: str, status: int = 400, **extra):
    return jsonify({"success": False, "error": msg, **extra}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# ── 1. API Info ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def api_info():
    return success({
        "name": "Flask Translator API",
        "version": "1.0.0",
        "description": "Multi-feature translation API with JWT auth",
        "features": {
            "translation": True,
            "language_detection": True,
            "text_to_speech": GTTS_AVAILABLE,
            "speech_to_text": STT_AVAILABLE,
            "ocr": OCR_AVAILABLE,
            "jwt_auth": JWT_AVAILABLE,
        },
        "language_count": len(LANGUAGES),
        "endpoints": [
            "GET  /",
            "POST /api/auth/token",
            "GET  /api/auth/verify",
            "POST /api/translate",
            "POST /api/detect-language",
            "POST /api/tts",
            "POST /api/stt",
            "POST /api/translate-image",
            "POST /api/extract-text",
            "GET  /api/languages",
            "GET  /api/language-codes",
            "GET  /api/language/<code>",
            "GET  /api/search-languages?q=<query>",
            "GET  /api/health",
        ],
    })


# ── 2. Auth: Get Token ───────────────────────────────────────────────────────
@app.route("/api/auth/token", methods=["POST"])
def get_token():
    """Generate a JWT token. Accepts any username/password for testing."""
    if not JWT_AVAILABLE:
        return error("JWT library not available. Run: pip install pyjwt", 503)

    data = request.get_json(silent=True) or {}
    username = data.get("username", "guest")
    # Any credentials are accepted (testing mode)

    now = datetime.datetime.utcnow()
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=TOKEN_EXPIRY_MINUTES),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    return success({
        "token": token,
        "token_type": "Bearer",
        "expires_in": TOKEN_EXPIRY_MINUTES * 60,
        "username": username,
    })


# ── 3. Auth: Verify Token ────────────────────────────────────────────────────
@app.route("/api/auth/verify", methods=["GET"])
@require_jwt
def verify_token():
    payload = request.jwt_payload
    return success({
        "valid": True,
        "username": payload.get("sub"),
        "expires_at": datetime.datetime.utcfromtimestamp(payload["exp"]).isoformat() + "Z",
    })


# ── 4. Translate Text ────────────────────────────────────────────────────────
@app.route("/api/translate", methods=["POST"])
@require_jwt
def translate():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    target = data.get("target_language", "").strip()
    source = data.get("source_language", "auto").strip()

    if not text:
        return error("'text' is required")
    if not target:
        return error("'target_language' is required")

    try:
        result = TranslationService.translate(text, target, source)
        return success({"translation": result})
    except ValueError as e:
        return error(str(e))
    except Exception as e:
        logger.exception("Translation error")
        return error(f"Translation failed: {str(e)}", 502)


# ── 5. Detect Language ───────────────────────────────────────────────────────
@app.route("/api/detect-language", methods=["POST"])
@require_jwt
def detect_language():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return error("'text' is required")

    try:
        result = TranslationService.detect(text)
        return success(result)
    except Exception as e:
        logger.exception("Detection error")
        return error(f"Detection failed: {str(e)}", 502)


# ── 6. Text-to-Speech ────────────────────────────────────────────────────────
@app.route("/api/tts", methods=["POST"])
@require_jwt
def text_to_speech():
    if not GTTS_AVAILABLE:
        return error("gTTS not installed. Run: pip install gtts", 503)

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    lang = data.get("language", "en").strip()

    if not text:
        return error("'text' is required")

    try:
        audio_b64 = SpeechService.text_to_speech(text, lang)
        return success({
            "audio_base64": audio_b64,
            "format": "mp3",
            "language": {"code": lang, "name": LANGUAGES.get(lang, lang)},
            "usage": "Decode base64 and play as audio/mpeg",
        })
    except Exception as e:
        logger.exception("TTS error")
        return error(f"TTS failed: {str(e)}", 502)


# ── 7. Speech-to-Text ────────────────────────────────────────────────────────
@app.route("/api/stt", methods=["POST"])
@require_jwt
def speech_to_text():
    if not STT_AVAILABLE:
        return error("SpeechRecognition not installed. Run: pip install SpeechRecognition", 503)

    data = request.get_json(silent=True) or {}
    audio_b64 = data.get("audio_base64", "").strip()
    lang = data.get("language", "en-US").strip()

    if not audio_b64:
        return error("'audio_base64' is required (base64-encoded WAV)")

    try:
        result = SpeechService.speech_to_text(audio_b64, lang)
        return success(result)
    except Exception as e:
        logger.exception("STT error")
        return error(f"STT failed: {str(e)}", 502)


# ── 8. Translate Image ───────────────────────────────────────────────────────
@app.route("/api/translate-image", methods=["POST"])
@require_jwt
def translate_image():
    if not OCR_AVAILABLE:
        return error("Pillow/pytesseract not installed. Run: pip install Pillow pytesseract", 503)

    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64", "").strip()
    target_lang = data.get("target_language", "en").strip()
    ocr_lang = data.get("ocr_language", "eng").strip()

    if not image_b64:
        return error("'image_base64' is required")

    try:
        result = ImageService.translate_image(image_b64, target_lang, ocr_lang)
        return success(result)
    except Exception as e:
        logger.exception("Image translation error")
        return error(f"Image translation failed: {str(e)}", 502)


# ── 9. Extract Text from Image ───────────────────────────────────────────────
@app.route("/api/extract-text", methods=["POST"])
@require_jwt
def extract_text():
    if not OCR_AVAILABLE:
        return error("Pillow/pytesseract not installed. Run: pip install Pillow pytesseract", 503)

    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64", "").strip()
    ocr_lang = data.get("ocr_language", "eng").strip()

    if not image_b64:
        return error("'image_base64' is required")

    try:
        text = ImageService.extract_text(image_b64, ocr_lang)
        return success({
            "extracted_text": text,
            "character_count": len(text),
            "ocr_language": ocr_lang,
        })
    except Exception as e:
        logger.exception("OCR error")
        return error(f"OCR failed: {str(e)}", 502)


# ── 10. All Languages ────────────────────────────────────────────────────────
@app.route("/api/languages", methods=["GET"])
def get_languages():
    langs = [{"code": k, "name": v} for k, v in sorted(LANGUAGES.items(), key=lambda x: x[1])]
    return success({"languages": langs, "count": len(langs)})


# ── 11. Language Codes Only ──────────────────────────────────────────────────
@app.route("/api/language-codes", methods=["GET"])
def get_language_codes():
    return success({"codes": sorted(LANGUAGES.keys()), "count": len(LANGUAGES)})


# ── 12. Language Info by Code ─────────────────────────────────────────────────
@app.route("/api/language/<string:code>", methods=["GET"])
def get_language(code):
    code = code.lower()
    if code not in LANGUAGES:
        return error(f"Language code '{code}' not found", 404)
    return success({"code": code, "name": LANGUAGES[code]})


# ── 13. Search Languages ─────────────────────────────────────────────────────
@app.route("/api/search-languages", methods=["GET"])
def search_languages():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return error("Query parameter 'q' is required", 400)

    results = [
        {"code": k, "name": v}
        for k, v in LANGUAGES.items()
        if query in k.lower() or query in v.lower()
    ]
    results.sort(key=lambda x: x["name"])
    return success({"query": query, "results": results, "count": len(results)})


# ── 14. Health Check ─────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return success({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "services": {
            "translation": True,
            "jwt": JWT_AVAILABLE,
            "tts": GTTS_AVAILABLE,
            "stt": STT_AVAILABLE,
            "ocr": OCR_AVAILABLE,
        },
        "language_count": len(LANGUAGES),
    })


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return error("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return error("Method not allowed", 405)


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return error("Internal server error", 500)


# ---------------------------------------------------------------------------
# Startup Banner
# ---------------------------------------------------------------------------
BANNER = r"""
  _____ _           _      _____                    _       _
 |  ___| | __ _ ___| | __ |_   _| __ __ _ _ __  __| |_   _| |_ ___  _ __
 | |_  | |/ _` / __| |/ /   | || '__/ _` | '_ \/ _` | | | | __/ _ \| '__|
 |  _| | | (_| \__ \   <    | || | | (_| | | | \__ \ | |_| | || (_) | |
 |_|   |_|\__,_|___/_|\_\   |_||_|  \__,_|_| |_|___/_|\__,_|\__\___/|_|

 Flask Translator API v1.0.0
 ================================================
 Supported languages : {lang_count}+
 JWT Auth            : {jwt}
 Text-to-Speech      : {tts}
 Speech-to-Text      : {stt}
 OCR / Image Trans.  : {ocr}

 Endpoints:
   GET  http://localhost:{port}/
   POST http://localhost:{port}/api/auth/token   <- Get JWT token
   GET  http://localhost:{port}/api/auth/verify  <- Verify token
   POST http://localhost:{port}/api/translate
   POST http://localhost:{port}/api/detect-language
   POST http://localhost:{port}/api/tts
   POST http://localhost:{port}/api/stt
   POST http://localhost:{port}/api/translate-image
   POST http://localhost:{port}/api/extract-text
   GET  http://localhost:{port}/api/languages
   GET  http://localhost:{port}/api/language-codes
   GET  http://localhost:{port}/api/language/<code>
   GET  http://localhost:{port}/api/search-languages?q=<query>
   GET  http://localhost:{port}/api/health

 Quick Start:
   1. Get a token:
      curl -X POST http://localhost:{port}/api/auth/token \\
           -H "Content-Type: application/json" \\
           -d '{{"username":"test","password":"any"}}'

   2. Translate text:
      curl -X POST http://localhost:{port}/api/translate \\
           -H "Authorization: Bearer <TOKEN>" \\
           -H "Content-Type: application/json" \\
           -d '{{"text":"Hello world","target_language":"es"}}'
 ================================================
"""


def print_banner():
    print(BANNER.format(
        lang_count=len(LANGUAGES),
        jwt="✓" if JWT_AVAILABLE else "✗ (pip install pyjwt)",
        tts="✓" if GTTS_AVAILABLE else "✗ (pip install gtts)",
        stt="✓" if STT_AVAILABLE else "✗ (pip install SpeechRecognition)",
        ocr="✓" if OCR_AVAILABLE else "✗ (pip install Pillow pytesseract)",
        port=PORT,
    ))


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_banner()
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
