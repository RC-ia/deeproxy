"""Configurações globais do DeepProxy."""
import json
import os

# ── .env ──────────────────────────────────────────────────────────────
_DOTENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.isfile(_DOTENV_PATH):
    with open(_DOTENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _val = _line.split("=", 1)
            _key = _key.strip()
            _val = _val.strip().strip("\"'")
            if not os.environ.get(_key):
                os.environ[_key] = _val
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.environ.get("DEEPROXY_BASE_DIR", os.path.dirname(__file__)))

CHROME_PROFILE_DIR = os.environ.get(
    "DEEPROXY_CHROME_PROFILE",
    os.path.join(BASE_DIR, "perfil_proxy"),
)

_ACCOUNTS_ENV = os.environ.get("DEEPROXY_ACCOUNTS", "")
if _ACCOUNTS_ENV:
    ACCOUNTS: dict[str, str] = json.loads(_ACCOUNTS_ENV)
else:
    ACCOUNTS = {"default": CHROME_PROFILE_DIR}

DEEPSEEK_URL = "https://chat.deepseek.com"

API_HOST = os.environ.get("DEEPROXY_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("DEEPROXY_PORT", "5001"))

DEFAULT_TIMEOUT = int(os.environ.get("DEEPROXY_TIMEOUT", "120"))
MAX_TIMEOUT = int(os.environ.get("DEEPROXY_MAX_TIMEOUT", "600"))

REPORTED_MODEL = os.environ.get("DEEPROXY_MODEL_NAME", "deepseek-chat")
