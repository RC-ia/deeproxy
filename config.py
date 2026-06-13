"""Configurações globais do DeepProxy."""
import os

# Diretório base onde os perfis do Chrome serão armazenados
BASE_DIR = os.path.abspath(os.environ.get("DEEPROXY_BASE_DIR", os.path.dirname(__file__)))

# Pasta do perfil do Chrome (mantém login persistente entre restarts)
CHROME_PROFILE_DIR = os.environ.get(
    "DEEPROXY_CHROME_PROFILE",
    os.path.join(BASE_DIR, "perfil_proxy"),
)

# URL do DeepSeek Chat
DEEPSEEK_URL = "https://chat.deepseek.com"

# Host/porta da API
API_HOST = os.environ.get("DEEPROXY_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("DEEPROXY_PORT", "4000"))

# Timeouts (segundos)
DEFAULT_TIMEOUT = int(os.environ.get("DEEPROXY_TIMEOUT", "120"))
MAX_TIMEOUT = int(os.environ.get("DEEPROXY_MAX_TIMEOUT", "600"))

# Modelo reportado nas respostas (compatibilidade OpenAI)
REPORTED_MODEL = os.environ.get("DEEPROXY_MODEL_NAME", "deepseek-chat")
