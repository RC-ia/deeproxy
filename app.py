"""
DeepProxy — API compatível com OpenAI Chat Completions que encaminha
requisições para o chat.deepseek.com via Selenium.

Endpoints:
  GET  /v1/models             -> lista de modelos (apenas 1: deepseek-chat)
  POST /v1/chat/completions   -> Chat Completions (com suporte a stream REAL)
  GET  /health                -> status do serviço

Uso:
  pip install -r requirements.txt
  python app.py

Na primeira execução, faça login manualmente na janela do Chrome que abrir.
"""
from __future__ import annotations

import json
import time
import traceback

from flask import Flask, Response, jsonify, request

import config
import proxy


app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_user_prompt(messages: list) -> str:
    """
    Extrai o conteúdo da última mensagem do usuário.

    Para compatibilidade com clientes OpenAI, converte o histórico completo
    em um único prompt concatenado quando há múltiplas mensagens.
    """
    if not messages:
        raise ValueError("'messages' está vazio.")

    users = [m for m in messages if m.get("role") == "user"]
    if len(messages) <= 2 and len(users) == 1 and not any(
        m.get("role") == "assistant" for m in messages
    ):
        return str(users[0].get("content", ""))

    linhas = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "system":
            linhas.append(f"[System Instructions]\n{content}\n")
        elif role == "user":
            linhas.append(f"[User]\n{content}\n")
        elif role == "assistant":
            linhas.append(f"[Assistant]\n{content}\n")
    return "\n".join(linhas).strip()


def _build_completion_payload(
    texto: str,
    model: str,
    finish_reason: str = "stop",
) -> dict:
    agora = int(time.time())
    return {
        "id": f"chatcmpl-{agora}",
        "object": "chat.completion",
        "created": agora,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": texto,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "deeproxy",
        "model": config.REPORTED_MODEL,
        "deepseek_url": config.DEEPSEEK_URL,
    })


@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": config.REPORTED_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
            }
        ],
    })


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": {"message": "Body JSON inválido.", "type": "invalid_request_error"}}), 400

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({
            "error": {
                "message": "'messages' é obrigatório e deve ser uma lista.",
                "type": "invalid_request_error",
            }
        }), 400

    model = data.get("model") or config.REPORTED_MODEL
    stream = bool(data.get("stream", False))
    timeout = int(data.get("timeout", config.DEFAULT_TIMEOUT))

    try:
        prompt = _extract_last_user_prompt(messages)
    except ValueError as e:
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error"}}), 400

    # --- STREAMING REAL (deltas capturados via MutationObserver) ---
    if stream:
        id_ = f"chatcmpl-{int(time.time())}"
        criado = int(time.time())

        def gerar():
            # Chunk 1: role
            yield (
                "data: " + json.dumps({
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": criado,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }) + "\n\n"
            )

            try:
                for delta in proxy.stream_prompt(prompt, timeout=timeout):
                    yield (
                        "data: " + json.dumps({
                            "id": id_,
                            "object": "chat.completion.chunk",
                            "created": criado,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }],
                        }) + "\n\n"
                    )
            except TimeoutError as e:
                # Envia erro como chunk final (cliente OpenAI normalmente só fecha conexão)
                yield (
                    "data: " + json.dumps({
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": criado,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\n[ERRO: {e}]"},
                            "finish_reason": "stop",
                        }],
                    }) + "\n\n"
                )
            except Exception as e:
                traceback.print_exc()
                yield (
                    "data: " + json.dumps({
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": criado,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\n[ERRO: {e}]"},
                            "finish_reason": "stop",
                        }],
                    }) + "\n\n"
                )

            # Chunk final: finish
            yield (
                "data: " + json.dumps({
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": criado,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }) + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return Response(gerar(), mimetype="text/event-stream")

    # --- NÃO-STREAMING ---
    try:
        texto = proxy.send_prompt(prompt, timeout=timeout)
    except TimeoutError as e:
        return jsonify({"error": {"message": str(e), "type": "timeout_error"}}), 504
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": f"Falha ao consultar o DeepSeek: {e}",
                "type": "server_error",
            }
        }), 500

    return jsonify(_build_completion_payload(texto, model))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"[DeepProxy] 🟢 Subindo API em http://{config.API_HOST}:{config.API_PORT}")
    print(f"[DeepProxy]    Modelo reportado: {config.REPORTED_MODEL}")
    print(f"[DeepProxy]    Timeout padrão: {config.DEFAULT_TIMEOUT}s")
    print(f"[DeepProxy]    Streaming: REAL (via MutationObserver)")

    try:
        proxy.get_driver()
    except Exception as e:
        print(f"[DeepProxy] ⚠️ Não foi possível iniciar o Chrome: {e}")

    try:
        app.run(host=config.API_HOST, port=config.API_PORT, threaded=True)
    finally:
        proxy.shutdown()


if __name__ == "__main__":
    main()
