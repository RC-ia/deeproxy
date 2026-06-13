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
import re
import time
import traceback
from typing import List, Optional, Tuple

from flask import Flask, Response, jsonify, request

import config
import proxy


app = Flask(__name__)


# ---------------------------------------------------------------------------
# Tool Call Parsing
# ---------------------------------------------------------------------------
# O DeepSeek web não tem suporte nativo a tool calls — quando o modelo
# decide usar uma ferramenta, ele emite algo como:
#   
#     {"name": "get_weather", "arguments": {...}}
#   
# ou
#   ```json
#   {"name": "get_weather", "arguments": {...}}
#   ```
# Este parser detecta esses padrões e converte pro formato OpenAI nativo:
#   message.tool_calls = [{id, type, function: {name, arguments}}]
#   finish_reason = "tool_calls"

_TC_PATTERNS = [
    # Hermes-style
    re.compile(
        r'\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*(?:\{[^{}]*\}|"[^"]*")\s*\}\s*',
        re.DOTALL,
    ),
    # JSON code-fence block
    re.compile(
        r'```(?:json)?\s*\n?\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*(?:\{.*?\}|".*?")\s*\}\s*\n?```',
        re.DOTALL,
    ),
]


def _try_parse_tool_json(bloco: str) -> Optional[dict]:
    """Tenta extrair um JSON de tool call de um bloco de texto."""
    # Remove fences de código
    bloco = re.sub(r'^```(?:json)?\s*\n?', '', bloco.strip())
    bloco = re.sub(r'\n?```\s*$', '', bloco.strip())
    # Remove tags XML
    bloco = re.sub(r'^[^{]*', '', bloco)
    bloco = re.sub(r'[^}]*$', '', bloco)
    try:
        obj = json.loads(bloco)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function")
    if not name:
        return None
    args = obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(args, dict):
        args = json.dumps(args, ensure_ascii=False)
    return {"name": str(name), "arguments": str(args)}


def _parse_xml_tool_call(name: str, body: str) -> Optional[dict]:
    """Extrai argumentos de sub-tags XML."""
    args = {}
    sub_re = re.compile(
        r'<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>',
        re.DOTALL,
    )
    for m in sub_re.finditer(body):
        args[m.group(1)] = m.group(2).strip()
    if not args:
        bs = body.strip()
        if bs.startswith("{"):
            try:
                args = json.loads(bs)
            except Exception:
                args = {"input": bs}
        elif bs:
            args = {"input": bs}
    return {
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
    }


def _extract_tool_calls(texto: str) -> Tuple[str, List[dict]]:
    """
    Varre o texto procurando tool calls (JSON ou XML). Retorna:
      (texto_sem_tool_calls, lista_de_tool_calls_no_formato_OpenAI)
    """
    if not texto:
        return "", []

    tool_calls = []
    texto_limpo = texto
    id_counter = 0

    # 1. XML-style tool calls
    xml_pat = (
        re.escape('<tool_call') +
        r'\\s+name\\s*=\\s*[' + chr(34) + chr(39) + r']([^' + chr(34) + chr(39) + r']+)[' + chr(34) + chr(39) + r']' + r'\\s*' +
        re.escape('>') +
        r'(.*?)' +
        re.escape('</tool_call>')
    )
    xml_re = re.compile(xml_pat, re.DOTALL)
    novo = texto_limpo
    for m in reversed(list(xml_re.finditer(novo))):
        parsed = _parse_xml_tool_call(m.group(1), m.group(2))
        if parsed:
            id_counter += 1
            tool_calls.insert(0, {
                "id": f"call_{int(time.time())}_{id_counter:03d}",
                "type": "function",
                "function": {"name": parsed["name"], "arguments": parsed["arguments"]},
            })
            novo = novo[:m.start()] + novo[m.end():]
    texto_limpo = novo

    # Remove wrapper tags se sobraram
    texto_limpo = texto_limpo.replace('<tool_calls>', "")
    texto_limpo = texto_limpo.replace('</tool_calls>', "")

    # 2. JSON-style (logica original)
    if '"name"' in texto_limpo:
        for pattern in _TC_PATTERNS:
            novo = texto_limpo
            for m in reversed(list(pattern.finditer(novo))):
                parsed = _try_parse_tool_json(m.group(0))
                if parsed:
                    id_counter += 1
                    tool_calls.insert(0, {
                        "id": f"call_{int(time.time())}_{id_counter:03d}",
                        "type": "function",
                        "function": {"name": parsed["name"], "arguments": parsed["arguments"]},
                    })
                    novo = novo[:m.start()] + novo[m.end():]
            texto_limpo = novo

    texto_limpo = re.sub(r'\n{3,}', '\n\n', texto_limpo).strip()
    return texto_limpo, tool_calls

def _build_completion_payload(
    texto: str,
    model: str,
    tool_calls: Optional[List[dict]] = None,
) -> dict:
    agora = int(time.time())
    finish_reason = "tool_calls" if tool_calls else "stop"
    message = {"role": "assistant"}
    if tool_calls:
        message["content"] = texto if texto else None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = texto
    return {
        "id": f"chatcmpl-{agora}",
        "object": "chat.completion",
        "created": agora,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
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

            texto_acumulado = ""
            try:
                for delta in proxy.stream_prompt(prompt, timeout=timeout):
                    texto_acumulado += delta
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
                yield "data: [DONE]\n\n"
                return
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
                yield "data: [DONE]\n\n"
                return

            # Verifica se há tool calls no texto acumulado
            _, tool_calls = _extract_tool_calls(texto_acumulado)
            finish = "tool_calls" if tool_calls else "stop"

            # Se encontrou tool calls, emite chunk com a estrutura
            if tool_calls:
                yield (
                    "data: " + json.dumps({
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": criado,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [
                                {
                                    "index": i,
                                    "id": tc["id"],
                                    "type": tc["type"],
                                    "function": tc["function"],
                                } for i, tc in enumerate(tool_calls)
                            ]},
                            "finish_reason": None,
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
                        "finish_reason": finish,
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

    # Extrai tool calls do texto gerado
    texto_limpo, tool_calls = _extract_tool_calls(texto)

    return jsonify(_build_completion_payload(texto_limpo, model, tool_calls=tool_calls))


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
