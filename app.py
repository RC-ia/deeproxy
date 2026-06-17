"""
DeepProxy — API compatível com OpenAI Chat Completions que encaminha
requisições para o chat.deepseek.com via Selenium.

Endpoints:
  GET  /v1/models             -> lista de modelos (apenas 1: deepseek-chat)
  POST /v1/chat/completions   -> Chat Completions (com suporte a stream REAL)
  POST /tools/call            -> Executa uma ferramenta
  GET  /tools/schema          -> Schemas das ferramentas
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

from flask import Flask, Response, jsonify, request

import config
import proxy
import tools

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    accounts_info: dict[str, dict] = {}
    for name in config.ACCOUNTS:
        driver = proxy.pool._drivers.get(name)
        accounts_info[name] = {
            "ready": driver is not None,
            "profile": config.ACCOUNTS[name],
        }
    return jsonify({
        "status": "ok",
        "service": "deeproxy",
        "model": config.REPORTED_MODEL,
        "deepseek_url": config.DEEPSEEK_URL,
        "accounts": accounts_info,
    })


@app.route("/tools/schema", methods=["GET"])
def tool_schemas():
    return jsonify({"tools": tools.TOOL_SCHEMAS})


@app.route("/tools/call", methods=["POST"])
def tool_call():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Body JSON inválido."}), 400
    name = data.get("name")
    args = data.get("args", {})
    if not name:
        return jsonify({"success": False, "error": "Campo 'name' é obrigatório."}), 400
    result = tools.execute_tool(name, args)
    return jsonify(result)


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
    account = request.headers.get("X-Account") or data.get("account") or None

    # --- STREAMING: repassa deltas brutos do DeepSeek ---
    if stream:
        id_ = f"chatcmpl-{int(time.time())}"
        criado = int(time.time())

        def _chunk(**kw):
            return "data: " + json.dumps({
                "id": id_, "object": "chat.completion.chunk",
                "created": criado, "model": model,
                "choices": [{"index": 0, **kw}],
            }) + "\n\n"

        def gerar():
            yield _chunk(delta={"role": "assistant"}, finish_reason=None)

            try:
                for delta in proxy.stream_prompt(messages, timeout=timeout, account=account):
                    if delta:
                        yield _chunk(delta={"content": delta}, finish_reason=None)
            except TimeoutError as e:
                yield _chunk(delta={"content": f"\n\n[ERRO: {e}]"}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                traceback.print_exc()
                yield _chunk(delta={"content": f"\n\n[ERRO: {e}]"}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

            yield _chunk(delta={}, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return Response(gerar(), mimetype="text/event-stream")

    # --- NÃO-STREAMING ---
    try:
        texto, tool_calls = proxy.send_prompt(messages, timeout=timeout, account=account)
    except TimeoutError as e:
        return jsonify({"error": {"message": str(e), "type": "timeout_error"}}), 504
    except ValueError as e:
        return jsonify({"error": {"message": str(e), "type": "invalid_request_error"}}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": f"Falha ao consultar o DeepSeek: {e}",
                "type": "server_error",
            }
        }), 500

    agora = int(time.time())
    
    # Constrói a mensagem no formato OpenAI
    message_data = {"role": "assistant", "content": texto}
    
    # Se houver tool calls, adiciona no formato OpenAI
    if tool_calls:
        openai_tool_calls = []
        for idx, xml_call in enumerate(tool_calls):
            # Extrai nome e argumentos do XML
            name_match = re.search(r'<tool_call\s+name=["\']([^"\']+)["\']', xml_call)
            if name_match:
                tool_name = name_match.group(1)
                # Extrai todos os parâmetros do XML
                args = {}
                param_matches = re.findall(r'<(\w+)>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</\w+>', xml_call, re.DOTALL)
                for param_name, param_value in param_matches:
                    if param_name != 'tool_call':  # Ignora a tag principal
                        args[param_name.strip()] = param_value.strip()
                
                openai_tool_calls.append({
                    "id": f"call_{idx + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args)
                    }
                })
        
        if openai_tool_calls:
            message_data["tool_calls"] = openai_tool_calls
    
    return jsonify({
        "id": f"chatcmpl-{agora}",
        "object": "chat.completion",
        "created": agora,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message_data,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"[DeepProxy] 🟢 Subindo API em http://{config.API_HOST}:{config.API_PORT}")
    print(f"[DeepProxy]    Modelo reportado: {config.REPORTED_MODEL}")
    print(f"[DeepProxy]    Timeout padrão: {config.DEFAULT_TIMEOUT}s")
    print(f"[DeepProxy]    Contas configuradas: {list(config.ACCOUNTS.keys())}")

    for name in config.ACCOUNTS:
        try:
            proxy.pool.get_driver(name)
            print(f"[DeepProxy] 🟢 Conta '{name}' pronta.")
        except Exception as e:
            traceback.print_exc()
            print(f"[DeepProxy] ⚠️ Conta '{name}' falhou ao iniciar: {e}")
        time.sleep(2)

    try:
        app.run(host=config.API_HOST, port=config.API_PORT, threaded=True)
    finally:
        proxy.pool.shutdown()


if __name__ == "__main__":
    main()
