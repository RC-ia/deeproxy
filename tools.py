"""Ferramentas de criação e edição de código expostas como API."""
from __future__ import annotations

import os
import subprocess
import traceback

import config

BASE_DIR = config.BASE_DIR


def _resolve_path(path: str) -> str:
    full = os.path.normpath(os.path.join(BASE_DIR, path))
    if not full.startswith(os.path.normpath(BASE_DIR)):
        raise PermissionError(f"Acesso negado: {path} está fora do diretório base.")
    return full


def write_file(path: str, content: str) -> dict:
    full = _resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return {"success": True, "path": path, "size": len(content)}


def read_file(path: str) -> dict:
    full = _resolve_path(path)
    if not os.path.isfile(full):
        return {"success": False, "error": f"Arquivo não encontrado: {path}"}
    with open(full, "r", encoding="utf-8") as f:
        content = f.read()
    return {"success": True, "path": path, "content": content, "size": len(content)}


def edit_file(path: str, old_string: str, new_string: str) -> dict:
    full = _resolve_path(path)
    if not os.path.isfile(full):
        return {"success": False, "error": f"Arquivo não encontrado: {path}"}
    with open(full, "r", encoding="utf-8") as f:
        content = f.read()
    if old_string not in content:
        return {"success": False, "error": f"Texto não encontrado no arquivo."}
    new_content = content.replace(old_string, new_string, 1)
    with open(full, "w", encoding="utf-8") as f:
        f.write(new_content)
    return {"success": True, "path": path, "size": len(new_content)}


def list_directory(path: str = "") -> dict:
    full = _resolve_path(path)
    if not os.path.isdir(full):
        return {"success": False, "error": f"Diretório não encontrado: {path}"}
    entries = []
    for entry in os.scandir(full):
        entries.append({
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else 0,
        })
    entries.sort(key=lambda e: (e["type"] != "directory", e["name"]))
    return {"success": True, "path": path, "entries": entries}


def run_command(command: str, cwd: str = "") -> dict:
    workdir = _resolve_path(cwd) if cwd else BASE_DIR
    try:
        r = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=60,
        )
        return {
            "success": r.returncode == 0,
            "returncode": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Comando excedeu o tempo limite de 60s."}
    except Exception as e:
        return {"success": False, "error": str(e)}


TOOL_SCHEMAS = [
    {
        "name": "write_file",
        "description": "Cria ou sobrescreve um arquivo com conteúdo.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho relativo ao projeto."},
                "content": {"type": "string", "description": "Conteúdo do arquivo."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Lê o conteúdo de um arquivo.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho relativo ao projeto."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edita um arquivo substituindo um trecho de texto por outro.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho relativo ao projeto."},
                "old_string": {"type": "string", "description": "Texto a ser substituído."},
                "new_string": {"type": "string", "description": "Novo texto."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_directory",
        "description": "Lista arquivos e pastas de um diretório.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho relativo ao projeto (vazio = raiz)."},
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": "Executa um comando no terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Comando a ser executado."},
                "cwd": {"type": "string", "description": "Diretório de trabalho (relativo ao projeto)."},
            },
            "required": ["command"],
        },
    },
]

TOOL_FUNCTIONS = {
    "write_file": write_file,
    "read_file": read_file,
    "edit_file": edit_file,
    "list_directory": list_directory,
    "run_command": run_command,
}


def execute_tool(name: str, args: dict) -> dict:
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return {"success": False, "error": f"Ferramenta desconhecida: {name}"}
    try:
        return fn(**args)
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
