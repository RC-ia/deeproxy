"""
Parser próprio para detectar e extrair tool calls da resposta do DeepSeek.

Suporta múltiplos formatos:
  - XML com atributo: <tool_call name="x"> <arg>val</arg> 
  - XML com sub-tag: <skill> <name>x</name> <param>val</param> </skill>
  - JSON blocks: {"name": "x", "arguments": {...}}
  - Texto estruturado: [Tool Call]: x\n\nArguments: {"key": "val"}
"""
from __future__ import annotations

import json
import re
from typing import Any


def _build_tool_call(name: str, args: str | dict, call_id_counter: list[int]) -> dict[str, Any]:
    """Constrói um tool call no formato OpenAI."""
    call_id = f"call_{call_id_counter[0]}"
    call_id_counter[0] += 1
    
    if isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps(args)
    
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": args_str,
        },
    }


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """
    Extrai tool calls de um texto de resposta do DeepSeek.
    
    Retorna uma lista de tool calls no formato OpenAI:
    [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "nome_da_ferramenta",
          "arguments": "{\"arg\": \"val\"}"
        }
      },
      ...
    ]
    """
    tool_calls = []
    call_id_counter = [0]
    
    # 1. Tentar XML com atributo name: <tool_call name="x">...
    xml_attr_pattern = r'<tool_call\s+name=["\']([^"\']+)["\'](.*?)'
    for match in re.finditer(xml_attr_pattern, text, re.DOTALL | re.IGNORECASE):
        name = match.group(1).strip()
        content = match.group(2).strip()
        
        # Tenta extrair argumentos como JSON ou XML interno
        args = {}
        json_match = re.search(r'\{[^{}]*\}', content)
        if json_match:
            try:
                args = json.loads(json_match.group())
            except json.JSONDecodeError:
                args = {"raw": content}
        else:
            # Tenta extrair tags XML internas como <arg>val</arg>
            arg_matches = re.findall(r'<(\w+)>([^<]*)</\w+>', content)
            if arg_matches:
                args = {k: v.strip() for k, v in arg_matches}
            else:
                args = {"raw": content}
        
        tool_calls.append(_build_tool_call(name, args, call_id_counter))
    
    # 2. Tentar XML com sub-tag <name>: <skill>...<name>x</name>...</skill>
    xml_tag_pattern = r'<(\w+)>\s*<name>([^<]+)</name>(.*?)</\1>'
    for match in re.finditer(xml_tag_pattern, text, re.DOTALL | re.IGNORECASE):
        tag_name = match.group(1)
        func_name = match.group(2).strip()
        content = match.group(3).strip()
        
        args = {}
        param_matches = re.findall(r'<(\w+)>([^<]*)</\1>', content)
        if param_matches:
            args = {k: v.strip() for k, v in param_matches}
        else:
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                try:
                    args = json.loads(json_match.group())
                except json.JSONDecodeError:
                    args = {"raw": content}
            else:
                args = {"raw": content}
        
        tool_calls.append(_build_tool_call(func_name, args, call_id_counter))
    
    # 3. Tentar JSON blocks: {"name": "x", "arguments": {...}}
    json_block_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}'
    for match in re.finditer(json_block_pattern, text, re.DOTALL):
        name = match.group(1).strip()
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}
        tool_calls.append(_build_tool_call(name, args, call_id_counter))
    
    # 4. Tentar formato de texto: [Tool Call]: nome\n\nArguments: {...}
    text_pattern = r'\[Tool Call\]:\s*([^\n]+)\s*\n\s*\n?\s*Arguments?:\s*(\{[^{}]*\})?'
    for match in re.finditer(text_pattern, text, re.IGNORECASE):
        name = match.group(1).strip()
        args_str = match.group(2) if match.group(2) else "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}
        tool_calls.append(_build_tool_call(name, args, call_id_counter))
    
    return tool_calls


def has_tool_call(text: str) -> bool:
    """Verifica se o texto contém pelo menos um tool call."""
    return len(parse_tool_calls_from_text(text)) > 0


def extract_tool_call_info(text: str) -> dict[str, Any] | None:
    """
    Extrai o primeiro tool call encontrado e retorna informações estruturadas.
    
    Retorna None se nenhum tool call for encontrado.
    """
    calls = parse_tool_calls_from_text(text)
    if not calls:
        return None
    
    return {
        "has_tool_call": True,
        "tool_calls": calls,
        "count": len(calls),
    }
