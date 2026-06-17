"""
Parser proprio para detectar e formatar tool calls no estilo XML estruturado.

Este parser converte tool calls detectados em um formato XML proprio que o
cliente pode reconhecer facilmente, sem dependencia do formato OpenAI.

Formato de saida:
    <tool_call name="write_file">
    <filepath>C:\\Users\\file.html</filepath>
    <content><![CDATA[...]]></content>
    
"""
from __future__ import annotations

import json
import re
from typing import Any


class ToolCallFormatter:
    """
    Parser e formatador de tool calls para o formato XML estruturado.
    """
    
    def __init__(self):
        self._call_id_counter = 0
    
    def _reset_counter(self):
        self._call_id_counter = 0
    
    def _extract_json_from_content(self, content: str) -> dict:
        """Extrai JSON de um conteúdo, lidando com braces aninhados."""
        start = content.find('{')
        if start < 0:
            return {}
        
        count = 0
        end = start
        for i, c in enumerate(content[start:], start):
            if c == '{':
                count += 1
            elif c == '}':
                count -= 1
                if count == 0:
                    end = i + 1
                    break
        
        json_str = content[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str_fixed = json_str.replace('\\', '\\\\')
                return json.loads(json_str_fixed)
            except json.JSONDecodeError:
                return {"raw": content.strip()}
    
    def _format_tool_call_to_xml(self, name: str, args: dict | str) -> str:
        """
        Converte um tool call para o formato XML estruturado.
        
        Args:
            name: nome da ferramenta
            args: argumentos como dict ou string JSON
            
        Returns:
            String formatada no estilo XML com CDATA para conteúdo
        """
        if isinstance(args, str):
            try:
                args_dict = json.loads(args)
            except json.JSONDecodeError:
                args_dict = {"raw": args}
        else:
            args_dict = args
        
        xml_parts = [f'<tool_call name="{name}">']
        
        priority_keys = ['filepath', 'path', 'file_path', 'content', 'code']
        other_keys = [k for k in args_dict.keys() if k not in priority_keys]
        sorted_keys = [k for k in priority_keys if k in args_dict] + other_keys
        
        for key in sorted_keys:
            value = args_dict[key]
            if isinstance(value, str) and ('<' in value or '>' in value or len(value) > 100):
                xml_parts.append(f'<{key}><![CDATA[{value}]]></{key}>')
            else:
                xml_parts.append(f'<{key}>{value}</{key}>')
        
        xml_parts.append('')
        return '\n'.join(xml_parts)
    
    def parse_and_format(self, text: str) -> tuple[str, list[str]]:
        """
        Extrai tool calls do texto e os formata no estilo XML.
        
        Args:
            text: texto da resposta do DeepSeek
            
        Returns:
            Tupla (texto_sem_tools, lista_de_tool_calls_xml)
        """
        self._reset_counter()
        tool_calls_xml = []
        cleaned_text = text
        
        # Padrão 1: <tool_call name="x"> seguido de JSON ou conteudo até o fim da linha ou outra tag
        # Captura tudo após a tag de abertura
        xml_attr_pattern = r'<tool_call\s+name=["\']([^"\']+)["\']\s*>([\s\S]*?)(?=<tool_call|$)'
        for match in re.finditer(xml_attr_pattern, cleaned_text, re.IGNORECASE):
            full_match = match.group(0)
            name = match.group(1).strip()
            content = match.group(2).strip()
            
            # Primeiro tenta extrair JSON diretamente do conteúdo (pode estar em uma única linha ou multilinha)
            args = self._extract_json_from_content(content)
            
            # Se encontrou JSON válido, usa-o; caso contrário, tenta tags XML
            if not args or "raw" in args:
                # Tenta extrair tags XML internas como <filepath>...</filepath>
                arg_matches = re.findall(r'<(\w+)>([^<]*)</\w+>', content)
                if arg_matches:
                    args = {k: v.strip() for k, v in arg_matches}
            
            xml_formatted = self._format_tool_call_to_xml(name, args)
            tool_calls_xml.append(xml_formatted)
            cleaned_text = cleaned_text.replace(full_match, '')
        
        # Padrão 2: Formato JSON {"name": "x", "arguments": {...}}
        json_block_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}'
        for match in re.finditer(json_block_pattern, cleaned_text, re.DOTALL):
            full_match = match.group(0)
            name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw": args_str}
            
            xml_formatted = self._format_tool_call_to_xml(name, args)
            tool_calls_xml.append(xml_formatted)
            cleaned_text = cleaned_text.replace(full_match, '')
        
        # Padrão 3: [Tool Call]: nome\n\nArguments: {...}
        text_pattern = r'\[Tool Call\]:\s*([^\n]+)\s*\n\s*\n?\s*Arguments?:\s*(\{[^{}]*\})?'
        for match in re.finditer(text_pattern, cleaned_text, re.IGNORECASE):
            full_match = match.group(0)
            name = match.group(1).strip()
            args_str = match.group(2) if match.group(2) else "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw": args_str}
            
            xml_formatted = self._format_tool_call_to_xml(name, args)
            tool_calls_xml.append(xml_formatted)
            cleaned_text = cleaned_text.replace(full_match, '')
        
        return cleaned_text.strip(), tool_calls_xml


# Instância global do formatter
formatter = ToolCallFormatter()


def parse_tool_calls_from_text(text: str) -> list[str]:
    """
    Extrai tool calls de um texto e retorna no formato XML estruturado.
    
    Retorna uma lista de strings formatadas como XML:
    [
      '<tool_call name="write_file">\\n<filepath>...</filepath>\\n<content><![CDATA[...]]></content>\\n',
      ...
    ]
    """
    _, tool_calls_xml = formatter.parse_and_format(text)
    return tool_calls_xml


def has_tool_call(text: str) -> bool:
    """Verifica se o texto contém pelo menos um tool call."""
    return len(parse_tool_calls_from_text(text)) > 0


def extract_tool_call_info(text: str) -> dict[str, Any] | None:
    """
    Extrai tool calls encontrados e retorna informações estruturadas.
    
    Retorna None se nenhum tool call for encontrado.
    """
    _, tool_calls_xml = formatter.parse_and_format(text)
    if not tool_calls_xml:
        return None
    
    return {
        "has_tool_call": True,
        "tool_calls_xml": tool_calls_xml,
        "count": len(tool_calls_xml),
    }
