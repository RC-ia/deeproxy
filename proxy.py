"""
proxy.py — núcleo do DeepProxy.

Abre o DeepSeek Chat via Selenium, envia mensagens (formato texto puro) e
captura a resposta. Suporta dois modos:
  - send_prompt(messages)      -> retorna a resposta completa (str)
  - stream_prompt(messages)    -> generator que yield deltas em tempo real

A captura incremental usa um MutationObserver injetado via JS que guarda
os fragmentos novos numa variável window.__deepseek_deltas. O Python faz
polling dessa variável a cada ~200ms e repassa os deltas ao cliente.

Este módulo inclui um parser próprio para detectar tool calls na resposta.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Iterator, Optional

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import config
import parser


# ---------------------------------------------------------------------------
# Account Pool — gerencia múltiplos drivers (um por conta/projeto)
# ---------------------------------------------------------------------------

class AccountPool:
    """Pool de navegadores Chrome, um por conta configurada.

    Cada conta tem seu próprio perfil Chrome (login persistente) e
    seu próprio lock, permitindo que requisições paralelas usem
    contas diferentes simultaneamente.
    """

    def __init__(self) -> None:
        self._drivers: dict[str, webdriver.Chrome] = {}
        self._locks: dict[str, threading.Lock] = {
            name: threading.Lock() for name in config.ACCOUNTS
        }
        self._rr_index = 0
        self._rr_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_driver(self, account: str | None = None) -> tuple[str, webdriver.Chrome]:
        """Retorna (nome_da_conta, driver) para a conta solicitada.

        Se ``account`` for None ou não existir, faz round-robin entre
        as contas disponíveis.
        """
        names = list(config.ACCOUNTS.keys())
        if account and account in names:
            name = account
        else:
            with self._rr_lock:
                name = names[self._rr_index % len(names)]
                self._rr_index += 1

        driver = self._drivers.get(name)
        if driver is None:
            profile = os.path.abspath(config.ACCOUNTS[name])
            print(f"[Proxy] 🚀 Iniciando Chrome para conta '{name}' (perfil: {profile})...")
            opts = Options()
            opts.add_argument(f"--user-data-dir={profile}")
            opts.add_argument("--window-size=1100,900")
            opts.add_argument("--no-first-run")
            opts.add_argument("--disable-default-apps")
            opts.add_argument("--remote-debugging-port=0")
            opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            opts.add_experimental_option("useAutomationExtension", False)
            try:
                driver = webdriver.Chrome(options=opts)
            except Exception as e:
                import traceback as _tb
                _tb.print_exc()
                raise
            driver.set_script_timeout(config.MAX_TIMEOUT)
            driver.get(config.DEEPSEEK_URL)
            print(f"[Proxy] 🟢 Conta '{name}' pronta. Faça login manualmente em {config.DEEPSEEK_URL}")
            self._drivers[name] = driver
        return name, driver

    def get_lock(self, account: str) -> threading.Lock:
        return self._locks[account]

    def shutdown(self) -> None:
        for name, driver in self._drivers.items():
            try:
                driver.quit()
            except Exception:
                pass
        self._drivers.clear()


pool = AccountPool()


def _click_search_button(driver: webdriver.Chrome) -> None:
    js = """
    const botoes = document.querySelectorAll('._58b31c9 .ds-atom-button');
    const btn = Array.from(botoes).find(b => b.textContent.trim().toLowerCase() === 'search');
    if (btn) { btn.click(); return true; }
    return false;
    """
    try:
        driver.execute_script(js)
    except Exception:
        pass


def _click_alternate_button(driver: webdriver.Chrome) -> None:
    js = """
(function() {
  const botaoNaoSelecionado = document.querySelector('._9f2341b._7ac2123:not(._31a22b0)');
  if (botaoNaoSelecionado) {
    const rect = botaoNaoSelecionado.getBoundingClientRect();
    const x = rect.left + (rect.width / 2);
    const y = rect.top + (rect.height / 2);
    const options = { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y };
    botaoNaoSelecionado.dispatchEvent(new MouseEvent('mousedown', options));
    botaoNaoSelecionado.dispatchEvent(new MouseEvent('mouseup', options));
    botaoNaoSelecionado.dispatchEvent(new MouseEvent('click', options));
  }
})();
"""
    try:
        driver.execute_script(js)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _find_textarea(driver: webdriver.Chrome):
    candidatos = [
        (By.CSS_SELECTOR, "textarea#chat-input"),
        (By.CSS_SELECTOR, "textarea[placeholder]"),
        (By.CSS_SELECTOR, "div[contenteditable='true']"),
    ]
    for by, sel in candidatos:
        try:
            el = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((by, sel))
            )
            return el
        except TimeoutException:
            continue
    raise RuntimeError("Não foi possível localizar o campo de input do DeepSeek.")


def _inject_prompt(driver: webdriver.Chrome, prompt: str) -> None:
    textarea = _find_textarea(driver)
    textarea.click()
    time.sleep(0.05)
    driver.execute_script(
        """
        const el = arguments[0];
        if (el.tagName === 'TEXTAREA') {
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(el, arguments[1]);
            el.dispatchEvent(new Event('input', { bubbles: true }));
        } else {
            el.innerText = arguments[1];
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }
        """,
        textarea,
        prompt,
    )


def _submit_prompt(driver: webdriver.Chrome) -> None:
    botoes_send = [
        (By.CSS_SELECTOR, "button[aria-label*='end' i]"),
        (By.CSS_SELECTOR, "button[aria-label*='ubmit' i]"),
        (By.CSS_SELECTOR, "button.ds-icon-button"),
    ]
    for by, sel in botoes_send:
        try:
            btn = driver.find_element(by, sel)
            if btn.is_displayed() and btn.is_enabled():
                btn.click()
                return
        except (NoSuchElementException, WebDriverException):
            continue
    try:
        textarea = _find_textarea(driver)
        textarea.send_keys(Keys.ENTER)
    except Exception as e:
        raise RuntimeError(f"Não foi possível submeter o prompt: {e}")


# ---------------------------------------------------------------------------
# Observador de deltas (MutationObserver injetado)
# ---------------------------------------------------------------------------

_JS_INSTALL_OBSERVER = """
// Reinicia estado
window.__deepseek_deltas = [];
window.__deepseek_last_text = '';
window.__deepseek_done = false;

// Para observer anterior se houver
if (window.__deepseek_observer) {
    try { window.__deepseek_observer.disconnect(); } catch(e) {}
}

const observer = new MutationObserver(() => {
    // Pega a ÚLTIMA mensagem do assistente (a que está sendo gerada agora)
    const msgs = document.querySelectorAll('.ds-markdown.ds-assistant-message-main-content');
    if (!msgs.length) return;
    const msg = msgs[msgs.length - 1];
    const currentText = msg.innerText || '';
    const last = window.__deepseek_last_text;

    if (currentText !== last && currentText.length > last.length) {
        // Só consideramos o delta se começar com o texto anterior
        // (evita pegarmos lixo quando o DOM é re-renderizado do zero)
        if (currentText.startsWith(last)) {
            const delta = currentText.substring(last.length);
            if (delta.length > 0) {
                window.__deepseek_deltas.push(delta);
            }
        } else {
            // Re-render: trata o texto inteiro como delta (primeira vez ou reset)
            if (last === '' && currentText.length > 0) {
                window.__deepseek_deltas.push(currentText);
            }
        }
        window.__deepseek_last_text = currentText;
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true
});

window.__deepseek_observer = observer;
return true;
"""

_JS_POLL_DELTAS = """
// Retorna os deltas acumulados e limpa o buffer.
const deltas = window.__deepseek_deltas || [];
window.__deepseek_deltas = [];

// Detecção robusta de "gerando":
// 1. Botão Stop visível
const stopBtn = document.querySelector('button[aria-label*="top" i], button[aria-label*="ancel" i], button[aria-label*="Stop" i], [class*="stop"]');
const stopVisible = stopBtn && stopBtn.offsetParent !== null;

// 2. Textarea desabilitado/readonly (sinal clássico de gerando)
const textarea = document.querySelector('textarea');
const isTextareaDisabled = textarea && (textarea.disabled || textarea.readOnly);

// 3. Cursor piscando ou indicador de thinking
const cursor = document.querySelector('.cursor-blink, [class*="typing"], [class*="thinking"]');
const hasCursor = !!cursor;

const msgs = document.querySelectorAll('.ds-markdown.ds-assistant-message-main-content, [class*="message-content"]');
const hasAssistant = msgs.length > 0;

// "Gerando" se: stop visível, OU textarea desabilitado, OU cursor presente
const isGenerating = stopVisible || isTextareaDisabled || hasCursor;

// done apenas se tem mensagem e NÃO está gerando
const done = hasAssistant && !isGenerating;

return {
    deltas: deltas,
    done: done,
    total_len: (window.__deepseek_last_text || '').length,
    is_generating: isGenerating
};
"""

_JS_STOP_OBSERVER = """
if (window.__deepseek_observer) {
    try { window.__deepseek_observer.disconnect(); } catch(e) {}
    window.__deepseek_observer = null;
}
const final_text = window.__deepseek_last_text || '';
const remaining = window.__deepseek_deltas || [];
window.__deepseek_deltas = [];
return { final_text: final_text, remaining: remaining };
"""


def _install_observer(driver: webdriver.Chrome) -> None:
    driver.execute_script(_JS_INSTALL_OBSERVER)


def _poll_deltas(driver: webdriver.Chrome) -> dict:
    try:
        result = driver.execute_script(_JS_POLL_DELTAS)
        return result or {"deltas": [], "done": False, "total_len": 0}
    except Exception:
        return {"deltas": [], "done": False, "total_len": 0}


def _stop_observer(driver: webdriver.Chrome) -> dict:
    try:
        result = driver.execute_script(_JS_STOP_OBSERVER)
        return result or {"final_text": "", "remaining": []}
    except Exception:
        return {"final_text": "", "remaining": []}


def _limpar_resposta(texto: str) -> str:
    texto = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", texto)
    texto = re.sub(r"\n```\s*$", "", texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# Message parsing (from OpenAI format to plain text prompt)
# ---------------------------------------------------------------------------

def _extract_last_user_prompt(messages: list) -> str:
    if not messages:
        raise ValueError("'messages' está vazio.")

    _clean = re.compile(r'<system-reminder>.*?</system-reminder>', re.DOTALL)

    users = [m for m in messages if m.get("role") == "user"]
    if len(messages) == 1 and len(users) == 1:
        return _clean.sub("", str(users[0].get("content", "")))

    linhas = []
    primeiro_user = False
    for m in messages:
        role = m.get("role", "user")
        content = _clean.sub("", m.get("content") or "")
        if role == "system":
            linhas.append(f"[System]\n{content}\n")
            continue
        elif role == "user":
            if not primeiro_user:
                primeiro_user = True
            linhas.append(f"[User]\n{content}\n")
        elif role == "assistant":
            if not primeiro_user:
                continue
            bloco = f"[Assistant]\n{content}\n" if content else "[Assistant]\n"
            for tc in (m.get("tool_calls") or []):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "{}")
                bloco += f"[Tool Call]: {name}\n\nArguments: {args}\n"
            linhas.append(bloco)
        elif role == "tool":
            if not primeiro_user:
                continue
            tool_name = m.get("name") or "tool"
            linhas.append(f"[Tool Result ({tool_name})]\n{content}\n")
    return "\n".join(linhas).strip()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def stream_prompt(
    messages: list,
    timeout: int = config.DEFAULT_TIMEOUT,
    account: str | None = None,
) -> Iterator[str]:
    """
    Envia mensagens (formato OpenAI) ao DeepSeek e yield deltas de texto em tempo real.

    Cada yield é um fragmento (string) novo que apareceu na resposta.
    O generator termina quando o DeepSeek acaba de gerar.

    Parâmetros:
        messages: histórico no formato OpenAI.
        timeout: timeout máximo em segundos.
        account: nome da conta a usar (ou None para round-robin).
    """
    prompt = _extract_last_user_prompt(messages)
    account_name, driver = pool.get_driver(account)
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt vazio.")

    timeout = max(10, min(int(timeout), config.MAX_TIMEOUT))

    with pool.get_lock(account_name):
        # SEMPRE redireciona para a URL raiz — isso força o DeepSeek a criar
        # um novo chat a cada chamada da API (cada requisição = conversa isolada).
        try:
            driver.get(config.DEEPSEEK_URL)
            # Aguarda o textarea ficar disponível na nova página
            WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "textarea"))
            )
        except TimeoutException:
            # Se não achou textarea, tenta o contenteditable como fallback
            try:
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "div[contenteditable='true']"))
                )
            except Exception:
                pass
        except Exception:
            pass

        time.sleep(0.1)
        _click_search_button(driver)
        _click_alternate_button(driver)

        # Instala o observer ANTES de submeter o prompt
        _install_observer(driver)

        _inject_prompt(driver, prompt)
        time.sleep(0.05)
        _submit_prompt(driver)

        deadline = time.time() + timeout
        primeira_resposta_vista = False
        sem_deltas_ha = 0.0
        ultimo_poll = time.time()
        stable_done_count = 0
        STABLE_POLLS_NEEDED = 4  # ~800ms de estabilidade sem deltas pra considerar concluído

        try:
            while time.time() < deadline:
                info = _poll_deltas(driver)
                deltas = info.get("deltas", []) or []
                done = bool(info.get("done", False))

                if deltas:
                    primeira_resposta_vista = True
                    sem_deltas_ha = 0.0
                    stable_done_count = 0  # reset se chegar delta novo
                    for delta in deltas:
                        yield delta
                else:
                    sem_deltas_ha += (time.time() - ultimo_poll)
                    if done:
                        stable_done_count += 1
                    else:
                        stable_done_count = 0

                ultimo_poll = time.time()

                # Timeout de "nunca começou a responder"
                if not primeira_resposta_vista and sem_deltas_ha > 45:
                    raise TimeoutError(
                        f"DeepSeek não começou a responder em 45s. "
                        "Verifique se o login ainda está ativo."
                    )

                # Se terminou de gerar de forma ESTÁVEL (evita cortes em micro-pausas)
                if done and primeira_resposta_vista and stable_done_count >= STABLE_POLLS_NEEDED:
                    # Último flush pra garantir
                    time.sleep(0.3)
                    info_final = _poll_deltas(driver)
                    for delta in (info_final.get("deltas", []) or []):
                        yield delta
                    return

                time.sleep(0.2)

            # Timeout total atingido: faz flush do que tiver
            info_final = _poll_deltas(driver)
            for delta in (info_final.get("deltas", []) or []):
                yield delta

        finally:
            _stop_observer(driver)


def send_prompt(
    messages: list,
    timeout: int = config.DEFAULT_TIMEOUT,
    account: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Modo não-streaming: concatena todos os deltas e retorna a resposta completa.
    
    Retorna uma tupla (texto_resposta, tool_calls) onde tool_calls é uma lista
    no formato OpenAI (pode ser vazia se nenhum tool call for detectado).
    """
    partes = []
    for delta in stream_prompt(messages, timeout=timeout, account=account):
        partes.append(delta)
    texto = "".join(partes)
    if not texto:
        raise RuntimeError("Resposta do DeepSeek veio vazia.")
    
    texto_limpo = _limpar_resposta(texto)
    tool_calls = parser.parse_tool_calls_from_text(texto_limpo)
    
    return texto_limpo, tool_calls


def send_prompt_with_tools(
    messages: list,
    timeout: int = config.DEFAULT_TIMEOUT,
    account: str | None = None,
) -> str:
    """
    Modo não-streaming simplificado: retorna apenas o texto da resposta.
    Mantido para compatibilidade com chamadas que não precisam de tool calls.
    """
    texto, _ = send_prompt(messages, timeout=timeout, account=account)
    return texto
