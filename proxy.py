"""
proxy.py — núcleo do DeepProxy.

Abre o DeepSeek Chat via Selenium, envia um prompt e captura a resposta.
Suporta dois modos:
  - send_prompt(prompt)        -> retorna a resposta completa (str)
  - stream_prompt(prompt)      -> generator que yield deltas em tempo real

A captura incremental usa um MutationObserver injetado via JS que guarda
os fragmentos novos numa variável window.__deepseek_deltas. O Python faz
polling dessa variável a cada ~200ms e repassa os deltas ao cliente.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Iterator, Optional

from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import config


_browser_lock = threading.Lock()
_driver: Optional[webdriver.Chrome] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _build_options() -> Options:
    opts = Options()
    opts.add_argument(f"--user-data-dir={os.path.abspath(config.CHROME_PROFILE_DIR)}")
    opts.add_argument("--window-size=1100,900")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    return opts


def get_driver() -> webdriver.Chrome:
    global _driver
    with _browser_lock:
        if _driver is None:
            print("[Proxy] 🚀 Iniciando Chrome...")
            _driver = webdriver.Chrome(options=_build_options())
            _driver.set_script_timeout(config.MAX_TIMEOUT)
            _driver.get(config.DEEPSEEK_URL)
            print(f"[Proxy] 🟢 Navegador pronto. Faça login manualmente em {config.DEEPSEEK_URL}")
        return _driver


def shutdown() -> None:
    global _driver
    with _browser_lock:
        if _driver is not None:
            try:
                _driver.quit()
            except Exception:
                pass
            _driver = None


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
            el = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((by, sel))
            )
            return el
        except TimeoutException:
            continue
    raise RuntimeError("Não foi possível localizar o campo de input do DeepSeek.")


def _inject_prompt(driver: webdriver.Chrome, prompt: str) -> None:
    textarea = _find_textarea(driver)
    textarea.click()
    time.sleep(0.2)
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
// Também detecta se a geração terminou.
const deltas = window.__deepseek_deltas || [];
window.__deepseek_deltas = [];

// Detecta fim da geração: botão "Stop" ausente E sem cursor piscando
const stopBtn = document.querySelector(
    'button[aria-label*="top" i], button[aria-label*="ancel" i], button[aria-label*="egenerate" i]'
);
const stopVisible = stopBtn && stopBtn.offsetParent !== null;
const cursor = document.querySelector('.cursor-blink, [class*="typing"]');

// Conferimos se ainda existe uma mensagem do assistente na tela
const msgs = document.querySelectorAll('.ds-markdown.ds-assistant-message-main-content');
const hasAssistant = msgs.length > 0;

// done = temos pelo menos 1 mensagem de assistente E nada está gerando
const done = hasAssistant && !stopVisible && !cursor;

return {
    deltas: deltas,
    done: done,
    total_len: (window.__deepseek_last_text || '').length
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
# API pública
# ---------------------------------------------------------------------------

def stream_prompt(prompt: str, timeout: int = config.DEFAULT_TIMEOUT) -> Iterator[str]:
    """
    Envia um prompt ao DeepSeek e yield deltas de texto em tempo real.

    Cada yield é um fragmento (string) novo que apareceu na resposta.
    O generator termina quando o DeepSeek acaba de gerar.
    """
    driver = get_driver()
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt vazio.")

    timeout = max(10, min(int(timeout), config.MAX_TIMEOUT))

    with _browser_lock:
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

        time.sleep(0.5)
        _click_search_button(driver)

        # Instala o observer ANTES de submeter o prompt
        _install_observer(driver)

        _inject_prompt(driver, prompt)
        time.sleep(0.3)
        _submit_prompt(driver)

        deadline = time.time() + timeout
        primeira_resposta_vista = False
        sem_deltas_ha = 0.0
        ultimo_poll = time.time()

        try:
            while time.time() < deadline:
                info = _poll_deltas(driver)
                deltas = info.get("deltas", []) or []
                done = bool(info.get("done", False))

                if deltas:
                    primeira_resposta_vista = True
                    sem_deltas_ha = 0.0
                    for delta in deltas:
                        yield delta
                else:
                    sem_deltas_ha += (time.time() - ultimo_poll)

                ultimo_poll = time.time()

                # Timeout de "nunca começou a responder"
                if not primeira_resposta_vista and sem_deltas_ha > 30:
                    raise TimeoutError(
                        f"DeepSeek não começou a responder em 30s. "
                        "Verifique se o login ainda está ativo."
                    )

                # Se terminou de gerar, faz um último poll pra pegar restos
                if done and primeira_resposta_vista:
                    # Pequeno grace period pra garantir que pegamos tudo
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


def send_prompt(prompt: str, timeout: int = config.DEFAULT_TIMEOUT) -> str:
    """
    Modo não-streaming: concatena todos os deltas e retorna a resposta completa.
    """
    partes = []
    for delta in stream_prompt(prompt, timeout=timeout):
        partes.append(delta)
    texto = "".join(partes)
    if not texto:
        raise RuntimeError("Resposta do DeepSeek veio vazia.")
    return _limpar_resposta(texto)
