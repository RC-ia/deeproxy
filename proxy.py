"""
proxy.py — núcleo do DeepProxy.

Abre o DeepSeek Chat via Selenium, envia um prompt, aguarda a resposta
completa e a devolve como string. Mantém o browser aberto entre chamadas
para aproveitar o login persistente.

Estratégia de captura:
  1. Localiza o textarea de entrada e injeta o prompt.
  2. Submete o formulário (Enter no textarea ou clique no botão Send).
  3. Aguarda o indicador de "digitando"/loading desaparecer.
  4. Lê o último bloco de mensagem do assistente.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

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
    # Evita banners de "Chrome está sendo controlado por automação"
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    return opts


def get_driver() -> webdriver.Chrome:
    """Retorna o driver singleton, criando-o se necessário."""
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
    """Encerra o browser (chamado no shutdown do Flask)."""
    global _driver
    with _browser_lock:
        if _driver is not None:
            try:
                _driver.quit()
            except Exception:
                pass
            _driver = None


def _click_search_button(driver: webdriver.Chrome) -> None:
    """Clica no botão 'Search' do DeepSeek se estiver visível (não-crítico)."""
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
# Envio + captura
# ---------------------------------------------------------------------------

def _find_textarea(driver: webdriver.Chrome) -> "WebElement":
    """Localiza o campo de input do DeepSeek. Usa vários seletores conhecidos."""
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
    """Injeta o texto no textarea via JS (mais confiável que send_keys para textos grandes)."""
    textarea = _find_textarea(driver)
    # Foca no elemento
    textarea.click()
    time.sleep(0.2)
    # Limpa e injeta via JS para preservar quebras de linha
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
    """Submete o prompt: tenta botão Send, senão Enter no textarea."""
    # Tenta clicar no botão de envio
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
    # Fallback: Enter
    try:
        textarea = _find_textarea(driver)
        textarea.send_keys(Keys.ENTER)
    except Exception as e:
        raise RuntimeError(f"Não foi possível submeter o prompt: {e}")


def _count_assistant_messages(driver: webdriver.Chrome) -> int:
    """Conta quantos blocos de resposta do assistente existem na tela."""
    js = """
    const nodes = document.querySelectorAll(
        '.ds-markdown, [class*="markdown-body"], [class*="answer-content"], [class*="message-content"]'
    );
    return nodes.length;
    """
    try:
        return int(driver.execute_script(js) or 0)
    except Exception:
        return 0


def _is_generating(driver: webdriver.Chrome) -> bool:
    """Retorna True se o DeepSeek ainda estiver gerando a resposta."""
    js = """
    // Botão "Stop" visível = gerando
    const stopBtn = document.querySelector(
        'button[aria-label*="top" i], button[aria-label*="ancel" i]'
    );
    if (stopBtn && stopBtn.offsetParent !== null) return true;
    // Cursor de digitação
    const cursor = document.querySelector('.cursor-blink, [class*="typing"]');
    if (cursor) return true;
    return false;
    """
    try:
        return bool(driver.execute_script(js))
    except Exception:
        return False


def _extract_last_assistant_text(driver: webdriver.Chrome) -> str:
    """Extrai o texto do último bloco de resposta do assistente."""
    js = """
    const nodes = document.querySelectorAll(
        '.ds-markdown, [class*="markdown-body"], [class*="answer-content"], [class*="message-content"]'
    );
    if (!nodes.length) return '';
    const last = nodes[nodes.length - 1];
    return (last.innerText || last.textContent || '').trim();
    """
    try:
        return driver.execute_script(js) or ""
    except Exception:
        return ""


def send_prompt(prompt: str, timeout: int = config.DEFAULT_TIMEOUT) -> str:
    """
    Envia um prompt ao DeepSeek e retorna a resposta completa.

    Fluxo:
      1. Garante que estamos na URL do chat.
      2. Conta quantas mensagens de assistente já existem.
      3. Injeta + submete o novo prompt.
      4. Aguarda aparecer uma NOVA mensagem de assistente.
      5. Aguarda o indicador de geração sumir.
      6. Extrai e retorna o texto da última mensagem.
    """
    driver = get_driver()
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt vazio.")

    timeout = max(10, min(int(timeout), config.MAX_TIMEOUT))

    with _browser_lock:
        # Garante que estamos no chat
        try:
            if not driver.current_url.startswith(config.DEEPSEEK_URL):
                driver.get(config.DEEPSEEK_URL)
                time.sleep(1)
                _click_search_button(driver)
        except Exception:
            driver.get(config.DEEPSEEK_URL)
            time.sleep(1)

        mensagens_antes = _count_assistant_messages(driver)

        # Injeta e submete
        _inject_prompt(driver, prompt)
        time.sleep(0.3)
        _submit_prompt(driver)

        # Aguarda aparecer uma nova mensagem de assistente (ou começar a gerar)
        deadline = time.time() + timeout
        while time.time() < deadline:
            agora = _count_assistant_messages(driver)
            if agora > mensagens_antes:
                break
            time.sleep(0.5)
        else:
            raise TimeoutError(
                f"DeepSeek não começou a responder em {timeout}s. "
                "Verifique se o login ainda está ativo."
            )

        # Aguarda terminar de gerar
        # Pequeno delay inicial pra evitar capturar o estado "antes de começar"
        time.sleep(0.8)
        while time.time() < deadline:
            if not _is_generating(driver):
                # Espera mais um pouco para garantir que o DOM foi atualizado
                time.sleep(0.4)
                if not _is_generating(driver):
                    break
            time.sleep(0.5)

        texto = _extract_last_assistant_text(driver)
        if not texto:
            raise RuntimeError("Resposta do DeepSeek veio vazia.")

        return _limpar_resposta(texto)


def _limpar_resposta(texto: str) -> str:
    """Remove artefatos comuns (blocos de código fences triplos envolvendo tudo, etc.)."""
    # Remove fence de abertura no início e de fechamento no fim, se envolverem TUDO
    texto = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", texto)
    texto = re.sub(r"\n```\s*$", "", texto)
    return texto.strip()
