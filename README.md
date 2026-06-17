# DeepProxy

API compatível com **OpenAI Chat Completions** que encaminha requisições para o
[chat.deepseek.com](https://chat.deepseek.com) via Selenium.

Permite usar clientes OpenAI-compatíveis (Open WebUI, Continue.dev, Cursor,
LiteLLM, etc.) apontando para o DeepSeek **sem precisar de chave de API oficial**
— desde que você tenha login ativo na conta web.

## Como funciona

```
[ Cliente OpenAI-compatível ]
          │  POST /v1/chat/completions
          ▼
    ┌──────────┐
    │ DeepProxy │  (Flask + Selenium)
    └──────────┘
          │  digita o prompt
          ▼
   chat.deepseek.com
          │  captura a resposta via MutationObserver
          ▼
    bufferiza tudo → analisa → reproduz a 250 tk/s
          │
          ├─ se for tool_call → stream só a estrutura tool_call
          └─ se for texto → stream do texto limpo
```

## Funcionalidades

- **Streaming real**: captura a resposta do DeepSeek via `MutationObserver` injetado, delta por delta
- **Tool calls automáticas**: detecta chamadas de ferramenta na resposta do modelo e converte para o formato OpenAI nativo
- **Formatos de tool call suportados**:
  - XML com atributo: `<tool_call name="x"> <arg>val</arg> </tool_call>`
  - XML com sub-tag `<name>`: `<skill> <name>x</name> <param>val</param> </skill>`
  - JSON blocks: `{"name": "x", "arguments": {...}}`
  - Texto: `[Tool Call]: x\n\nArguments: {"key": "val"}`
- **250 tk/s simulado**: a resposta é bufferizada e reproduzida a 250 tokens/s para o cliente
- **Histórico multi-turno**: suporta `role: system`, `user`, `assistant` e `tool`
- **Persistência de login**: perfil Chrome salvo entre execuções

## Instalação

```bash
# Chrome + ChromeDriver precisam estar instalados e compatíveis
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

Na **primeira execução**, uma janela do Chrome vai abrir em
`https://chat.deepseek.com` — **faça login manualmente** nela. O login fica
salvo no perfil local (`perfil_proxy/`) e será reaproveitado nas próximas vezes.

Depois, basta apontar seu cliente para:

```
Base URL: http://localhost:4000/v1
Model:    deepseek-chat   (aceita qualquer string, é só decorativo)
```

### Exemplo com curl

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "Explique o que é uma closure em JavaScript."}
    ]
  }'
```

### Streaming

```bash
curl -N http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "Conte uma piada."}]
  }'
```

### Tool calls

```bash
curl -N http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "Adicione 'estudar' na minha lista de tarefas."}]
  }'
```

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `DEEPROXY_HOST` | `0.0.0.0` | Host onde a API escuta |
| `DEEPROXY_PORT` | `4000` | Porta da API |
| `DEEPROXY_TIMEOUT` | `120` | Timeout padrão por requisição (s) |
| `DEEPROXY_MAX_TIMEOUT` | `600` | Timeout máximo permitido (s) |
| `DEEPROXY_MODEL_NAME` | `deepseek-chat` | Nome do modelo reportado nas respostas |
| `DEEPROXY_CHROME_PROFILE` | `./perfil_proxy` | Pasta do perfil do Chrome |

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status do serviço |
| `GET` | `/v1/models` | Lista de modelos (OpenAI compat) |
| `POST` | `/v1/chat/completions` | Chat Completions (com/sem stream) |

## Limitações conhecidas

- **Uma requisição por vez** (há um lock global no browser).
- **Sem contagem de tokens** — o campo `usage` sempre retorna 0.
- **Depende do DOM do DeepSeek** — se eles mudarem os seletores, `proxy.py` precisa ser ajustado.
- **Login manual na primeira vez** (o DeepSeek não permite login automatizado fácil).
- **Rate limit do próprio DeepSeek** — respeite os limites da conta web.

## Estrutura

```
deeproxy/
├── app.py            # Flask + rotas + parsers de tool call
├── proxy.py          # Lógica Selenium + MutationObserver
├── config.py         # Configurações via env
├── requirements.txt
└── README.md
```
