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
          │  captura a resposta
          ▼
    resposta JSON (SSE se stream=true)
```

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
Base URL: http://localhost:8000/v1
Model:    deepseek-chat   (aceita qualquer string, é só decorativo)
```

### Exemplo com curl

```bash
curl http://localhost:8000/v1/chat/completions \
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
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "Conte uma piada."}]
  }'
```

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `DEEPROXY_HOST` | `0.0.0.0` | Host onde a API escuta |
| `DEEPROXY_PORT` | `8000` | Porta da API |
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
├── app.py            # Flask + rotas
├── proxy.py          # Lógica Selenium
├── config.py         # Configurações
├── requirements.txt
└── README.md
```

## Licença

MIT
