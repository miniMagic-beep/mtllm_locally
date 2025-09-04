# Tiny FastAPI Server — Quick Start

Minimal steps to install dependencies and run the server.

### Install dependencies

    pip install fastapi "uvicorn[standard]" accelerate outlines transformers peft datasets

### Run the API

    uvicorn app:app --host 0.0.0.0 --port 7000 --reload

## API Endpoints


| Method | Endpoint            | Description | Request Example | Response Example |
|--------|---------------------|-------------|-----------------|------------------|
| **GET** | `/healthz` | Service health check | – | `{ "status": "ok" }` |
| **GET** | `/which` | Get current operating mode | – | `{ "mode": "global" }` |
| **POST** | `/mode` | Change operating mode (`global`, `local`, `eval`, `idle`) | `{ "new_mode": "local" }` | `{ "mode": "local" }` |
| **POST** | `/eval` | Update evaluation state based on a verdict<br>- `verdict=true`: increments eval count<br>- If count > 10 → switch to local<br>- `verdict=false`: reset to global | `{ "verdict": true }` | `{ "eval_count": 3 }` |
| **POST** | `/chat/completions` | OpenAI-compatible chat completions<br>- Stores training data in global mode<br>- Triggers training if threshold reached<br>- Otherwise responds locally | ```json { "messages": [ { "role": "system", "content": "You are a helpful assistant." }, { "role": "user", "content": "Hello!" } ] }``` | ```json { "choices": [ { "index": 0, "message": { "content": "Hi there!" } } ] }``` |

