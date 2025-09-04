# Tiny FastAPI Server — Quick Start

Minimal steps to install dependencies and run the server.

### Install dependencies

    pip install fastapi "uvicorn[standard]" accelerate outlines transformers peft datasets

### Run the API

    uvicorn app:app --host 0.0.0.0 --port 7000 --reload

## API Endpoints

### 1. Health Check
*GET* /healthz  
Returns service health status.

*Response*
json
{ "status": "ok" }


---

### 2. Current Mode
*GET* /which  
Check the current operating mode.

*Response*
json
{ "mode": "global" }


---

### 3. Set Mode
*POST* /mode  
Change the operating mode (global, local, eval, idle).

*Request*
json
{ "new_mode": "local" }


*Response*
json
{ "mode": "local" }


---

### 4. Evaluation Counter
*POST* /eval  
Update evaluation state based on a verdict.  
- If verdict = true: increments eval count.  
- If count > 10: switches to local mode.  
- If verdict = false: resets to global.

*Request*
json
{ "verdict": true }


*Response*
json
{ "eval_count": 3 }


---

### 5. Chat Completions (OpenAI-Compatible)
*POST* /chat/completions  
Submit a conversation in OpenAI’s format.  
- Stores training data when in global mode.  
- Starts training if data threshold is reached.  
- Otherwise, generates a local model response.

*Request*
json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ]
}


*Response*
json
{
  "choices": [
    {
      "index": 0,
      "message": { "content": "Hi there!" }
    }
  ]
}
