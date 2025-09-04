from byllm.llm import Model
from byllm.mtir import MTIR

import logging



log = logging.getLogger("my_plugin")
log.setLevel(logging.ERROR)  

#SOMETHINGS I DONT KNOW HOW IT WORKS BUT IT WORKS!!!
def _to_part_dict(part) -> dict:
    # Accepts dict / Pydantic / string / other
    if hasattr(part, "model_dump"):  # Pydantic v2
        part = part.model_dump()
    elif hasattr(part, "dict"):      # Pydantic v1
        part = part.dict()

    if isinstance(part, dict):
        # Already a dict. Ensure it matches {"type":"text","text":...}
        if part.get("type") == "text" and "text" in part:
            return part
        # Try to coerce common alternatives
        if "content" in part:
            return {"type": "text", "text": part["content"]}
        if "text" in part and not part.get("type"):
            return {"type": "text", "text": part["text"]}
        # Fallback: stringify
        return {"type": "text", "text": json.dumps(part, ensure_ascii=False)}

    if isinstance(part, str):
        return {"type": "text", "text": part}

    # Fallback for unknown objects
    return {"type": "text", "text": str(part)}

def _normalize_content(content) -> list[dict]:
    # Ensure a list[{"type":"text","text": "..."}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [_to_part_dict(p) for p in content]
    # Pydantic object or other
    if hasattr(content, "model_dump"):
        d = content.model_dump()
        if isinstance(d, dict):
            return [_to_part_dict(d)]
    return [{"type": "text", "text": str(content)}]

def normalize_messages(messages):
    """Normalize any SDK/Pydantic message objects to the JSON shape LiteLLM expects."""
    norm = []
    for m in messages:
        if hasattr(m, "model_dump"):
            m = m.model_dump()
        elif hasattr(m, "dict"):
            m = m.dict()

        if not isinstance(m, dict):
            # Last resort: pull role/content attrs
            role = getattr(m, "role", None)
            content = getattr(m, "content", "")
            m = {"role": role, "content": content}

        m["content"] = _normalize_content(m.get("content", ""))
        norm.append({"role": m["role"], "content": m["content"]})
    return norm

#This I wrote
def evaluate_local_model(model: Model, local_mtir: MTIR) -> bool:
    """Ask the big LLM to check whether the answer given by the local LLM is correct."""
    # Ensure messages are plain dicts with OpenAI/Gemini-compatible content parts
    local_messages = normalize_messages(local_mtir.messages)
    # Remove any existing system messages from the local run
    local_messages = [m for m in local_messages if m.get("role") != "system"]
    
    # Append evaluator instruction as a SYSTEM message
    local_messages = [{
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are an evaluator. Look at the original instruction and the assistant's answer. "
                    "Decide if the assistant's answer correctly follows the instruction. "
                    "Return only a JSON boolean: true if correct, false otherwise. No explanation.Do NOT be very strict.If it matches the output say True"
                )
            }
        ]
    }] + local_messages

    global_mtir = MTIR(
        messages=local_messages,
        tools=local_mtir.tools,
        resp_type=bool,          # expect a boolean back
        stream=local_mtir.stream,
        call_params=local_mtir.call_params,
    )

    verdict = model.invoke(global_mtir)
    log.debug(f"Evaluator verdict: {verdict}")
    return verdict