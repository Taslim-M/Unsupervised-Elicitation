import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests


BASE_URL = "http://localhost:8000"
IN_PATH = Path("alpaca_test.json")
OUT_PATH = Path("results_alpaca_test.json")

# ---------- HTTP helpers ----------

def get_models(base_url: str) -> List[str]:
    """Return available model ids from an OpenAI-compatible vLLM server."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception as e:
        print(f"[warn] Could not fetch models: {e}")
        return []

def chat_completion(
    base_url: str,
    model: str,
    user_content: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: int = 30
) -> str:
    """Call /v1/chat/completions. Returns text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def completion_fallback(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: int = 30
) -> str:
    """Fallback to /v1/completions if chat endpoint is unavailable."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
    }
    r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["text"].strip()

# ---------- Core logic ----------

SYSTEM_PROMPT = (
    "You are a strict boolean classifier. "
    "Given an instruction and an input claim, output exactly one of: True or False. "
    "No punctuation, no extra words."
)

def build_user_prompt(instruction: str, input_text: str) -> str:
    # We intentionally do NOT include any existing 'output' field from the data.
    return f"{instruction}\n\nInput:\n{input_text}\n\nAnswer with exactly one token: True or False."

def generate_for_item(
    base_url: str, model: str, item: Dict[str, Any]
) -> str:
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    user_prompt = build_user_prompt(instruction, input_text)

    # Try chat-completions first
    try:
        return chat_completion(
            base_url,
            model,
            user_prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[warn] chat endpoint failed, falling back to completions: {e}")

    # Fallback to completions
    prompt = f"{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:"
    return completion_fallback(
        base_url,
        model,
        prompt,
        max_tokens=8,
        temperature=0.0,
    )

def main(
    in_path: Path = IN_PATH,
    out_path: Path = OUT_PATH,
    base_url: str = BASE_URL
):
    # Load input
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of Alpaca-style records.")

    # Pick a model
    models = get_models(base_url)
    if not models:
        # If API doesn't expose models, allow a dummy name; many vLLM servers accept it anyway.
        model = "local-model"
        print("[warn] Could not list models; using model='local-model'.")
    else:
        model = models[0]
        print(f"[info] Using model: {model}")

    results = []
    for i, item in enumerate(data, 1):
        try:
            gen = generate_for_item(base_url, model, item)
            # Keep only the exact True/False token if extra whitespace/non-alpha slips in
            gen_clean = gen.strip().split()[0]
            if gen_clean.lower().startswith("true"):
                gen_clean = "True"
            elif gen_clean.lower().startswith("false"):
                gen_clean = "False"
            # Append result with an added "generated_output"
            new_item = dict(item)
            new_item["generated_output"] = gen_clean
            results.append(new_item)
        except Exception as e:
            print(f"[error] item {i} failed: {e}")
            # Still append with a placeholder to preserve indexing
            new_item = dict(item)
            new_item["generated_output"] = ""
            results.append(new_item)
        # Small polite delay (tune/remove as you prefer)
        time.sleep(0.01)

    # Save output
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {out_path} ({len(results)} items).")

if __name__ == "__main__":
    # Optional CLI: python script.py [in_json] [out_json] [base_url]
    in_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    url_arg = sys.argv[3] if len(sys.argv) > 3 else BASE_URL
    main(in_arg, out_arg, url_arg)
