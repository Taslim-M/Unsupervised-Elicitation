import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import math

import requests

BASE_URL = "http://localhost:8000"
IN_PATH = Path("fold1_test_alpaca.json")
TRAIN_PATH = Path("fold1_train_icm_alpaca.json")  # NEW: train file
OUT_PATH = Path("results_icm_few_logs_fold1.json")

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
    timeout: int = 30,
    top_logprobs: int = 5,   
) -> Dict[str, Any]:
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
        "logprobs": True,           
        "top_logprobs": top_logprobs,  
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    return {
        "text": data["choices"][0]["message"]["content"].strip(),
        "logprobs": data["choices"][0].get("logprobs"),  
    }


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

# ---------- Few-shot helpers ----------

COUNTRY_RE = re.compile(r"People\s+from\s+([A-Za-z][A-Za-z\s\-\&\.']+?)\s+would\s+answer", re.IGNORECASE)

def extract_country(input_text: str) -> Optional[str]:
    """Extract country name from test input like: 'People from Brazil would answer ...'"""
    m = COUNTRY_RE.search(input_text or "")
    if not m:
        return None
    # Normalize whitespace + strip quotes/punctuation
    country = m.group(1).strip().strip(" '\"\t\r\n")
    return country

def index_train_by_country(
    train_data: List[Dict[str, Any]],
    use_icm: bool
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build a dict: country -> list of (input, label_str) from train records.
    Uses icm_output or gt_output depending on `use_icm`.
    """
    key = "icm_output" if use_icm else "gt_output"
    by_country: Dict[str, List[Tuple[str, str]]] = {}
    for r in train_data:
        inp = r.get("input", "")
        lbl = r.get(key, None)
        if lbl is None:
            # fall back to 'output' if provided; else skip
            lbl = r.get("output", None)
        if not inp or lbl is None:
            continue
        c = extract_country(inp)
        if not c:
            continue
        by_country.setdefault(c, []).append((inp, str(lbl).strip()))
    return by_country

def build_fewshot_block(country: Optional[str], shots: List[Tuple[str, str]]) -> str:
    """
    Create a compact labeled few-shot block.
    Each example is:
    Example i:
    Input: ...
    Label: True/False
    """
    n = len(shots)
    header = f"Few-shot examples for country: {country or 'N/A'} (count={n})"
    lines = [header]
    for i, (inp, lbl) in enumerate(shots, 1):
        lines.append(f"Example {i}:\nInput: {inp}\nLabel: {lbl}")
    return "\n\n".join(lines)

# ---------- Core logic ----------

SYSTEM_PROMPT = (
    "You are a strict boolean classifier. "
    "Given an instruction and an input claim, output exactly one of: True or False. "
    "No punctuation, no extra words."
)

def build_user_prompt_with_fewshot(
    instruction: str,
    input_text: str,
    fewshot_text: str
) -> str:
    """
    Compose the full user message with:
      - the few-shot labeled block,
      - then the actual task (without including the test 'output').
    """
    return (
        f"{fewshot_text}\n\n"
        f"Task:\n{instruction}\n\n"
        f"Input:\n{input_text}\n\n"
        "Answer with exactly one token: True or False."
    )

def generate_for_item(
    base_url: str,
    model: str,
    instruction: str,
    input_text: str,
    fewshot_text: str
) -> str:
    user_prompt = build_user_prompt_with_fewshot(instruction, input_text, fewshot_text)

    # Prefer chat; fall back to completions
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

    prompt = f"{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:"
    return completion_fallback(
        base_url,
        model,
        prompt,
        max_tokens=8,
        temperature=0.0,
    )

def extract_tf_probs(logprobs_obj):
    """
    Extract P(True) / P(False) from vLLM-style chat logprobs.
    Returns dict with probabilities or None if unavailable.
    """
    if not logprobs_obj:
        return None

    content = logprobs_obj.get("content", [])
    if not content:
        return None

    # Only the FIRST token matters ("True"/"False")
    first = content[0]
    top = first.get("top_logprobs", [])

    tf_logprobs = {}

    for t in top:
        tok = t["token"].strip()  # handles " True", "\nTrue", etc.
        if tok == "True":
            tf_logprobs["True"] = t["logprob"]
        elif tok == "False":
            tf_logprobs["False"] = t["logprob"]

    if not tf_logprobs:
        return None

    # Convert logprobs → normalized probabilities
    max_lp = max(tf_logprobs.values())
    exps = {k: math.exp(v - max_lp) for k, v in tf_logprobs.items()}
    Z = sum(exps.values())

    return {k: v / Z for k, v in exps.items()}
def main(
    in_path: Path = IN_PATH,
    train_path: Path = TRAIN_PATH,
    out_path: Path = OUT_PATH,
    base_url: str = BASE_URL,
    use_icm: bool = False,
    max_shots: Optional[int] = None,  # None = use all for the country
):
    # Load files
    test_data = json.loads(in_path.read_text(encoding="utf-8"))
    train_data = json.loads(train_path.read_text(encoding="utf-8"))

    if not isinstance(test_data, list):
        raise ValueError("Test JSON must be a list of Alpaca-style records.")
    if not isinstance(train_data, list):
        raise ValueError("Train JSON must be a list of Alpaca-style records with gt_output/icm_output.")

    # Build country index for train
    by_country = index_train_by_country(train_data, use_icm)
    print(f"[info] Indexed train by country: {len(by_country)} countries.")

    # Pick a model
    models = get_models(base_url)
    if not models:
        model = "local-model"
        print("[warn] Could not list models; using model='local-model'.")
    else:
        model = models[0]
        print(f"[info] Using model: {model}")

    results = []
    for i, item in enumerate(test_data, 1):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "") or ""
        test_country = extract_country(input_text)

        # Pull few-shots for this country
        shots = by_country.get(test_country, [])
        if max_shots is not None and len(shots) > max_shots:
            shots = shots[:max_shots]

        # Build few-shot text (and visibly include the count)
        fewshot_text = build_fewshot_block(test_country, shots)

        # Also log the count for this sample
        print(f"[fewshot] sample={i} country={test_country or 'N/A'} shots={len(shots)}")

        # Generate
        try:
            resp = generate_for_item(base_url, model, instruction, input_text, fewshot_text)
            gen = resp["text"]
            gen_clean = gen.strip().split()[0]
            if gen_clean.lower().startswith("true"):
                gen_clean = "True"
            elif gen_clean.lower().startswith("false"):
                gen_clean = "False"
            else:
                # fallback if model rambles
                low = gen.lower()
                gen_clean = "True" if ("true" in low and "false" not in low) else \
                            "False" if ("false" in low and "true" not in low) else ""
        except Exception as e:
            print(f"[error] item {i} failed: {e}")
            gen_clean = ""

        new_item = dict(item)
        new_item["generated_output"] = gen_clean
        # (Optionally include bookkeeping for analysis)
        new_item["_fewshot_country"] = test_country
        new_item["_fewshot_count"] = len(shots)
        tf_probs = extract_tf_probs(resp.get("logprobs"))
        if tf_probs:
            pred = max(tf_probs, key=tf_probs.get)
            new_item["pred_confidence"] = tf_probs[pred]
        results.append(new_item)

        time.sleep(0.01)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Wrote {out_path} ({len(results)} items).")

if __name__ == "__main__":
    # CLI:
    # python script.py [test_json] [train_json] [out_json] [base_url] [use_icm:0/1] [max_shots]
    in_arg    = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    train_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_PATH
    out_arg   = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_PATH
    url_arg   = sys.argv[4] if len(sys.argv) > 4 else BASE_URL
    use_icm   = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    max_shots = int(sys.argv[6]) if len(sys.argv) > 6 else None
    main(in_arg, train_arg, out_arg, url_arg, use_icm, max_shots)
