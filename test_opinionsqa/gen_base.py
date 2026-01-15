import json, sys, time, requests, re
from pathlib import Path
from typing import Optional
import math

BASE_URL = "http://localhost:8000"
IN_PATH = Path("alpaca_test.json")
OUT_PATH = Path("results_alpaca_test.json")
MODEL = "meta-llama/Llama-3.1-8B"  # set to your loaded model id if needed

SYSTEM_HINT = (
    "You are a strict boolean classifier. "
    "Given an instruction and an input claim, output exactly one of: True or False."
)

def build_prompt(instruction: str, input_text: str) -> str:
    # Self-contained prompt for base (non-chat) models
    return (
        f"{SYSTEM_HINT}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{input_text}\n\n"
        "Answer:"
    )

def request_with_retries(url: str, payload: dict, timeout: int = 60, max_retries: int = 3, backoff: float = 0.5) -> requests.Response:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            # vLLM usually uses proper HTTP status codes; raise for non-2xx
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            # backoff and retry on transient errors
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                raise
    # Shouldn't reach here
    raise last_err

def complete(base_url: str, model: str, prompt: str, max_tokens: int = 8, logprobs_k: int = 5):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "n": 1,
        "logprobs": logprobs_k,  
        "stop": ["\nInstruction:", "\nInput:", "\nTask:", "\nLabel:", "\nAnswer", "\n\n"]
    }
    r = request_with_retries(f"{base_url}/v1/completions", payload)
    data = r.json()

    choices = (data or {}).get("choices") or []
    if not choices or "text" not in choices[0]:
        return "", None

    text = (choices[0]["text"] or "").strip()
    lp = choices[0].get("logprobs")  
    return text, lp

_TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)

def normalize_bool(txt: str) -> str:
    """
    Normalize any model text to exactly 'True' or 'False'.
    Returns '' if indeterminate/blank.
    """
    if not txt or not txt.strip():
        return ""
    # Fast path: first token
    first_tokens = txt.strip().split()
    if first_tokens:
        first = first_tokens[0].strip(" .,:;!?\"'()[]{}").lower()
        if first.startswith("true"):
            return "True"
        if first.startswith("false"):
            return "False"
    # Fallback: search anywhere (but avoid both present)
    low = txt.lower()
    has_t = "true" in low
    has_f = "false" in low
    if has_t and not has_f:
        return "True"
    if has_f and not has_t:
        return "False"
    # Regex scan for first occurrence
    m = _TRUE_FALSE_RE.search(txt)
    if m:
        return "True" if m.group(1).lower() == "true" else "False"
    return ""

def extract_pred_confidence_from_completion_logprobs(logprobs_obj, pred_label: str) -> Optional[float]:
    if not logprobs_obj:
        return None

    top = logprobs_obj.get("top_logprobs")
    if not top or not isinstance(top, list) or len(top) == 0:
        return None

    first_top = top[0]
    if not isinstance(first_top, dict):
        return None

    # token variants that sometimes appear
    true_keys  = ["True", " True", "true", " true"]
    false_keys = ["False", " False", "false", " false"]

    lp_true = next((first_top[k] for k in true_keys  if k in first_top), None)
    lp_false = next((first_top[k] for k in false_keys if k in first_top), None)

    if lp_true is None and lp_false is None:
        return None

    # normalize over (True, False) only
    vals = {}
    if lp_true is not None:
        vals["True"] = lp_true
    if lp_false is not None:
        vals["False"] = lp_false

    m = max(vals.values())
    probs = {k: math.exp(v - m) for k, v in vals.items()}
    Z = sum(probs.values())
    probs = {k: v / Z for k, v in probs.items()}

    return probs.get(pred_label)

def main(in_path=IN_PATH, out_path=OUT_PATH, base_url=BASE_URL, model=MODEL):
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    assert isinstance(data, list), "Input must be a list of Alpaca-style records."

    results = []
    for i, item in enumerate(data, 1):
        instruction = item.get("instruction", "")
        input_text  = item.get("input", "")

        prompt = build_prompt(instruction, input_text)
        raw, lp = "", None
        try:
            raw, lp = complete(base_url, model, prompt, max_tokens=8, logprobs_k=5)
            gen = normalize_bool(raw)
            if not gen:
                # Second attempt with slightly larger budget if we got blank
                raw, lp = complete(base_url, model, prompt, max_tokens=16, logprobs_k=5)
                gen = normalize_bool(raw)
        except Exception as e:
            print(f"[error] item {i} failed: {e}")
            gen = ""

        if not gen:
            low = (raw or "").lower()
            gen = "True" if ("true" in low and "false" not in low) else \
                  "False" if ("false" in low and "true" not in low) else ""

        new_item = dict(item)
        new_item["generated_output"] = gen
        conf = extract_pred_confidence_from_completion_logprobs(lp, gen) if gen in ("True", "False") else None
        new_item["pred_confidence"] = conf
        results.append(new_item)

        # small pacing to be nice to local server
        time.sleep(0.005)

    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Wrote {out_path} ({len(results)} items).")

if __name__ == "__main__":
    in_arg  = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    url_arg = sys.argv[3] if len(sys.argv) > 3 else BASE_URL
    model   = sys.argv[4] if len(sys.argv) > 4 else MODEL
    main(in_arg, out_arg, url_arg, model)
