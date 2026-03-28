import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import math
import requests


BASE_URL = "http://localhost:8000"
IN_PATH = Path("fold1_test_opinionsqa.json")
OUT_PATH = Path("results_zeroshot_chat_fold1.json")

# ---------- HTTP helpers ----------

def get_models(base_url: str) -> List[str]:
    """Return available model ids from an OpenAI-compatible vLLM server."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=30)
        r.raise_for_status()
        data = r.json()
        return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except Exception as e:
        print(f"[warn] Could not fetch models: {e}")
        return []


def request_with_retries(
    url: str,
    payload: dict,
    timeout: int = 60,
    max_retries: int = 3,
    backoff: float = 0.5,
) -> requests.Response:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                raise
    raise last_err


def chat_completion(
    base_url: str,
    model: str,
    user_content: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Call /v1/chat/completions. Returns dict with 'text' and 'logprobs'."""
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
        "top_logprobs": 5,
    }
    r = request_with_retries(f"{base_url}/v1/chat/completions", payload, timeout=timeout)
    data = r.json()
    return {
        "text": data["choices"][0]["message"]["content"].strip(),
        "logprobs": data["choices"][0].get("logprobs"),
    }


def completion_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 8,
    temperature: float = 0.0,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Call /v1/completions and return dict with 'text' and completion logprobs."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
        "logprobs": 5,
        "stop": ["\nQuestion:", "\n\nQuestion:", "\n\n"],
    }
    r = request_with_retries(f"{base_url}/v1/completions", payload, timeout=timeout)
    data = r.json()
    choices = (data or {}).get("choices") or []
    if not choices or "text" not in choices[0]:
        raise RuntimeError("/v1/completions returned no choices")
    return {
        "text": (choices[0].get("text") or "").strip(),
        "logprobs": choices[0].get("logprobs"),
    }


def extract_pred_confidence_from_chat_logprobs(logprobs_obj, pred_label: str) -> Optional[float]:
    """Return P(pred_label) from chat top_logprobs for the first generated token."""
    if not logprobs_obj or pred_label not in ("True", "False"):
        return None

    content = logprobs_obj.get("content", [])
    if not content:
        return None

    first = content[0]
    top = first.get("top_logprobs", [])
    if not top:
        return None

    tf_lp = {}
    for t in top:
        tok = str(t.get("token", "")).strip()
        if tok == "True":
            tf_lp["True"] = float(t["logprob"])
        elif tok == "False":
            tf_lp["False"] = float(t["logprob"])

    if not tf_lp:
        return None

    m = max(tf_lp.values())
    exps = {k: math.exp(v - m) for k, v in tf_lp.items()}
    Z = sum(exps.values())
    probs = {k: v / Z for k, v in exps.items()}
    return probs.get(pred_label)


def extract_pred_confidence_from_completion_logprobs(logprobs_obj, pred_label: str) -> Optional[float]:
    """Return P(pred_label) from completion top_logprobs for the first generated token."""
    if not logprobs_obj or pred_label not in ("True", "False"):
        return None

    top = logprobs_obj.get("top_logprobs")
    if not top or not isinstance(top, list) or len(top) == 0:
        return None

    first_top = top[0]
    if not isinstance(first_top, dict):
        return None

    true_keys = ["True", " True", "true", " true"]
    false_keys = ["False", " False", "false", " false"]

    lp_true = next((first_top[k] for k in true_keys if k in first_top), None)
    lp_false = next((first_top[k] for k in false_keys if k in first_top), None)

    if lp_true is None and lp_false is None:
        return None

    vals = {}
    if lp_true is not None:
        vals["True"] = lp_true
    if lp_false is not None:
        vals["False"] = lp_false

    m = max(vals.values())
    exps = {k: math.exp(v - m) for k, v in vals.items()}
    Z = sum(exps.values())
    probs = {k: v / Z for k, v in exps.items()}
    return probs.get(pred_label)


_TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def normalize_bool(txt: str) -> str:
    """Normalize model output to exactly 'True' or 'False', or '' if indeterminate."""
    if not txt or not txt.strip():
        return ""

    first_tokens = txt.strip().split()
    if first_tokens:
        first = first_tokens[0].strip(" .,:;!?\"'()[]{}").lower()
        if first.startswith("true"):
            return "True"
        if first.startswith("false"):
            return "False"

    low = txt.lower()
    has_t = "true" in low
    has_f = "false" in low
    if has_t and not has_f:
        return "True"
    if has_f and not has_t:
        return "False"

    m = _TRUE_FALSE_RE.search(txt)
    if m:
        return "True" if m.group(1).lower() == "true" else "False"
    return ""


# ---------- Core logic ----------

SYSTEM_PROMPT = None


def normalize_persona_prompt(input_text: str) -> str:
    if not input_text:
        return ""
    return input_text if input_text.endswith(" ") else f"{input_text} "


def build_user_prompt(instruction: str, input_text: str) -> str:
    _ = instruction
    return normalize_persona_prompt(input_text)


def build_completion_prompt(instruction: str, input_text: str) -> str:
    _ = instruction
    return normalize_persona_prompt(input_text)


def generate_for_item(base_url: str, model: str, item: Dict[str, Any]) -> Dict[str, Any]:
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")

    # Prefer completions by default.
    try:
        prompt = build_completion_prompt(instruction, input_text)
        resp = completion_request(
            base_url,
            model,
            prompt,
            max_tokens=8,
            temperature=0.0,
        )
        return {"text": resp["text"], "logprobs": resp.get("logprobs"), "source": "completion"}
    except Exception as e:
        print(f"[warn] completion endpoint failed, falling back to chat: {e}")

    user_prompt = build_user_prompt(instruction, input_text)
    chat_resp = chat_completion(
        base_url,
        model,
        user_prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=8,
        temperature=0.0,
    )
    chat_resp["source"] = "chat"
    return chat_resp


def main(
    in_path: Path = IN_PATH,
    out_path: Path = OUT_PATH,
    base_url: str = BASE_URL,
    model_name: Optional[str] = None,
):
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of Alpaca-style records.")

    if model_name:
        model = model_name
        print(f"[info] Using specified model: {model}")
    else:
        models = get_models(base_url)
        if not models:
            model = "local-model"
            print("[warn] Could not list models; using model='local-model'.")
        else:
            model = models[0]
            print(f"[info] Using model: {model}")

    results = []
    for i, item in enumerate(data, 1):
        try:
            resp = generate_for_item(base_url, model, item)
            gen_clean = normalize_bool(resp.get("text", ""))

            pred_conf = None
            if gen_clean in ("True", "False"):
                if resp.get("source") == "chat":
                    pred_conf = extract_pred_confidence_from_chat_logprobs(resp.get("logprobs"), gen_clean)
                else:
                    pred_conf = extract_pred_confidence_from_completion_logprobs(resp.get("logprobs"), gen_clean)

            new_item = dict(item)
            new_item["generated_output"] = gen_clean
            new_item["pred_confidence"] = pred_conf
            results.append(new_item)
        except Exception as e:
            print(f"[error] item {i} failed: {e}")
            new_item = dict(item)
            new_item["generated_output"] = ""
            new_item["pred_confidence"] = None
            results.append(new_item)

        time.sleep(0.01)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] Wrote {out_path} ({len(results)} items).")


if __name__ == "__main__":
    # Optional CLI: python script.py [in_json] [out_json] [base_url] [model]
    in_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    url_arg = sys.argv[3] if len(sys.argv) > 3 else BASE_URL
    model_arg = sys.argv[4] if len(sys.argv) > 4 else None
    main(in_arg, out_arg, url_arg, model_arg)
