import json
import sys
import time
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests

BASE_URL = "http://localhost:8000"
IN_PATH = Path("fold1_test_opinionsqa.json")
TRAIN_PATH = Path("fold1_train_icm_opinionsqa.json")
OUT_PATH = Path("results_icm_few_fold1.json")

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

# Persona label in this dataset is embedded in the input text as:
# "Human Response Preference: ..."
PREFERENCE_RE = re.compile(
    r"\bHuman\s+Response\s+Preference\s*:\s*([^\n\r]+)",
    re.IGNORECASE,
)

# Keep party extraction for compatibility with older OpinionQA-shaped files.
PARTY_RE = re.compile(
    r"\bmajority\s+of\s+(Democrats?|Republicans?|Independent(?:\s*\([^)]*\))?)\b",
    re.IGNORECASE,
)

def _normalize_preference(raw_pref: str) -> Optional[str]:
    if not raw_pref:
        return None
    s = " ".join(raw_pref.strip().split())
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ", ".join(parts) if parts else s

def extract_preference(input_text: str) -> Optional[str]:
    """Extract persona preference label from input text."""
    m = PREFERENCE_RE.search(input_text or "")
    if not m:
        return None
    return _normalize_preference(m.group(1))

def _normalize_party(raw_party: str) -> Optional[str]:
    if not raw_party:
        return None
    s = raw_party.strip().lower()
    if s.startswith("democrat"):
        return "Democrat"
    if s.startswith("republican"):
        return "Republican"
    if s.startswith("independent"):
        return "Independent"
    return None

def extract_party(input_text: str) -> Optional[str]:
    """Extract party label from claim text like: 'The majority of Democrats ...'"""
    m = PARTY_RE.search(input_text or "")
    if not m:
        return None
    return _normalize_party(m.group(1))

def _extract_label(record: Dict[str, Any], use_icm: bool) -> Optional[str]:
    """Pick label field with graceful fallback across dataset variants."""
    key = "icm_output" if use_icm else "gt_output"
    lbl = record.get(key, None)
    if lbl is None:
        lbl = record.get("output", None)
    if lbl is None:
        return None
    s = str(lbl).strip()
    return s if s else None

def index_train_examples(
    train_data: List[Dict[str, Any]],
    use_icm: bool
) -> Tuple[Dict[str, List[Tuple[str, str]]], str, List[Tuple[str, str]]]:
    """
    Build grouped train examples with automatic schema detection.
    Priority: persona preference grouping -> party grouping -> global pool.
    """
    by_pref: Dict[str, List[Tuple[str, str]]] = {}
    by_party: Dict[str, List[Tuple[str, str]]] = {}
    all_examples: List[Tuple[str, str]] = []

    for r in train_data:
        inp = r.get("input", "")
        lbl = _extract_label(r, use_icm)
        if not inp or lbl is None:
            continue

        all_examples.append((inp, lbl))

        pref = extract_preference(inp)
        if pref:
            by_pref.setdefault(pref, []).append((inp, lbl))

        party = extract_party(inp)
        if party:
            by_party.setdefault(party, []).append((inp, lbl))

    if by_pref:
        return by_pref, "preference", all_examples
    if by_party:
        return by_party, "party", all_examples
    return {"ALL": all_examples}, "global", all_examples

def build_fewshot_block(
    group_type: str,
    group_value: Optional[str],
    shots: List[Tuple[str, str]]
) -> str:
    """
    Create a compact labeled few-shot block.
    Each example is:
    Example i:
    Input: ...
    Label: True/False
    """
    n = len(shots)
    header = f"Few-shot examples for {group_type}: {group_value or 'N/A'} (count={n})"
    lines = [header]
    for i, (inp, lbl) in enumerate(shots, 1):
        lines.append(f"Example {i}:\nInput: {inp}\nLabel: {lbl}")
    return "\n\n".join(lines)

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
) -> Dict[str, Any]:
    user_prompt = build_user_prompt_with_fewshot(instruction, input_text, fewshot_text)

    # Prefer completions; fall back to chat
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:"
        text = completion_fallback(
            base_url,
            model,
            prompt,
            max_tokens=8,
            temperature=0.0,
        )
        return {"text": text, "logprobs": None}
    except Exception as e:
        print(f"[warn] completion endpoint failed, falling back to chat: {e}")

    return chat_completion(
        base_url,
        model,
        user_prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=8,
        temperature=0.0,
    )

def main(
    in_path: Path = IN_PATH,
    train_path: Path = TRAIN_PATH,
    out_path: Path = OUT_PATH,
    base_url: str = BASE_URL,
    use_icm: bool = False,
    max_shots: Optional[int] = None,
    model_name: Optional[str] = None,
):
    # Load files
    test_data = json.loads(in_path.read_text(encoding="utf-8"))
    train_data = json.loads(train_path.read_text(encoding="utf-8"))

    if not isinstance(test_data, list):
        raise ValueError("Test JSON must be a list of Alpaca-style records.")
    if not isinstance(train_data, list):
        raise ValueError("Train JSON must be a list of Alpaca-style records with gt_output/icm_output.")

    # Build grouped index for train (preference for persona, party for older datasets).
    grouped_train, group_mode, all_examples = index_train_examples(train_data, use_icm)
    print(f"[info] Indexed train by {group_mode}: {len(grouped_train)} groups.")

    # Pick a model
    if model_name:
        model = model_name
        print(f"[info] Using specified model: {model}")
    else:
        models = get_models(base_url)
        if not models:
            model = "meta-llama/Llama-3.1-70B"
            print("[warn] Could not list models; using model='meta-llama/Llama-3.1-70B'.")
        else:
            model = models[0]
            print(f"[info] Using model: {model}")

    results = []
    for i, item in enumerate(test_data, 1):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "") or ""

        # Pick bucket by active grouping strategy.
        if group_mode == "preference":
            group_value = extract_preference(input_text)
            shots = grouped_train.get(group_value, [])
        elif group_mode == "party":
            group_value = extract_party(input_text)
            shots = grouped_train.get(group_value, [])
        else:
            group_value = "ALL"
            shots = grouped_train.get("ALL", [])

        # If this sample misses a group label, back off to all available examples.
        if not shots and all_examples:
            shots = all_examples

        if max_shots is not None and len(shots) > max_shots:
            shots = shots[:max_shots]

        # Build few-shot text (and visibly include the count)
        fewshot_text = build_fewshot_block(group_mode, group_value, shots)

        # Also log the count for this sample
        print(
            f"[fewshot] sample={i} group_type={group_mode} "
            f"group_value={group_value or 'N/A'} shots={len(shots)}"
        )

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
            
            # Extract confidence
            pred_conf = None
            tf_probs = extract_tf_probs(resp.get("logprobs"))
            if tf_probs and gen_clean in ("True", "False"):
                pred_conf = tf_probs[gen_clean]
                
        except Exception as e:
            print(f"[error] item {i} failed: {e}")
            gen_clean = ""
            pred_conf = None

        new_item = dict(item)
        new_item["generated_output"] = gen_clean
        new_item["pred_confidence"] = pred_conf
        # Bookkeeping for analysis
        new_item["_fewshot_group_type"] = group_mode
        new_item["_fewshot_group_value"] = group_value
        new_item["_fewshot_party"] = group_value if group_mode == "party" else None
        new_item["_fewshot_count"] = len(shots)
        results.append(new_item)

        time.sleep(0.01)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Wrote {out_path} ({len(results)} items).")

if __name__ == "__main__":
    # CLI:
    # python script.py [test_json] [train_json] [out_json] [base_url] [use_icm:0/1] [max_shots] [model]
    in_arg    = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    train_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_PATH
    out_arg   = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_PATH
    url_arg   = sys.argv[4] if len(sys.argv) > 4 else BASE_URL
    use_icm   = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    max_shots = int(sys.argv[6]) if len(sys.argv) > 6 else None
    model_arg = sys.argv[7] if len(sys.argv) > 7 else None
    main(in_arg, train_arg, out_arg, url_arg, use_icm, max_shots, model_arg)
