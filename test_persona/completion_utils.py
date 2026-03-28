#!/usr/bin/env python3
import math
import re
import time
from typing import Optional

import requests


COMPLETION_STOP = ["\nQuestion:", "\n\nQuestion:", "\n\n"]
_TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def request_with_retries(
    url,
    payload,
    timeout=60,
    max_retries=3,
    backoff=0.5,
):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            else:
                raise
    raise last_err


def complete(
    base_url,
    model,
    prompt,
    max_tokens=8,
    logprobs_k=5,
    timeout=60,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "n": 1,
        "logprobs": logprobs_k,
        "stop": COMPLETION_STOP,
    }
    response = request_with_retries(f"{base_url}/v1/completions", payload, timeout)
    data = response.json()
    choices = (data or {}).get("choices") or []
    if not choices or "text" not in choices[0]:
        return "", None
    return (choices[0]["text"] or "").strip(), choices[0].get("logprobs")


def normalize_bool(text):
    if not text or not text.strip():
        return ""

    first_tokens = text.strip().split()
    if first_tokens:
        first = first_tokens[0].strip(" .,:;!?\"'()[]{}").lower()
        if first.startswith("true"):
            return "True"
        if first.startswith("false"):
            return "False"

    lowered = text.lower()
    has_true = "true" in lowered
    has_false = "false" in lowered
    if has_true and not has_false:
        return "True"
    if has_false and not has_true:
        return "False"

    match = _TRUE_FALSE_RE.search(text)
    if match:
        return "True" if match.group(1).lower() == "true" else "False"
    return ""


def extract_pred_confidence_from_completion_logprobs(
    logprobs_obj,
    pred_label,
) -> Optional[float]:
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

    values = {}
    if lp_true is not None:
        values["True"] = lp_true
    if lp_false is not None:
        values["False"] = lp_false

    maximum = max(values.values())
    probs = {key: math.exp(value - maximum) for key, value in values.items()}
    normalizer = sum(probs.values())
    probs = {key: value / normalizer for key, value in probs.items()}
    return probs.get(pred_label)
