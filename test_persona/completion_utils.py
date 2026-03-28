#!/usr/bin/env python3
import math
import re
import time
from typing import Optional

import requests


COMPLETION_STOP = ["\nQuestion:", "\n\nQuestion:"]
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
        return {
            "raw_text": "",
            "text": "",
            "logprobs": None,
            "finish_reason": None,
        }
    return {
        "raw_text": choices[0].get("text") or "",
        "text": (choices[0].get("text") or "").strip(),
        "logprobs": choices[0].get("logprobs"),
        "finish_reason": choices[0].get("finish_reason"),
    }


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


def run_completion_flow(
    base_url,
    model,
    prompt,
    max_tokens_list=(8, 16),
    logprobs_k=5,
    timeout=60,
):
    attempts = []
    selected_attempt = None

    for max_tokens in max_tokens_list:
        attempt = {
            "max_tokens": max_tokens,
            "raw_completion": "",
            "normalized_prediction": "",
            "finish_reason": None,
            "error_message": None,
        }
        try:
            response = complete(
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                logprobs_k=logprobs_k,
                timeout=timeout,
            )
            attempt["raw_completion"] = response["raw_text"]
            attempt["finish_reason"] = response["finish_reason"]
            attempt["normalized_prediction"] = normalize_bool(response["raw_text"])
            attempt["_logprobs"] = response["logprobs"]
        except Exception as exc:
            attempt["error_message"] = str(exc)
            attempt["_logprobs"] = None

        attempts.append(attempt)
        if attempt["normalized_prediction"]:
            selected_attempt = attempt
            break

    if selected_attempt is None and attempts:
        selected_attempt = attempts[-1]

    prediction = selected_attempt["normalized_prediction"] if selected_attempt else ""
    pred_confidence = None
    if prediction in ("True", "False") and selected_attempt is not None:
        pred_confidence = extract_pred_confidence_from_completion_logprobs(
            selected_attempt.get("_logprobs"),
            prediction,
        )

    error_messages = [attempt["error_message"] for attempt in attempts if attempt["error_message"]]
    debug_attempts = []
    for attempt in attempts:
        debug_attempts.append(
            {
                "max_tokens": attempt["max_tokens"],
                "raw_completion": attempt["raw_completion"],
                "normalized_prediction": attempt["normalized_prediction"],
                "finish_reason": attempt["finish_reason"],
                "error_message": attempt["error_message"],
            }
        )

    return {
        "prediction": prediction,
        "pred_confidence": pred_confidence,
        "raw_completion": selected_attempt["raw_completion"] if selected_attempt else "",
        "finish_reason": selected_attempt["finish_reason"] if selected_attempt else None,
        "error_message": " | ".join(error_messages) if error_messages else None,
        "completion_attempts": debug_attempts,
    }
