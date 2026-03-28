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


def score_suffix_via_echo(
    base_url,
    model,
    prompt,
    suffix,
    logprobs_k=5,
    timeout=60,
):
    full_prompt = f"{prompt}{suffix}"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "max_tokens": 0,
        "temperature": 0.0,
        "n": 1,
        "logprobs": logprobs_k,
        "echo": True,
    }
    response = request_with_retries(f"{base_url}/v1/completions", payload, timeout)
    data = response.json()
    choices = (data or {}).get("choices") or []
    if not choices:
        raise ValueError("Completion response did not contain any choices.")

    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    token_logprobs = logprobs.get("token_logprobs")
    text_offsets = logprobs.get("text_offset")
    tokens = logprobs.get("tokens") or []
    if not isinstance(token_logprobs, list) or not isinstance(text_offsets, list):
        raise ValueError("Completion response did not include echoed token logprobs.")

    prompt_len = len(prompt)
    suffix_indices = [
        index
        for index, offset in enumerate(text_offsets)
        if isinstance(offset, int) and offset >= prompt_len
    ]
    if not suffix_indices:
        raise ValueError("Could not locate suffix tokens in echoed completion response.")

    total_logprob = 0.0
    suffix_tokens = []
    for index in suffix_indices:
        token_logprob = token_logprobs[index]
        if token_logprob is None:
            raise ValueError("Echoed completion returned None logprob for a suffix token.")
        total_logprob += float(token_logprob)
        if index < len(tokens):
            suffix_tokens.append(tokens[index])

    return {
        "total_logprob": total_logprob,
        "suffix_token_count": len(suffix_indices),
        "suffix_tokens": suffix_tokens,
        "raw_text": choice.get("text") or "",
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


def run_binary_scoring_flow(
    base_url,
    model,
    prompt,
    labels=("True", "False"),
    logprobs_k=5,
    timeout=60,
    allow_generation_fallback=True,
):
    candidate_attempts = []
    candidate_scores = {}

    for label in labels:
        attempt = {
            "label": label,
            "total_logprob": None,
            "suffix_token_count": 0,
            "suffix_tokens": [],
            "error_message": None,
        }
        try:
            score_info = score_suffix_via_echo(
                base_url,
                model,
                prompt,
                label,
                logprobs_k=logprobs_k,
                timeout=timeout,
            )
            attempt["total_logprob"] = score_info["total_logprob"]
            attempt["suffix_token_count"] = score_info["suffix_token_count"]
            attempt["suffix_tokens"] = score_info["suffix_tokens"]
            candidate_scores[label] = score_info["total_logprob"]
        except Exception as exc:
            attempt["error_message"] = str(exc)
        candidate_attempts.append(attempt)

    if all(label in candidate_scores for label in labels):
        maximum = max(candidate_scores.values())
        probs = {
            label: math.exp(score - maximum)
            for label, score in candidate_scores.items()
        }
        normalizer = sum(probs.values())
        probs = {
            label: value / normalizer
            for label, value in probs.items()
        }
        prediction = max(candidate_scores, key=candidate_scores.get)
        return {
            "prediction": prediction,
            "pred_confidence": probs.get(prediction),
            "raw_completion": "",
            "finish_reason": "scored",
            "error_message": None,
            "completion_attempts": candidate_attempts,
            "candidate_scores": candidate_scores,
            "scoring_mode": "echo_logprob_binary",
        }

    error_messages = [
        f"{attempt['label']}: {attempt['error_message']}"
        for attempt in candidate_attempts
        if attempt["error_message"]
    ]
    if allow_generation_fallback:
        fallback_result = run_completion_flow(
            base_url,
            model,
            prompt,
            logprobs_k=logprobs_k,
            timeout=timeout,
        )
        fallback_result["error_message"] = " | ".join(error_messages) if error_messages else fallback_result["error_message"]
        fallback_result["candidate_scores"] = candidate_scores
        fallback_result["scoring_mode"] = "generation_fallback"
        fallback_result["completion_attempts"] = candidate_attempts + [
            {
                "label": "generation_fallback",
                "attempts": fallback_result["completion_attempts"],
            }
        ]
        return fallback_result

    return {
        "prediction": "",
        "pred_confidence": None,
        "raw_completion": "",
        "finish_reason": None,
        "error_message": " | ".join(error_messages) if error_messages else "Binary scoring failed.",
        "completion_attempts": candidate_attempts,
        "candidate_scores": candidate_scores,
        "scoring_mode": "echo_logprob_binary",
    }
