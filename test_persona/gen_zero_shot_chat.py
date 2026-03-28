#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

from completion_utils import (
    complete,
    extract_pred_confidence_from_completion_logprobs,
    normalize_bool,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BASE_URL = "http://localhost:8000"
IN_PATH = ROOT_DIR / "data" / "persona_eval_data" / "DSN_fold1_test_persona.json"
OUT_PATH = SCRIPT_DIR / "results" / "DSN_results_zeroshot_chat_fold1.json"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main(in_path=IN_PATH, out_path=OUT_PATH, base_url=BASE_URL, model=MODEL):
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list.")

    results = []
    for idx, item in enumerate(data, 1):
        raw = ""
        logprobs = None
        prompt = item.get("prompt", "")
        try:
            raw, logprobs = complete(base_url, model, prompt, max_tokens=8, logprobs_k=5)
            prediction = normalize_bool(raw)
            if not prediction:
                raw, logprobs = complete(
                    base_url, model, prompt, max_tokens=16, logprobs_k=5
                )
                prediction = normalize_bool(raw)
        except Exception as exc:
            print(f"[error] item {idx} failed: {exc}")
            prediction = ""

        if not prediction:
            prediction = normalize_bool(raw)

        result = dict(item)
        result["generated_output"] = prediction
        result["pred_confidence"] = (
            extract_pred_confidence_from_completion_logprobs(logprobs, prediction)
            if prediction in ("True", "False")
            else None
        )
        results.append(result)
        time.sleep(0.005)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] Wrote {out_path} ({len(results)} items).")


if __name__ == "__main__":
    in_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_PATH
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    url_arg = sys.argv[3] if len(sys.argv) > 3 else BASE_URL
    model_arg = sys.argv[4] if len(sys.argv) > 4 else MODEL
    main(in_arg, out_arg, url_arg, model_arg)
