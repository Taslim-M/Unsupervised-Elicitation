#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

from completion_utils import (
    run_binary_scoring_flow,
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
        prompt = item.get("prompt", "")
        completion_info = None
        try:
            completion_info = run_binary_scoring_flow(base_url, model, prompt)
            prediction = completion_info["prediction"]
        except Exception as exc:
            print(f"[error] item {idx} failed: {exc}")
            prediction = ""
            completion_info = {
                "pred_confidence": None,
                "raw_completion": "",
                "finish_reason": None,
                "error_message": str(exc),
                "completion_attempts": [],
                "candidate_scores": {},
                "scoring_mode": "error",
            }

        result = dict(item)
        result["generated_output"] = prediction
        result["pred_confidence"] = completion_info["pred_confidence"]
        result["raw_completion"] = completion_info["raw_completion"]
        result["finish_reason"] = completion_info["finish_reason"]
        result["error_message"] = completion_info["error_message"]
        result["completion_attempts"] = completion_info["completion_attempts"]
        result["candidate_scores"] = completion_info["candidate_scores"]
        result["scoring_mode"] = completion_info["scoring_mode"]
        result["_prompt_char_len"] = len(prompt)
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
