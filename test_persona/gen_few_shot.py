#!/usr/bin/env python3
import json
import re
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
TRAIN_PATH = ROOT_DIR / "data" / "persona_eval_data" / "DSN_fold1_train_icm_persona.json"
OUT_PATH = SCRIPT_DIR / "results" / "DSN_results_icm_few10_fold1.json"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LABEL_MODE_RE = re.compile(r"_train_(icm|gold)_persona$")


def infer_label_mode(train_path):
    match = LABEL_MODE_RE.search(Path(train_path).stem)
    if match:
        return match.group(1)
    return "unknown"


def build_prompt(demonstrations, example):
    demo_prefix = "".join(
        f"{demo['prompt']}{demo['output']}\n\n" for demo in demonstrations
    )
    return f"{demo_prefix}{example['prompt']}"


def main(
    in_path=IN_PATH,
    train_path=TRAIN_PATH,
    out_path=OUT_PATH,
    base_url=BASE_URL,
    max_shots=None,
    model=MODEL,
):
    test_data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    train_data = json.loads(Path(train_path).read_text(encoding="utf-8"))
    if not isinstance(test_data, list):
        raise ValueError("Test data must be a JSON list.")
    if not isinstance(train_data, list):
        raise ValueError("Train data must be a JSON list.")

    train_size = len(train_data)
    requested_shots = train_size if max_shots is None else int(max_shots)
    used_shots = min(requested_shots, train_size)
    demonstrations = train_data[:used_shots]
    label_mode = infer_label_mode(train_path)

    print(
        f"[info] train_size={train_size} requested_shots={requested_shots} "
        f"used_shots={used_shots} label_mode={label_mode}"
    )

    results = []
    for idx, item in enumerate(test_data, 1):
        prompt = build_prompt(demonstrations, item)
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
        result["_requested_shots"] = requested_shots
        result["_used_shots"] = used_shots
        result["_train_size"] = train_size
        result["_train_label_mode"] = label_mode
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
    train_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN_PATH
    out_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_PATH
    url_arg = sys.argv[4] if len(sys.argv) > 4 else BASE_URL
    max_shots_arg = int(sys.argv[5]) if len(sys.argv) > 5 else None
    model_arg = sys.argv[6] if len(sys.argv) > 6 else MODEL
    main(in_arg, train_arg, out_arg, url_arg, max_shots_arg, model_arg)
