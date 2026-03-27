import asyncio
import json
import math
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import argparse

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment
from src.experiments.ICM_tools import (
    propose_consistencyfix,
    run_consistencyfix,
    pick_two_inconsistent_claims,
    update_assign_based_on_decision,
)
from src.model_querying.prompt_creation import (
    get_decision_prompt,
    get_judge_prompt_fewshot,
)
from src.model_querying.solution_extraction import (
    extract_claim_logprobs,
    extract_decision_logprobs,
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.tools.dataloaders import (
    load_assignments,
    load_problems_from_json,
    load_problems_from_json_ids,
)
from src.tools.path_utils import get_default_results_directory, get_root_directory


def calculate_accuracy(train_data, inconsistent_pairs):
    train_probs = []
    for i in train_data.values():
        if i["label"] is None:
            continue
        if i["label"] == 1:
            train_probs.append(i["score"])
        else:
            train_probs.append(-i["score"])
    if len(train_probs) == 0:
        train_prob = 0
    else:
        train_prob = np.mean(train_probs)

    return {
        "train_accuracy": 0
        if len(train_data) == 0
        else np.mean([i["label"] == i["vanilla_label"] for i in train_data.values()]),
        "train_label_distribution": Counter(
            [i["vanilla_label"] for i in train_data.values()]
        ),
        "train_predict_distribution": Counter(
            [i["label"] for i in train_data.values()]
        ),
        "train_prob": train_prob,
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs),
    }


def update_assign(data):
    for key, value in data.items():
        if value["score"] > 0:
            value["label"] = 1
        else:
            value["label"] = 0
    return data


def fix_inconsistency(demonstrations, cur_metric, name, alpha, iter=0, K=20):
    backup_metric = deepcopy(cur_metric)
    if cur_metric["inconsistent_num"] == 0:
        return demonstrations, cur_metric

    cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}
    assignment = cur_pool
    
    best_metric = cur_metric
    best_assignment = assignment
    best_decision_id = None
    for k in range(K):
        pipeline = propose_consistencyfix(
            args.model,
            name=name,
            iter=f"{iter}-{k}",
            assignment=assignment,
        )
        results = asyncio.run(pipeline.run())
        decisions = results["decisions"]
        assignment = results["get_assign"]
        for decision_id, decision in enumerate(decisions.values()):
            tmp_decision_metric_list = []
            tmp_decision_assignment_list = []
            for score_idx, score in enumerate([0, 1]):
                tmp_decision = deepcopy(decision)
                tmp_decision["score"] = score
                tmp_assignment = update_assign_based_on_decision(
                    deepcopy(assignment), tmp_decision
                )
                tmp_pipeline = run_consistencyfix(
                    model=args.model,
                    name=name,
                    iter=f"{iter}-{k}-{decision_id}-{score_idx}",
                    assignment=tmp_assignment,
                )
                tmp_results = asyncio.run(tmp_pipeline.run())
                tmp_metric = tmp_results["evaluate"]
                tmp_decision_metric_list.append(tmp_metric)
                tmp_decision_assignment_list.append(tmp_assignment)
            tmp_best_decision_id = np.argmax(
                [get_energy(i, args.alpha) for i in tmp_decision_metric_list]
            )
            tmp_assignment = tmp_decision_assignment_list[tmp_best_decision_id]
            tmp_metric = tmp_decision_metric_list[tmp_best_decision_id]

            if get_energy(tmp_metric, args.alpha) >= get_energy(best_metric, args.alpha):
                best_decision_id = decision_id
                best_metric = tmp_metric
                best_assignment = tmp_assignment
                break
        if best_decision_id is None:
            break
        elif best_metric["inconsistent_num"] == 0:
            assignment = best_assignment
            break
        else:
            assignment = best_assignment

    for k in assignment:
        demonstrations[k] = assignment[k]

    return demonstrations, best_metric


def get_pipeline(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
):
    pipeline_name = f"iterative-truth-assign-iter-{iter}"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    ROOT_DIR = get_root_directory()
    DATA_DIR = ROOT_DIR / "data"


    pipeline_config = PipelineConfig(
        pipeline_name,
        anthropic_num_threads=40,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    def add_train_demonstrations(train_data):
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        values = list(copy_data.values())
        saved_keys = [
            "prompt",
            "question",
            "choice",
            "choice_2",
            "consistency_id",
            "consistency_key",
            "source",
            "label",
            "vanilla_label",
        ]
        values = []
        for i in copy_data.values():
            values.append({saved_key: i[saved_key] for saved_key in saved_keys if saved_key in i})

        for idx, key in enumerate(keys):
            tmp_keys, tmp_values = [], []
            for j, (prev_key, prev_value) in enumerate(zip(keys, values)):
                if j != idx:
                    tmp_keys.append(prev_key)
                    tmp_values.append(prev_value)

            demos = {
                prev_key: prev_value
                for j, (prev_key, prev_value) in enumerate(zip(tmp_keys, tmp_values))
            }

            sorted_demos = {}
            for k, v in demos.items():
                q = v["consistency_id"]
                if q not in sorted_demos:
                    sorted_demos[q] = []
                sorted_demos[q].append((k, v))

            out_sorted_demos = {}
            for group in sorted_demos.values():
                for k, v in group:
                    out_sorted_demos[k] = v
                    
            copy_data[key]["demonstration"] = out_sorted_demos

        return copy_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    get_train_preds = pipeline.add_query_step(
        "get_train_preds",
        model,
        get_judge_prompt_fewshot,
        extract_claim_logprobs,
        dependencies=[merged_train_data],
        logprobs=20,
        max_tokens=1,
        use_cache=use_cache,
    )

    pick_claims = pipeline.add_transformation_step(
        "pick_two_inconsistent_claims",
        pick_two_inconsistent_claims,
        dependencies=[initial_assign],
    )

    eval_preds = pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds, pick_claims],
    )
    return pipeline


async def predict_assignment(model, example, demonstrations):
    demos = [
        v
        for k, v in demonstrations.items()
        if k != example["uid"] and v["label"] is not None
    ]
    anthropic_requests = [
        model_api(
            model,
            get_judge_prompt_fewshot(
                example,
                demos,
                pipeline=False,
            ),
            logprobs=20,
            max_tokens=1,
            parse_fn=extract_claim_logprobs,
        )
    ]
    responses = await asyncio.gather(*anthropic_requests)
    score = responses[0][0]["score"]
    new_label = score > 0
    return int(new_label)


def get_temperature(
    iteration, initial_temp, final_temp, decay_rate, schedule="exp"
):
    """
    Calculate the temperature for simulated annealing.

    Parameters:
    - iteration: Current iteration number.
    - initial_temp: Initial temperature.
    - decay_rate: Rate at which the temperature decreases.

    Returns:
    - Current temperature.
    """
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate**iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=30)
    parser.add_argument("--seed", type=int, default=27565976)
    parser.add_argument("--testbed", type=str, default="gsm8k")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--K", type=int, default=3000)
    parser.add_argument("--consistency_fix_K", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="log")
    parser.add_argument("--file_name", type=str, default=None, help="Override the dataset file name in data/ (e.g., train_truthfulqa.json)")
    parser.add_argument("--use_goldseed", type=int, default=0) 
    parser.add_argument("--continue_from_existing", type=int, default=0)
    parser.add_argument("--use_interactive_restart", type=int, default=0)
    args = parser.parse_args()
    return args


def sanitize_name_component(value):
    return str(value).replace("\\", "-").replace("/", "-").replace(" ", "_")


def resolve_dataset_file(args):
    if args.file_name is None:
        raise ValueError("--file_name is required for this experiment.")

    data_dir = get_root_directory() / "data"
    requested_path = Path(args.file_name)

    if requested_path.is_absolute():
        try:
            requested_path = requested_path.relative_to(data_dir)
        except ValueError:
            requested_path = Path(requested_path.name)
    elif requested_path.parts and requested_path.parts[0] == "data":
        requested_path = Path(*requested_path.parts[1:])

    normalized_path = requested_path
    if args.testbed == "persona":
        if not (requested_path.parts and requested_path.parts[0] == "persona_tailor_icm"):
            normalized_path = Path("persona_tailor_icm") / requested_path.name
    elif args.testbed == "OpinionQA":
        direct_path = data_dir / requested_path
        if not direct_path.exists() and not (
            requested_path.parts and requested_path.parts[0] == "processed_OpinionQA"
        ):
            normalized_path = Path("processed_OpinionQA") / requested_path.name

    resolved_path = data_dir / normalized_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {resolved_path}")

    args.file_name = normalized_path.as_posix()
    args.file_tag = sanitize_name_component(normalized_path.as_posix())
    return resolved_path


def load_data(args):
    data_path = resolve_dataset_file(args)
    
    if args.testbed == "alpaca":
        with open(data_path) as f:
            train = json.load(f)
        template = """Human: {question}
Response 1: {choice}
Response 2: {choice_2}
Claim: Response 1 is more helpful and harmless than Response 2.
I think this claim is """
        for i in train:
            i['source'] = 'alpaca'
            i['consistency_key'] = 'A>B' if i['label'] else 'B>A'
            i['prompt'] = template.format(question=i['question'], choice=i['choice'].replace("\n", ""), choice_2=i['choice_2'].replace("\n", ""))
        
        args.GROUP_SIZE = 2
        
    elif args.testbed == 'gsm8k':
        with open(data_path) as f:
            train = json.load(f)
        template = """Question: {question}
Claim: {answer}
I think this claim is """

        for i in train:
            i['source'] = 'gsm8k'
            i['consistency_key'] = i['answer']
            i['prompt'] = template.format(question=i['question'], answer=i['choice'])

        args.GROUP_SIZE = 4
        
    elif args.testbed == 'truthfulQA':
        with open(data_path) as f:
            train = json.load(f)
        template = """Question: {question}
Claim: {answer}
I think this claim is """

        for i in train:
            i['source'] = 'truthfulQA'
            i['consistency_key'] = 'A' if i['label'] else 'B'
            i['prompt'] = template.format(question=i['question'], answer=i['choice'])
        args.GROUP_SIZE = 4
        
    elif args.testbed == 'truthfulQA-preference':
        with open(data_path) as f:
            train = json.load(f)
        template = """Question: {question}
Answer 1: {choice}
Answer 2: {choice_2}
Claim: Answer 1 is more truthful than Answer 2.
I think this claim is """

        for i in train:
            i['source'] = 'truthfulQA-preference'
            i['consistency_key'] = 'A>B' if i['label'] else 'B>A'
            i['prompt'] = template.format(question=i['question'], choice=i['choice'], choice_2=i['choice_2'])
        args.GROUP_SIZE = 2

    elif args.testbed == 'OpinionQA':
        with open(data_path) as f:
            train = json.load(f)
        template = """Question: {question}
Claim: {choice}
I think this claim is """

        for i in train:
            i['source'] = 'OpinionQA'
            i['prompt'] = template.format(question=i['question'], choice=i['choice'])
        args.GROUP_SIZE = 2

    elif args.testbed == "persona":
        with open(data_path) as f:
            train = json.load(f)
        template = """Question: {question}
Response 1: {choice}
Response 2: {choice_2}
Claim: Response 1 better matches the human response preference than Response 2.
I think this claim is """

        for i in train:
            i["source"] = "persona"
            i["consistency_key"] = "A>B" if i["label"] else "B>A"
            i["prompt"] = template.format(
                question=i["question"],
                choice=i["choice"],
                choice_2=i["choice_2"],
            )
        args.GROUP_SIZE = 2
        if args.batch_size > len(train):
            print(
                f"Clamping persona batch_size from {args.batch_size} to full fold size {len(train)}"
            )
            args.batch_size = len(train)
        
    train_map = {}
    for i in train:
        if i['consistency_id'] not in train_map:
            train_map[i['consistency_id']] = []
        train_map[i['consistency_id']].append(i)

    # Sample at the consistency_id-group level, then flatten.
    group_keys = list(train_map.keys())
    target_num_groups = args.batch_size // args.GROUP_SIZE
    if target_num_groups <= 0:
        raise ValueError(
            f"Invalid batch_size/GROUP_SIZE: batch_size={args.batch_size}, GROUP_SIZE={args.GROUP_SIZE}"
        )

    # When continuing or interactively restarting, force-include any groups that already
    # have ICM labels, so the run does not depend on reproducing the original seed sampling.
    force_include_group_keys = []
    if bool(args.continue_from_existing) or bool(args.use_interactive_restart):
        for key in group_keys:
            group = train_map[key]
            if any(ex.get("icm_label", None) is not None for ex in group):
                force_include_group_keys.append(key)

    if len(force_include_group_keys) > target_num_groups:
        raise ValueError(
            "Too many consistency_id groups with existing icm_label to fit into the requested batch. "
            f"Have {len(force_include_group_keys)} labeled groups, but batch can only hold {target_num_groups} groups "
            f"(batch_size={args.batch_size}, GROUP_SIZE={args.GROUP_SIZE}). Increase batch_size or reduce labeled groups in the file."
        )

    remaining_group_keys = [
        k for k in group_keys if k not in set(force_include_group_keys)
    ]
    num_to_sample = target_num_groups - len(force_include_group_keys)
    if len(remaining_group_keys) < num_to_sample:
        raise ValueError(
            "Not enough remaining consistency_id groups to fill the requested batch. "
            f"Need {num_to_sample} more groups, but only have {len(remaining_group_keys)} available. "
            "Decrease batch_size or provide a larger dataset file."
        )

    sampled_group_keys = (
        random.sample(remaining_group_keys, num_to_sample) if num_to_sample > 0 else []
    )
    selected_group_keys = force_include_group_keys + sampled_group_keys

    # Optional sanity check: group sizes should match GROUP_SIZE.
    bad_groups = [k for k in selected_group_keys if len(train_map[k]) != args.GROUP_SIZE]
    if len(bad_groups) > 0:
        raise ValueError(
            "Some selected consistency_id groups do not match GROUP_SIZE. "
            f"GROUP_SIZE={args.GROUP_SIZE}, offending consistency_id(s)={bad_groups[:5]}" +
            (" (truncated)" if len(bad_groups) > 5 else "")
        )

    selected_train = []
    for key in selected_group_keys:
        selected_train.extend(train_map[key])

    fewshot_ids = list(range(len(selected_train)))
    return selected_train, fewshot_ids

def initialize(train, fewshot_ids, args):
    demonstrations = {}
    unlabeled_ids = []
    whole_ids = []
    seed_ids = []

    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)

    # First build items with stable uids for this run
    items = []
    for uid, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"]  # store dataset labels to measure agreement during the searching process
        item["uid"] = uid
        whole_ids.append(uid)
        items.append(item)
        demonstrations[uid] = item

    if bool(args.continue_from_existing):
        # Keep any existing icm_label as labeled *in addition* to a fresh random seed set.
        # This ensures initial pool size = num_seed + (# existing icm_label entries in the batch).
        icm_labeled_items = []
        for item in items:
            if item.get("icm_label", None) is not None:
                item["label"] = item["icm_label"]
                item["type"] = "predict"
                icm_labeled_items.append(item)
            else:
                item["label"] = None
                item["type"] = "predict"

        unlabeled_pool = [it for it in items if it.get("label", None) is None]
        if len(unlabeled_pool) < args.num_seed:
            raise ValueError(
                "Not enough unlabeled examples in the selected batch to create the requested seed set. "
                f"Need num_seed={args.num_seed}, but only have {len(unlabeled_pool)} unlabeled examples. "
                "Increase batch_size or decrease num_seed."
            )

        print(f"ICM labeled items: {len(icm_labeled_items)}, their vanilla labels match: {sum(1 for it in icm_labeled_items if it['label'] == it['vanilla_label'])}/{len(icm_labeled_items)}")
        
        chosen_seed_items = random.sample(unlabeled_pool, args.num_seed)
        for idx, item in enumerate(chosen_seed_items):
            item["type"] = "seed"
            item["label"] = random_init_labels[idx]
            seed_ids.append(item["uid"])

        unlabeled_ids = [it["uid"] for it in items if it.get("label", None) is None]
        return demonstrations, unlabeled_ids, whole_ids, seed_ids

    if bool(args.use_interactive_restart):
        # Restart from existing interactive icm_label seeds, then top up with random seeds to reach num_seed.
        for item in items:
            if item.get("icm_label", None) is not None:
                item["label"] = item.get("icm_label", None)
                item["type"] = "seed"
                seed_ids.append(item["uid"])
            else:
                item["label"] = None
                item["type"] = "predict"

        remaining = max(0, args.num_seed - len(seed_ids))
        if remaining > 0:
            unlabeled_pool = [it for it in items if it.get("label", None) is None]
            if len(unlabeled_pool) < remaining:
                raise ValueError(
                    "Not enough unlabeled examples in the selected batch to top up the seed set. "
                    f"Need {remaining} more seeds (num_seed={args.num_seed}, existing_icm={len(seed_ids)}), "
                    f"but only have {len(unlabeled_pool)} unlabeled examples."
                )
            chosen_seed_items = random.sample(unlabeled_pool, remaining)
            # Use a fresh random label list sized for the top-up portion.
            topup_labels = [1] * (remaining // 2) + [0] * (remaining - (remaining // 2))
            random.shuffle(topup_labels)
            for idx, item in enumerate(chosen_seed_items):
                item["type"] = "seed"
                item["label"] = topup_labels[idx]
                seed_ids.append(item["uid"])

        unlabeled_ids = [it["uid"] for it in items if it.get("label", None) is None]
        return demonstrations, unlabeled_ids, whole_ids, seed_ids

    # Default behavior (fresh run): first num_seed are seeds, rest unlabeled.
    for uid, item in enumerate(items):
        if uid >= args.num_seed:
            item["label"] = None
            item["type"] = "predict"
            unlabeled_ids.append(uid)
        elif bool(args.use_goldseed):
            item["type"] = "seed"
            item["label"] = item["vanilla_label"]
            seed_ids.append(uid)
        else:
            item["type"] = "seed"
            item["label"] = random_init_labels[uid]
            seed_ids.append(uid)

    return demonstrations, unlabeled_ids, whole_ids, seed_ids


def main(args):
    train, fewshot_ids = load_data(args)
    
    # If continuing from existing or restarting, force-include all samples with icm_label
    if bool(args.continue_from_existing) or bool(args.use_interactive_restart):
        # Find all samples with icm_label in the entire dataset
        icm_labeled_indices = [idx for idx, item in enumerate(train) if item.get("icm_label", None) is not None]
        
        # Add missing icm_label samples to fewshot_ids
        existing_ids_set = set(fewshot_ids)
        for idx in icm_labeled_indices:
            if idx not in existing_ids_set:
                fewshot_ids.append(idx)
        
        print(f"Force-included {len(icm_labeled_indices)} samples with icm_label. Total batch size: {len(fewshot_ids)}")

    demonstrations, unlabeled_ids, whole_ids, seed_ids = initialize(train, fewshot_ids, args)
    
    # Debug: show icm_label samples after initialize
    if bool(args.continue_from_existing) or bool(args.use_interactive_restart):
        icm_items = {k: v for k, v in demonstrations.items() if v.get("icm_label", None) is not None}
        print(f"After initialize, icm_label samples: {[(k, v['label'], v['vanilla_label']) for k, v in icm_items.items()]}")
    
    cur_metric = {
        "train_prob": -1e6,
        "inconsistent_num": 100000,
        "train_accuracy": 1.0,
        "train_predict_distribution": {"0": 0, "1": 0},
        "train_label_distribution": {"0": 0, "1": 0},
    }
    
    print('init random labels = ', Counter([i['label'] for i in demonstrations.values() if i['type'] == 'seed']), 'init label acc = ', np.mean([i['label'] == i['vanilla_label'] for i in demonstrations.values() if i['type'] == 'seed']))
    name = f"{args.testbed}-llama70b-K{args.K}-bc{args.batch_size}_seed{args.seed}-initialsize{args.num_seed}-weighted{args.alpha}-decay{args.decay}-initialT{args.initial_T}-finalT{args.final_T}-scheduler{args.scheduler}-usegoldseed{args.use_goldseed}-file{getattr(args, 'file_tag', sanitize_name_component(args.file_name))}"

    iter = 0
    flip_cnt = 0
    example_id = 0

    for _ in tqdm(range(args.K), desc="searching"):
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
        initial_demos = deepcopy(demonstrations)
        if iter == 0:
            pipeline = get_pipeline(
                args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=cur_pool,
            )
            results = asyncio.run(pipeline.run())
            cur_metric = results["evaluate"]
            
            # Debug: check icm_label samples before fix
            if bool(args.continue_from_existing) or bool(args.use_interactive_restart):
                icm_items = {k: v for k, v in demonstrations.items() if v.get("icm_label", None) is not None}
                print(f"Before fix_inconsistency, icm_label samples: {[(k, v['label'], v.get('score', 'N/A')) for k, v in icm_items.items()]}")
            
            demonstrations, cur_metric = fix_inconsistency(
                demonstrations, cur_metric, name, args.alpha, iter=iter, K=args.consistency_fix_K
            )
            
            # Debug: check icm_label samples after fix
            if bool(args.continue_from_existing) or bool(args.use_interactive_restart):
                icm_items = {k: v for k, v in demonstrations.items() if v.get("icm_label", None) is not None}
                print(f"After fix_inconsistency, icm_label samples: {[(k, v['label'], v.get('score', 'N/A')) for k, v in icm_items.items()]}")
            
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
       
        while True: # weighted sampling
            # if(args.continue_from_existing):
            #     candidate_ids_unlocked = [i for i in whole_ids if not demonstrations[i]["label_locked"]]
            #     candidates_ids = candidate_ids_unlocked
            # else:
            candidates_ids = whole_ids
            # if(args.continue_from_existing):
            #     candidate_ids_unlocked = [i for i in whole_ids if not demonstrations[i]["label_locked"]]
            #     candidates_ids = candidate_ids_unlocked
            # else:
            candidates_ids = whole_ids
            weights = [1 for _ in range(len(candidates_ids))]
            for i in candidates_ids:
                if i in cur_pool:
                    same_consistency_group_ids = [j for j in candidates_ids if demonstrations[j]["consistency_id"] == demonstrations[i]["consistency_id"]]
                    for j in same_consistency_group_ids:
                        if j not in cur_pool:
                            weights[j] = 100

            example_id = random.choices(candidates_ids, k=1, weights=weights)[0]
            break

        new_label = asyncio.run(
            predict_assignment(
                args.model,
                demonstrations[example_id],
                cur_pool,
            )
        )

        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label
            dummy_metric = {
                "train_prob": -1e6,
                "inconsistent_num": 100000,
                "train_accuracy": 1.0,
                "train_predict_distribution": {"0": 0, "1": 0},
                "train_label_distribution": {"0": 0, "1": 0},
            }

            tmp_demonstrations, _ = fix_inconsistency(
                tmp_demonstrations,
                dummy_metric,
                name + "newlabelexplore",
                args.alpha,
                iter=iter,
                K=10,
            )
            
            tmp_pool = {
                k: v
                for k, v in tmp_demonstrations.items()
                if v["label"] is not None
            }
            pipeline = get_pipeline(
                model=args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=tmp_pool,
            )
            results = asyncio.run(pipeline.run())
            metric = results["evaluate"]
            T = get_temperature(
                flip_cnt, args.initial_T, args.final_T, args.decay, schedule=args.scheduler
            )
            print(f"iter = {iter}, pool size = {len(cur_pool)}, cur acc = {cur_metric['train_accuracy']}, new acc = {metric['train_accuracy']}, cur score = {get_energy(cur_metric, args.alpha)}, new score = {get_energy(metric, args.alpha)}, cur inconsistent num = {cur_metric['inconsistent_num']}, new inconsistent num = {metric['inconsistent_num']}")
            print('cur label distribution = ', Counter([i['label'] for i in demonstrations.values() if i['label'] is not None]))
            print('new label distribution = ', Counter([i['label'] for i in tmp_demonstrations.values() if i['label'] is not None]))

            accept_prob = math.exp((get_energy(metric, args.alpha) - get_energy(cur_metric, args.alpha)) / T)
            print("accept prob = ", accept_prob)
            if random.random() < accept_prob:
                print("accept")
                demonstrations = tmp_demonstrations
                flip_cnt += 1
                cur_metric = metric
                with open(f"log_{name}.jsonl", "a") as f:
                    f.write(json.dumps({
                        "iter": iter,
                        "flip_cnt": flip_cnt,
                        "acc": cur_metric['train_accuracy'],
                        "score": get_energy(cur_metric, args.alpha),
                    }) + "\n")
            else:
                print("reject")
        
        print("=" * 100)
        iter += 1
    
    # Re-score final pool so each labeled item has a 'score'
    cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}
    pipeline = get_pipeline(model=args.model, name=name, num_problems=None, iter="final", assignment=cur_pool)
    results = asyncio.run(pipeline.run())
    
    preds = results["get_train_preds"]
    
    # Build a robust uid->score map
    final_scores = {}
    for k, d in preds.items():
        # Preferred: use the uid stored in metadata (always present)
        meta_uid = d.get("metadata", {}).get("uid", None)
        if meta_uid is None:
            # Fallback: handle keys like "123-0" by stripping the suffix
            try:
                meta_uid = int(str(k).split("-")[0])
            except Exception:
                continue
        final_scores[int(meta_uid)] = d.get("score", None)
    
    # Attach scores back by integer uid
    for uid in cur_pool.keys():
        demonstrations[uid]["score"] = final_scores.get(int(uid), None)
        
    # Save labels to results/<name>/final_labels.jsonl
    save_dir = get_root_directory() / name
    os.makedirs(save_dir, exist_ok=True)
    out_path = save_dir / "final_labels.jsonl"
    with open(out_path, "w") as f:
        for uid, ex in sorted(demonstrations.items()):
            f.write(json.dumps({
                "uid": uid,
                "consistency_id": ex.get("consistency_id"),
                "consistency_key": ex.get("consistency_key"),
                "source": ex.get("source"),
                "label": ex.get("label"),                # ICM’s final label (0/1 or None)
                "vanilla_label": ex.get("vanilla_label"),# dataset label if present (for eval)
                "score": ex.get("score"),                # model True-vs-False logprob diff
                "question": ex.get("question"),
                "choice": ex.get("choice"),
                "choice_2": ex.get("choice_2"),
                "persona_cluster_id": ex.get("persona_cluster_id"),
                "prompt": ex.get("prompt"),
                "type": ex.get("type"),
            }) + "\n")

    print(f"Saved final labels to: {out_path}")


if __name__ == "__main__":
    setup_environment(logger_level="error")
    model_api = ModelAPI(anthropic_num_threads=20, openai_fraction_rate_limit=0.99)
    args = get_args()  
    print("task: ", args.testbed)
    random.seed(args.seed)
    main(args)
