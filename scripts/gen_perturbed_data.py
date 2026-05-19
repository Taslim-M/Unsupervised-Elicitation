import json
import os
import glob
import random

# ================= 配置区域 =================
INPUT_DIR = './icm_results'       # 原始 .jsonl 文件路径
OUTPUT_DIR = './perturbed_icm_data'   # 输出路径
RANDOM_SEED = 42                  # 固定种子

os.makedirs(OUTPUT_DIR, exist_ok=True)

def map_label_to_str(val):
    """将 0/1 映射为 'False'/'True'"""
    if val == 0: return "False"
    if val == 1: return "True"
    return str(val)

def invert_label_str(label_str):
    """翻转标签: True <-> False"""
    if label_str == "True": return "False"
    if label_str == "False": return "True"
    return label_str

def process_folds():
    # 全局设置种子，确保每次运行结果一致
    random.seed(RANDOM_SEED)
    
    # 1. 获取源文件
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    if not all_files:
        print(f"Error: No .jsonl files found in {INPUT_DIR}")
        return

    print(f"Found {len(all_files)} files. Starting generation...\n")

    for fold_idx in range(1, 5):
        print(f"--- Processing Fold {fold_idx} ---")
        
        # 定义当前Fold的文件后缀
        fold_suffix = f"{fold_idx}of4.jsonl"
        
        # 临时容器
        test_items = []         # 存放测试集
        train_candidates = []   # 存放训练集原始数据 (用于后续生成 Gold 和 Perturbed)
        
        # -------------------------------------------------
        # 步骤 1: 读取并分流数据 (Test vs Train)
        # -------------------------------------------------
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        row = json.loads(line)
                    except:
                        continue
                    
                    # 基础清洗：如果没有 label 或 vanilla_label，直接丢弃
                    if row.get('label') is None or row.get('vanilla_label') is None:
                        continue
                    
                    # 提取通用字段
                    instruction = "Label the input claim as True or False"
                    input_text = row.get('prompt', '')
                    
                    # 获取两种标签的字符串形式
                    lbl_icm = map_label_to_str(row['label'])
                    lbl_gold = map_label_to_str(row['vanilla_label'])
                    
                    # 构建基础数据对象
                    item_base = {
                        "instruction": instruction,
                        "input": input_text,
                        "_vanilla": lbl_gold,  # 暂存真实标签
                        "_icm": lbl_icm        # 暂存ICM标签
                    }

                    # 分流
                    if filename.endswith(fold_suffix):
                        # 是当前Fold -> 归入 Test Set
                        # Test Set 必须使用 Vanilla Label
                        final_item = {
                            "instruction": instruction,
                            "input": input_text,
                            "output": lbl_gold
                        }
                        test_items.append(final_item)
                    else:
                        # 不是当前Fold -> 归入 Train Candidates
                        train_candidates.append(item_base)

        # -------------------------------------------------
        # 步骤 2: 处理并保存 Test Set
        # -------------------------------------------------
        random.shuffle(test_items) # Test集也可以shuffle一下
        test_filename = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_test_opinionsqa.json")
        with open(test_filename, 'w', encoding='utf-8') as f:
            json.dump(test_items, f, indent=4, ensure_ascii=False)
        print(f"  [Test] Generated {os.path.basename(test_filename)} ({len(test_items)} samples)")

        # -------------------------------------------------
        # 步骤 3: 生成 Train Gold (对照组)
        # -------------------------------------------------
        train_gold_items = []
        for cand in train_candidates:
            train_gold_items.append({
                "instruction": cand["instruction"],
                "input": cand["input"],
                "output": cand["_vanilla"] # 始终使用真实标签
            })
        
        random.shuffle(train_gold_items) # 必须 Shuffle
        gold_filename = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_train_gold_opinionsqa.json")
        with open(gold_filename, 'w', encoding='utf-8') as f:
            json.dump(train_gold_items, f, indent=4, ensure_ascii=False)
        print(f"  [Gold] Generated {os.path.basename(gold_filename)} ({len(train_gold_items)} samples)")

        # -------------------------------------------------
        # 步骤 4: 生成 Train Perturbed (实验组)
        # -------------------------------------------------
        # 逻辑：
        # 1. 找出原本ICM标错的 (Wrong) 和标对的 (Correct)
        # 2. 统计错的数量 N
        # 3. 把 N 个错的改成对的 (Fix)
        # 4. 把 N 个对的改成错的 (Break)
        # 5. 剩下的对的保持原样
        
        original_wrong = []
        original_correct = []
        
        for cand in train_candidates:
            if cand["_icm"] != cand["_vanilla"]:
                original_wrong.append(cand)
            else:
                original_correct.append(cand)
        
        count_wrong = len(original_wrong)
        count_correct = len(original_correct)
        
        # 确定翻转数量 (防止极少数情况下正确样本不够用)
        count_to_flip = min(count_wrong, count_correct)
        
        train_perturbed_items = []
        
        # A. 修复原本的错误 (Original Wrong -> Fixed to Vanilla)
        for item in original_wrong:
            train_perturbed_items.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["_vanilla"] # 修正为正确
            })
            
        # B. 破坏原本的正确 (Original Correct -> Corrupted to Inverse Vanilla)
        random.shuffle(original_correct) # 打乱正确组，随机挑选牺牲者
        
        # 前 N 个牺牲者
        for item in original_correct[:count_to_flip]:
            train_perturbed_items.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": invert_label_str(item["_vanilla"]) # 故意改错
            })
            
        # 后面的幸存者
        for item in original_correct[count_to_flip:]:
            train_perturbed_items.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["_vanilla"] # 保持正确
            })
            
        random.shuffle(train_perturbed_items) # 最后必须整体 Shuffle
        
        pert_filename = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_train_perturbed_opinionsqa.json")
        with open(pert_filename, 'w', encoding='utf-8') as f:
            json.dump(train_perturbed_items, f, indent=4, ensure_ascii=False)
        
        print(f"  [Perturbed] Generated {os.path.basename(pert_filename)} ({len(train_perturbed_items)} samples)")
        print(f"     -> Swapped {count_to_flip} labels (Fixed {count_to_flip} errors, Created {count_to_flip} new errors)")
        print("")

if __name__ == "__main__":
    process_folds()
    print("All 12 files generated successfully.")