import json
import os
import glob
import random
import re

# ================= 配置区域 =================
# 输入文件所在的文件夹路径 (请确保12个jsonl文件都在这里)
INPUT_DIR = './qwen30' 
# 输出文件的保存路径
OUTPUT_DIR = './qwen30_shuffled_results'
# 随机种子 (保证每次运行生成的 Shuffle 顺序一致，便于复现)
RANDOM_SEED = 42

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def map_label(val):
    """将数字标签转换为字符串 True/False"""
    if val == 0:
        return "False"
    elif val == 1:
        return "True"
    else:
        return str(val)

def get_fold_index(filename):
    """从文件名中提取 fold 编号，兼容 1of4 和 fold1 两种命名。"""
    m = re.search(r'(\d)of4\.jsonl$', filename)
    if m:
        return int(m.group(1))

    m = re.search(r'fold(\d)\.jsonl$', filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None

def process_folds():
    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 获取目录下所有的 jsonl 文件
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    if not all_files:
        print(f"错误: 在 {INPUT_DIR} 目录下没有找到 .jsonl 文件。请检查路径。")
        return

    print(f"找到 {len(all_files)} 个文件，开始处理...")

    # 循环处理 Fold 1 到 Fold 4
    for fold_idx in range(1, 5):
        print(f"\n--- 正在处理 Fold {fold_idx} ---")
        
        # 初始化列表
        test_data = []
        train_icm_data = []
        train_gold_data = []

        # 遍历所有文件 (包含所有 Party: Democrat, Republican, Independent)
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # 1. 数据清洗: 如果 label 为 null，直接丢弃
                    if row.get('label') is None:
                        continue

                    # 准备通用字段
                    instruction = "Label the input claim as True or False"
                    input_text = row.get('prompt', '') # 使用 prompt 字段作为 input
                    
                    # 获取标签 (ICM用预测值，Gold用真实值)
                    try:
                        label_val_icm = map_label(row['label'])
                        label_val_gold = map_label(row['vanilla_label'])
                    except KeyError:
                        continue

                    # 构建数据项
                    # Test item (始终使用真实标签 vanilla_label)
                    item_test = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": label_val_gold
                    }
                    
                    # Train ICM item (使用预测标签 label)
                    item_train_icm = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": label_val_icm
                    }

                    # Train Gold item (使用真实标签 vanilla_label)
                    item_train_gold = {
                        "instruction": instruction,
                        "input": input_text,
                        "output": label_val_gold
                    }

                    # 2. 分配到 Train 还是 Test
                    row_fold_idx = get_fold_index(filename)
                    if row_fold_idx == fold_idx:
                        # 如果是当前 Fold 的文件 -> 放入测试集
                        test_data.append(item_test)
                    else:
                        # 如果是其他 Fold 的文件 -> 放入训练集
                        # 这里会自动混合所有 Party 的数据
                        train_icm_data.append(item_train_icm)
                        train_gold_data.append(item_train_gold)

        # 3. Shuffle (打乱数据) - 关键步骤！
        # 测试集通常不需要打乱，但为了保险也可以打乱
        random.shuffle(test_data)
        
        # 训练集必须打乱，以混合不同 Party 的数据，避免 Near-bias
        random.shuffle(train_icm_data)
        
        # Gold 训练集也独立打乱
        random.shuffle(train_gold_data)

        # 4. 写入文件
        # (1) Test Set
        test_filename = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_test_opinionsqa.json")
        with open(test_filename, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=4, ensure_ascii=False)
        print(f"生成 Test: {os.path.basename(test_filename)} (样本数: {len(test_data)})")

        # (2) Train ICM Set
        icm_filename = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_train_icm_opinionsqa.json")
        with open(icm_filename, 'w', encoding='utf-8') as f:
            json.dump(train_icm_data, f, indent=4, ensure_ascii=False)
        print(f"生成 Train ICM: {os.path.basename(icm_filename)} (样本数: {len(train_icm_data)})")

        # (3) Train Gold Set
        gold_filename = os.path.join(OUTPUT_DIR, f"fold{fold_idx}_train_gold_opinionsqa.json")
        with open(gold_filename, 'w', encoding='utf-8') as f:
            json.dump(train_gold_data, f, indent=4, ensure_ascii=False)
        print(f"生成 Train Gold: {os.path.basename(gold_filename)} (样本数: {len(train_gold_data)})")

if __name__ == "__main__":
    process_folds()
    print("\n所有处理完成！")