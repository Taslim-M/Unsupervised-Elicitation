#!/bin/bash

# Array of all JSON file names
files=(
Brazil_18_part4.json
Britain_28_part4.json
France_28_part4.json
Germany_28_part4.json
Indonesia_24_part4.json
Japan_28_part4.json
Jordan_28_part4.json
Lebanon_28_part4.json
Mexico_28_part4.json
Nigeria_18_part4.json
Pakistan_28_part4.json
Russia_28_part4.json
Turkey_28_part4.json
)


total=${#files[@]}
count=0

for f in "${files[@]}"; do
    ((count++))
    batch_size=$(echo "$f" | grep -oE '_[0-9]+_' | tr -d '_')
    
    echo "[$count/$total] Running batch_size=$batch_size for file=$f..."
    
    # Run command quietly
    python ICM.py \
        --testbed truthfulQA \
        --alpha 70 \
        --K 500 \
        --model meta-llama/Llama-3.1-8B \
        --num_seed 8 \
        --batch_size "$batch_size" \
        --file_name "$f" \
        > /dev/null 2>&1

    # Print success or failure
    if [ $? -eq 0 ]; then
        echo "✅ [$count/$total] Completed: $f"
    else
        echo "❌ [$count/$total] Failed: $f"
    fi

    echo "--------------------------------------"
done

echo "🎉 All $total files processed!"
