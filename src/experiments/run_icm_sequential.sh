#!/bin/bash

# Array of all JSON file names
files=(Brazil_12_part3.json Brazil_14_part1.json Brazil_18_part4.json Brazil_20_part2.json 
Britain_28_part1.json Britain_28_part2.json Britain_28_part3.json Britain_28_part4.json 
France_28_part1.json France_28_part2.json France_28_part3.json France_28_part4.json 
Germany_28_part1.json Germany_28_part2.json Germany_28_part3.json Germany_28_part4.json 
Indonesia_18_part3.json Indonesia_22_part1.json Indonesia_24_part4.json Indonesia_28_part2.json 
Japan_28_part1.json Japan_28_part2.json Japan_28_part3.json Japan_28_part4.json 
Jordan_28_part1.json Jordan_28_part2.json Jordan_28_part3.json Jordan_28_part4.json 
Lebanon_28_part1.json Lebanon_28_part2.json Lebanon_28_part3.json Lebanon_28_part4.json 
Mexico_28_part1.json Mexico_28_part2.json Mexico_28_part3.json Mexico_28_part4.json 
Nigeria_12_part3.json Nigeria_14_part1.json Nigeria_16_part2.json Nigeria_18_part4.json 
Pakistan_28_part1.json Pakistan_28_part2.json Pakistan_28_part3.json Pakistan_28_part4.json 
Russia_28_part1.json Russia_28_part2.json Russia_28_part3.json Russia_28_part4.json 
Turkey_28_part1.json Turkey_28_part2.json Turkey_28_part3.json Turkey_28_part4.json)



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
        --use_goldseed 1 \
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
