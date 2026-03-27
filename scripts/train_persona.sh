#!/bin/bash

# Default model if no argument is provided
MODEL=${1:-llama70b-gpu0}

# Default seed
SEED=${2:-27565976}

echo "Starting persona training with Model: $MODEL Seed: $SEED"

for FILE_PATH in $(find data/persona_tailor_icm -maxdepth 1 -type f -name '*.json' | sort)
do
    FILE_NAME=$(basename "$FILE_PATH")

    echo "----------------------------------------------------------------"
    echo "Running ICM for file: $FILE_NAME"
    echo "----------------------------------------------------------------"

    python src/experiments/ICM.py \
        --testbed persona \
        --alpha 50 \
        --file_name "$FILE_NAME" \
        --K 500 \
        --model "$MODEL" \
        --batch_size 256 \
        --seed "$SEED"

    if [ $? -eq 0 ]; then
        echo "Successfully finished $FILE_NAME"
    else
        echo "Error running $FILE_NAME"
        exit 1
    fi
done

echo "All persona folds finished."
