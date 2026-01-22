#!/bin/bash

# Default model if no argument is provided
MODEL=${1:-llama70b-gpu0}

# Default seed
SEED=${2:-27565976}

echo "Starting training with Model: $MODEL Seed: $SEED"

# Loop through parties
for PARTY in Democrat Republican Independent
do
    echo "Processing Party: $PARTY"

    # Loop through parts 1 to 4
    for i in {1..4}
    do
        FILE_NAME="preferences_POLPARTY_binary_noRefused_${PARTY}_part${i}of4.json"
        
        echo "----------------------------------------------------------------"
        echo "Running ICM for file: $FILE_NAME"
        echo "----------------------------------------------------------------"
        
        python src/experiments/ICM.py \
            --testbed OpinionQA \
            --alpha 50 \
            --file_name "$FILE_NAME" \
            --K 500 \
            --model "$MODEL" \
            --batch_size 128 \
            --seed "$SEED"
            
        if [ $? -eq 0 ]; then
            echo "Successfully finished $FILE_NAME"
        else
            echo "Error running $FILE_NAME"
            exit 1
        fi
    done
done

echo "All parts finished."
