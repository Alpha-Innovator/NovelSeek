#!/bin/bash

export OPENAI_API_KEY={your_openai_key}
export S2_API_KEY={your_semantic_scholar_key}

python launch_dolphin.py \
    --model gpt-4o-2024-11-20 \
    --code_model gpt-4o-2024-11-20 \
    --experiment exp_name (e.g., point_classification_modelnet) \
    --topic "your topic" \
    --rag \
    --num-ideas 3 \
    --round 0 \
    --check_similarity \
    --embedding_model sentence-transformers/all-roberta-large-v1 \
    --save_name {your_save_name} \
    | tee launch_dolphin.txt
