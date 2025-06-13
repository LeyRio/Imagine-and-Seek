CUDA_VISIBLE_DEIVCES=0 python generate_layout.py \
    --output_path OUTPUT_PATH \
    --llm_path CHECKPOINT_FOR_YOUR_LLM_MODEL \
    --model_type qwen \
    --batch_size 1 \
    --mode circo_test \
    --dataset_path PATH_OF_RETRIEVAL_DATASET 