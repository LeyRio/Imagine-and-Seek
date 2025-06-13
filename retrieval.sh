CUDA_VISIBLE_DEVICES=0 python src/retrieval_circo.py \
    --submission-name circo_aug_G --dataset-path DATASET_PATH \
    --s_w 1.0 --t_w 1.0 --a_w 1.0 --fusion_weight 0.3 \
    --aug_dir AUG_DIR_PATH \
    --layout_path AUG_DIR_JSON \
    --type G --eval-type LDRE-G --with_aug 