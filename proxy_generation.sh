CUDA_VISIBLE_DEVICES=0 python generate_proxy_migc_elite.py \
    --layout_file LAYOUT_FILE_GENERATED_IN_STEP_1 \
    --image_source CIRCO_CIRR_DATAPATH \
    --output_path OUTPUT_PATH \
    --aug_caption "high quality image" \
    --img_per_mode 3 \
    --MIGCsteps 25 \
    --guidance_scale 7.5 \
    --NaiveFuserSteps 50
