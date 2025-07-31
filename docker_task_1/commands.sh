#!/bin/bash
echo input folders:
ls  /input/
ls  /input/images/
echo model folder:
ls /opt/ml/model

echo app folder:
ls /opt/app

# echo "huggingface cache folder:"
# ls /home/user/.cache/huggingface


# echo "Running TRIDENT"
# cd /opt/app/TRIDENT
# python run_batch_of_slides.py \
#     --task all \
#     --wsi_dir /input/images/prostatectomy-wsi \
#     --wsi_cache /output/cache \
#     --job_dir /output/trident_processed \
#     --slide_encoder titan \
#     --patch_encoder conch_v15 \
#     --patch_encoder_ckpt /home/user/.cache/huggingface/modules/transformers_modules/titan/conch_v1_5_pytorch_model.bin \
#     --mag 10 \
#     --patch_size 1024 \
#     --batch_size 32 


echo running task_1_inference_template.py
cd /opt/app
# python -u task1_inference_template.py
python -u clinical_only_inference.py
echo finished