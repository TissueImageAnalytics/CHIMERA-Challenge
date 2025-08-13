#!/usr/bin/env bash
set -euo pipefail   # -e: exit on error, -u: unset var is error, pipefail: catch pipe errors
# optional: nicer error info
trap 'echo "Error on line $LINENO (exit $?)"; exit 1' ERR

echo input folders:
ls  /input/
ls  /input/images/

echo app folder:
ls /opt/app


echo "===== SYSTEM INFO ====="
uname -a

echo
echo "===== CPU INFO ====="
lscpu

echo
echo "===== MEMORY INFO ====="
free -h

echo
echo "===== DISK INFO ====="
df -h

echo
echo "===== GPU INFO ====="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No GPU detected or nvidia-smi not installed."
fi


# echo "Running convert_tissue_mask_for_trident.py"
# python convert_tissue_mask_for_trident.py
# echo "Finished convert_tissue_mask_for_trident.py"

echo "Running TITAN"
cd /opt/app/TRIDENT

python run_batch_of_slides.py \
  --task all \
  --max_workers 8 \
  --wsi_dir /input/images/bladder-cancer-tissue-biopsy-wsi \
  --job_dir /output/trident_processed \
  --segmenter grandqc \
  --slide_encoder titan \
  --patch_encoder conch_v15 \
  --patch_encoder_ckpt /home/user/.cache/huggingface/modules/transformers_modules/titan/conch_v1_5_pytorch_model.bin \
  --mag 10 \
  --patch_size 1024 \
  --batch_size 32

echo "TITAN finished OK"

# echo "Running PRISM"
# cd /opt/app/TRIDENT
# python run_batch_of_slides.py \
#     --task all \
#     --max_workers 8 \
#     --wsi_dir /input/images/prostatectomy-wsi \ 
#     --job_dir /output/trident_processed \
#     --slide_encoder prism \
#     --patch_encoder virchow \
#     --patch_encoder_ckpt /home/user/.cache/huggingface/modules/transformers_modules/virchow/pytorch_model.bin \
#     --mag 10 \
#     --patch_size 896 \
#     --batch_size 64 


cd /opt/app
echo "running task3_inference_114.py"
# Use exec for the last long-running process so PID 1 is Python (proper signals/exit code)
exec python -u task3_inference_114.py