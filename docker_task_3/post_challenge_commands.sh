#!/bin/bash
echo input folders:
ls  /input/
ls  /input/images/

echo app folder:
ls /opt/app

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


cd /opt/app
echo "running post_challenge_clin2_wsi_rna.py"
exec python -u post_challenge_clin2_wsi_rna.py