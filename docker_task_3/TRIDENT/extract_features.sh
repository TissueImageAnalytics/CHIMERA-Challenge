python run_batch_of_slides.py \
    --task all \
    --wsi_dir /media/u1910100/data/slides/ \
    --custom_list_of_wsis /home/u1910100/GitHub/TRIDENT/slides.csv \
    --job_dir ./trident_processed \
    --slide_encoder titan \
    --patch_encoder conch_v15 \
    --patch_encoder_ckpt /home/u1910100/GitHub/TRIDENT/conchv1_5/pytorch_model_vision.bin \
    --mag 20 \
    --patch_size 1024


# python run_single_slide.py \
#     --slide_path /media/u1910100/data/slides/TCGA-55-7725-01Z-00-DX1.4d678777-63b1-4f4a-932a-7fccabf504c7.svs \
#     --job_dir ./trident_processed \
#     --mag 20 \
#     --patch_size 1024 \
#     --patch_encoder conch_v15 \
#     --slide_encoder titan