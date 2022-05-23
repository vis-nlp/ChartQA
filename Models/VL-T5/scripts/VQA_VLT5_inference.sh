# The name of experiment
name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python3.8 -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa_inference.py \
        --distributed --multiGPU \
        --test test \
        --num_workers 8 \
        --backbone 't5-base' \
        --output '/home/masry20/output/predictions/' \
        --load BEST \
        --num_beams 5 \
        --valid_batch_size 64 \
        --src_folder "data/" \
        --raw_label \
        --fp16 \
        --use_vis_order_embedding False \
