# The name of experiment
name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python3.8 -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqa.py \
        --distributed --multiGPU \
        --train train \
        --valid validation \
        --test test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 10 \
        --num_workers 16 \
        --backbone 't5-base' \
        --output '/home/masry20/output/' \
        --load Epoch30 \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 64 \
        --src_folder "data/" \
        --raw_label \
        --fp16 \
