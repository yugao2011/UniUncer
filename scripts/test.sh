bash ./tools/dist_test.sh \
    projects/configs/uniuncer_stage2.py \
    ckpt/uniuncer_final_iter_11720.pth \
    2 \
    --deterministic \
    --eval bbox
