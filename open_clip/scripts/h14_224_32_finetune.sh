# 64k batchsize for 2.048e-3 lr
TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 1 -m training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data 'jiviai/xray_caption_conv' \
    --dataset-type "hf" \
    --lr "2.048e-4" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 166 \
    --wd 0.2 \
    --batch-size 64 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.5 gray_scale_prob=0.5 \
    --epochs=10 \
    --workers=1 \
    --model "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378" \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 224 \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logsdir ./logs/ \
    --name 'DFN-ImageFailLatest2' \
    --report-to "wandb" \
    --wandb-project-name "DFN-finetune-chest" \
    --use-last4
    # --val-data 'jiviai/xray_caption_conv' \
