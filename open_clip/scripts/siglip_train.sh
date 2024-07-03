# 64k batchsize for 2.048e-3 lr
TORCH_CUDNN_V8_API_ENABLED=1 torchrun --nproc_per_node 1 -m training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data 'jiviai/RSNA_Refined' \
    --dataset-type "hf" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 166 \
    --wd 0.2 \
    --batch-size 64 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.5 gray_scale_prob=0.5 \
    --epochs=10 \
    --workers=1 \
    --model "ViT-SO400M-14-SigLIP-384" \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 384 \
    --grad-checkpointing \
    --log-every-n-steps 32 \
    --seed 0 \
    --logsdir ./logs/ \
    --name 'siglip_test_only_image_bglr' \
    --report-to "wandb" \
    --wandb-project-name "siglip-finetune-pneumonia" \
    --vision_encoder 10 \
    --siglip \
    --lr "2.048e-3" \
    # --resume "/home/jayant/Desktop/jivi/sig_lip_train/open_clip/logs/siglip_test_only_image_2/checkpoints/epoch_latest.pt" 
    


    # --text_encoder 7 \
    # --val-data 'jiviai/xray_caption_conv' \
    # --lr "2.048e-4" \
