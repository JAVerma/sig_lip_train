host_api:
	uvicorn app:app --port 8097

train_hf_transformers_clip:
	python transformers/examples/pytorch/contrastive-image-text/run_clip.py \
    --output_dir ./clip_jivi_xray_trained_new_may7_only_caption_change \
    --resume_from_checkpoint ./clip_jivi_xray_trained_new_may6 \
    --model_name_or_path /home/ubuntu/partition/gitartha/clip/scripts/dual_train \
    --train_file /home/ubuntu/partition/gitartha/xray_clip_data/train_tl.json \
    --validation_file /home/ubuntu/partition/gitartha/xray_clip_data/val_tl.json \
    --test_file /home/ubuntu/partition/gitartha/xray_clip_data/test.json \
    --image_column image_path \
    --num_train_epochs 7 \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval --do_predict\
    --per_device_train_batch_size="40" \
    --per_device_eval_batch_size="40" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --overwrite_cache True

eyes_train_hf_transformers_clip:
	python transformers/examples/pytorch/contrastive-image-text/run_clip.py \
    --output_dir ./clip_jivi_eyes_trained_new_june28_withpp \
    --model_name_or_path /home/ubuntu/partition/gitartha/clip/scripts/dual_train \
    --train_file /home/ubuntu/partition/shubham/Datasets/train.json \
    --validation_file /home/ubuntu/partition/shubham/Datasets/val.json \
    --test_file /home/ubuntu/partition/shubham/Datasets/test.json \
    --image_column image_path \
    --caption_column caption \
    --num_train_epochs 10 \
    --remove_unused_columns=False \
    --do_train  --do_eval --do_predict\
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --save_steps 270 \

skin_train_hf_transformers_clip:
	python transformers/examples/pytorch/contrastive-image-text/run_clip.py \
    --output_dir ./clip_jivi_derma_recaption_june30_more_layers \
    --model_name_or_path /home/ubuntu/partition/gitartha/clip/scripts/dual_train \
    --train_file /home/ubuntu/partition/gitartha/derma_recaption/train.json \
    --validation_file /home/ubuntu/partition/gitartha/derma_recaption/validation.json \
    --test_file /home/ubuntu/partition/gitartha/derma_recaption/validation.json \
    --image_column image_path \
    --caption_column caption \
    --num_train_epochs 20 \
    --remove_unused_columns=False \
    --do_train  --do_eval --do_predict\
    --per_device_train_batch_size="12" \
    --per_device_eval_batch_size="12" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --save_steps 2452 \

rsna_train_hf_transformers_clip:
	python transformers/examples/pytorch/contrastive-image-text/run_clip.py \
    --output_dir ./clip_jivi_rsna_refined_captions \
    --model_name_or_path /home/ubuntu/partition/gitartha/clip/scripts/dual_train \
    --train_file /home/ubuntu/shubhamk/RSNA_Refine/train.json \
    --validation_file /home/ubuntu/shubhamk/RSNA_Refine/val.json \
    --test_file /home/ubuntu/shubhamk/RSNA_Refine/test.json \
    --image_column image_path \
    --caption_column caption \
    --num_train_epochs 20 \
    --remove_unused_columns=False \
    --do_train  --do_eval --do_predict\
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --save_steps 601 \
