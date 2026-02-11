#!/bin/bash

# set path firstly
lightx2v_path=./
model_path=/data/ai-models/lightx2v/wan2.2/Wan2.2-Distill-Models/

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

prompt="
#### Scene 1: Palm Touching the Talisman (4 seconds)
**Prompt:** A young cultivator named Han Li in ancient robes sits in a dimly lit bamboo hut, his palm gently touching a glowing jade peace talisman on a wooden table. Close-up on the hand: a refreshing, ethereal blue energy wave spreads from the talisman into his skin, symbolizing a penetrating clear-minded sensation entering his body and mind. Camera slowly zooms in on the contact point, soft volumetric light highlights the calming aura, no dialogue, serene ambient glow.
**Shot:** Extreme close-up on hand, subtle energy effect for detail.
**Transition:** Fade to wide shot.

#### Scene 2: Inner Calm Restored (5 seconds)
**Prompt:** Han Li's agitated face relaxes instantly in the hut, eyes closing briefly as inner turmoil fades. Medium shot: waves of dark misty energy (representing depression and discomfort) dissipate from his body like smoke, all abnormal phenomena vanishing—heart rate slows, muscles unclench. Background shows flickering candlelight stabilizing, conveying total peace and normalcy restored. Cinematic style with gentle pan down his torso, soft blue hues for tranquility.
**Shot:** Medium close-up on face and upper body, focus on emotional shift.
**Transition:** Smooth dissolve.
"

torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls wan2.2_moe_distill \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/examplex/wan2.2/wan_moe_i2v_distill_quant.json \
--prompt  ${prompt} \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path /data/downloads/singer5.jpg \
--save_result_path ${lightx2v_path}/save_results/wan_moe_i2v_distill_quant-v3.mp4

# torchrun --nproc_per_node=2 -m lightx2v.infer \
# --model_cls wan2.2_moe_distill \
# --task i2v \
# --model_path $model_path \
# --config_json $lightx2v_path/examplex/wan2.2/wan_moe_i2v_distill.json \
# --prompt "一位 成年中国女性，黑色长发在海风中轻轻飘动，发丝清晰分离，边缘被阳光勾勒出细微高光。身穿黑色轻薄外套，布料随风产生自然褶皱，织物纹理清楚可见。皮肤呈现真实质感，面部有自然阴影与反射。背景为开阔海岸线，淡蓝色海水层次分明，水面有细小波纹与光斑反射，整体画面偏电影级写实。" \
# --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
# --image_path "${lightx2v_path}/assets/inputs/imgs/chendulin.jpg" \
# --save_result_path ${lightx2v_path}/save_results/wan_moe_i2v_distill.mp4
