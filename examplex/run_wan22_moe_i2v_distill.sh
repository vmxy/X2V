#!/bin/bash

# set path firstly
lightx2v_path=./
model_path=/data/ai-models/lightx2v/wan22/Wan2.2-Distill-Models/

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
#torchrun --nproc_per_node=2 -m lightx2v.infer \
#--model_cls wan2.2_moe_distill \
#--task i2v \
#--model_path $model_path \
#--config_json ${lightx2v_path}/configs/wan22/wan_moe_i2v_distill.json \
#--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
#--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
#--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
#--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_distill.mp4

torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls wan2.2_moe_distill \
--task i2v \
--model_path $model_path \
--config_json $lightx2v_path/examplex/wan_moe_i2v_distill.json \
--prompt "一位 成年中国女性，黑色长发在海风中轻轻飘动，发丝清晰分离，边缘被阳光勾勒出细微高光。身穿黑色轻薄外套，布料随风产生自然褶皱，织物纹理清楚可见。皮肤呈现真实质感，面部有自然阴影与反射。背景为开阔海岸线，淡蓝色海水层次分明，水面有细小波纹与光斑反射，整体画面偏电影级写实。" \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path "${lightx2v_path}/assets/inputs/imgs/chendulin.jpg" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_distill_chendulin.mp4
