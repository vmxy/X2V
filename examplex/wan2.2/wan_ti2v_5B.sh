#!/bin/bash

# set path firstly
lightx2v_path=./
model_path=/data/ai-models/wan2.2/Wan2.2-TI2V-5B

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls wan2.2 \
--task t2v \
--model_path $model_path \
--config_json ${lightx2v_path}/examplex/wan2.2/wan_ti2v_5B.json \
--prompt  "The video features a man and a woman standing by a bench in the park,Separate the two characters by a distance, Pan appropriately to frame the speaker at the center of the screen, their expressions tense and voices raised as they argue. The man gestures with both hands, his arms swinging slightly as if to emphasize each heated word, while the woman stands with her hands on her waist, her brows furrowed in frustration. The background is a wide expanse of sunlit grass, the golden light contrasting with the sharp energy of their quarrel. Their voices seem to clash in the air, and the rhythm of their hand movements and body postures interweaves with the rising tension, creating a vivid scene of confrontation." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path /data/downloads/talk7.jpg \
--save_result_path ${lightx2v_path}/save_results/wan_ti2v_5B-cat.mp4

# torchrun --nproc_per_node=2 -m lightx2v.infer \
# --model_cls wan2.2_moe_distill \
# --task i2v \
# --model_path $model_path \
# --config_json $lightx2v_path/examplex/wan2.2/wan_moe_i2v_distill.json \
# --prompt "一位 成年中国女性，黑色长发在海风中轻轻飘动，发丝清晰分离，边缘被阳光勾勒出细微高光。身穿黑色轻薄外套，布料随风产生自然褶皱，织物纹理清楚可见。皮肤呈现真实质感，面部有自然阴影与反射。背景为开阔海岸线，淡蓝色海水层次分明，水面有细小波纹与光斑反射，整体画面偏电影级写实。" \
# --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
# --image_path "${lightx2v_path}/assets/inputs/imgs/chendulin.jpg" \
# --save_result_path ${lightx2v_path}/save_results/wan_moe_i2v_distill.mp4
