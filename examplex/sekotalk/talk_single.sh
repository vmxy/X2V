#!/bin/bash

lightx2v_path=./
model_path=/data/ai-models/SekoTek-V2.5

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

#--image_path ${lightx2v_path}/assets/inputs/audio/multi_person/seko_input.png \
#torchrun --nproc-per-node 2 -m lightx2v.infer \
#--model_cls seko_talk \
#--task s2v \
#--model_path $model_path \
#--config_json ${lightx2v_path}/examplex/sekotalk/talk_single.json \
#--prompt  "The video features a man and a woman standing by a bench in the park,Separate the two characters by a distance, Pan appropriately to frame the speaker at the center of the screen, their expressions tense and voices raised as they argue. The man gestures with both hands, his arms swinging slightly as if to emphasize each heated word, while the woman stands with her hands on her waist, her brows furrowed in frustration. The background is a wide expanse of sunlit grass, the golden light contrasting with the sharp energy of their quarrel. Their voices seem to clash in the air, and the rhythm of their hand movements and body postures interweaves with the rising tension, creating a vivid scene of confrontation." \
#--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
#--image_path /data/downloads/talk7.jpg \
#--audio_path ${lightx2v_path}/assets/inputs/audio/multi_person \
#--save_result_path ${lightx2v_path}/save_results/seko_talk_multi_person_dist_fp8_v10.mp4

#torchrun --nproc-per-node 2
python -m lightx2v.infer \
--model_cls seko_talk \
--task s2v \
--model_path $model_path \
--config_json ${lightx2v_path}/examplex/sekotalk/talk_single.json \
--prompt  "The video features a female speaking to the camera with arms spread out, a slightly furrowed brow, and a focused gaze, Generate a stage, appropriately scale the person down and place them in the center of the stage, occupying about 30% of the space. The camera should circle around the person as they act drunk." \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--image_path ${lightx2v_path}/assets/inputs/audio/seko_input.png  \
--audio_path ${lightx2v_path}/assets/inputs/audio/seko_input.mp3 \
--save_result_path ${lightx2v_path}/save_results/singer5_v3.mp4

echo "finished"