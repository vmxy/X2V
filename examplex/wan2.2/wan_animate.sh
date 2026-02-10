#!/bin/bash

# 添加命令行参数解析
preprocess=false  # 默认执行预处理

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -pre|--preprocess)
            preprocess=true
            shift
            ;;
        -no-pre|--no-preprocess)
            preprocess=false
            shift
            ;;
        *)
            # 忽略其他参数
            shift
            ;;
    esac
done

# set path firstly
lightx2v_path="./"
model_path=/data/ai-models/wan2.2/Wan2.2-Animate-14B
video_path=./save_results/talk_v1.mp4
refer_path=/data/downloads/singer4.jpg

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

echo "=== process input resource ==="
# process 1280 780

export CUDA_VISIBLE_DEVICES=0,1

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh

if [ "$preprocess" = true ]; then
    echo "=== preprocess ==="
    # process
    python ${lightx2v_path}/tools/preprocess/preprocess_data.py \
        --ckpt_path ${model_path}/process_checkpoint \
        --video_path $video_path  \
        --refer_path $refer_path \
        --save_path ${lightx2v_path}/save_results/animate_replace/process_results \
        --resolution_area 832 480 \
        --fps 16 \
        --iterations 3 \
        --k 7 \
        --w_len 1 \
        --h_len 1 \
        --replace_flag
fi
echo "=== inter ==="
torchrun --nproc_per_node=2 -m lightx2v.infer \
--model_cls wan2.2_animate \
--task animate \
--model_path $model_path \
--config_json ${lightx2v_path}/examplex/wan2.2/wan_animate.json \
--src_pose_path ${lightx2v_path}/save_results/animate_replace/process_results/src_pose.mp4 \
--src_face_path ${lightx2v_path}/save_results/animate_replace/process_results/src_face.mp4 \
--src_ref_images ${lightx2v_path}/save_results/animate_replace/process_results/src_ref.png \
--src_bg_path ${lightx2v_path}/save_results/animate_replace/process_results/src_bg.mp4 \
--src_mask_path ${lightx2v_path}/save_results/animate_replace/process_results/src_mask.mp4 \
--image_path ${lightx2v_path}/save_results/animate_replace/process_results/src_ref.png \
--prompt "视频中的人在做动作" \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_replace.mp4
