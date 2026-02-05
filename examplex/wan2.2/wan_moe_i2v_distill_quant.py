"""
an2.1 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for I2V generation.
"""

import random
import time
from lightx2v import LightX2VPipeline
model_path="/data/ai-models/lightx2v/wan2.2"
pipe = LightX2VPipeline(
    model_cls="wan2.2_moe_distill",
    model_path=f"{model_path}/Wan2.2-Distill-Models/",
    task="i2v"
)
#pipe.enable_compile
""" 
pipe.enable_offload(
    offload_granularity="block",
    cpu_offload=True,
    vae_offload=True,
    text_encoder_offload=True,
    #image_encoder_offload=True
)

pipe.enable_quantize(
    dit_quantized=True,
    quant_scheme="fp8-torchao",                                                             
    high_noise_quantized_ckpt=f"{model_path}/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_1030.safetensors",
    low_noise_quantized_ckpt=f"{model_path}/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
)
pipe.enable_parallel(
    seq_p_size=1,
    seq_p_attn_type="ulysses",
    cfg_p_size=2,
) 
 """
start_time = time.time()
pipe.create_generator(
    config_json=f"examplex/wan2.2/wan_moe_i2v_distill_quant.json",
    #attn_mode="sage_attn3",
    #infer_steps=4,
    #num_frames=81,
    #height=480,
    #width=832,
    #guidance_scale=5,
    #sample_shift=5.0,
    #fps=16,
    #aspect_ratio="16:9",
    #boundary=0.9,
    #boundary_step_index=2,
    #denoising_step_list=[1000, 750, 500, 250],
   
    #rope_type="torch",
    #resize_mode=None,
    #audio_fps=24000,
    #double_precision_rope=True,
    #norm_modulate_backend="torch",
    #distilled_sigma_values=None,
)
print(f"======================= create pipe cost={time.time() - start_time:.0f}s =======================")
seed = 49
prompt="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path=f"save_results/wan_moe_i2v_distill-cat-{seed}.mp4"
image_path="assets/inputs/imgs/img_0.jpg"
start_time = time.time()
pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    image_path=image_path,
    save_result_path=save_result_path,
    image_strength=None,
    last_frame_path=None,
    audio_path=None,
    src_ref_images=None,
    src_video=None,
    src_mask=None,
    return_result_tensor=False,
    target_shape=None,
)
print(f"======================= pipe generate video cost={time.time() - start_time:.0f}s =======================")