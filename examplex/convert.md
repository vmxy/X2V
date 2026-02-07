# 模型量化



## 量化为fp8
```bash
#  sekotalk 量化为fp8
python tools/convert/quant_adapter.py \
    --model_path /data/ai-models/SekoTek-V2.5/audio_adapter_model.safetensors \
    --output_path audio_adapter_model_fp8.safetensors
 

```


## 量化wan2.1
```bash

## 把模型Wan2.1-I2V-14B-480P量化为fp8
python tools/convert/converter.py \
    --device cpu \
    --source /data/ai-models/wan2.1/Wan2.1-I2V-14B-480P \
    --output checkpoints/wan2.1 \
    --output_ext .safetensors \
    --output_name wan2.1_i2v_14b_480p_scaled-fp8 \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --copy_no_weight_files \
    --single_file  


```


## 量化wan2.2
```bash
## 量化 Wan2.2-Animate-14B 为fp8
python tools/convert/converter.py \
    --device cpu \
    --source /data/ai-models/wan2.2/Wan2.2-Animate-14B \
    --output /data/ai-models/lightx2v/wan2.2/Wan2.2-Animate-14B \
    --output_ext .safetensors \
    --output_name wan2.2_animate_14b_scaled-fp8-test \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_animate_dit \
    --quantized \
    --single_file  
    
#     --copy_no_weight_files \

python tools/convert/converter.py \
    --device cpu \
    --source /data/ai-models/wan2.2/Wan2.2-Animate-14B \
    --output /data/ai-models/lightx2v/wan2.2 \
    --output_ext .safetensors \
    --output_name wan2.2_animate_14b_scaled-fp8 \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_animate_dit \
    --quantized \
    --single_file  

## 量化 Wan2.2-S2V-14B 为fp8
python tools/convert/converter.py \
    --device cpu \
    --source /data/ai-models/wan2.2/Wan2.2-S2V-14B \
    --output /data/ai-models/lightx2v/wan2.2/Wan2.2-S2V-14B \
    --output_ext .safetensors \
    --output_name wan2.2_s2v_14b_scaled-fp8 \
    --linear_type fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_s2v_dit \
    --quantized \
    --single_file  

```