
## 命令demo
```bash

export CUDA_VISIBLE_DEVICES=0,1 

## 使用nvfp4量化的模型
torchrun --nproc_per_node=2  examplex/wan2.2/wan_moe_i2v_distill.py

```
