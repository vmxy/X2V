
## 命令demo
```bash

export CUDA_VISIBLE_DEVICES=0,1 

## 使用nvfp4量化的模型
torchrun --nproc_per_node=2  examplex/wan2.1/wan_i2v_nvfp4.py

```
