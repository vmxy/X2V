# LightX2V 安装指南

## 使用 uv 环境安装
```bash
## --python=3.11 torch=2.8 有flash-attn=2.8
## --python=3.12 torch=2.9 有flash-attn=2.8
uv venv --python 3.12 .venv
TORCH_VERSION=2.9.1
uv sync
uv pip install flash-attn --no-build-isolation "torch~=$TORCH_VERSION"
uv pip install sageattention "torch~=$TORCH_VERSION"
source .venv/bin/activate

# 2. Install NVFP4 Kernel
uv pip install scikit_build_core
cd lightx2v_kernel
git clone https://github.com/NVIDIA/cutlass.git
# 依赖 cutlass
MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
uv build --wheel \
  -Cbuild-dir=build . \
  -Ccmake.define.CUTLASS_PATH=./cutlass \
  --verbose --color=always --no-build-isolation
uv pip install dist/*whl --force-reinstall --no-deps

# 安装 sageattention
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention/sageattention3_blackwell
#export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=12 
CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
cd ../../

# 3. Run inference
python examples/wan/wan_i2v_nvfp4.py   # Image-to-Video
python examples/wan/wan_t2v_nvfp4.py   # Text-to-Video
```


## 验证安装
```python

python << EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

import torchada
print("TorchADA version:", torchada.__version__)

import torchaudio
print("Torchaudio version:", torchaudio.__version__)

import torchvision
print("Torchvision version:", torchvision.__version__)

import flash_attn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
print("Flash Attention version:", flash_attn.__version__)
print("Flash Attention flash_attn_func:", flash_attn_func)


try:
  from sageattention import sageattn
  print("SageAttention version: sageattn2")
except ImportError:
  try:
    from sageattn3 import sageattn3_blackwell
    print("SageAttention sageattn3_blackwell:", sageattn3_blackwell)
  except ImportError:
     print("no install SageAttention ")

EOF

```