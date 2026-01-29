#!/bin/bash

## --python=3.11 torch=2.8 有flash-attn=2.8
## --python=3.12 torch=2.9 有flash-attn=2.8
uv venv --python 3.11 .venv
TORCH_VERSION=2.8.0

sed -i 's/VIRTUAL_ENV_PROMPT=.*/VIRTUAL_ENV_PROMPT="X2V"/g' .venv/bin/activate
source .venv/bin/activate

uv sync

uv pip install flash-attn --no-build-isolation "torch~=$TORCH_VERSION"

mkdir .deps

## 安装 sageattention
git clone https://github.com/thu-ml/SageAttention.git .deps/SageAttention
### 安装 sageattention2
cd .deps/SageAttention
CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
cd ../../
### 安装 sageattention3
#cd sageattention3_blackwell
#CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
#cd ../../../


## 2. Install NVFP4 Kernel
uv pip install scikit_build_core
cd lightx2v_kernel
git clone https://github.com/NVIDIA/cutlass.git
### 依赖 cutlass
MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    uv build --wheel \
        -Cbuild-dir=build . \
        -Ccmake.define.CUTLASS_PATH=./cutlass \
        --verbose --color=always --no-build-isolation
uv pip install dist/*whl --force-reinstall --no-deps





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
