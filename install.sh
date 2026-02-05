#!/bin/bash

sudo apt install -y \
  libavdevice-dev \
  libavformat-dev \
  libavcodec-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  pkg-config

## --python=3.11 torch=2.8 有flash-attn=2.8
## --python=3.12 torch=2.9 有flash-attn=2.8
uv venv --python 3.12 .venv
TORCH_VERSION=2.9.1
mkdir -p .deps

sed -i 's/VIRTUAL_ENV_PROMPT=.*/VIRTUAL_ENV_PROMPT="X2V"/g' .venv/bin/activate
source .venv/bin/activate



uv sync
uv pip install torch torchaudio torchvision "torch~=$TORCH_VERSION"

## 安装 flash_attn2
uv pip install flash-attn --no-build-isolation "torch~=$TORCH_VERSION"
#uv pip install flashinfer-python  "torch~=$TORCH_VERSION"

## 安装flash_attn3
#git clone https://github.com/Dao-AILab/flash-attention.git .deps/flash-attention
#cd .deps/flash-attention/hopper
#CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
#cd ../../../

## 安装 sageattention
git clone https://github.com/thu-ml/SageAttention.git .deps/SageAttention
### 安装 sageattention2
cd .deps/SageAttention
rm -rf build
CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install 
#cd ../../
### 安装 sageattention3
cd sageattention3_blackwell
CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
cd ../../../

## 安装 spas_sage_attn 
git clone https://github.com/thu-ml/SpargeAttn.git .deps/SpargeAttn
cd .deps/SpargeAttn
CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install


## 2. Install NVFP4 Kernel
uv pip install scikit_build_core
cd lightx2v_kernel
rm -rf build
git clone https://github.com/NVIDIA/cutlass.git
### 依赖 cutlass
MAX_JOBS=$(nproc) CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
    uv build --wheel \
        -Cbuild-dir=build . \
        -Ccmake.define.CUTLASS_PATH=./cutlass \
        --verbose --color=always --no-build-isolation
uv pip install dist/*whl --force-reinstall --no-deps
cd ../


#git clone https://github.com/Lightricks/LTX-Video-Q8-Kernels.git .deps/LTX-Video-Q8-Kernels
#cd .deps/LTX-Video-Q8-Kernels
#CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install

#git clone https://github.com/vllm-project/vllm.git .deps/vllm
#cd .deps/vllm
#uv pip install -e . "torch~=$TORCH_VERSION"
uv pip install vllm "torch~=$TORCH_VERSION"
uv pip install sgl-kernel "torch~=$TORCH_VERSION"

#uv pip install -U torchcodec "torch~=$TORCH_VERSION" --no-build-isolation --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128 
git clone https://github.com/meta-pytorch/torchcodec.git .deps/torchcodec
cd .deps/torchcodec
uv pip install pybind11 "torch~=$TORCH_VERSION"
export pybind11_DIR=$(python -m pybind11 --cmakedir)
CFLAGS="-O2" CXXFLAGS="-O2" python setup.py install
cd ../../



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

try:
  import flash_attn
  #from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
  print("Flash Attention 2 version:", flash_attn.__version__)
except ImportError:
   print("no install flash_attn2")

try:
  import flash_attn_interface
  #from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
  print("Flash Attention 3 version:", flash_attn_interface)
except ImportError:
   print("no install flash_attn3")

try:
  from sageattention import sageattn
  print("sageattn2 installed")
except ImportError:
  print("no install sageattn2 ")
    
try:
  from sageattn3 import sageattn3_blackwell
  print("sageattn3_blackwell installed")
except ImportError:
  print("no install sageattn3_blackwell ")

try:
  from q8_kernels.functional.linear import q8_linear
  print(f"install q8_kernels")
except ImportError:
  print(f"no install q8_kernels")

EOF

