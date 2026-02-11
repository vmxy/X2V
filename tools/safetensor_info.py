#!/usr/bin/env python3
"""
Safetensors Key 查看工具 - 简化版
用法: python safetensors_viewer.py -i <文件路径>
"""

from safetensors import safe_open
import argparse
import sys

def view_safetensors_keys(file_path):
    """
    查看 safetensors 文件中的所有 key
    """
    try:
        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
            print(f"文件: {file_path}")
            print(f"总张量数: {len(keys)}")
            print("=" * 60)
            
            for i, key in enumerate(keys, 1):
                tensor = f.get_tensor(key)
                # 获取张量信息
                shape = f.get_slice(key).get_shape()
                dtype = f.get_slice(key).get_dtype()
                # 计算参数量
                params = 1
                for dim in shape:
                    params *= dim
                # 打印信息
                print(f"{i:4d}. {key} 形状: {shape} 类型: {dtype}  参数量: {params:,} {tensor if params == 1 else ''}")
                
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='查看 Safetensors 文件中的 key')
    parser.add_argument('-i', '--input', required=True, help='safetensors 文件路径')
    
    args = parser.parse_args()
    
    # 调用查看函数
    view_safetensors_keys(args.input)

if __name__ == "__main__":
    main()