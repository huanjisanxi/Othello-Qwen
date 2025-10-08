# 使用包含 CUDA 12.8 和 Ubuntu 20.04 的官方基础镜像
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖项
# 修改后的命令（禁用交互）
# 正确的格式：反斜杠后无空格，包名作为 apt-get install 的参数延续
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.12 为默认版本
RUN ln -s /usr/bin/python3.12 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# 安装 Python 包管理工具
RUN python -m ensurepip --upgrade

# 升级 pip 到最新版本
RUN pip install --no-cache-dir --upgrade pip

# # 复制项目文件并安装依赖
# COPY . /workspace/
# RUN pip install --no-cache-dir -r requirements.txt

# # 默认执行命令
# CMD ["python3", "scripts/train_stf.py"]
