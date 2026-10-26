<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

面向大语言模型（LLM）的**轻量级且高性能**推理框架。

[English](./README.md) | **简体中文**

---

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现版本，旨在揭开现代大语言模型在线推理服务系统底层复杂性的神秘面纱。凭借仅 **约 5,000 行 Python 代码**的精炼体量，它既能作为功能完备的高性能推理引擎直接运行，又是供研究人员和开发者深入理解系统架构的透明参考实现。

## ✨ 核心特性 (Key Features)

- **卓越的性能表现**：借助多项前沿系统级优化，实现业界领先（SOTA）的吞吐量与低延迟表现。
- **轻量且极易阅读**：采用清晰、模块化且全类型注解（Type-Annotated）的代码库设计，极易阅读、理解与二次修改。
- **前沿系统级优化**：
  - **基数树缓存 (Radix Cache)**：跨请求自动复用共享前缀（Shared Prefixes）的 KV Cache。
  - **分块预填充 (Chunked Prefill)**：显著降低长上下文推理服务时的显存峰值占用。
  - **重叠调度 (Overlap Scheduling)**：将 CPU 调度开销隐藏在 GPU 计算过程背后，实现零开销调度。
  - **张量并行 (Tensor Parallelism)**：支持跨多张 GPU 水平扩展大模型推理。
  - **深度优化的计算算子**：深度集成 **FlashAttention** 与 **FlashInfer**，最大化硬件计算效率。
  - ……

## 🚀 快速开始 (Quick Start)

> **⚠️ 平台支持说明**：Mini-SGLang 目前**仅支持 Linux 系统**（`x86_64` 与 `aarch64`）。由于依赖 Linux 专用的底层 CUDA 算子包（`sgl-kernel`、`flashinfer`），暂不支持原生 Windows 与 macOS。建议 Windows 用户使用 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 或通过 Docker 容器实现跨平台运行。

### 1. 环境准备 (Environment Setup)

我们推荐使用 `uv` 进行快速且可靠的依赖安装（注：`uv` 与 `conda` 环境互不冲突）。

```bash
# 创建虚拟环境（推荐使用 Python 3.10+）
uv venv --python=3.12
source .venv/bin/activate
```

**前置条件**：Mini-SGLang 依赖即时编译（JIT）的 CUDA 算子。请确保您的系统已安装 **NVIDIA CUDA Toolkit**，且其版本与显卡驱动版本匹配。您可以通过运行 `nvidia-smi` 检查驱动所支持的 CUDA 版本。

### 2. 安装指南 (Installation)

直接从源码仓库克隆并安装 Mini-SGLang：

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 在 Windows (WSL2) 上安装</b></summary>

由于 Mini-SGLang 依赖 Linux 专属组件，Windows 用户请使用 WSL2 环境：

1. **安装 WSL2**（若尚未安装）：
   ```powershell
   # 在以管理员身份运行的 PowerShell 中执行
   wsl --install
   ```

2. **在 WSL2 中安装 CUDA**：
   - 请遵循 [NVIDIA 官方 WSL2 CUDA 安装指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - 确保您的 Windows 宿主机 GPU 驱动支持 WSL2

3. **在 WSL2 中安装 Mini-SGLang**：
   ```bash
   # 在 WSL2 终端内执行
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **从 Windows 宿主机访问**：服务启动后，可直接从 Windows 浏览器或应用程序访问 `http://localhost:8000`。

</details>

<details>
<summary><b>🐳 使用 Docker 运行</b></summary>

**前置条件**：
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **构建 Docker 镜像**：
   ```bash
   docker build -t minisgl .
   ```

2. **启动推理服务**：
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **以交互式命令行模式运行**：
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **挂载 Docker 数据卷以持久化缓存**（推荐，可大幅加快后续启动速度）：
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. 在线推理服务 (Online Serving)

只需一行命令即可启动兼容 OpenAI 协议的 API 服务端：

```bash
# 在单张 GPU 上部署 Qwen/Qwen3-0.6B
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 使用张量并行（TP=4）在 4 张 GPU 上部署 meta-llama/Llama-3.1-70B-Instruct，并监听 30000 端口
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

服务启动后，您即可使用 `curl` 等标准工具或任意兼容 OpenAI SDK 的客户端发送请求。

### 4. 交互式终端对话 (Interactive Shell)

添加 `--shell` 标志即可直接在终端中与模型进行交互式对话：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

在对话过程中，您还可以输入 `/reset` 清空当前对话历史。

## 📊 性能基准评测 (Benchmark)

### 离线推理 (Offline inference)

更多评测细节请参阅 [bench.py](./benchmark/offline/bench.py)。如需针对重叠调度（Overlap Scheduling）进行消融实验，可设置环境变量 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`。

测试配置：

- 硬件环境：1x H200 GPU
- 测试模型：Qwen3-0.6B、Qwen3-14B
- 请求总数：256 条序列
- 输入长度：在 100–1024 tokens 之间随机采样
- 输出长度：在 100–1024 tokens 之间随机采样

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理 (Online inference)

更多评测细节请参阅 [benchmark_qwen.py](./benchmark/online/bench_qwen.py)。

测试配置：

- 硬件环境：4x H200 GPU（通过 NVLink 互联）
- 测试模型：Qwen3-32B
- 评测数据集：[Qwen Trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl)，回放前 1,000 条真实请求

服务启动命令：

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **提示**：如果您在从 Hugging Face 下载模型时遇到网络连接问题，可以尝试添加 `--model-source modelscope` 参数改从魔搭社区（ModelScope）下载模型权重：
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 深入了解 (Learn More)

- **[功能特性详解](./docs/features.md)**：探索全部支持的功能特性与命令行参数说明。
- **[系统架构设计](./docs/structures.md)**：深入剖析 Mini-SGLang 的模块设计与内部数据流转机制。

---

> _注：本简体中文文档由社区志愿者翻译维护。若与英文原版存在差异，请以 [README.md](README.md) 为准。_
