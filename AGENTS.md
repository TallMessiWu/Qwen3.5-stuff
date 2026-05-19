# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Running Environment

**此仓库中的代码无法在本地运行。** 目标硬件为华为昇腾 NPU（Atlas 800I A2/A3、Atlas A2/A3 Training 系列），所有代码在远程服务器上执行，环境为：
- CANN == 8.5.1
- PyTorch == 2.9.0 + torch-npu == 2.9.0
- Python >= 3.10, < 3.12

用户贴出的报错信息和性能数据均来自远程服务器。不要尝试在本地运行或编译代码。

## 代码仓结构

本目录 (`Qwen3.5/`) 包含 `vllm-ascend/` 子仓，是 vLLM 的昇腾 NPU 硬件插件，用于在昇腾 NPU 上跑 Qwen3.5 等大模型。

主要工作代码在 `vllm-ascend/` 下：

```
vllm-ascend/
├── vllm_ascend/
│   ├── patch/            # 上游 vLLM 补丁（核心集成机制）
│   │   ├── platform/     # Worker 启动前应用的平台级补丁
│   │   └── worker/       # Worker 初始化时应用的模型级补丁
│   ├── worker/           # NPU Model Runner（v1 和 v2）
│   │   └── v2/
│   ├── ops/              # NPU 自定义算子（layernorm、MLA、rotary embedding 等）
│   ├── quantization/     # 量化方法（W8A8、W4A8、W4A4、W8A16、KV C8）
│   ├── distributed/      # HCCL 通信、KV transfer（P2P/offload/LMCache/Mooncake）
│   ├── compilation/      # ACL Graph 融合 pass（allreduce+rmsnorm、norm+quant 等）
│   ├── eplb/             # MoE 专家并行负载均衡
│   ├── model_loader/     # NetLoader（分布式权重加载）和 rfork loader
│   ├── _310p/            # 310P 设备专用代码
│   ├── envs.py           # 所有 VLLM_ASCEND_* 环境变量必须在此定义
│   ├── platform.py       # NPUPlatform：主平台适配器
│   └── __init__.py       # 插件注册入口
└── tests/
    ├── ut/               # 单元测试（无需 NPU 硬件，需 COMPILE_CUSTOM_KERNELS=0）
    └── e2e/              # 端到端系统测试（需要 NPU 硬件）
        ├── singlecard/
        ├── multicard/2-cards/
        ├── multicard/4-cards/
        └── nightly/      # 夜间大模型基准测试
```

## Commands

```bash
# 安装开发依赖
pip install -e .[dev]

# Lint 检查
ruff check vllm_ascend/

# 格式化
ruff format vllm_ascend/

# 完整格式检查（推送前必须通过，含 markdown lint）
bash format.sh ci

# 运行单个单元测试文件
pytest -sv tests/ut/ops/test_prepare_finalize.py

# 运行单个测试函数
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# 运行 e2e 系统测试（需要服务器上的 NPU 硬件）
pytest -sv tests/e2e/singlecard/test_piecewise_res_consistency.py
```

## 架构关键点

### 插件入口

- `vllm_ascend/__init__.py` — 将 `NPUPlatform` 注册为 vLLM 平台，同时注册 KV transfer connector 和 model loader。
- `vllm_ascend/platform.py` — `NPUPlatform` 类，处理平台级补丁和配置。

### 补丁系统（最重要的集成机制）

vllm-ascend 通过 monkey-patch 方式适配上游 vLLM：

- **`patch/platform/`** — 在 Worker 启动前应用，由 `NPUPlatform.pre_register_and_update()` 调用。涵盖分布式 op、调度、config 验证等。
- **`patch/worker/`** — 在每个 Worker 初始化时应用。涵盖模型前向、attention、triton op 替换、权重加载等。

**`vllm_ascend/patch/__init__.py`** 记录了所有补丁的 Why/How/Future Plan，排查异常行为或新增补丁时必须先读这个文件。

### Model Runner

- `vllm_ascend/worker/model_runner_v1.py` — vLLM v1 model runner（`NPUModelRunner` 继承自 `GPUModelRunner`）
- `vllm_ascend/worker/v2/model_runner.py` — vLLM v2 model runner

### 新增功能的正确模式

1. **优先继承**：`NPUModelRunner(GPUModelRunner)`、`AscendSampler` 等
2. **必要时 patch**：参考 `patch/` 下现有模式，patch 必须经过架构评审并文档化
3. **不直接添加模型文件**：模型特定行为通过 patch 或 inheritance 实现

## NPU 特有规则

1. **`tensor.item()` 热路径**：触发 NPU→CPU 同步传输，严重影响吞吐。热路径中避免使用，优先用 `torch.max`、`torch.argmax` 等设备端操作。
2. **避免热路径中的 CPU-NPU 数据传输**：批量操作以减少同步频率。
3. **环境变量**：必须在 `vllm_ascend/envs.py` 的 `env_variables` dict 中定义；命名遵循 `VLLM_ASCEND_*`；使用时 `from vllm_ascend import envs`。
4. **新 patch**：需要架构评审；在 `vllm_ascend/patch/__init__.py` 中补充文档（Why/How/Future Plan）。
5. **model_runner 新行为**：同样需要架构评审，说明为何不能放在 patch 中。

## 代码风格

- 行宽 120 字符（ruff 强制）
- 所有 import 在文件顶部；type-only import 用 `TYPE_CHECKING` 守卫；循环依赖或 worker 隔离场景才用内联 import
- 不新增可变全局状态；常量用 `ALL_UPPER_CASE`
- 命名：类 `PascalCase`，函数/变量 `snake_case`，常量 `ALL_UPPER_CASE`

## Commit 与 PR 格式

提交需要 sign-off 并遵循 Conventional Commits：

```
<type>: <summary>

<body - 说明改了什么、为什么>

Signed-off-by: Name <email>
```

有效类型：`feat`、`fix`、`perf`、`refactor`、`test`、`docs`、`chore`

PR 标题格式：`[Type][Module] Description`，例如 `[Bugfix] Fix CPU binding logic`
