# OC-RAP v50.2：External Baseline cuDNN SDPA Hotfix

## 1. 直接原因

失败并非数据集、候选 mask、loss 或 batch size 导致。Wayformer 与 GameFormer 在不同模型入口同时进入
`torch.nn.functional.scaled_dot_product_attention`，随后由 PyTorch 自动选择 cuDNN SDPA；当前服务器的
PyTorch/CUDA/cuDNN/GPU/attention-shape 组合无法为该图构造 execution plan，因而抛出：

```text
RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error: No execution plans support the graph.
```

原代码没有显式设置 SDPA backend，且 AMP 默认硬编码为 BF16。因此：

- 新版 PyTorch 可以自动选择不兼容的 cuDNN attention backend；
- 不支持原生 BF16 的 GPU 也没有自动降级；
- Safe 的 Wayformer/GameFormer 并行进程会在首个 forward 同时失败。

## 2. 修复内容

### 2.1 安全 SDPA backend

新增 `src/ocrap/external_baselines/runtime.py`。默认 `safe` 模式：

- cuDNN SDPA：关闭；
- FlashAttention：开启；
- memory-efficient SDPA：开启；
- math SDPA：开启，作为最终 fallback。

这里只关闭发生错误的 cuDNN attention backend，不关闭整个 cuDNN，也不禁用 LSTM 等正常 cuDNN 算子。

### 2.2 AMP 自动选择

`OCRAP_AMP_DTYPE=auto`：

- GPU 支持 BF16：使用 BF16；
- GPU 不支持 BF16：自动使用 FP16，并启用 GradScaler；
- 不再把所有 CUDA GPU 都假定为 BF16-capable。

### 2.3 训练和推理一致

训练入口与 checkpoint 加载/闭环推理入口都会调用同一 CUDA runtime 配置，防止训练通过后离线评测或闭环再次选择 cuDNN SDPA。

### 2.4 配置与脚本

以下配置默认加入：

```yaml
training:
  sdpa_backend: safe
  amp_dtype: auto
```

总控、Safe 和 Near-contact optimized 脚本会显式传播：

```bash
OCRAP_SDPA_BACKEND=safe
OCRAP_AMP_DTYPE=auto
```

### 2.5 诊断工具

新增：

```bash
python tools/check_external_baseline_cuda_runtime.py
```

它会运行与 baseline 相同规模的 pre-norm Transformer forward + backward，并输出 PyTorch、CUDA、cuDNN、GPU、BF16 支持与有效 SDPA backend。

## 3. 是否需要删除或重训 checkpoint

本次日志显示错误发生在第一批数据的 forward，尚未完成 optimizer step，也没有生成有效 `best.pt`。不需要删除整个 `$OUT`。

重新运行时：

- 有效 checkpoint 会被复用；
- 缺失、零字节或不满足 deployable contract 的 checkpoint 会自动重训；
- 本次失败的 Safe Wayformer/GameFormer 会重新开始训练；
- BeTop 是否训练取决于它是否已有有效 checkpoint。

## 4. 推荐命令

```bash
cd /home/senzeyu2/code/OC-RAP
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OCRAP_SDPA_BACKEND=safe
export OCRAP_AMP_DTYPE=auto

CUDA_VISIBLE_DEVICES=0 python tools/check_external_baseline_cuda_runtime.py \
  --sdpa-backend safe --amp-dtype auto \
  --output "$OUT/cuda_runtime_gpu0.json"

CUDA_VISIBLE_DEVICES=1 python tools/check_external_baseline_cuda_runtime.py \
  --sdpa-backend safe --amp-dtype auto \
  --output "$OUT/cuda_runtime_gpu1.json"

bash scripts/run_all_regime_external_baselines_optimized.sh
```

运行日志开头应出现类似：

```text
{'event': 'external_baseline_cuda_runtime',
 'sdpa_backend': 'safe',
 'cudnn_sdp': False,
 'flash_sdp': True,
 'mem_efficient_sdp': True,
 'math_sdp': True,
 'amp_dtype': 'bfloat16'}
```

若服务器的 Flash/memory-efficient backend 仍有兼容问题，使用最保守 fallback：

```bash
export OCRAP_SDPA_BACKEND=math
export OCRAP_AMP_DTYPE=fp16
bash scripts/run_all_regime_external_baselines_optimized.sh
```

`math` 会更慢，只应作为兼容性后备。

## 5. 验证结果

- Python compileall：通过；
- Bash syntax：通过；
- YAML parse：通过；
- Wayformer/GameFormer/BeTop dummy forward：通过；
- 相关回归测试：24 passed；
- runtime 诊断工具 CPU smoke test：通过。

当前环境没有用户服务器上的 CUDA GPU，因此无法在这里执行目标 GPU 的真实 cuDNN/Flash kernel；新增的 GPU preflight 工具用于在正式训练前验证该服务器。
