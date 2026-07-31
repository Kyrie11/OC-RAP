# OC-RAP v48.24 SUPPORT-BRIDGE 变更清单

## 算法层

- `src/ocrap/models/losses.py`
  - 新增 final admission 的 continuous safe-utility regression。
  - 新增 nominal+top-k safe-utility listwise/KL。
  - harmful action 使用严格负 safe-utility target。
- `src/ocrap/cli/train.py`
  - 接入三项 safe-utility 配置。
- `scripts/train_ocrap_v48_trac_sr.sh`
  - 接入命令行配置；legacy group opportunity 默认权重 0。
- `scripts/adapt_ocrap_v48_24_support_variant.sh`
  - top-8、safe-benefit label、safe-positive group sampling、direct safe utility、轻量 frontier。

## 证书与诊断

- `tools/calibrate_policy_risk_v48.py`
  - 新增 k=1/3/5/8/active proposal support curve。
  - 新增 fit-only diagnostic selector；不得用于 deployment/test/stress。
  - 证书同步输出完整 selector contract。
- `scripts/calibrate_v48_24_certificate_pool.sh`
  - safe-benefit opportunity、top-8 Evidence rerank、结构支持失败明确归一为 RC=30。

## 工程修复

- `scripts/run_ocrap_v48_trac_sr.sh`
  - runtime 加载 score/opp/harm/rank-margin/top-k/rerank/conditional-ranking 全合同。
  - `DEV_SHADOW_DIAGNOSTIC=1` 允许 RC=20 的 adaptation-dev-only 诊断；正式部署仍严格拒绝无效证书。
- `scripts/run_v48_23_dev_shadow_closed_loop.sh`
  - 修复上一版 shadow 无法启动的问题。
- `scripts/run_v48_24_dev_shadow_closed_loop.sh`
  - 双卡并发 variant、单卡内 Near→Contact 顺序执行。
- `scripts/run_v48_24_support_dedicated.sh`
  - v48.24 主控制器。
- `scripts/run_v48_24_parallel_ablations.sh`
  - 四 wave、每张卡一次一个任务、每卡四任务。
- `scripts/run_v48_24_stress_if_authorized.sh`
  - 仅 NEXT_COMMANDS 授权后读取 held-out stress/test。

## 测试

- `tests/test_v48_24_support_bridge.py`
  - harmful/safe target 梯度方向。
  - support curve 与 diagnostic selector。
  - runtime 完整 selector contract。
  - safe label、Noisy-OR 关闭及双卡 wave 调度。

## 未修改

- 不重建 train/val/test/calibration 数据集。
- 不读取 held-out test/stress 进行模型选择。
- 不加入 regime ID/router、bucket-specific residual 或事后放宽 v48.23 gate。
