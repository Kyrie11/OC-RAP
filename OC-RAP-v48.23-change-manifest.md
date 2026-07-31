# OC-RAP v48.23 FRONTIER-BRIDGE 代码变更清单

## 核心模型与训练

- `src/ocrap/models/ocrap.py`
  - 新增 FRONTIER 模式与低风险 component prior；
  - component harm 改为先验加有界 residual；
  - admission prior 改为以 harm prior 为中心的 identity-preserving 形式；
  - 保留 v48.18–v48.22 legacy checkpoint 路径。

- `src/ocrap/models/losses.py`
  - 新增 frozen-top-k continuous PCD listwise/KL loss；
  - 新增 safe-beneficial 对 beneficial-but-harmful 的 frontier pairwise loss；
  - 新增 nominal + top-k categorical one-action group objective；
  - legacy noisy-OR 路径仍可用于历史模型兼容。

- `src/ocrap/cli/train.py`
  - 新增 FRONTIER loss/config plumbing；
  - 新增 high-opportunity harmful mass 和 frontier-aware checkpoint risk；
  - checkpoint selection 不再仅依赖 broad harm 或固定阈值。

- `configs/default.yaml`
  - 新增 FRONTIER、component prior、listwise、frontier contrast、categorical group policy 和 checkpoint 权重配置。

## 证书、诊断与闭环指标

- `tools/calibrate_policy_risk_v48.py`
  - 新增 v48.23 method version；
  - 新增 proposal-constrained oracle fit/verify gate audit；
  - 枚举可行选择量，区分 proposal/gate contract 与 learned selector failure；
  - oracle audit 明确标记忽略 macro constraint，仅作为必要条件。

- `src/ocrap/simulation/closed_loop_runner.py`
  - Near 新增最小间距/TTC危险暴露时长与亏损积分；
  - Contact 新增接触持续时间、最长接触 run、撞后净空积分、持续脱离和脱离时间；
  - 修复 `secondary_overlap_event` 跨场景取 max 的错误，改为场景率；
  - 新增二次接触率、稳定停车率、持续脱离率；
  - 删除重复场景分位数计算；
  - 保留轻量 inference sample view 和共享场景特征以减少闭环复制开销。

- `tools/compare_paired_closed_loop.py`
  - 加入新 Near/Contact 物理指标；
  - 为 lower-is-better 指标输出方向正确的 improvement fraction；
  - 保留 raw paired 差值和 bootstrap CI。

- `tools/check_v48_23_regime_targets.py`
  - 新增 Near/Contact 开发与投稿目标检查；
  - publication 模式要求 paired directional confidence interval。

## 新增主实验与消融脚本

- `scripts/adapt_ocrap_v48_23_frontier_variant.sh`
  - FRONTIER 统一模型训练入口；
  - 默认 semantic prior、categorical objective、continuous ranking 和 frontier loss；
  - 主模型无 regime ID 或 regime-specific residual。

- `scripts/calibrate_v48_23_certificate_pool.sh`
  - 预注册并保持原 primary Natural gate；
  - 并行校准 Balanced/Precision；
  - 生成 proposal oracle audit；
  - 严格校验非空、scene-disjoint、support feasibility 和工件完整性。

- `scripts/calibrate_v48_23_learning_gate.sh`
  - v48.23 learning-gate controller 包装。

- `scripts/run_v48_23_frontier_dedicated.sh`
  - 主实验 Balanced→GPU0、Precision→GPU1 并发；
  - 默认 batch size 96、3 workers、prefetch 3、28 epochs；
  - 双分支必须完成；0/20/30 返回码严格归一化；
  - `test_roots_read=false` seal。

- `scripts/run_v48_23_parallel_ablations.sh`
  - 新增四组非重复消融：A/B/C/D；
  - 8 个任务一次性启动；GPU0/GPU1 各 4 个；
  - 默认每任务 batch 56、1 worker、host 线程限制 2；
  - 输出 `TASK_GPU_ASSIGNMENT.txt` 与完整任务状态。

- `scripts/run_v48_23_dev_shadow_closed_loop.sh`
  - 仅 adaptation-dev 的非论文 shadow closed loop；
  - 不读取 certificate/test/stress；
  - Balanced/Precision 分别占用两张卡；每卡 Near/Contact 顺序执行；
  - 自动生成 paired physical diagnostics。

- `scripts/run_v48_23_stress_if_authorized.sh`
  - 只有独立 gate 自动生成 `NEXT_COMMANDS.txt` 后才能运行 held-out stress。

- `scripts/train_ocrap_v48_trac_sr.sh`
  - 新配置参数传递；
  - 保留 AMP、persistent workers、pin memory 和 prefetch；
  - checkpoint metric 支持 FRONTIER。

- `scripts/run_ocrap_v48_trac_sr.sh`
  - generic closed-loop bucket split 改为可配置 `BUCKET_SPLIT`，支持 adaptation-dev shadow；
  - held-out 默认仍为 test。

## 测试与日志

- `tests/test_v48_23_frontier_bridge.py`
  - semantic risk prior；
  - admission identity；
  - continuous listwise gradient；
  - frontier contrast gradient；
  - categorical objective；
  - frontier checkpoint metric；
  - oracle/dev-shadow/8-task plumbing；
  - Contact event scene-rate aggregation。

- `ALGORITHM_CHANGELOG.md`
  - 新增 v48.22 结果归因、gate/model 判断、工程缺陷、v48.23 算法、消融、指标、决策与不重复规则。

## 随代码附带的分析文件

- `OC-RAP-v48.22-results-audit-and-v48.23-FRONTIER-BRIDGE-plan-ZH.md`
- `OC-RAP-v48.22-results-audit-summary.json`
- `OC-RAP-v48.23-run-commands-ZH.txt`

## 验证结果

- pytest：216 passed，5 warnings；
- compileall：通过；
- 所有 shell `bash -n`：通过；
- 缺失 protocol 故障注入：正确归一化为 RC=30；
- ZIP 完整性与 SHA256：见交付校验文件。
