# OC-RAP v48.22 COVENANT-BRIDGE 代码变更清单

## 核心算法

- 新增第三个 safe-admission hypothesis；raw benefit、component harm、final admission 不再共享语义。
- Admission prior 使用 detached benefit/harm，加零初始化有界 residual，避免稀疏 admission 梯度破坏两条基础 evidence。
- 修复 v48.21 opportunity-only group MIL；完整模型使用 admission noisy-OR，两头消融使用 `P(benefit)*(1-P(harm))`。
- 训练、early stopping、calibration、evaluation、selector 与 closed-loop 统一使用显式 admission score。
- 新增 epoch-zero checkpoint evaluation 和 COVENANT threshold-free selection risk。
- Sampler 使用 safe-positive group，但 raw-benefit head 保持 raw benefit target。
- 增加 safety-frontier diagnostics。

## 控制器与实验

- 新增 v48.22 主实验、certificate、learning-gate、stress 授权脚本。
- 新增四组非重复消融；8 个任务一次同时启动，GPU0/GPU1 各 4 个。
- 工程异常统一归一化为 RC=30；禁止单分支带病进入 certificate。
- 新增 admission branch 梯度、bucket invariance、容量、MIL、安全采样、epoch-zero 和 GPU 分配测试。

## 文件级差异

- `Files /mnt/data/ocrap_v4821_work/code/ALGORITHM_CHANGELOG.md and /mnt/data/ocrap_v4822_work/code/ALGORITHM_CHANGELOG.md differ`
- `Files /mnt/data/ocrap_v4821_work/code/ALGORITHM_CHANGELOG_V48.md and /mnt/data/ocrap_v4822_work/code/ALGORITHM_CHANGELOG_V48.md differ`
- `Only in /mnt/data/ocrap_v4822_work/code: OC-RAP-v48.21-results-audit-and-v48.22-COVENANT-BRIDGE-plan-ZH.md`
- `Only in /mnt/data/ocrap_v4822_work/code: OC-RAP-v48.21-results-audit-summary.json`
- `Only in /mnt/data/ocrap_v4822_work/code: OC-RAP-v48.22-run-commands-ZH.txt`
- `Files /mnt/data/ocrap_v4821_work/code/configs/default.yaml and /mnt/data/ocrap_v4822_work/code/configs/default.yaml differ`
- `Only in /mnt/data/ocrap_v4822_work/code/scripts: adapt_ocrap_v48_22_covenant_variant.sh`
- `Only in /mnt/data/ocrap_v4822_work/code/scripts: calibrate_v48_22_certificate_pool.sh`
- `Only in /mnt/data/ocrap_v4822_work/code/scripts: calibrate_v48_22_learning_gate.sh`
- `Only in /mnt/data/ocrap_v4822_work/code/scripts: run_v48_22_covenant_dedicated.sh`
- `Only in /mnt/data/ocrap_v4822_work/code/scripts: run_v48_22_parallel_ablations.sh`
- `Only in /mnt/data/ocrap_v4822_work/code/scripts: run_v48_22_stress_if_authorized.sh`
- `Files /mnt/data/ocrap_v4821_work/code/scripts/train_ocrap_v48_trac_sr.sh and /mnt/data/ocrap_v4822_work/code/scripts/train_ocrap_v48_trac_sr.sh differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/cli/train.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/cli/train.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/config/defaults.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/config/defaults.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/evaluation/evaluator.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/evaluation/evaluator.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/models/inference.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/models/inference.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/models/losses.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/models/losses.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/models/ocrap.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/models/ocrap.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/planning/selector.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/planning/selector.py differ`
- `Files /mnt/data/ocrap_v4821_work/code/src/ocrap/simulation/closed_loop_runner.py and /mnt/data/ocrap_v4822_work/code/src/ocrap/simulation/closed_loop_runner.py differ`
- `Only in /mnt/data/ocrap_v4822_work/code/tests: test_v48_22_covenant_bridge.py`
- `Files /mnt/data/ocrap_v4821_work/code/tools/calibrate_policy_risk_v48.py and /mnt/data/ocrap_v4822_work/code/tools/calibrate_policy_risk_v48.py differ`
