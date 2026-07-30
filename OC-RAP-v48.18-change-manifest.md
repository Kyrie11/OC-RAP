# OC-RAP v48.18 DUET-BRIDGE 变更清单

## 工程修复

| 文件 | 变更 |
|---|---|
| `scripts/adapt_ocrap_v48_17_bridge_variant.sh` | 将错误的固定 20,000 参数上限改为可配置 guard；记录 context source。 |
| `scripts/run_v48_17_bridge_dedicated.sh` | adaptation 双失败时也写 `learning_gates_v48_17.json` 和 `V48_17_COMPLETE.json`，明确 RC=30。 |
| `scripts/recover_v48_17_after_param_guard.sh` | 复用已训练 v48.17 checkpoint，补写 marker 并执行独立 certificate，无需重训。 |
| `tools/check_v48_16_learning_gates.py` | 支持版本字段；报告 pipeline/adaptation failure，避免用文件缺失猜测 gate。 |
| `scripts/run_v48_18_stress_if_authorized.sh` | stress 继续强制要求 `NEXT_COMMANDS.txt` 和有效 gate 授权。 |

## v48.18 算法实现

| 文件 | 变更 |
|---|---|
| `src/ocrap/models/ocrap.py` | Recovery Set Tournament 暴露冻结 contextual embedding；新增 tournament/relative context source；新增 independent benefit/harm bounded residual；nominal logits 校准后重新钉扎为 0。 |
| `src/ocrap/models/losses.py` | 新增 independent-tail BCE；新增 per-regime/per-class balanced objective；支持严格替换 dead-zone-dominated Evidence ERM。 |
| `src/ocrap/cli/train.py` | 传播新 loss/config；checkpoint 保存 context source；新增跨 Near/Contact 的 `direct_duet_selection_risk`。 |
| `src/ocrap/models/inference.py` | 从 checkpoint 恢复 context source，保证训练和推理结构一致。 |
| `scripts/train_ocrap_v48_trac_sr.sh` | 传播 dual tails、balanced replacement、context source、cross-regime metric 参数。 |
| `scripts/adapt_ocrap_v48_18_duet_variant.sh` | 新增 DUET-BRIDGE 适配入口；默认 48 维 tournament context、1,532 参数校准器、strict balance 和双尾目标。 |
| `scripts/run_v48_18_duet_dedicated.sh` | 两张 A30 并行 Balanced/Precision；始终写 gate/completion 状态；遵守 RC 0/20/30。 |
| `scripts/run_v48_18_parallel_ablations.sh` | 4 组 × 2 variant 共 8 个任务一次启动，GPU0/GPU1 各 4 个任务，每任务默认 1 worker。 |
| `tests/test_v48_18_duet_bridge.py` | 覆盖 identity、1,532 参数量、tournament context、nominal pin、非 simplex 双尾和跨 regime checkpoint metric。 |

## 消融定义

- `A_dual_scalar`：双尾 + 四个 scalar。
- `B_dual_tournament`：A + 冻结 tournament context。
- `C_dual_tournament_balanced`：B + stratified batch + strict balanced replacement。
- `D_full_duet`：C + cross-regime checkpoint selection。

## 不重复项

不再重复：78,630 参数 raw-context calibrator、simplex-only correction、balanced-as-auxiliary、完整 Evidence 重训、阈值放宽、绕过 Natural gate 运行 test/stress。
