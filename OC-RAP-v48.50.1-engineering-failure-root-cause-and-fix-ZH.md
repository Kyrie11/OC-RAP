# OC-RAP v48.50.1：ENGINEERING FAILURE 根因、修复、全链路审计与重跑说明

日期：2026-08-18

## 1. 结论先行

本轮上传的 v48.50 A/B/C/D **全部不能用于算法归因**。四臂 authoritative status 都是 `RC=30`、`pipeline_valid=false`，统一失败在 `certificate` stage；Balanced/Precision 的训练/适配已经成功，真正崩溃点是 v48.50 新增的 calibration native-certificate diagnostic。

唯一一致 Python 异常：

```text
TypeError: 'ComponentVetoTolerances' object is not subscriptable
```

位置：`tools/calibrate_policy_risk_v48.py` 原 v48.50 约 755--757 行。

`ComponentVetoTolerances` 是 frozen dataclass，却被当作 tuple 用 `component_tolerances[0/1/2]` 访问。异常发生在 `direct_value_risk_near_v48.json` / `direct_value_risk_contact_v48.json` 写出之前，因此 certificate controller 随后看到缺失 calibration artifacts，并按 fail-closed 规则返回 RC30。A/B/C/D、Balanced/Precision、Near/Contact 的 traceback 都是同一根因。

已修成 named fields：

```python
native_pair_margins = [
    float(n_native[0] - r_native[0] - component_tolerances.drs),
    float(n_native[1] - r_native[1] - component_tolerances.deployability_gate),
    float(n_native[3] - r_native[3] - component_tolerances.gap_discount),
]
```

这只是工程修复：不改变算法、阈值、loss、数据、2×2 因子、risk budget、proposal top-k、calibration protocol、Natural gate 或输出目录。

## 2. 我对论文与算法主线的理解

论文核心问题不是“碰撞后怎样做一套单独策略”，而是：规划器在当前可观测信息下，是否仍保存了一个**可部署、观测一致、低尾鲁棒**的 recovery option。传统 branch-wise/oracle recovery 会偷用 hidden future identity；当多个 future 在 post-prefix observation 下不可区分、却要求不同 recovery action 时，oracle recoverability 会过度乐观。

论文主链条可以整理为：

1. recovery-sufficient latent roots 表示决策相关未来，而非完整 future reconstruction；
2. post-prefix observation-equivalence kernel 限制不可区分 roots 必须共享 recovery choice；
3. affordance-conditioned signed recovery margin 描述具体 recovery option 的可行 headroom；
4. OC-MERO 做 observation-consistent、lower-tail、existential aggregation，得到 deployable/oracle recoverability 与 gap；
5. CRISP 将 recoverability 当作 admission property，而不是把它变成唯一 utility；
6. v48.50 的 DCP-DRFC-DE 进一步要求 certificate 在 learned transport 中保持 **decision-equivalent**：最终 admission 使用的 sign、ordering、decision boundary 不能被 smooth proxy 再次改写。

因此最强的 CCF-A story 应是 **Observation-Consistent, Decision-Sufficient Recoverability**，并把 v48.50 的技术核心解释为 **Decision-Equivalent Certificate Transport**，而不是“OC-MERO + DRFC + NCP + DCP + 一串模块”。

论文 Appendix 当前仍有 `Regime-conditioned recovery admission` 段落；它和这条主线以及目标约束冲突。Safe / near-contact / contact 应只作为同一个 planning primitive 在 normal→critical continuum 上的 dataset/evaluation strata，不能进入 policy/router/threshold/risk-budget。当前 v48.50 model/factor contract 已明确 fail-close regime conditioning。

## 3. 数据集理解与当前证据边界

canonical 数据集是 4 split × 3 regime：

| split | safe | near_contact | contact |
|---|---:|---:|---:|
| train | 20,000 | 13,324 | 16,790 |
| val | 2,328 | 3,445 | 6,477 |
| calibration | 2,544 | 6,039 | 16,843 |
| test | 3,216 | 4,723 | 6,687 |

`reports.zip` 额外还有一个 `traincontact.json`（16,306 samples / 486 scenes），与 canonical `train_contact.json`（16,790 / 500）不一致；它应视为历史/冗余 report，不能混入 12 个 split×regime 的主表。

Near/contact reports 支持 observation-consistency、alias incompatibility、FRA/ODG/DRS 等机制研究；safe 主要支持 DRS/normal-regime 检查。12 个 canonical reports 均没有 failure，但它们都报告 `supports_waymax_runtime_claim=true`、`supports_womd_primary_claim=false`。所以论文中“primary benchmark is built from WOMD and Waymax”在投稿前仍需 builder/source manifest 的独立证据，不能只由当前 reports 支撑。

## 4. 本轮四臂为什么全部 ENGINEERING FAILURE

四个上传目录：

- `ocrap_v48_50_dcp_de_ablation_A`
- `ocrap_v48_50_dcp_de_ablation_B`
- `ocrap_v48_50_dcp_de_ablation_C`
- `ocrap_v48_50_dcp_de_main`

共同状态：

- authoritative `RC=30`；
- `pipeline_valid=false`；
- failure stage=`certificate`；
- `test_roots_read=false`，test seal 没有被破坏；
- A/B/C/D 的 factor contract 本身正确；
- Balanced/Precision × Near/Contact 的 diagnostic log 全部出现同一个 `ComponentVetoTolerances` TypeError；
- 没找到第二种 Python traceback、OOM、CUDA error、segfault 或 dataset-missing signature。

因此不能比较 B−A、C−A、D−B−C+A，也不能把当前 D/Main 的失败写成“DEFC/E-NAP 无效”。

## 5. 根因的代码语义

`src/ocrap/algorithms/evidence_targets.py` 中：

```python
@dataclass(frozen=True)
class ComponentVetoTolerances:
    drs: float = 0.05
    deployability_gate: float = 0.05
    gap_discount: float = 0.05
    hard_violation: float = 0.05
    harm_proxy: float = 0.05
```

v48.50 diagnostic 使用的 native coordinate 顺序是：

```text
[hard DRS, sigmoid(R_dep), smooth boundary DRS, exp(-relu(gap))]
```

所以 candidate-vs-nominal 三个安全 component margin 正确对应：

- hard DRS → `.drs`；
- sigmoid(R_dep) → `.deployability_gate`；
- gap-quality（native index 3）→ `.gap_discount`。

旧代码不仅 Python 类型错误，而且 `component_tolerances[2]` 从可读性上也容易让人误以为对应 native coordinate index 2；named-field 修复同时去除了这种潜在语义歧义。

## 6. v48.50 2×2 和 no-regime-conditioning 审计

当前 factor design 保持：

- A：v48.49-C reference = old DRFC + smooth NAP；
- B：A + DEFC；
- C：A + Exact NAP；
- D/Main：A + DEFC + Exact NAP；
- 四臂 MC-NCP 都关闭。

D/Main uploaded contract 的 expected/actual inference contract 一致；关键约束包括：

- `direct_recovery_value_regime_conditioning=false`；
- `strategy_regime_conditioning=false`；
- native certificate preservation=true；
- D 的 exact native advantage=true；
- `test_roots_read=false`。

所以这次工程事故并没有暴露 2×2 factor 串线或 regime-conditioned policy 泄漏。

## 7. calibration / fail-closed / 输出目录审计

`calibrate_v48_36_shared_certificate_pool.sh` 的主流程仍然是：每个 variant 使用临时 calibration 目录 → near/contact development proposal diagnostic → pooled shared rule → standard calibration / verification certificates → required-artifact 与 shared-rule SHA 检查 → 原子替换到最终 `calibration` 目录。

Unexpected RC 会在访问不完整 certificate pool 前 fail-close 为 RC30；Balanced/Precision 通过子进程隔离，避免 `set -e`/环境变量泄漏。当前 bug 正是被这套 fail-closed 机制正确暴露，而不是生成了一份表面“成功”但内容残缺的 certificate。

`run_v48_50_postgate_if_authorized.sh` 也重新做了 synthetic validation：

- RC20：拒绝，`NEXT_COMMANDS.txt` 不执行；
- RC0 + valid D factor + certificate/gate executed：授权执行。

根双 GPU 指令和 arm scripts 仍使用原输出目录：

```text
/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_50_dcp_de_ablation_A
/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_50_dcp_de_ablation_B
/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_50_dcp_de_ablation_C
/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_50_dcp_de_main
```

本次没有改实验输出目录。

## 8. 性能优化：只保留结果不变的部分

已把 `predict_samples` / `predict_sample` 从 `torch.no_grad()` 改成 `torch.inference_mode()`。它只降低纯 inference 的 autograd/version-counter overhead，不改变模型、候选顺序、group index、dtype、算子或输出定义。

在固定 random seed、相同 synthetic candidate set、完全相同模型参数上，优化前/后以下字段全部逐 bit 相同，最大绝对差 0：

`r_dep, r_orc, gap, q, root_probs, c_star, margins, direct_recovery_value, direct_recovery_std`。

我没有启用以下可能改变数值或语义的“加速”：AMP/FP16/BF16、TF32、跨 scene/group batching、候选重排、改变 proposal top-k、改 DataLoader shuffle、异步近似 certificate、缓存 teacher 以外的模型输出。跨 group batching 尤其不能直接做，因为当前 `predict_samples` 明确假设一个完整 scene-time candidate set 共用 `group_index=0`。

A30 上真实吞吐提升需要你机器上用相同 run 做 wall-clock/profile；当前环境只有 CPU，不能给不可靠的 GPU 加速百分比。

## 9. 修复后验证范围

最终代码至少完成：

- v48.47--v48.50 focused algorithm regression：36 passed；
- 新增 calibrator named-tolerance regression；
- inference hotpath 定向 test：1 passed；
- optimization 前/后 synthetic prediction bitwise equivalence：PASS；
- `python -m compileall -q src tools tests`：PASS；
- 全仓库 115 个 `*.sh`：`bash -n` PASS；
- 根 v48.50 双 GPU command file：`bash -n` PASS；
- RC20/RC0 post-gate synthetic contract：PASS；
- 在性能优化前，本轮还完成过 active terminal/stage-isolation + v48.47--v48.50 关键 suites：93 passed, 1 skipped。

一次把更多 slow/historical tests 合并执行时触及当前工具 120s 上限；更广的 historical suite 还包含旧版本缺失脚本/package-expectation debt。因此本交付**不声称 full historical pytest 全通过**。这不影响当前 v48.50 active-path 修复结论，但避免把测试范围说得过头。

## 10. 重跑方式

你可以删除原四个 v48.50 结果目录，然后用交付代码覆盖/放到：

```bash
/home/senzeyu2/code/OC-RAP-v48.50-DCP-DRFC-DE
```

继续执行**原文件名、原输出目录**的指令：

```bash
cd /home/senzeyu2/code/OC-RAP-v48.50-DCP-DRFC-DE
bash OC-RAP-v48.50-DCP-DRFC-DE-two-GPU-run-commands-ZH.txt
```

只有四臂分别满足 authoritative RC∈{0,20}、`pipeline_valid=true`、certificate/gate executed、factor identity valid 后，才做算法归因。

## 11. 正确结果回来后的分析顺序

第一优先不是先看 D/Main 的最终 recall，而是：

1. **B−A**：DEFC 是否改善 upstream exact native coordinate calibration / false-veto；
2. **C−A**：Exact NAP 是否把 downstream PCD transport 的 sign/centering 拉回 teacher/evaluator 语义，同时保留 v48.49-C 的 ranking benefit；
3. **D−B−C+A**：两者是否有真正互补或冲突；
4. 然后看 Near/Contact 的 joint sign、certificate recall、harm UCB 和 native component diagnostics；
5. 只有 D RC0 才进入 Safe paired scene-disjoint non-inferiority + stress/closed-loop。

如果 B/D 仍不能把 Contact development positive sign 拉起来，就按已预注册 stop rule 停止 selector/OCAF/threshold/top-k 搜索，转到 teacher PCD correctness、constraint/margin normalization、model root-probability calibration、recovery-option coverage / candidate feasibility audit。不要改成 regime-conditioned policy 来“救”三个 regime 的分数。
