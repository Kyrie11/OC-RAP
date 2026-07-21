# OC-RAP v42 实验结果诊断与 v43 OC-RSC 方案

日期：2026-07-21

## 1. 结论

v42 的结论应分成两层：

- **排序模块层面：部分生效。** 三个模型的 top-1 opportunity capture 为 70.5%–88.6%，明显不是随机排序；说明候选 score head 学到了一部分正恢复候选的相对顺序。
- **端到端规划层面：没有生效。** 所有 Near/Contact calibration 的 challenge rate 都是 0；offline Near 276/276 次保持 nominal，Contact 449 次中仅 1 次干预且原因不是 direct value。两个主模型的 offline 指标完全一致。因此 v42 没有改变执行策略，也没有产生可归因于 OCSAVA 的物理收益。

Stage 2 阻止后续 Waymax closed-loop 是正确行为。当前没有 v42 闭环结果，不能把“未恶化”解释成“优化有效”。

## 2. v42 训练与校准证据

| 模型 | Regime | 最优 epoch / 完成 epoch | 训练时间 | q | opportunity rate | top-1 capture | 全候选 advantage MAE | challenge rate | top-1 teacher advantage 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairwise balanced | Near | 4 / 9 | 81.2 min | 0.5223 | 12.03% | 78.13% | 0.2785 | 0 | -0.1187 |
| pairwise balanced | Contact | 4 / 9 | 81.2 min | 0.4983 | 11.96% | 70.45% | 0.2655 | 0 | -0.0891 |
| pairwise hard | Near | 2 / 7 | 25.4 min | 0.4842 | 12.03% | 84.38% | 0.2876 | 0 | -0.1146 |
| pairwise hard | Contact | 2 / 7 | 25.4 min | 0.4495 | 11.96% | 72.73% | 0.2759 | 0 | -0.0907 |
| no raw adapter | Near | 2 / 7 | 48.7 min | 0.5093 | 12.03% | 87.50% | 0.2741 | 0 | -0.1296 |
| no raw adapter | Contact | 2 / 7 | 48.7 min | 0.4893 | 11.96% | 88.64% | 0.2619 | 0 | -0.1033 |

关键解释：

1. top-1 capture 较高，只说明“当组内存在正恢复候选时，模型有时能把它排到第一”；它不等于模型选择的 top-1 通常有正收益。
2. 三个模型的 top-1 teacher advantage 均值仍为负，说明多数 scene-time 上模型选出的最佳动作仍劣于 nominal。
3. additive q 为 0.45–0.52，而部署 score advantage 没有足够大的正尾部，最终全部被 LCB 阈值过滤。
4. no-raw 模型的 capture 最好、MAE 相当，raw adapter 没有带来准确率收益。其训练时间介于 balanced 和 hard 之间，因此不能单凭耗时判断，但没有理由继续保留未经物理归一化的 raw flatten branch。

## 3. v42 offline 结果

| Regime | NUP | FRA | DRS | PCD | intervention | direct-value reasons | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Safe | 1.0000 | 0 | 1.0000 | 0.61626 | 0 | 0 | 方向性达标，但来自 nominal hard lock |
| Near-contact | 1.0000 | 0.07609 | 0.92391 | 0.57352 | 0 | 0 | 276/276 nominal，新模块未使用 |
| Contact | 1.0000 | 0.07795 | 0.92205 | 0.57227 | 0.00223 | 0 | 唯一干预来自旧 calibrated admission，不是 direct value |

balanced 与 hard 在三个 regime 上完全相同，证明模型分数没有传导到最终策略。

### 当前候选集的经验 teacher frontier

| Regime | 当前 PCD | teacher PCD | PCD 可见差距 | 当前 FRA | teacher FRA | FRA 可见差距 | teacher intervention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Near | 0.57352 | 0.58798 | +0.01447 | 0.07609 | 0.05435 | -0.02174 | 44.57% |
| Contact | 0.57227 | 0.58961 | +0.01734 | 0.07795 | 0.05122 | -0.02673 | 44.99% |

这不是严格数学上界，但它是重要的经验天花板：此前提出的 Near `PCD +0.02`、Contact `PCD +0.03` 已超过当前 teacher selector 展示出的差距。要达到这种绝对提升，必须同时改善 candidate/recovery frontier，而不能只改 selector。当前更合理的投稿目标是：在很低的干预率和高 NUP 下关闭 50% 左右的 teacher gap，并用多 seed 与置信区间证明稳定性。

## 4. 三个 regime 的修订目标

### Safe

- 开发门槛：intervention=0，NUP>=0.999，FRA/DRS/PCD 不退化。
- 投稿证据：3 seeds；主系统可保留 Safe nominal guard，但必须增加“关闭 hard lock”的安全消融，避免结果被认为是规则硬编码。

### Near-contact

- 6/12 rollout 门槛：direct path 非零；所有执行动作 deviation>=0.002；NUP>=0.995；干预率不高于约 4%–6%；paired PCD 或 regret 至少改善 0.005。
- 投稿目标：FRA 相对下降至少约 15%（0.07609 -> 约 0.0647），或 PCD 提升约 0.007–0.010；相当于关闭约一半当前 teacher gap。

### Contact

- 6/12 rollout 门槛：direct path 非零；NUP>=0.985；干预率不高于约 8%；无 no-op；paired PCD/regret 至少改善 0.005。
- 投稿目标：FRA 相对下降约 20%（0.07795 -> 约 0.0624），或 PCD 提升约 0.009–0.012；同时报告 secondary collision、stable stop、yaw/rejoin 等接触后指标。

这些是项目目标，不是 CCF-A 的统一官方阈值。最终说服力来自：强基线、成对场景、多 seed、统计区间、机制消融，以及收益确实来自论文提出的新模块。

## 5. 根因

### 5.1 additive conformal 仍然是零覆盖证书

v42 将 max-residual 改为 top-1 residual，q 确实从 v41 的约 0.57 降到 0.45–0.52，但仍大于所有可部署正 advantage，challenge rate 仍为 0。

### 5.2 calibration 与 selector admission 存在结构矛盾

v42 calibration 在 actionable 候选中选择 top-1；selector 随后又执行 `direct_value_challenge &= admitted`。offline 的 admission 通常只有 nominal，因此即使 score head 找到候选，旧 OC-MERO admission 仍会把它删除。value head 实际只是“已认证集合内偏好”，无法修复旧证书的 false abstention。

### 5.3 sampler 没有针对真正的正优势组

v42 的 hard-group sampler 使用恢复候选的绝对 `r_dep` 阈值，而不是 `r_dep(candidate)-r_dep(nominal)`。大量 nominal 同样好或更好的组被高权重采样，稀释了约 12% 的真实机会组。

### 5.4 raw candidate/action adapter 没有物理归一化

raw prefix states 和 controls 直接 flatten/pad 后进入 adapter，不同量纲、轨迹长度和 padding 模式会成为噪声。no-raw 消融的 capture 更高，故 v43 默认拒绝该分支。

### 5.5 当前 candidate frontier 也限制上限

teacher 需要约 45% 干预才能得到上述改善，说明当前 candidate set 中“收益高且可低频选择”的动作仍稀缺。v43 先解决 selector 零使用；若 v43 能关闭大部分现有 teacher gap但仍达不到论文目标，下一版本应优化 contact-specific candidate generation，而不是继续堆 selector gate。

## 6. v43：OC-RSC

全称：**Observation-Consistent Risk-controlled Selective Certificate**。

### 6.1 选择性风险证书

- 每个 scene-time 先确定唯一的 actionable top-1 recovery candidate。
- 使用稳定 scene-time hash 将校准组分成 fit/verify 两个不重叠 fold。
- fit fold 搜索 score-advantage threshold，要求最小覆盖、正收益 precision 和最大 harmful-selected rate。
- verify fold 检查非零选择、positive precision，并计算 harmful challenge **group exposure** 的单侧 90% Wilson 上界。
- 只有 `valid_for_deployment=true` 才允许离线或闭环运行；否则自动停止。

该证书不再宣称 additive conformal LCB，而是明确的 held-out selective-risk contract，避免“形式上保守、实际永不决策”。

### 6.2 stress-only admission augmentation

新增 `selection.direct_value_risk_controlled_admission`：

- Safe 永远关闭；
- Near/Contact 中，只有通过 feasible、hard/harm、macro、trajectory deviation、deterministic top-1 和 held-out threshold 的候选可以扩展 admission；
- rank-2 不回退，避免选择规则与校准事件不一致；
- 保留 actionability、budget、cooldown 和 nominal challenge 逻辑。

### 6.3 正优势采样与轻量重训

- v43 sampler 按 `r_dep(candidate)-r_dep(nominal)` 识别正优势组。
- 默认 `candidate_concat`，不使用 raw flatten adapter。
- 两个短候选最多 10 epochs、patience 3。
- 首先尝试复用 v42 `no_raw_adapter` 和 `pairwise_hard` checkpoint；只有校准或 offline direct-use 门槛失败才重训。

### 6.4 统计有效性限制

fit/verify hash fold 解决了阈值拟合与阈值验证之间的 scene-time 重叠，但若同一 validation root 已用于 early stopping，则仍不应把它作为最终论文中的独立风险保证。脚本支持：

```bash
export RSC_CAL_NEAR_DATA=/path/to/dedicated_cal_near
export RSC_CAL_CONTACT_DATA=/path/to/dedicated_cal_contact
```

论文正式实验必须使用未参与训练和 checkpoint 选择的专用 calibration roots。默认 `val_*` 仅用于开发筛选。

## 7. Closed-loop 加速

原流程直接从 offline 进入 6-rollout，再进入 12-rollout。v43 改成：

1. **无重训风险校准**：复用现有 checkpoint。
2. **offline zero-use gate**：Safe/NUP、证书有效性、Near/Contact direct reasons；不通过不运行 Waymax。
3. **3-rollout selected-only probe**：8 candidates、6 recovery options、8 steps、replan interval=2，仅标注实际选择候选。
4. **6-rollout mechanism gate**：10 candidates、top-5 labels、正常 replanning，要求 paired 方向性收益。
5. **12-rollout confirmation**：12 candidates、top-8 labels；此时才运行 compact Safe closed-loop。
6. 通过后才做 ablation 和 3-seed publication runs。

所有阶段继续使用两张 GPU：Near 在 GPU0，Contact 在 GPU1；保留 partial JSON、scene JSONL 和 resume。消融不重复 scalar baseline。

注意：3-rollout 探针只用于排除零使用/no-op/严重退化，不能作为论文结果。6/12 rollout 仍是开发证据，正式实验需要更大样本与多 seed。

## 8. 交付与验证

新增或修改：

- `ALGORITHM_CHANGELOG.md`：根目录永久更新日志。
- `tools/calibrate_direct_value_risk_v43.py`
- `tools/select_v43_candidate.py`
- `tools/check_v43_quick_gate.py`
- `scripts/calibrate_ocrap_v43_rsc.sh`
- `scripts/train_ocrap_v43_rsc.sh`
- `scripts/run_ocrap_v43_rsc.sh`
- `run_v43_two_gpu_fast_commands.txt`
- `run_v43_targeted_ablation_commands.txt`
- selector、baseline config、group sampler 与单元测试。

本地验证：

```text
python -m compileall -q src tools tests
bash -n ...
PYTHONPATH=src pytest -q
77 passed, 1 warning
```

当前环境没有 WOMD/Waymax/GPU，因此没有虚构 v43 数值。下一轮的第一目标不是直接达到投稿数值，而是先验证：现有 v42 checkpoint 是否能产生有效 held-out threshold、offline direct reasons 和真实非 no-op 闭环选择。
