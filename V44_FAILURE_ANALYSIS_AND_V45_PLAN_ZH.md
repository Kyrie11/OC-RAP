# v44 失败分析与 v45 OC-RAVE 方案

## 1. 结论

v44 在进入 offline/Waymax 前被正确停止。两个 checkpoint 都不是因为 closed-loop 太严格而失败，而是在机会概率与价值分数的校准阶段没有产生任何可验证候选。

v44 的失败由三层问题叠加造成：

1. **校准器存在固定机会概率下限错误。** v44 只搜索 `opportunity >= 0.05` 的候选；所有经过物理过滤的候选输出均低于该下限，因此 `num_top1_after_opportunity_gate=0`，风险阈值根本没有开始拟合。
2. **机会头监督量与部署量不一致。** 监督事件是候选相对本组 nominal 的 PCD 优势，但 v44 预测的是候选绝对机会概率。不同场景的 nominal 质量变化很大，绝对候选 logit 无法直接表达相对事件。
3. **Near/Contact 仍由一个无条件 value head 处理。** v44 修复了跨 bucket 的组内混排，但相似当前观察在不同压力未来下仍可得到不同 teacher 标签。一个冻结共享表征上的单头仍承受任务冲突。

因此只把 `0.05` 改成 `0` 不足以构成根本修复；它只能让校准器看见候选，不能保证候选排序正确。

## 2. v44 结果证据

| Variant | Regime | Eligible groups | Positive-opportunity groups | Pair MAE | Top-1 after opportunity gate | Held-out selected | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 246 | 30 | 0.3124 | 0 | 0 | No |
| Balanced | Contact | 357 | 43 | 0.3078 | 0 | 0 | No |
| Precision | Near | 246 | 30 | 0.3291 | 0 | 0 | No |
| Precision | Contact | 357 | 43 | 0.3289 | 0 | 0 | No |

两个变体均在 epoch 3 取得最佳 checkpoint。Balanced 的 validation direct loss 为 5.3875，Precision 为 9.5148，但较低 loss 没有转化为可部署选择规则，说明单看训练 loss 无法判断 selector 是否可用。

## 3. 数据集诊断

### 3.1 Train/val 可用于当前开发

- `train_near_contact`：post-contact 占 1.88%。
- `val_near_contact`：post-contact 占 1.70%。
- `train_contact` 和 `val_contact`：post-contact 均为 100%。

因此当前 train/val 的三类语义隔离基本可用于开发训练和早停。

### 3.2 当前 test 不满足论文主实验契约

- `test_near_contact` 中 post-contact 为 `1270/2058 = 61.71%`，主体实际属于 Contact。
- Near 的 test-val `r_dep` 均值偏移为 1.474，hard-violation 均值偏移为 0.409。
- Contact 的 test-val `r_dep` 均值偏移为 1.582，hard-violation 均值偏移为 0.371。
- Safe 的 val/test 只有 22/28 个 scene，不足以支撑论文级非劣性结论。

该问题不导致 v44 calibration 失败，因为 calibration 使用的是 val；但它会让后续 Near/Contact 外部基线和最终主表失去清晰含义。

## 4. v45：OC-RAVE

全称：**Observation-Consistent Regime-Expert Value Abstention**。

### 4.1 一套共享主模型，不是三套完整模型

v45 保留冻结的共享 OC-MERO encoder 和安全证书，只增加两个轻量 stress expert：

- Near-contact value/opportunity expert；
- Contact value/opportunity expert。

Safe 不使用 direct branch，继续由冻结 OC-MERO 与 nominal-preserving selector 负责。因此计算主体仍共享，不是三套 planner。

### 4.2 相对 nominal 的机会概率

机会概率改为：

`sigmoid(opportunity_logit(candidate) - opportunity_logit(nominal))`

训练、calibration、offline selector 和 closed-loop 使用完全相同的量。这样模型学习的是“该候选是否比当前 nominal 更值得恢复”，而不是绝对意义上的好候选。

### 4.3 经验支持上的阈值搜索

- 删除固定 0.05 机会概率下限；
- 从当前模型实际输出的最小值开始搜索；
- 即使没有规则通过，也保存 opportunity、predicted advantage、teacher advantage 分布和 macro 诊断；
- 防止再次出现只有 `inf` 阈值、却不知道模型输出在哪里的问题。

### 4.4 训练策略

- group key 继续使用 `(bucket_id, scene_hash, time_index)`；
- Safe 不参加 direct-head 训练；
- group epoch 改为不放回，保证每个 Near/Contact scene-time 每个 epoch 都被访问；
- 不再使用 `r_dep` 代理过采样；正负机会由真实 PCD advantage loss 加权；
- 候选宏动作覆盖 brake、yield、merge、pull_over、stabilize；calibration 与执行 allowlist 完全一致。

### 4.5 风险与动作证书不变

v45 不删除：

- trajectory-deviation actionability；
- hard/harm 上界；
- held-out fit/verify；
- Safe nominal lock；
- intervention budget；
- 2→4→8 rollout 逐级门槛。

## 5. 是否现在加入平滑 Regime 切换

暂不加入 soft-MoE 或连续 attention routing。

当前 v45 的训练任务由 Near/Contact 数据集选择对应轻量 expert；closed-loop 则根据当前可观测 clearance、TTC、contact 状态选择 expert，不把 teacher outcome 或测试集标签输入网络。这是 observation-derived hard routing。

先验证两个 expert 是否同时满足：

1. held-out predicted/teacher advantage 关系不再为负；
2. calibration 有非零 verified selections；
3. offline direct reason 非零；
4. 2-rollout 中真实执行且无 no-op。

若这些基础条件通过，下一阶段再比较：

- hard threshold routing；
- 带 hysteresis 的 routing；
- continuous risk token + soft expert mixture。

否则提前增加 soft routing 会把“expert 学不会”和“router 切换错误”混在一起。

## 6. 测试集重建要求

最终论文主表前必须重建：

### Clean Safe

- 样本必须为 normal；
- 禁止 near_contact、post_contact、oracle_artifact；
- 至少 100 个独立 scene，推荐 200+；
- 与 train/val scene 完全不重合。

### Clean Near-contact

主集要求：

- `require_any_regimes=[near_contact]`；
- `forbid_any_regimes=[post_contact,oracle_artifact]`；
- 不使用 artifact margin override 代替真实 regime 判断；
- future metrics 与 val_near_contact 保持同一 teacher/backend/targeted-future 定义。

Oracle-artifact/override 样本应单独生成 appendix stress set，不能混入 Near 主表。

### Clean Contact

主集要求：

- `require_any_regimes=[post_contact]`；
- 排除 oracle_artifact 主表污染；
- contact impulse、secondary collision、control delay、low friction 的 teacher 配置与 val_contact 对齐；
- 不允许 artifact pass 跳过 future metrics。

重建后必须执行：

```bash
PYTHONPATH=src python tools/check_regime_dataset_contract.py /path/to/new_diagnostics --scope all --mode paper
```

只有 `RESULT: PASS` 才能用于最终外部基线和主实验。

## 7. 下一步

开发运行：

```bash
cd /home/senzeyu2/code/OC-RAP
DATASET_DIAGNOSTICS_DIR=/path/to/unzipped/dataset_diagnostics \
  bash run_v45_two_gpu_fast_commands.txt
```

当前 test contract 失败时，脚本只使用 val roots 做 development zero-use 筛选，并明确不把结果当成论文 held-out 证据。

v45 仍可能失败；如果失败，新的 calibration JSON 会提供完整概率/优势分布、相关性和 macro 诊断，可以判断是 expert 学习失败、机会稀缺，还是风险约束不可能满足，而不再只能看到零选择。
