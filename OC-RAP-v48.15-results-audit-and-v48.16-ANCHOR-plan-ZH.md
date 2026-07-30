# v48.15结果审计与v48.16 OC-TRAC-ANCHOR设计

## 1. 最关键的结论

本轮看到的`certificate recovery rc=20`并不是一次有效的Natural gate拒绝。专用协议把样本角色写成`certificate_pool`，但标准校准器和风险校准器只接受字面值`calibration`或`val`，导致Near的2,412个NPZ和Contact的6,929个NPZ全部被过滤，最终风险JSON均为`num_groups=0、num_scenes=0`。旧controller仍安装了这些空JSON并写出`GATE_FAILED.json`。

因此，本轮尚不能回答v48.15在真实dedicated certificate pool上是否通过Natural gate。第一步应使用v48.16修复后的校准代码，对已经训练的v48.15 checkpoint重新运行certificate，无需重新训练。

上传的v48.15主目录也不包含candidate checkpoint、adaptation log、certificate JSON或完成标记；`learning_gates_v48_15.json`明确显示`artifact_present=false、calibration_complete=false`。它同样不能作为Natural gate结论。

## 2. Safe实验为何无效

Safe结果读取到120个离线target，但扫描2,000个原始场景后匹配数为0，最终scene和decision均为0。原因包括：

1. Waymo WOMD validation应使用`validation_tfexample.tfrecord@150`；
2. dedicated calibration target来自较后的validation场景，`SAFE_RAW_MAX_SCENARIOS=2000`只扫描开头场景，很容易一个也匹配不到；
3. 旧runner在`require_bucket_targets=true`时只检查是否加载到target，没有在最终匹配数为0时失败。

v48.16要求150个shard文件全部存在，默认完整扫描validation，并在0匹配时硬失败。

## 3. v48.14/v48.15算法证据

由于最终certificate为空，只能分析adaptation-dev指标，不能分析正式gate或闭环。

### 3.1 v48.14高容量adapter

高容量adapter在部分设置下降低harm和false intervention，但总体趋于过度保守：

- Balanced Near：positive recall 0.111，raw admission 0.025；
- Balanced Contact：positive recall 0.036；
- Precision Contact：positive recall 0；
- 约39万参数由Near仅16个正group、Contact仅44个正group监督，目标域样本不足。

结论：目标域适配的方向成立，但完整重训Evidence函数无效，不应重复。

### 3.2 v48.15 tiny calibrator

132参数、零初始化和有界残差是合理的：它不破坏源proposal和源Evidence，并显著降低部分harm/false intervention。但其目标仍然被dead-zone多数类主导，出现近似always-abstain：

- Balanced Near recall 0；Contact 0.036；
- Precision Near 0.111；Contact 0.036；
- C与D的最佳epoch指标完全相同，当前hard-harm/hard-benefit设置没有可测增益。

结论：低容量残差校正值得保留；原损失函数和强hard mining需要替换。

## 4. 三个regime的当前状态

### Safe

算法方向应继续nominal lock。当前没有任何有效paired scene，无法验证collision/offroad非劣、route progression、NUP、jerk、yaw-rate或intervention置信区间。Safe问题主要是实验匹配与统计，不是恢复策略学习。

### Near-contact

Near的主要矛盾是正机会稀少且幅度小。目标域适配容易降低harm的同时把coverage和positive recall压到零。为达到投稿前门槛，需要在独立scene上同时获得：非零coverage、recall约0.35以上、正平均teacher advantage、precision LCB提升以及harmful UCB下降。

### Contact

Contact源模型已有较强benefit信号，但目标域适配容易遗忘，尤其把beneficial和dead-zone一起压成abstain。Contact需要保留源Evidence，只学习小幅harm/dead与benefit边界校正，而不是重训整个Evidence函数。

### 统一方法

三个regime可以统一为：

- Safe：nominal lock加paired non-inferiority certificate；
- Near/Contact：冻结高召回top-k proposal与源Evidence；
- 用regime-specific低容量目标域校正；
- 用独立certificate pool做scene-disjoint统计证书；
- 证据不足时abstain。

## 5. v48.16 ANCHOR

ANCHOR全称为**Adaptation with Nominal-preserving Class-balanced Held-out Ordinal Risk**。

1. **Class-balanced ordered evidence**：在每个proposal group中分别平均harmful、dead-zone、beneficial损失，再对存在的类别平均，避免dead-zone多数类主导。
2. **Bipolar probability margins**：beneficial proposal必须达到最低benefit概率，harmful proposal必须达到最低harm概率，防止全部预测为abstain。
3. **Source-residual anchor**：对目标域calibrator residual施加L2锚定，没有目标域证据时保持源模型。
4. **更小的校正器**：hidden从8降至4，残差范围从0.30降至0.20；hard-harm与hard-benefit只保留温和权重。
5. **本轮不改proposal**：避免Evidence与排序同时变化，下一轮可明确归因。

## 6. 是否在gate失败时运行closed-loop

不建议在真实test/stress roots上绕过Natural gate。这样会消耗封闭测试集，并可能把失败策略产生的结果用于后续调参，破坏论文协议。

在且仅在certificate数据有效、返回码确实为20后，可以在development/val roots上做明确标记为`diagnostic_only`的少量定性闭环，用于观察失败模式；不得作为论文指标、阈值选择或最终模型选择依据。v48.16标准stress wrapper仍强制检查`NEXT_COMMANDS.txt`。

## 7. 退出码语义

- `0`：dedicated certificate数据有效，Near和Contact Natural gate均通过；
- `20`：dedicated certificate数据有效，但至少一个Natural gate真实拒绝；
- `30`：数据角色、空certificate、缺失checkpoint或controller等工程失败。

只有退出码0会生成`NEXT_COMMANDS.txt`。
