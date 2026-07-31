# OC-RAP v48.23 实验联合审计与 v48.24 SUPPORT-BRIDGE 优化报告

## 1. 结论先行

这次 `RC=20` 是真实的 Natural-gate 拒绝，但根因与上轮预判相比更靠前了一层：**不仅 learned gate 没学会安全准入，当前 frozen top-3 + raw-benefit opportunity + component-veto + gate 的 fit 支持本身也不够。**

- Near fit 共有 127 个组，top-3 内只有 **3 个 safe-positive 组**。即使 oracle 选择 10 个，最多只有 3 个真阳性，precision LCB 仅 **0.1538**，不可能达到 gate。
- Contact fit 共有 384 个组，top-3 内只有 **10 个 safe-positive 组**。oracle 选择 16 个时 precision LCB 为 **0.4652**，仍低于 0.5。
- Near/Contact verify 的 oracle 都可行，说明问题集中在 fit 的有限样本与候选支持，而不是所有证书数据完全没有机会。

因此，v48.23 的 RC=20 由三层共同造成：

1. **首要结构缺陷：** raw-benefit top-3 并不等于 safe-positive top-3；上一轮“proposal hit≈0.97–1.00”只证明原始收益召回，不证明安全收益支持。
2. **首要学习缺陷：**最终 admission 没有直接学习连续 safe utility，而是依赖 benefit、harm、frontier 等多个间接目标恰好组合正确。
3. **数据贡献因素：**safe-positive 极稀疏且 train→val/cal 分布有明显变化；本轮不重建数据，因此改用 safe-positive 分层采样和目标重定义。

Macro action 不是这次 RC=20 的直接根因。oracle audit 本来就忽略 macro concentration；现有宏动作集合包含 brake/yield/merge/stabilize 等恢复语义，真正缺的是同一宏动作连续参数中“安全且有收益”的变体进入 top-k 并被正确排序。简单新增宏动作名称没有证据支持。

## 2. 论文与实现合同的核对

论文的核心不是“选最大收益动作”，而是把 recoverability 当作 admission constraint：nominal 若满足约束则保持 nominal；只有 nominal 低 headroom 时，语义上具有保护性的恢复前缀才可在非劣 deployability 坐标下准入。代码修复因此遵循三条原则：

- 训练目标必须对应部署时“nominal 或一个 recovery action”的 one-action 决策；
- harmful 不能被更高 benefit 补偿；
- certificate、runtime 和 closed-loop 必须执行同一 top-k/rerank/threshold 合同。

## 3. v48.23 主实验：Near 是否把强收益信号转成安全准入

| 变体 | Candidate benefit AUC | Learned benefit AUC | Learned safe-benefit AUC | Learned harm AUC | Conditional harm AUC | score-teacher corr | harmful switch | top-1 regret | verify selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced Near | 0.851 | 0.826 | 0.832 | 0.364 | 0.669 | -0.080 | 0.0% | 0.005 | 0 |
| Precision Near | 0.789 | 0.773 | 0.792 | 0.593 | 0.527 | 0.044 | 49.2% | 0.091 | 0 |

判断：**没有完成安全准入转换。**

- Balanced 的 raw benefit 信号仍强，但 learned harm AUC 只有 0.364，score-teacher correlation 为 -0.080，最终完全 abstain。它不是“安全所以不选”，而是风险与排序表示失真。
- Precision 的 conditional harm AUC、harmful switch、相关性相较 v48.22 只有小幅变化，harmful switch 仍约 49.2%，verify coverage 仍为 0。
- 由于 dev shadow 工程错误，本轮没有任何物理 paired 结果，不能声称 minimum clearance、TTC、危险暴露、PCD/DRS/NUP 已改善。

## 4. Contact 风险与收益是否改善

| 变体 | Candidate benefit AUC | Learned benefit AUC | Learned safe-benefit AUC | Candidate harm AUC | Learned harm AUC | Conditional harm AUC | corr | harmful switch | regret | verify selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced Contact | 0.551 | 0.384 | 0.514 | 0.411 | 0.397 | 0.522 | -0.241 | 40.0% | 0.182 | 0 |
| Precision Contact | 0.561 | 0.422 | 0.496 | 0.640 | 0.650 | 0.502 | -0.062 | 42.7% | 0.106 | 0 |

判断：**上一轮修改没有建立 Contact 收益判断，且 Balanced Contact 更差。**

- Precision 的 broad harm AUC 仍约 0.65，但 high-opportunity conditional harm AUC 从上轮约 0.558 降到 0.502，接近随机。
- Learned benefit AUC 仍约 0.422，safe-benefit AUC 约 0.496，相关性变为 -0.062，regret 仍约 0.106。
- Balanced Contact 的 learned benefit AUC 0.384、相关性 -0.241、regret 0.182，均比上轮更差。
- 因而“识别广义风险”与“选出真正可迁移的撞后安全收益动作”仍是两件没有连接起来的事。

## 5. v48.23 消融的明确归因

8 个任务全部完成、全部 gate fail、fit/verify selected 均为 0。结论不是某个权重略小，而是目标没有形成有效链路：

- A 的 semantic prior + centered identity + categorical policy 恢复了部分 broad harm AUC，说明工程修正确实必要。
- B 加 raw-benefit listwise 后，score-teacher correlation 和 top-1 并未系统改善；原始收益排序会把 harmful-beneficial 一并向上推。
- C 的 frontier contrast 只有有限局部增益，稀疏 pairwise 信号不足以承担最终准入学习。
- D 没有同时优于 B/C，说明 listwise 与 contrast 在当前写法下不是互补模块。

完整 8×2 指标已写入 `OC-RAP-v48.23-ablation-metrics.csv`。

## 6. 数据集、模型设计、macro action 的责任划分

### 6.1 模型设计：主要可修复原因

v48.23 同时保留 categorical one-action 和权重 1.25 的 legacy Noisy-OR group objective。一个要求概率集中到 nominal+单个 action，另一个允许多个弱机会相加，梯度合同冲突。另一方面，continuous listwise 学的是 raw benefit，frontier contrast 只是稀疏辅助，最终 admission 并未被连续 safe utility 直接监督。

### 6.2 数据集：重要贡献因素，但本轮不重建

| Regime/Split | samples | groups | scenes | artifact | negative deployable | oracle recoverable | incompatible alias pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Near train | 13324 | 1800 | 600 | 18.9% | 55.3% | 63.6% | 16.1% |
| Near val | 3445 | 433 | 176 | 24.6% | 50.4% | 74.2% | 20.4% |
| Near calibration | 6039 | 765 | 316 | 24.0% | 44.8% | 79.2% | 19.6% |
| Contact train | 16790 | 2000 | 500 | 16.6% | 54.3% | 62.3% | 9.5% |
| Contact val | 6477 | 723 | 211 | 21.9% | 46.1% | 75.7% | 13.5% |
| Contact calibration | 16843 | 1896 | 543 | 21.2% | 41.7% | 79.5% | 14.2% |

train 与 val/cal 的 artifact、negative deployable、oracle recoverable、alias incompatibility 都有明显变化；safe-positive 又很稀疏，因此模型很容易学到 broad risk，而学不到高收益安全前沿。v48.24 不重建数据，只做 safe-positive group sampling、safe-benefit target 和 support-width 诊断。

### 6.3 Macro action：次要结构因素

论文已经限制 relaxed protective admission 只能用于保护性恢复语义；代码中的 deployable macros 也不是无限制的 nominal perturbation。当前 oracle fit 失败发生在忽略 macro concentration 的乐观条件下，所以宏动作集中度不是根因。更合理的修复是把 top-k 从 3 扩到 8，观察 safe variants 是否在更深候选中；若 top-8 仍 oracle fail，才有证据重新设计 proposal 连续参数或宏动作，而不是盲目新增类别。

## 7. 工程错误与修复

### 7.1 adaptation-dev shadow closed loop

日志中的 `invalid OC-TRAC-SR certificate` 来自工程错误：shadow 脚本调用了 deployment-only loader，而 RC=20 后证书必然 `valid_for_deployment=false`。你的命令按提供的指令执行，没有运行错。

修复后：

- `DEV_SHADOW_DIAGNOSTIC=1` 只允许读取 fit-derived diagnostic selector；
- 仍然只读 `evidence_adapt_dev`，禁止 certificate/test/stress；
- deployment/stress 路径仍要求正式有效证书，不会被诊断后门绕过。

### 7.2 certificate 与 runtime 策略不一致

v48.23 certificate 使用 top-3 + Evidence rerank，但 runtime 只读取三项阈值，默认变成 top-1、no-rerank。即便未来 gate 通过，closed-loop 也不会执行被认证的策略。现在 loader 同步读取：score threshold、opportunity threshold、harm threshold、rank margin、proposal top-k、Evidence rerank、conditional ranking。

### 7.3 目标冲突

v48.23 保留的 Noisy-OR 已默认关闭。SUPPORT-BRIDGE 只保留 nominal+top-k categorical one-action 主目标。

## 8. v48.24 SUPPORT-BRIDGE 的算法修改

1. **Top-8 safe support：**冻结 proposal 不重训，只把可重排支持从 top-3 扩到 top-8。
2. **Proposal support curve：**证书输出 k=1/3/5/8/active 的 fit/verify optimistic oracle，直接判断结构可行性。
3. **Safe-benefit opportunity：**只有 `PCD delta>positive_gain` 且 component-harm=false 才是正机会。
4. **Direct safe-utility regression：**safe action target=连续 PCD advantage；harmful action target=`-max(abs(delta), positive_gain)`。
5. **Direct safe-utility listwise：**在 exact deployed top-k 内对 nominal+候选做连续分布排序。
6. **Safe-positive batch：**提高 safe-positive group 被看到的频率，但不复制证书样本、不读取 test。
7. **Light frontier contrast：**只作为 0.25 的辅助，不再让稀疏 pairwise 独自承担准入。
8. **Runtime-true policy：**训练、certificate、dev shadow、held-out stress 使用同一 selector contract。

理论上变成 RC=0 的必要链条是：

`top-8 fit/verify oracle support pass → learned safe-utility correlation/conditional harm改善 → fit frontier满足 → verify gate满足 → RC=0`

这不是结果保证。若 top-8 oracle 仍失败，当前固定数据与新合同仍没有足够 safe-positive 支持，代码会明确返回 RC=30，而不是再浪费一轮 calibrator。

## 9. 下一轮判读顺序

1. 先看每个证书的 `proposal_support_curve`，特别是 k=3→5→8 的 safe-positive 数与 oracle LCB。
2. 再看 learned top-k safe-benefit AUC、continuous correlation、positive regret、conditional harm AUC、harmful switch。
3. 再看 fit nearest frontier，不只看 selected=0。
4. RC=20 时运行修复后的 adaptation-dev shadow，检查 Near clearance/TTC/exposure/deficit 与 Contact secondary contact/overlap/escape/stable-stop。
5. 消融应按 A→B→C→D：B>A 说明 top-k 支持有效；C>B 说明 direct safe utility 有效；D>C 说明轻量 frontier 仍有增益。

## 10. 本地验证与限制

- `PYTHONPATH="$PWD/src" pytest -q`：**220 passed，5 warnings**。
- `python -m compileall -q src tools tests`：通过。
- 所有 Shell `bash -n`：通过。
- 当前环境没有真实 WOMD/Waymax 数据和两张 A30，因此没有声称 v48.24 已经 RC=0，也没有伪造 closed-loop 结果。
