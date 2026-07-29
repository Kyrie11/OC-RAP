# v48.14结果审计与v48.15 PRISM-CC设计

## 1. 最重要的结论

本轮不能把“没有`NEXT_COMMANDS.txt`”解释成Natural gate已经被算法拒绝。上传的八个消融任务在真正运行Near/Contact校准之前都发生了相同的Shell异常：

```text
scripts/calibrate_v48_14_certificate_pool.sh: line 23: variant: unbound variable
```

根因是`set -u`下在同一个`local`命令中引用了尚未完成赋值的局部变量：

```bash
local variant="$1" gpu="$2" run="$OUTPUTDIR/candidates/$variant"
```

因此本轮实际状态是：

- Evidence adaptation训练完成；
- certificate calibration没有启动；
- Near/Contact risk JSON没有生成；
- Natural gate没有被评估；
- 旧controller把产物缺失误写成了`GATE_FAILED.json`。

v48.15已经把状态拆成：

- 返回码0：校准完整且Natural gate通过；
- 返回码20：校准完整但Natural gate确实拒绝；
- 返回码30：产物、脚本或controller错误，并写`CALIBRATION_FAILED.json`。

所以第一步不是重新训练，而是使用修复后的脚本对现有v48.14 checkpoint重跑certificate阶段。

## 2. Safe结果同样存在工程性失真

上传的Safe paired结果报告8个paired scenes，并且candidate与scalar baseline完全相同。但两个闭环JSON同时显示：

```text
bucket_dataset=/data0/senzeyu2/dataset/OCRAP/calibration_safe
bucket_target_count=0
bucket_matched_rollouts=0
```

旧脚本强制设置`closed_loop.bucket_split=test`，而`calibration_safe`的manifest并不是`test` split。目标加载器得到0个目标后，闭环runner静默退化成任意WOMD场景评估。因此这8个scene只能证明Safe nominal lock在随机场景中工作，不能作为calibration-safe paired non-inferiority证据。

v48.15的修复包括：

1. Safe默认不强制split；
2. 指定bucket dataset时必须匹配到至少一个target，否则立即报错；
3. Safe wrapper默认`CL_RESUME=0`，避免沿用旧的8-scene partial；
4. 新增scene-level jerk p95与yaw-rate p95；
5. route progression仍需真实sdc path支持，缺失时继续明确报告不可用，不以其他指标替代。

## 3. v48.14中仍可进行的算法判断

虽然最终Natural gate没有运行，但Evidence adaptation的独立dev结果仍然能说明模型学习趋势。

### 3.1 专用目标域适配方向成立

将Evidence适配到dedicated calibration的train/dev分布后，部分harmful-switch与false-intervention诊断下降。这说明此前判断的train-to-calibration合同漂移确实存在，使用目标域数据适配轻量风险模块是合理方向。

### 3.2 v48.14适配器容量明显过大

v48.14只训练`direct_delta_adapters`，但实际可训练参数约392,892个。对应的目标域正机会支持只有：

| Regime | Deployable positive groups | Positive scenes |
|---|---:|---:|
| Near | 16 | 10 |
| Contact | 44 | 17 |

即使用几十个正group更新近40万参数。结果是模型在adaptation dev上变得极度保守：

- Near positive admission recall约0～0.333；
- Contact positive admission recall约0～0.036；
- 某些设置能降低harm，但几乎不再允许正恢复动作。

这不是可靠证书，而是典型的小样本目标域过拟合/灾难性遗忘与all-abstain倾向。

### 3.3 强hard-harm mining有局部安全作用，但损害coverage

C组相较普通adaptation，部分Near false intervention和harmful switch下降；但positive recall进一步降低。说明false-safe hard mining可以保留为辅助项，但权重2.5不适合继续作为主适配机制。

### 3.4 same-group counterfactual没有稳定增益

D组相较C组的dev指标几乎相同，部分指标略差。原因是一个top-k proposal中并不总同时存在beneficial、dead和harmful成员，可用同组pair过少。v48.15默认关闭该项，避免重复投入。

## 4. 三个regime的当前状态

### 4.1 Safe

当前唯一可确认的是nominal lock和零干预在8个任意场景中成立。下列投稿目标仍未得到有效验证：

- calibration-safe目标是否真实匹配；
- collision/offroad paired非劣；
- 至少100个paired scenes的置信区间；
- route progression；
- jerk/yaw-rate p95；
- intervention episode上界。

v48.15先修复目标匹配，再使用120个target运行paired probe。

### 4.2 Near-contact

Near的目标域正机会支持最少。当前主要问题不是完全没有收益信号，而是：

- 适配器参数量远高于有效正group数量；
- 强风险训练迅速退化为abstain；
- dev正机会只有个位数到十余个，checkpoint方差很大；
- 最终certificate尚未真正运行，因此precision LCB、harm UCB和verify recall未知。

Near距离投稿目标的关键差距仍是同时获得：非零coverage、positive recall≥0.35、precision LCB足够高、harmful-selected UCB足够低，然后才有资格评估clearance、TTC、DRS、PCD、FRA和ODG。

### 4.3 Contact

Contact有更多正机会group，且此前v48.13已经表现出较强benefit识别与top-k proposal recall。但v48.14 dev上的positive admission recall仍只有0～3.6%，说明全量适配器把风险边界推得过于保守。

Contact真正需要的是在不破坏源模型benefit排序的前提下，对目标域harm/dead边界做小幅校正，而不是重新学习整个Evidence函数。

## 5. 统一算法优化：v48.15 PRISM-CC

PRISM-CC保留已经有效的三层结构：

```text
高召回top-k proposal
→ ordinal evidence
→ 独立scene-disjoint certificate / abstention
```

但目标域适配改为低容量残差校正。

### 5.1 冻结proposal与源Evidence

冻结：

- encoder；
- Recovery Set Tournament；
- source direct-delta/evidence experts；
- 其它value、harm、opportunity heads。

这避免dedicated calibration小样本破坏源模型已经学到的Contact benefit信号和top-k proposal。

### 5.2 Tiny regime-specific evidence calibrator

每个regime的小校正器输入：

- 冻结Evidence center；
- 冻结Evidence width；
- 冻结Preference score；
- 冻结Preference runner-up gap。

输出对center和width的有界残差。最后一层严格零初始化，因此启用校正器时首个forward与源checkpoint完全相同。两套校正器总state参数共132个参数，而v48.14适配器约392k。

这一设计使模型学习的是：

> 在目标域中，应如何小幅修正源模型的风险置信度？

而不是：

> 用几十个正group重新学习完整Evidence函数。

### 5.3 防止再次退化为always-abstain

- hard-harm附加权重由2.5降为1.0；
- beneficial class权重提高；
- checkpoint risk中的missed-opportunity权重由0.25提高到0.55；
- Natural gate本身的precision、harm、support和scene-disjoint门槛不变。

这是改变学习目标的平衡，不是放宽部署门槛。

## 6. 必须按层验证，而不是只看NEXT_COMMANDS

### 层0：工程完整性

- calibration worker无异常；
- gamma、四个标准calibration JSON、Near/Contact risk JSON齐全；
- `CERTIFICATE_CALIBRATION_COMPLETE.json`存在；
- `CALIBRATION_FAILED.json`不存在。

### 层1：目标域适配

相较v48.14全量adapter，tiny calibrator应：

- 保持或提高positive admission recall；
- 不显著增加harmful switch；
- 不破坏源proposal与rank输出；
- Near/Contact dev risk都稳定，而非只改善一个regime。

### 层2：独立certificate

- verify selection非零；
- selected mean teacher advantage为正；
- precision LCB与harmful-selected UCB同时改善；
- macro excess concentration通过；
- 同一个variant在Near和Contact都`valid_for_deployment=true`。

### 层3：闭环

只有层2通过并生成`NEXT_COMMANDS.txt`，才运行Near/Contact stress closed-loop。Natural gate阈值没有被降低，也无法科学地保证一定通过。

## 7. v48.15消融

| 组 | 目的 |
|---|---|
| A source dedicated | 只修复工程并用dedicated certificate评价源模型 |
| B full adapter PRISM | 复现v48.14近40万参数适配 |
| C tiny calibrator | 只验证低容量残差校正 |
| D full PRISM-CC | tiny校正 + 温和hard-harm/missed-benefit平衡 |

每个variant wave的四组同时运行：GPU0运行A/C，GPU1运行B/D；每张A30同时两个约1GB进程。Balanced完成后再运行Precision，以避免八个任务同时争抢CPU与磁盘。

## 8. 结论边界

本轮不能声称v48.14通过或未通过Natural gate，因为gate根本没有运行。也不能声称Safe通过非劣，因为0个calibration-safe target被匹配。v48.15先修复这些工程事实，再用低容量Evidence correction解决v48.14已显示的过度保守和小样本过拟合问题。
