# v48.34 RC=30 工程审计与 v48.34.1 修复报告

## 1. 本轮结论边界

v48.34 主实验的真实状态为 `pipeline_exit_code=30`。Balanced 与 Precision adaptation 均返回0，但流水线在 `model_inference_contract` 阶段退出；`certificate_executed=false`、`gate_evaluated=false`、`test_roots_read=false`。因此本轮不能分析 BARRIER-CROSSFIT 的算法优劣，也不能把任何训练期指标解释为 Natural gate 结果。

## 2. 确定性根因

控制器向 `tools/check_v48_32_model_contract.py` 传入：

```text
--expect-admission-prior-mode barrier_gated_slack
```

但旧检查器的 argparse 枚举只包含：

```text
risk_centered, benefit_only, safety_slack
```

因此检查器在加载checkpoint之前就以原始退出码2失败。顶层控制器将它规范化为RC=30。模型实现和已保存的Stage架构均支持 `barrier_gated_slack`；这不是算法数值发散、训练失败、显存问题或gate拒绝，而是版本化合同检查器落后于模型版本。

## 3. 不重训修复

v48.34.1增加专用的 `check_v48_34_model_contract.py`，并让主实验和消融控制器使用它。新增repair脚本只接受这一个已知失败签名，并在继续执行前检查：

- 原状态必须是pipeline RC=30且失败阶段为 `model_inference_contract`；
- Balanced/Precision adaptation退出码必须均为0；
- 两个最终checkpoint、训练完成元数据、Stage transfer、support contract和policy contract必须存在；
- checkpoint文件SHA必须与元数据一致；
- Stage transfer必须有效。

全部满足后只重跑模型/训练合同并从certificate calibration继续，不重跑Stage-1/Stage-2。上传日志中Balanced约使用1265.45秒factor训练和483.73秒identity训练；Precision约使用1081.52秒和866.81秒。若服务器checkpoint仍在，可避免约61.6 GPU分钟、约32.5分钟并行墙钟时间的重复训练。

## 4. Closed-loop、baseline与可视化代码审计

### 4.1 原代码中会误导结果的问题

1. adaptation-dev Contact可能默认读取 `validation_interactive`，但该数据集来自标准validation，可能导致目标完全匹配失败。
2. adaptation-dev数据可能继承 `BUCKET_SPLIT=test`，使 `evidence_adapt_dev` 目标无法匹配。
3. 旧进度对比使用selected/top-k在线teacher标签；这些标签不是物理闭环指标所必需，却会成为主要耗时。
4. 上传的Near/Contact baseline结果没有与OC-RAP使用相同target-key集合；现有汇总中的 `bucket_matched_rollouts=0` 不能作为直接数值对比。
5. 旧报告没有显式scalar control行，也没有统一的三regime展示表，容易只展示有利的方向化差值而隐藏绝对量级。
6. 视频选择允许缺失指标按0参与评分，且正例/失败例可能重复；同scene不同target time还可能覆盖同名视频。

### 4.2 v48.34.1修复后的合同

- adaptation-dev Near/Contact统一使用标准validation与 `evidence_adapt_dev` split；
- 执行前检查dataset split、source role、official ID/source index、TFRecord解析以及 `@N` 扫描上限；
- 所有方法在同一target-key集合、同一horizon上重跑，任何缺失、重复或scene-set不一致均fail-closed；
- Near/Contact两个variant分别使用GPU0/GPU1并行，外部方法两两并发；
- 进度闭环默认 `label_mode=fast`、在线teacher标签数为0，保留Waymax物理指标；
- 每个regime输出逐指标长表、宽表、compact CSV/Markdown展示表和full CSV；
- 报告包含scalar绝对值、方法绝对值、raw paired delta、方向化delta与paired bootstrap区间；
- critical scene同时输出正向toy example与failure case，且正例必须有实际干预、完整指标、正向综合变化、没有新增overlap/offroad/re-contact；
- 视频使用同一坐标范围并排显示Control/OC-RAP，包含SDC轨迹、动作、选择原因、TTC/clearance等信息，并生成JSON/CSV索引。

## 5. 三个regime的展示指标

- Safe：NUP、intervention、overlap/offroad、route progression、acceleration/jerk/yaw rate、clearance和TTC。
- Near-contact：NUP、intervention、TTC与terminal TTC、clearance与terminal clearance、critical TTC exposure、near-zero clearance exposure、clearance/TTC deficit AUC、overlap/offroad。
- Contact：NUP、intervention、post-contact terminal clearance、normalized free-space AUC、clearance gain/deficit AUC、escape、re-contact、stable-stop quality、overlap duration、overlap/offroad。

compact表适合给导师展示，full表保留全部指标。所有表明确标注为探索性进度材料；Natural gate未通过时不得作为正式部署或论文主结论。

## 6. 下一轮判定

- repair或clean run返回30：立即停止，不做算法分析，也不运行消融、shadow、test或stress。
- 返回20：pipeline与certificate有效，把完整结果交回进行下一轮算法分析；baseline/video可作为明确标注的进度材料。
- 返回0：只执行自动生成的 `NEXT_COMMANDS.txt` 进行正式后续实验。
