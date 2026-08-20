# OC-RAP v48.45.2 pipeline failure 与 v48.45.3 工程修复

## 结论

本轮上传的 source rebuild 与 A/B/C **不能用于算法归因**。主故障发生在 S0 shared recovery backbone 的模型构造阶段、epoch 1 之前；不是 OOM，也不是 Natural gate failure。A/B/C 的 RC=30 是 S0 主故障导致 source checkpoint 缺失后的二次 fail-closed 结果。

## 一手故障链

`ocrap_v48_45_source_rebuild_s7/logs/train_shared_recovery_backbone.log` 已完成数据扫描：pooled train 50114 个样本、validation 12250 个样本；train/development scene overlap audit 为 0。随后在 `OCRAPModel.__init__` 中立即出现：

```text
ValueError: could not convert string to float: 'None'
```

触发字段为 `direct_recovery_evidence_component_reliability`。

根因不是 recovery 算法，而是 CLI 空字符串配置契约：shell 中 `--set ...component_reliability=""` 被 YAML parser 解析为 Python `None`，`train.py` 又执行 `str(None)` 得到字面量 `"None"`，最终 reliability CSV parser 尝试 `float("None")`。

因此 S0 没有 `best.pt` / `TRAINING_COMPLETE.json`，S1 没有 Balanced / Precision `best.pt`，也没有 `SOURCE_REBUILD_COMPLETE.json`。随后 A/B/C 都在 `source_checkpoint_contract` 处以 RC=30 退出；三个 arm 的 adaptation exit code 都是 null，certificate 没执行，gate 没评估，test roots 没读取。

## v48.45.3 修复范围

这是纯工程 hotfix，v48.45 SOWR 算法和 v48.45.2 source-rebuild attribution protocol 不变：

1. `--set key=` 现在保留为空字符串；若确实需要 null，仍可显式写 `key=null` 或 `key=~`。
2. train/model/loss/inference 的 component reliability 路径都对 `None`、空字符串以及旧的 textual null 做兼容；未指定 reliability 仍使用历史默认的全 1，不改变非空 CSV 的数值语义。
3. source rebuild 在失败时新增 `SOURCE_REBUILD_FAILED.json`，记录真实失败阶段，避免以后只看到下游 checkpoint 缺失。
4. 新执行指令会先验证 empty-override code contract；若 source 未 seal，则删除旧的 incomplete source 并从头构建；source rebuild 非零或缺少 `SOURCE_REBUILD_COMPLETE.json` 时立即 RC=30，绝不继续启动 A/B/C/D。
5. 不改变 S0/S1 数据混合、loss、架构、ROCT、SOWR 开关、top-k、shared Natural rule、risk budget、certificate 或 gate。

## 下一轮归因边界

请丢弃本轮失败的 A/B/C。用 v48.45.3 从零建立一个新的 sealed source；只要 `SOURCE_REBUILD_COMPLETE.json`、`SOURCE_CHECKPOINT_CONTRACT_RECHECK.json`、`SOURCE_QUALITY_CONTRACT_RECHECK.json` 全部通过，再启动 A/B/C/D。此后 RC=0/20 才是可用于算法归因的结果，RC=30 仍是工程失败。
