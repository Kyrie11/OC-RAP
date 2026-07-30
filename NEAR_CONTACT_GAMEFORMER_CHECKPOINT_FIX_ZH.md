# Near-contact GameFormer `best.pt` 缺失修复说明

## 根因

旧版 `src/ocrap/external_baselines/train.py` 的保存顺序是：

1. 每个 epoch 无条件写入 `latest.pt`；
2. 仅当 `val_loss <= best_val` 时写入 `best.pt`。

`best_val` 初值为正无穷，因此只要验证损失是普通有限数值，第一轮一定会生成 `best.pt`。在训练命令正常返回、目录中存在 `latest.pt`、但始终没有 `best.pt` 的情况下，最主要原因是 `val_loss=NaN`：任何与 NaN 的大小比较都为 false。

GameFormer 数据适配中有若干处使用无参数的 `np.nan_to_num()`。NumPy 会把正负无穷转换成 float32 的最大有限值，而不是 0。该极大值进入局部坐标历史或 `prefix_traj` 后，会在 Gaussian trajectory NLL 的平方项中溢出，并可能使梯度和模型参数变为 NaN。旧训练器没有非有限值检查，因此仍保存了无效的 `latest.pt`，最后才由 shell 脚本提示缺少 `best.pt`。

另一个较低概率原因是进程恰好在 `latest.pt` 写完、`best.pt` 写入前被中断。修复版增加了原子保存和严格 checkpoint 校验，用于区分这两种情况。

## 修改内容

- GameFormer 历史、prefix trajectory 和 topology 特征中的 NaN/Inf 统一显式替换为 0，不再转换为 float32 最大值。
- trajectory loss 使用有限目标掩码和 `torch.where`，避免无效时间步通过 `NaN * 0` 污染损失。
- 每个训练 batch 在 backward 前同步检查所有 loss；发现 NaN/Inf 时，两张 GPU 一起停止，并输出 rank、阶段、epoch、batch 和具体 loss 名称。
- optimizer step 前检查梯度范数是否有限。
- 验证损失非有限时明确报错，不再静默结束。
- 第一轮有限验证结果必定先原子写入 `best.pt`，随后再写 `latest.pt`。
- `FORCE_RETRAIN_GAMEFORMER=true` 时删除旧的 `best.pt/latest.pt/train_summary.json`，防止旧失败产物被误认为本次训练结果。
- 闭环前验证 checkpoint 必需字段、全部模型权重有限、`val_loss` 有限，并验证 deployable input contract。
- 闭环结束后检查 8 个 near-contact 方法的 JSON 是否全部存在、可解析且包含非空场景和决策。

## 推荐运行命令

```bash
FORCE_RETRAIN_GAMEFORMER=true \
GAMEFORMER_TRAIN_GPUS=2 \
GAMEFORMER_GLOBAL_BATCH_SIZE=64 \
GAMEFORMER_NUM_WORKERS_TOTAL=8 \
CUDA_DEVICES=0,1 \
bash scripts/run_near_contact_external_baselines_2gpu_optimized.sh
```

运行成功时，应先看到类似：

```text
{"event":"checkpoint_valid", ..., "checkpoint":".../gameformer_lite/best.pt"}
```

随后完成离线评测和 8 个闭环任务，最后输出：

```text
{'event': 'near_contact_closed_loop_complete', 'methods': [...]}
```

## 检查旧 `latest.pt`

仅用于判断旧模型是否可用，不建议替代重新训练：

```bash
python tools/validate_external_checkpoint.py \
  --checkpoint runs/near_contact_external_baselines_optimized/gameformer_lite/best.pt \
  --promote-from runs/near_contact_external_baselines_optimized/gameformer_lite/latest.pt \
  --require-deployable-contract \
  --allow-promotion
```

工具只有在 `latest.pt` 的权重、验证损失和输入 contract 全部有效时才会生成 `best.pt`。若提示 `val_loss is not finite` 或 `non-finite model tensors`，必须重新训练，不能用于正式闭环结果。

## 验证状态

- 完整单元测试：178 passed。
- 额外 CPU smoke test：GameFormer 训练 1 epoch 后同时生成 `best.pt` 和 `latest.pt`，checkpoint 校验通过。
- 当前环境没有你的 `/data0/...` 数据和双 GPU，因此正式 30 epoch DDP 与 Waymax 闭环需在训练服务器运行。
