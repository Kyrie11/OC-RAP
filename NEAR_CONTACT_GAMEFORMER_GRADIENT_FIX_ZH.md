# Near-contact GameFormer 首批次非有限梯度修复

## 现象

训练前向 loss 有限，但在 epoch 1 / batch 1 的 backward 后报：

```text
Non-finite gradient norm detected ...
```

这与验证 loss 为 NaN 导致缺少 `best.pt` 是不同阶段的问题。

## 两个根因

1. `GameFormerFutureEncoder` 原先令首时间步的前一位置等于首位置，因此每条轨迹首步必然得到 `dxy=(0,0)`，随后计算 `atan2(0,0)`。该点导数未定义，在部分 CUDA/PyTorch kernel 上可出现“前向有限、反向 NaN”。
2. 原训练器直接使用 `torch.nn.utils.clip_grad_norm_`。当梯度元素均为有限 float32、但幅值较大时，L2 范数内部平方求和仍可能溢出为 `Inf`。PyTorch 随后会用近似 0 的系数缩放梯度，旧代码再把它误判成“梯度本身非有限”。

## 修复

- 首步位移改为相对当前 ego 原点；静止位移在进入 `atan2` 前替换为常量方向 `(1,0)`，避免未定义梯度。
- 使用稳定范数计算先检测异常；仅在失败时逐参数定位真实 NaN/Inf，避免正常训练的额外同步开销。
- 使用按全局最大梯度缩放后的 float64 累加计算全局 L2 norm，再按原有 `grad_clip=5.0` 语义缩放。
- 若仍出现真实非有限梯度，日志会打印具体参数名、最大有限梯度和当前 batch 的输入数值范围。

正常数据和正常梯度下，新的 clipper 与原 global-L2 clipping 目标一致；它不会启用 AMP、TF32，也没有修改 batch size、候选数或训练轮数。
