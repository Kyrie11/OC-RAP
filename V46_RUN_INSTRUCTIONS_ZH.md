# OC-RAP v46 OC-RACE 运行指令

## 0. 先检查旧 clean base 是否真的全量解冻

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('runs/ocrap_v39_ocrac_clean_safe/model_v39_ocrac/train_summary.json')
d = json.load(p.open())
print('freeze_param_prefixes =', d.get('freeze_param_prefixes'))
if d.get('freeze_param_prefixes'):
    raise SystemExit('旧 clean base 仍冻结了模块，必须用 v46 重新训练。')
PY
```

旧 summary 非空并不奇怪：v45 的 shell 空字符串 bug 会导致该情况。

## 1. 当前数据先跑 development screening

当前 reports 没有独立 calibration split，Contact calibration/test scene 数也不足 final contract，因此先运行：

```bash
cd /path/to/OC-RAP-v46-OC-RACE-optimized

FINAL_RUN=0 \
RETRAIN_CLEAN_BASE=1 \
CLEAN_BASE_RUN=runs/ocrap_v39_ocrac_clean_safe_v46 \
RUN_HELDOUT_TEST=0 \
DATASET_DIAGNOSTICS_DIR=/path/to/extracted/reports \
bash run_v46_two_gpu_fast_commands.txt
```

如果不使用 diagnostics preflight，可去掉 `DATASET_DIAGNOSTICS_DIR`。development mode 会使用 val roots 做快速 calibration，但输出明确标记为非 publication evidence。

## 2. Final contract 的前置条件

需要先构建完全 scene-disjoint 的：

- `cal_near_contact`：建议至少 200 groups、100 scenes；
- `cal_contact`：建议至少 200 groups、100 scenes；
- 对应 held-out test 不参与 checkpoint/threshold 选择；
- calibration sample 的 `split_id` 应为 `calibration`；
- Contact 若仍全是 `post_contact_counterfactual`，论文必须限定 claim。

然后运行：

```bash
cd /path/to/OC-RAP-v46-OC-RACE-optimized

FINAL_RUN=1 \
RETRAIN_CLEAN_BASE=1 \
CLEAN_BASE_RUN=runs/ocrap_v39_ocrac_clean_safe_v46 \
RACE_CAL_NEAR_DATA=/data0/senzeyu2/dataset/OCRAP/cal_near_contact \
RACE_CAL_CONTACT_DATA=/data0/senzeyu2/dataset/OCRAP/cal_contact \
RUN_HELDOUT_TEST=1 \
DATASET_DIAGNOSTICS_DIR=/path/to/final/reports \
bash run_v46_two_gpu_fast_commands.txt
```

## 3. 只复用已验证 clean base

只有当以下文件存在，且其内容对应 `freeze_param_prefixes=[]` 的后置验证时，才可跳过 Stage-0：

```bash
runs/ocrap_v39_ocrac_clean_safe_v46/CLEAN_DATASET_RETRAINED
```

命令：

```bash
FINAL_RUN=0 \
RETRAIN_CLEAN_BASE=0 \
BASE_V39=runs/ocrap_v39_ocrac_clean_safe_v46 \
RUN_HELDOUT_TEST=0 \
bash run_v46_two_gpu_fast_commands.txt
```

## 4. Gate 通过标准

Development 与 final 都必须同时满足：

- Near、Contact 均有有限 opportunity/score thresholds；
- scene-disjoint fit/verify；
- 最低 verify selections 与 precision；
- harmful/all-groups UCB；
- harmful/selected-actions conditional UCB；
- pred/teacher advantage correlation；
- offline direct-value path 实际被使用；
- Safe intervention=0、NUP 不退化；
- staged 2→4→8 Waymax gate 无动作性/PCD/regret 回退。

Final 默认更严格：100 scenes、200 groups、25 verify selections、precision≥0.80、conditional harmful UCB90≤0.10。

## 5. 首轮结束后优先查看

```bash
cat runs/ocrap_v46_race_balanced/model_v46_race/train_summary.json
cat runs/ocrap_v46_race_precision/model_v46_race/train_summary.json

cat runs/ocrap_v46_race_balanced/calibration/direct_value_risk_near_v46.json
cat runs/ocrap_v46_race_balanced/calibration/direct_value_risk_contact_v46.json
cat runs/ocrap_v46_race_precision/calibration/direct_value_risk_near_v46.json
cat runs/ocrap_v46_race_precision/calibration/direct_value_risk_contact_v46.json
```

重点字段：

- `loss_direct_recovery_value_near/contact/worst`；
- `direct_router_accuracy`；
- `pred_teacher_advantage_correlation`；
- `supported_macro_ids`；
- `verify.num_selected`；
- `verify.challenge_precision`；
- `verify.harmful_selected_ucb90`；
- `valid_for_active_contract`。

若 Near 仍无有限 rule，先计算 oracle candidate frontier，不要放宽 gate。
