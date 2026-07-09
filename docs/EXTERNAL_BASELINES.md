# External baseline package for OC-RAP

This package adds external baseline adapters without changing the main OC-RAP
model or selector.  The new CLI commands are:

```bash
python -m ocrap.cli train-baseline \
  --config configs/external_baselines/route_bc_lite.yaml \
  --dataset "$TRAIN_MIX" --val-dataset "$VAL_MIX" \
  --baseline route_bc_lite --output runs/external_baselines/route_bc_lite

python -m ocrap.cli train-baseline \
  --config configs/external_baselines/gameformer_lite.yaml \
  --dataset "$TRAIN_MIX" --val-dataset "$VAL_MIX" \
  --baseline gameformer_lite --output runs/external_baselines/gameformer_lite

python -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/all_external_baselines.yaml \
  --dataset "$OCRAP_ROOT/test_near_contact" \
  --checkpoint runs/external_baselines/gameformer_lite/best.pt \
  --split test \
  --output runs/external_baselines/eval_near_contact_external_all.json \
  --baselines route_bc_lite,gameformer_lite,marc_lite,racp_lite,expected_risk_filter,cvar_risk_filter,dro_cvar_filter,postimpact_mpc_lite
```

`route_bc_lite` and `gameformer_lite` are learned candidate-set Transformer
baselines.  MARC/RACP/CVaR/DRO/post-impact MPC are optimization-style adapters
that use the same candidate prefixes, hidden roots, targeted futures, and
teacher margins saved in OC-RAP `.npz` samples.

## Intended regime use

- Safe: `route_bc_lite`, log replay / nominal preservation, plus OC-RAP.
- Near-contact: `gameformer_lite`, `marc_lite`, `racp_lite`, expected-risk,
  CVaR-risk, and DRO-CVaR filters.
- Contact: `marc_lite`, `racp_lite`, CVaR/DRO filters, and
  `postimpact_mpc_lite`.

The output JSON reports the same core quantities used by OC-RAP evaluation:
FRA, DRS, ODG, bounded NUP, intervention rate, selected-admitted rate, and
post-contact deployability.  Contact-oriented proxy fields are also included:
secondary collision rate, stable-stop success, yaw-rate violation proxy,
route-rejoin success, and mean harm proxy.
