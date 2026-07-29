#!/usr/bin/env python3
"""Layered PRISM-CC diagnostic: adaptation, certificate artifacts, Natural gate."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def finite(v: Any) -> float | None:
    try: x=float(v)
    except Exception: return None
    return x if math.isfinite(x) else None


def best_val(summary: dict[str, Any]) -> dict[str, Any]:
    ep=int(summary.get('best_epoch') or 0)
    for row in summary.get('history') or []:
        if int(row.get('epoch') or -1)==ep: return row.get('val') or {}
    return {}


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); report={'version':'v48.15-PRISM-CC','run':str(a.run),'variants':{}}
    for variant in ('balanced','precision'):
        root=a.run/'candidates'/variant; summary=load(root/'model_v48_trac_sr'/'train_summary.json'); val=best_val(summary)
        item={'adaptation':{},'certificate':{},'natural_gate_passed':False}
        for regime in ('near','contact'):
            item['adaptation'][regime]={
                'positive_admission_recall':finite(val.get(f'direct_positive_admission_recall_{regime}')),
                'harmful_switch_rate':finite(val.get(f'direct_harmful_switch_rate_{regime}')),
                'false_intervention_rate':finite(val.get(f'direct_false_intervention_rate_{regime}')),
                'certificate_risk':finite(val.get(f'direct_certificate_risk_mean_{regime}')),
                'positive_groups':finite(val.get(f'direct_positive_group_count_{regime}')),
            }
            risk=load(root/'calibration'/f'direct_value_risk_{regime}_v48.json'); verify=risk.get('verify') or {}
            item['certificate'][regime]={
                'artifact_present':bool(risk),
                'valid_for_deployment':bool(risk.get('valid_for_deployment',False)),
                'candidate_positive_auc':finite(risk.get('candidate_positive_auc')),
                'group_top1_correlation':finite(risk.get('unconstrained_group_top1_correlation')),
                'num_selected':finite(verify.get('num_selected',verify.get('selected'))),
                'precision':finite(verify.get('precision')),
                'precision_lcb90':finite(verify.get('precision_lcb90')),
                'positive_recall':finite(verify.get('positive_recall')),
                'harmful_selected_rate':finite(verify.get('harmful_selected_rate')),
                'harmful_selected_ucb90':finite(verify.get('harmful_selected_ucb90')),
                'teacher_advantage_selected_mean':finite(verify.get('teacher_advantage_selected_mean')),
            }
        item['natural_gate_passed']=all(item['certificate'][r]['valid_for_deployment'] for r in ('near','contact'))
        item['calibration_complete']=(root/'calibration'/'CERTIFICATE_CALIBRATION_COMPLETE.json').is_file()
        report['variants'][variant]=item
    report['next_commands_present']=(a.run/'NEXT_COMMANDS.txt').is_file()
    report['calibration_failed']=(a.run/'CALIBRATION_FAILED.json').is_file()
    report['gate_failed']=(a.run/'GATE_FAILED.json').is_file()
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
