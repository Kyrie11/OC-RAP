#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[]
    for task in sorted((args.root/'tasks').glob('*')):
        if not task.is_dir(): continue
        parts=task.name.rsplit('_',1); variant=parts[-1]; group=parts[0]
        row={'task':task.name,'group':group,'variant':variant,'complete':(task/'TASK_COMPLETE.json').is_file()}
        for bucket in ('near','contact'):
            p=task/'candidates'/variant/'calibration'/f'direct_value_risk_{bucket}_v48.json'
            try:d=json.load(open(p))
            except Exception as e: row[bucket]={'missing':str(e)}; continue
            row[bucket]={
                'valid':d.get('valid_for_deployment'),'candidate_auc':d.get('candidate_positive_auc'),
                'harm_auc':d.get('candidate_harm_auc'),'top1_corr':d.get('unconstrained_group_top1_correlation'),
                'proposal_oracle_best_hit':d.get('proposal_oracle_best_hit_rate_positive'),
                'proposal_any_positive_hit':d.get('proposal_any_positive_hit_rate_positive'),
                'verify':d.get('verify'),'warnings':d.get('warnings'),
            }
        rows.append(row)
    doc={'version':'v48.14-PRISM','tasks':rows,'num_tasks':len(rows),'complete_tasks':sum(bool(x['complete']) for x in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(doc,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
