#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrap.external_baselines.provenance import registry_dict


def esc(x: object) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description="Write an auditable paper/code/fidelity manifest for OC-RAP external baselines.")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--regimes", default="safe,near,contact")
    args = ap.parse_args()
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    rows = registry_dict(regimes)
    doc = {
        "schema_version": 1,
        "regimes": regimes,
        "reporting_rule": "Only implementation_kind/fidelity in this manifest may be used in tables. Native adaptations are not author-official reproductions.",
        "baselines": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# External baseline fidelity audit", "",
        "| Regime | Reporting name | Paper | Official code | Implementation | Fidelity | Retained core | Known gaps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        paper = r["paper_title"] + (f" ({r['paper_year']})" if r.get("paper_year") else "")
        if r.get("paper_url"):
            paper = f"[{paper}]({r['paper_url']})"
        code = f"[repository]({r['official_code_url']})" if r.get("official_code_url") else "not identified / not applicable"
        lines.append("| " + " | ".join([
            esc(", ".join(r["regimes"])), esc(r["reporting_name"]), esc(paper), esc(code),
            esc(r["implementation_kind"]), esc(r["fidelity"]), esc("; ".join(r["core_retained"])), esc("; ".join(r["known_gaps"]) or "none"),
        ]) + " |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "external_baseline_fidelity_audit", "count": len(rows), "json": str(args.output_json), "markdown": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
