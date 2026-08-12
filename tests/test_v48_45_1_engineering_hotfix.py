from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_checkpoint_contract_valid_and_missing(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "ocrap_v48_13_terra_proxy_4801"
    for variant in ("balanced", "precision"):
        p = source / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((variant + "-checkpoint").encode())
    out = tmp_path / "contract.json"
    ok = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_source_checkpoint_contract.py"),
            "--source-run",
            str(source),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    doc = json.loads(out.read_text())
    assert doc["valid"] is True
    assert doc["checks"]["balanced"]["sha256"]
    assert doc["checks"]["precision"]["sha256"]
    assert doc["test_roots_read"] is False

    (source / "candidates" / "precision" / "model_v48_trac_sr" / "best.pt").unlink()
    bad = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_source_checkpoint_contract.py"),
            "--source-run",
            str(source),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode != 0
    doc = json.loads(out.read_text())
    assert doc["valid"] is False
    assert doc["checks"]["balanced"]["exists"] is True
    assert doc["checks"]["precision"]["exists"] is False


def test_dedicated_missing_source_fails_before_dataset_and_adaptation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    missing_source = tmp_path / "persistent-runs" / "ocrap_v48_13_terra_proxy_4801"
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "OCRAP_REPO": str(ROOT),
            "OUTPUTDIR": str(run),
            "SOURCE_RUN": str(missing_source),
            "PROTOCOL_ROOT": str(tmp_path / "missing-protocol"),
            "CAL_SAFE": str(tmp_path / "missing-safe"),
            "RESUME_AFTER_ADAPTATION": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 30, completed.stdout + completed.stderr
    failed = json.loads((run / "PIPELINE_FAILED.json").read_text())
    contract = json.loads((run / "SOURCE_CHECKPOINT_CONTRACT.json").read_text())
    assert failed["stage"] == "source_checkpoint_contract"
    assert failed["certificate_executed"] is False
    assert failed["gate_evaluated"] is False
    assert contract["valid"] is False
    assert not (run / "DATASET_ROOT_CONTRACT.json").exists()
    assert not (run / "candidates").exists()


def _make_fake_parallel_repo(tmp_path: Path, rc_map: dict[str, int]) -> Path:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "run_v48_45_sowr_2x2_parallel.sh", scripts / "run_v48_45_sowr_2x2_parallel.sh")
    cases = "\n".join(f"  {arm}) exit {rc} ;;" for arm, rc in rc_map.items())
    (scripts / "run_v48_45_sowr_ablation_arm.sh").write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\ncase \"$1\" in\n" + cases + "\n  *) exit 2 ;;\nesac\n",
        encoding="utf-8",
    )
    os.chmod(scripts / "run_v48_45_sowr_ablation_arm.sh", 0o755)
    return repo


def test_parallel_launcher_accepts_rc20_as_valid_ablation(tmp_path: Path) -> None:
    repo = _make_fake_parallel_repo(tmp_path, {"A": 20, "B": 20, "C": 20, "D": 20})
    out = tmp_path / "runs"
    completed = subprocess.run(
        ["bash", str(repo / "scripts" / "run_v48_45_sowr_2x2_parallel.sh")],
        cwd=repo,
        env={**os.environ, "OCRAP_REPO": str(repo), "BASE_OUT": str(out), "MAX_PARALLEL_ARMS": "2"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("valid Natural-gate failure") == 4
    status = json.loads((out / "ocrap_v48_45_sowr_parallel_status.json").read_text())
    assert status["engineering_failed"] is False
    assert status["any_natural_gate_failure"] is True
    assert all(row["classification"] == "valid_natural_gate_failure" for row in status["arms"].values())


def test_parallel_launcher_rejects_engineering_failure(tmp_path: Path) -> None:
    repo = _make_fake_parallel_repo(tmp_path, {"A": 20, "B": 30, "C": 0, "D": 20})
    out = tmp_path / "runs"
    completed = subprocess.run(
        ["bash", str(repo / "scripts" / "run_v48_45_sowr_2x2_parallel.sh")],
        cwd=repo,
        env={**os.environ, "OCRAP_REPO": str(repo), "BASE_OUT": str(out), "MAX_PARALLEL_ARMS": "2"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "arm B ENGINEERING FAILED: RC=30" in completed.stderr
    status = json.loads((out / "ocrap_v48_45_sowr_parallel_status.json").read_text())
    assert status["engineering_failed"] is True
    assert status["arms"]["B"]["classification"] == "engineering_failure"


def test_v4845_arm_resolves_rebuilt_source_before_missing_historical_source() -> None:
    text = (ROOT / "scripts" / "run_v48_45_sowr_ablation_arm.sh").read_text()
    assert 'SOURCE_RUN_BASENAME="${V4845_SOURCE_RUN_BASENAME:-ocrap_v48_13_terra_proxy_4801}"' in text
    assert 'REBUILT_SOURCE_BASENAME="${V4845_REBUILT_SOURCE_BASENAME:-ocrap_v48_45_source_rebuild_s7}"' in text
    rebuilt = '$BASE_OUT/$REBUILT_SOURCE_BASENAME/SOURCE_REBUILD_COMPLETE.json'
    historical = '$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt'
    assert rebuilt in text and historical in text
    assert text.index(rebuilt) < text.index(historical)
    assert 'export SOURCE_RUN' in text

def test_v4845_sowr_controls_explicitly_cross_variant_process_boundary() -> None:
    text = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    assert 'V4845_SOWR_MARGIN_WITNESS="${V4845_SOWR_MARGIN_WITNESS:-0}"' in text
    assert 'V4845_SOWR_OBS_KERNEL="${V4845_SOWR_OBS_KERNEL:-0}"' in text
    assert 'SOWR_LR="${SOWR_LR:-0.00005}"' in text
    assert text.index("check_v48_36_source_checkpoint_contract.py") < text.index("check_v48_36_dataset_root_contract.py")


def test_sowr_env_command_has_no_backslash_comment_break():
    text = (ROOT / "scripts" / "adapt_ocrap_v48_45_sowr_stage.sh").read_text()
    # A comment immediately after a backslash-continued assignment silently
    # terminates the env-assignment command after line-continuation removal.
    assert not re.search(r"\\\n\s*#", text)
    assert "SKIP_POST_TRAIN_CALIBRATION=1 \\" in text


def test_source_contract_enforces_rebuild_manifest_hashes(tmp_path: Path) -> None:
    import hashlib
    source = tmp_path / "source"
    variants = {}
    for variant in ("balanced", "precision"):
        p = source / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = (variant + "-rebuilt").encode()
        p.write_bytes(payload)
        variants[variant] = {"sha256": hashlib.sha256(payload).hexdigest()}
    (source / "SOURCE_REBUILD_COMPLETE.json").write_text(json.dumps({"variants": variants}) + "\n")
    out = tmp_path / "contract.json"
    ok = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_36_source_checkpoint_contract.py"),
         "--source-run", str(source), "--output", str(out)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    doc = json.loads(out.read_text())
    assert doc["source_rebuild_manifest_present"] is True
    assert doc["checks"]["balanced"]["manifest_hash_match"] is True

    p = source / "candidates" / "balanced" / "model_v48_trac_sr" / "best.pt"
    p.write_bytes(b"mutated")
    bad = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_36_source_checkpoint_contract.py"),
         "--source-run", str(source), "--output", str(out)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert bad.returncode != 0
    doc = json.loads(out.read_text())
    assert doc["valid"] is False
    assert doc["checks"]["balanced"]["manifest_hash_match"] is False


def test_scratch_init_is_explicit_and_normal_training_remains_fail_closed() -> None:
    text = (ROOT / "scripts" / "train_ocrap_v48_trac_sr.sh").read_text()
    assert 'ALLOW_SCRATCH_INIT="${ALLOW_SCRATCH_INIT:-0}"' in text
    assert 'empty INIT_CKPT requires ALLOW_SCRATCH_INIT=1' in text
    assert '--set training.init_checkpoint="$INIT_CKPT"' in text
    assert '--set training.encoder_anchor_weight="$ENCODER_ANCHOR_WEIGHT"' in text


def test_source_rebuild_freezes_one_common_backbone_before_ablation_source_heads() -> None:
    text = (ROOT / "scripts" / "rebuild_v48_45_shared_source.sh").read_text()
    assert "ALLOW_SCRATCH_INIT=1 INIT_CKPT=" in text
    assert "DIRECT_ONLY_FAST_PATH=false DIRECT_VALUE_WEIGHT=0" in text
    assert "LOSS_MARGIN=2.0" in text and "LOSS_OBS=1.0" in text and "LOSS_OPTION_Q=0.5" in text
    assert 'BACKBONE_TRAIN_MIX="$TRAIN_SAFE,$TRAIN_NEAR,$TRAIN_CONTACT"' in text
    assert 'BACKBONE_VAL_MIX="$DEV_SAFE,$DEV_NEAR,$DEV_CONTACT"' in text
    assert 'POLICY_TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT"' in text
    assert "train_source_variant balanced" in text and "train_source_variant precision" in text
    assert 'INIT_CKPT="$BACKBONE_CKPT"' in text
    assert "TRAINABLE_PARAM_PREFIXES='direct_value_heads,direct_preference_set_ranker,direct_delta_adapters'" in text
    assert "STRICT_INIT_PREFIXES='direct_preference_set_ranker,direct_delta_adapters'" in text
    assert "SET_TOURNAMENT_ENABLED=true" in text
    assert "DELTA_REGIME_EXPERTS=true" in text
    assert 'BACKBONE_DONE="$BACKBONE_RUN/TRAINING_COMPLETE.json"' in text
    assert "best.pt exists without TRAINING_COMPLETE.json" in text
    assert "SOURCE_REBUILD_COMPLETE.json" in text
    assert "test_roots_read':False" in text


def test_v48453_source_rebuild_empty_string_config_contract() -> None:
    from ocrap.config.overrides import parse_cli_overrides
    cfg = parse_cli_overrides([
        "training.init_checkpoint=",
        "training.freeze_param_prefixes=",
        "training.trainable_param_prefixes=",
        "model.direct_recovery_evidence_component_reliability=",
        "training.direct_value_ordinal_evidence_component_reliability=",
    ])
    assert cfg["training"]["init_checkpoint"] == ""
    assert cfg["training"]["freeze_param_prefixes"] == ""
    assert cfg["training"]["trainable_param_prefixes"] == ""
    assert cfg["model"]["direct_recovery_evidence_component_reliability"] == ""
    assert cfg["training"]["direct_value_ordinal_evidence_component_reliability"] == ""


def test_v48453_model_accepts_unspecified_reliability_without_string_none() -> None:
    from ocrap.models.encoders import FlatFeatureLayout
    from ocrap.models.ocrap import OCRAPModel

    common = dict(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=3,
        num_options=4,
        d_model=12,
        d_obs=6,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=3,
        dropout=0.0,
        direct_recovery_delta_head=True,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_delta_regime_experts=True,
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=3,
    )
    for value in (None, "", "None", "null", "~"):
        model = OCRAPModel(
            **common,
            direct_recovery_evidence_component_reliability=value,
        )
        assert model.direct_recovery_evidence_component_reliability == (1.0, 1.0, 1.0)


def test_v48453_source_rebuild_writes_failure_stage_marker() -> None:
    text = (ROOT / "scripts" / "rebuild_v48_45_shared_source.sh").read_text()
    assert 'SOURCE_REBUILD_STAGE="S0_shared_recovery_backbone"' in text
    assert 'SOURCE_REBUILD_STAGE="S1_source_policy_heads"' in text
    assert 'SOURCE_REBUILD_FAILED.json' in text
    assert 'implementation_version": "v48.45.4-s1-nounset-hotfix"' in text


def test_v48453_operator_commands_fail_closed_before_ablation() -> None:
    text = (ROOT / "OC-RAP-v48.45.3-source-rebuild-and-SOWR-run-commands-ZH.txt").read_text()
    assert "V48.45.3 EMPTY-OVERRIDE CONTRACT PASS" in text
    assert "source_rc=${PIPESTATUS[0]}" in text
    assert "SOURCE REBUILD ENGINEERING FAILURE" in text
    assert '[[ -s "$SOURCE_RUN/SOURCE_REBUILD_COMPLETE.json" ]]' in text
    assert text.index("SOURCE REBUILD ENGINEERING FAILURE") < text.index("run_v48_45_sowr_2x2_parallel.sh")
    assert 'if [[ -d "$SOURCE_RUN" && ! -f "$SOURCE_RUN/SOURCE_REBUILD_COMPLETE.json" ]]' in text



def test_v48454_s1_local_initialization_is_nounset_safe() -> None:
    text = (ROOT / "scripts" / "rebuild_v48_45_shared_source.sh").read_text()
    assert "local variant gpu run" in text
    assert 'variant="$1"' in text
    assert 'gpu="$2"' in text
    assert 'run="$SOURCE_OUT/candidates/$variant"' in text
    assert 'local variant="$1" gpu="$2" run="$SOURCE_OUT/candidates/$variant"' not in text
    assert "S1_SOURCE_POLICY_STATUS.json" in text
    assert "v48.45.4-s1-nounset-hotfix" in text


def test_v48454_no_same_local_command_self_dependency_under_nounset() -> None:
    """Catch the exact Bash class that caused uploaded RC=30.

    In `local a=... b=...$a`, every RHS is expanded before the local builtin
    assigns `a`; with `set -u` this aborts the function if `a` was not already set.
    Inspect only each local builtin up to its command separator to avoid flagging
    ordinary later references such as `local a=$1; echo $a`.
    """
    import shlex

    problems: list[tuple[str, int, str, str]] = []
    for path in sorted((ROOT / "scripts").glob("*.sh")):
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "local " not in line or line.lstrip().startswith("#"):
                continue
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
            i = 0
            while i < len(tokens):
                if tokens[i] != "local":
                    i += 1
                    continue
                j = i + 1
                segment: list[str] = []
                while j < len(tokens) and tokens[j] != ";":
                    segment.append(tokens[j])
                    j += 1
                assignments: list[tuple[str, str]] = []
                for token in segment:
                    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", token, flags=re.S)
                    if m:
                        assignments.append((m.group(1), m.group(2)))
                local_names = {name for name, _ in assignments}
                for target, rhs in assignments:
                    for ref in local_names:
                        if re.search(rf"\${re.escape(ref)}\b", rhs) or re.search(rf"\${{{re.escape(ref)}}}", rhs):
                            problems.append((str(path.relative_to(ROOT)), lineno, target, ref))
                i = max(j + 1, i + 1)
    assert problems == []


def test_v48454_parallel_default_uses_two_training_processes_total() -> None:
    text = (ROOT / "scripts" / "run_v48_45_sowr_2x2_parallel.sh").read_text()
    assert 'MAX_PARALLEL_ARMS="${MAX_PARALLEL_ARMS:-1}"' in text
    assert "1 recommended: each arm already uses GPU0/GPU1 in parallel" in text


def test_v48454_operator_commands_preserve_reusable_s0_and_compare_2x2() -> None:
    text = (ROOT / "OC-RAP-v48.45.4-source-rebuild-and-SOWR-run-commands-ZH.txt").read_text()
    assert not re.search(r'^\s*rm -rf \"\$SOURCE_RUN\"\s*$', text, flags=re.M)
    assert "reusable S0 detected; S0 WILL NOT be retrained" in text
    assert "S1_SOURCE_POLICY_STATUS.json" in text
    assert "MAX_PARALLEL_ARMS=1" in text
    assert "compare_v48_45_sowr_2x2.py" in text
    assert text.index("SOURCE HASH + QUALITY CONTRACT PASS") < text.index("run_v48_45_sowr_2x2_parallel.sh")
